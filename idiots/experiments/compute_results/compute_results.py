import json
import os
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import neural_tangents as nt
import numpy as np
import optax
import orbax.checkpoint as ocp
from einops import rearrange
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Kernel,
    NormalizedKernelMixin,
    StationaryKernelMixin,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import trange

from idiots.dataset.dataloader import DataLoader
from idiots.experiments.classification.training import restore as mnist_restore
from idiots.experiments.gradient_flow.init import restore as gradient_flow_restore
from idiots.experiments.grokking.training import loss_fn
from idiots.experiments.grokking.training import restore as algorithmic_restore
from idiots.utils import metrics

warnings.filterwarnings("ignore")


def eval_checkpoint(apply_fn, params, config, train_loader, test_loader):
    """Compute training/test accuracy/loss for each timestep"""

    @jax.jit
    def eval_step(batch) -> dict:
        y_pred = apply_fn(params, batch["x"])
        losses = loss_fn(y_pred, batch["y"], variant=config.loss_variant)
        acc = jnp.argmax(y_pred, axis=-1) == batch["y"]
        return {"eval_loss": losses, "eval_accuracy": acc}

    def eval_loss_acc(loader):
        for batch in loader:
            logs = eval_step(batch)
            metrics.log(**logs)
        [losses, accuracies] = metrics.collect("eval_loss", "eval_accuracy")
        loss = jnp.concatenate(losses).mean().item()
        acc = jnp.concatenate(accuracies).mean().item()
        return loss, acc

    train_loss, train_acc = eval_loss_acc(train_loader)
    test_loss, test_acc = eval_loss_acc(test_loader)

    return train_loss, train_acc, test_loss, test_acc


def generate_analysis_dataset(
    X_test,
    Y_test,
    num_analysis_training_samples,
    num_analysis_test_samples,
    experiment_type,
):
    analysis_X_train, X_test, analysis_Y_train, Y_test = train_test_split(
        X_test, Y_test, train_size=num_analysis_training_samples, stratify=Y_test
    )
    analysis_X_test, X_test, analysis_Y_test, Y_test = train_test_split(
        X_test, Y_test, train_size=num_analysis_test_samples, stratify=Y_test
    )

    if experiment_type == "mnist" or experiment_type == "gradient_flow_mnist":
        analysis_X_train = rearrange(analysis_X_train, "b h w -> b (h w)")
        analysis_X_test = rearrange(analysis_X_test, "b h w -> b (h w)")

    if experiment_type == "cifar":
        analysis_X_train = rearrange(analysis_X_train, "b h w c -> b (h w c)")
        analysis_X_test = rearrange(analysis_X_test, "b h w c -> b (h w c)")

    return (analysis_X_train, analysis_Y_train, analysis_X_test, analysis_Y_test)


def compute_kernel_trace_axes_fn(transformer_state_apply_fn, batch_size):
    # Return a batched kernel function where trace_axes=() [for calculating DOTS]
    kernel_fn_trace_axes = nt.empirical_ntk_fn(
        transformer_state_apply_fn,
        vmap_axes=0,
        trace_axes=(),
        implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,
    )
    return nt.batch(kernel_fn_trace_axes, batch_size=batch_size)


def compute_kernel_fn(transformer_state_apply_fn, batch_size):
    # Return a batched kernel function where trace_axes is not defined [for computing everything other than DOTS]
    kernel_fn = nt.empirical_ntk_fn(
        transformer_state_apply_fn,
        vmap_axes=0,
        implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,
    )
    return nt.batch(kernel_fn, batch_size=batch_size)


@jax.jit
def compute_dots(kernel_trace_axes):
    """Compute DOTS on the kernel_trace_axes matrix"""
    ntk_flat = rearrange(kernel_trace_axes, "b1 b2 d1 d2 -> (b1 d1) (b2 d2)")
    S = jnp.linalg.svd(ntk_flat, full_matrices=False, compute_uv=False)
    m, n = ntk_flat.shape

    tol_1 = S.max(-1) * np.max([m, n]).astype(S.dtype) * jnp.finfo(S.dtype).eps
    tol_2 = 0.5 * S.max(-1) * jnp.finfo(S.dtype).eps * jnp.sqrt(m + n + 1)
    dots_1 = jnp.sum(S > tol_1)
    dots_2 = jnp.sum(S > tol_2)

    s_dist = S / jnp.sum(S)
    dots_3 = jnp.exp(-jnp.sum(s_dist * jnp.log(s_dist)))

    return dots_1, dots_2, dots_3, S


# Given the a custom kernel and training/test data, compute the accuracy of an SVM
def compute_svm_accuracy(
    custom_kernel_fn,
    analysis_X_train,
    analysis_Y_train,
    analysis_X_test,
    analysis_Y_test,
):
    svc = SVC(kernel=custom_kernel_fn)
    svc.fit(analysis_X_train, analysis_Y_train)

    train_accuracy = svc.score(analysis_X_train, analysis_Y_train)
    test_accuracy = svc.score(analysis_X_test, analysis_Y_test)
    return train_accuracy, test_accuracy


# Given the a custom kernel and training/test data, compute the accuracy of a Gaussian Process
def compute_gp_accuracy(
    custom_kernel_fn,
    analysis_X_train,
    analysis_Y_train,
    analysis_X_test,
    analysis_Y_test,
    num_y_classes,
):
    analysis_Y_train_one_hot = jax.nn.one_hot(analysis_Y_train, num_y_classes)

    class CustomKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
        """GP kernel object (for compatability with sklearn.gaussian_proccess)"""

        def __init__(self) -> None:
            super().__init__()

        def __call__(self, X, Y=None, eval_gradient=False):
            kernel = np.array(custom_kernel_fn(X, Y))

            if eval_gradient:
                return kernel, np.zeros(X.shape)
            else:
                return kernel

    custom_gp_kernel = CustomKernel()
    gaussian_process_classifier = GaussianProcessRegressor(
        kernel=custom_gp_kernel, alpha=1e-7
    )
    gaussian_process_classifier.fit(analysis_X_train, analysis_Y_train_one_hot)

    predictions = gaussian_process_classifier.predict(analysis_X_test).argmax(axis=-1)
    accuracy = accuracy_score(analysis_Y_test, predictions)
    return accuracy


@jax.jit
def compute_kernel_alignment(kernel, analysis_Y_test):
    """Compute the kernel alignment metric.

    Source: Shan 2022: A Theory of Neural Tangent Kernel Alignment and Its Influence on Training
    We use the "traced" kernel here.
    """
    Y = jax.nn.one_hot(analysis_Y_test, kernel.shape[-1])
    YYT = Y @ Y.T
    K = jnp.trace(kernel, axis1=-2, axis2=-1) / kernel.shape[-1]
    kernel_alignment = jnp.sum(YYT * K) / (jnp.linalg.norm(K) * jnp.linalg.norm(YYT))
    return kernel_alignment


def restore_checkpoint(checkpoint_dir: Path, experiment_type: str):
    if experiment_type == "algorithmic":
        mngr, config, state, ds_train, ds_test = algorithmic_restore(checkpoint_dir, 0)

        def get_params(step: int):
            if step == 0:
                return state.params
            else:
                return mngr.restore(step, args=ocp.args.StandardRestore(state)).params

        return config, state.apply_fn, get_params, ds_train, ds_test, mngr.all_steps()

    elif experiment_type == "mnist" or experiment_type == "cifar":
        mngr, config, state, ds_train, ds_test = mnist_restore(checkpoint_dir, 0)

        def get_params(step: int):
            if step == 0:
                return state.params
            else:
                return mngr.restore(step, args=ocp.args.StandardRestore(state)).params

        return config, state.apply_fn, get_params, ds_train, ds_test, mngr.all_steps()

    elif experiment_type.startswith("gradient_flow_"):
        apply_fn, init_params, ds_train, ds_test, mngr, config = gradient_flow_restore(
            checkpoint_dir
        )

        def get_params(step: int):
            if step == 0:
                return init_params
            else:
                return mngr.restore(step, args=ocp.args.StandardRestore(init_params))

        return config, apply_fn, get_params, ds_train, ds_test, mngr.all_steps()
    else:
        raise ValueError(f"Experiment type {experiment_type} not valid.")


def compute_results(experiments, add_kernel, kernel_batch_size=32):
    """
    experiments: (
        experiment_json_file_name: e.g. "mnist-100" to save file as "mnist-100.json",
        experiment_checkpoint_path: e.g. "checkpoints/mnist-100/checkpoints",
        experiment_type: either "algorithmic" for modular division or S5, or "mnist" for mnist,
        step_distance: distance between checkpoints you want to analyse,
        total_steps: value of the highest checkpoint you want to analyse,
        kernel_samples: number of samples used to compute the DOTS and kernel alignment,
        num_analysis_training_samples: number of training samples used in the remaining analysis: when fitting the SVM and GP,
        num_analysis_test_samples: number of test samples used in the remaining analysis: when fitting the SVM and GP,
    )
    add_kernel: whether the kernel should be computed and added to the log file (can take up a large amount of space)
    kernel_batch_size: batch size when computing the kernels using nt.batch
    """
    for (
        experiment_json_file_name,
        experiment_checkpoint_path,
        experiment_type,
        step_distance,
        total_steps,
        kernel_samples,
        num_analysis_training_samples,
        num_analysis_test_samples,
    ) in experiments:
        print(f"Experiment: {experiment_json_file_name}")

        config, apply_fn, get_params, ds_train, ds_test, _ = restore_checkpoint(
            Path(experiment_checkpoint_path), experiment_type
        )
        if len(ds_train) > len(ds_test):
            ds_train = ds_train.select(range(len(ds_test)))
        train_loader = DataLoader(ds_train, 256)
        test_loader = DataLoader(ds_test, 256)

        init_weight_norm = optax.global_norm(get_params(0)).item()

        kernel_fn = compute_kernel_fn(apply_fn, kernel_batch_size)
        kernel_trace_axes_fn = compute_kernel_trace_axes_fn(apply_fn, kernel_batch_size)

        # kernel dataset is used for computing the kernels used in DOTS and kernel alignment
        # analysis datasets are used for the remaining analysis: SVM, GP
        X_test, Y_test = jnp.array(ds_test["x"]), jnp.array(ds_test["y"])
        (
            analysis_X_train,
            analysis_Y_train,
            analysis_X_test,
            analysis_Y_test,
        ) = generate_analysis_dataset(
            X_test,
            Y_test,
            num_analysis_training_samples,
            num_analysis_test_samples,
            experiment_type,
        )

        all_metrics = []
        for step in trange(0, total_steps + 1, step_distance):
            params = get_params(step)
            train_loss, train_acc, test_loss, test_acc = eval_checkpoint(
                apply_fn, params, config, train_loader, test_loader
            )
            svm_train_acc, svm_test_acc = compute_svm_accuracy(
                lambda x1, x2: kernel_fn(x1, x2, params),
                analysis_X_train,
                analysis_Y_train,
                analysis_X_test,
                analysis_Y_test,
            )
            gp_acc = compute_gp_accuracy(
                lambda x1, x2: kernel_fn(x1, x2, params),
                analysis_X_train,
                analysis_Y_train,
                analysis_X_test,
                analysis_Y_test,
                ds_train.features["y"].num_classes,
            )

            kernel_X = ds_test["x"][:kernel_samples]
            kernel_Y = ds_test["y"][:kernel_samples]
            kernel = kernel_trace_axes_fn(kernel_X, None, params)
            dots_1, dots_2, dots_3, S = compute_dots(kernel)
            kernel_alignment = compute_kernel_alignment(kernel, kernel_Y)

            all_metrics.append(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "training_acc": train_acc,
                    "test_acc": test_acc,
                    "svm_train_accuracy": svm_train_acc,
                    "svm_accuracy": svm_test_acc,
                    "gp_accuracy": gp_acc,
                    "dots": dots_1.item(),
                    "dots_2": dots_2.item(),
                    "dots_3": dots_3.item(),
                    "eigenvalues": S.tolist(),
                    "kernel_alignment": kernel_alignment.item(),
                    "kernel": kernel.tolist() if add_kernel else None,
                    "weight_norm": optax.global_norm(params).item(),
                    "relative_weight_norm": optax.global_norm(params).item()
                    / init_weight_norm,
                }
            )

        out_file = Path(logs_base_path, "results", f"{experiment_json_file_name}.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            # Convert list[dict] -> dict[list]
            metrics = {k: [d[k] for d in all_metrics] for k in all_metrics[0]}
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # logs_base_path = Path("/home/dm894/idiots/logs/")
    logs_base_path = Path("logs")
    # logs_base_path = Path("../../../logs")
    # logs_base_path = Path("/home/dc755/idiots/logs/")

    experiments = [
        # (
        #     "cifar-10",
        #     logs_base_path / "checkpoints/mnist/exp52/checkpoints",
        #     "cifar",
        #     1_000,
        #     100_000,
        #     512,
        #     128,
        #     512,
        # ),
        # (
        #     "mnist-gd-grokking-2",
        #     logs_base_path / "checkpoints/mnist_gd_grokking/exp55/checkpoints",
        #     "mnist",
        #     1_000,
        #     100_000,
        #     512,
        #     64,
        #     512,
        # ),
        # (
        #     "mnist-adamw",
        #     logs_base_path / "checkpoints/mnist/mnist_adamw/checkpoints",
        #     "mnist",
        #     1_000,
        #     50_000,
        #     512,
        #     64,
        #     256,
        # ),
        # (
        #     "mnist-grokking-slower-2",
        #     logs_base_path / "checkpoints/mnist_grokking_slower/exp26/checkpoints",
        #     "mnist",
        #     1_000,
        #     100_000,
        #     512,
        #     64,
        #     512,
        # ),
        # (
        #     "mnist-gf",
        #     logs_base_path / "checkpoints/gradient_flow/exp34/checkpoints",
        #     "gradient_flow_mnist",
        #     50,
        #     1000,
        #     512,
        #     64,
        #     512,
        # ),
        # (
        #     "mnist-adamw",
        #     logs_base_path / "checkpoints/mnist/exp66/checkpoints",
        #     "mnist",
        #     5000,
        #     100_000,
        #     512,
        #     64,
        #     512,
        # ),
        # (
        #     "mnist-sgd",
        #     logs_base_path / "checkpoints/mnist/exp70/checkpoints",
        #     "mnist",
        #     5000,
        #     100_000,
        #     512,
        #     64,
        #     512,
        # ),
        # (
        #     "mnist-sgd-16",
        #     logs_base_path / "checkpoints/mnist/exp72/checkpoints",
        #     "mnist",
        #     5000,
        #     100_000,
        #     512,
        #     64,
        #     512,
        # ),
        # (
        #     "mnist-sgd-8",
        #     logs_base_path / "checkpoints/mnist/exp71/checkpoints",
        #     "mnist",
        #     5000,
        #     100_000,
        #     512,
        #     64,
        #     512,
        # ),
        # (
        #     "mnist-adamw-64",
        #     logs_base_path / "checkpoints/mnist/exp75/checkpoints",
        #     "mnist",
        #     5000,
        #     100_000,
        #     512,
        #     64,
        #     512,
        # ),
        # (
        #     "mnist-adamw-256",
        #     logs_base_path / "checkpoints/mnist/exp76/checkpoints",
        #     "mnist",
        #     5000,
        #     100_000,
        #     512,
        #     64,
        #     512,
        # ),
        # (
        #     "addition-gf",
        #     logs_base_path / "checkpoints/gradient_flow/exp36/checkpoints",
        #     "gradient_flow_algorithmic",
        #     120_000,
        #     1_500_000,
        #     64,
        #     256,
        #     256,
        # ),
        # (
        #     "addition-adamw",
        #     logs_base_path / "checkpoints/grokking/exp100/checkpoints",
        #     "algorithmic",
        #     5_000,
        #     50_000,
        #     64,
        #     256,
        #     256,
        # ),
        # (
        #     "division-adamw-mlp",
        #     logs_base_path / "checkpoints/grokking/exp131/checkpoints",
        #     "algorithmic",
        #     1_000,
        #     50_000,
        #     128,
        #     128,
        #     256,
        # ),
        # (
        #     "division-gf-mlp",
        #     logs_base_path / "checkpoints/gradient_flow/exp39/checkpoints",
        #     "gradient_flow_algorithmic",
        #     12_000,
        #     600_000,
        #     128,
        #     128,
        #     256,
        # ),
        # (
        #     "division-adamw-transformer",
        #     logs_base_path
        #     / "checkpoints/grokking/division_adamw_transformer/checkpoints",
        #     "algorithmic",
        #     1_000,
        #     50_000,
        #     128,
        #     128,
        #     256,
        # ),
        # (
        #     "division-adamw-mlp-1",
        #     logs_base_path
        #     / "checkpoints/grokking/division_adamw_mlp_1_layer/checkpoints",
        #     "algorithmic",
        #     1_000,
        #     200_000,
        #     128,
        #     128,
        #     256,
        # ),
        # (
        #     "mnist-adamw",
        #     logs_base_path / "checkpoints/mnist/exp83/checkpoints",
        #     "mnist",
        #     1000,
        #     100_000,
        #     512,
        #     64,
        #     512,
        # ),
    ]

    # for exp in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27]:
    #     experiments.append(
    #         (
    #             f"mnist-fixed-norm-{exp}",
    #             logs_base_path / f"checkpoints/mnist_fixed_norm/exp{exp}/checkpoints",
    #             "mnist",
    #             10_000,
    #             10_000,
    #             512,
    #             64,
    #             512,
    #         ),
    #     )

    # for exp in range(43, 61):
    #     experiments.append(
    #         (
    #             f"mnist-fixed-norm-gf-{exp}",
    #             logs_base_path / f"checkpoints/gradient_flow/exp{exp}/checkpoints",
    #             "gradient_flow_mnist",
    #             10_000,
    #             10_000,
    #             512,
    #             64,
    #             512,
    #         ),
    #     )

    compute_results(experiments, add_kernel=False, kernel_batch_size=64)
