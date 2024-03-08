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
    RBF,
    Kernel,
    NormalizedKernelMixin,
    StationaryKernelMixin,
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from idiots.dataset.dataloader import DataLoader
from idiots.experiments.classification.training import restore as mnist_restore
from idiots.experiments.grokking.training import TrainState, eval_step
from idiots.experiments.grokking.training import restore as algorithmic_restore
from idiots.utils import metrics

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

warnings.filterwarnings("ignore")


def eval_checkpoint(state, config, train_loader, test_loader):
    """Compute training/test accuracy/loss for each timestep"""

    def eval_loss_acc(loader):
        for batch in loader:
            logs = eval_step(state, batch, config.loss_variant)
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

    if experiment_type == "mnist":
        analysis_X_train = rearrange(analysis_X_train, "b h w -> b (h w)")
        analysis_X_test = rearrange(analysis_X_test, "b h w -> b (h w)")

    return (analysis_X_train, analysis_Y_train, analysis_X_test, analysis_Y_test)


# Return a batched kernel function where trace_axes=() [for calculating DOTS]
def compute_kernel_trace_axes_fn(transformer_state_apply_fn, batch_size):
    kernel_fn_trace_axes = nt.empirical_ntk_fn(
        transformer_state_apply_fn,
        vmap_axes=0,
        trace_axes=(),
        implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,
    )
    return nt.batch(kernel_fn_trace_axes, batch_size=batch_size)


# Return a batched kernel function where trace_axes is not defined [for computing everything other than DOTS]
def compute_kernel_fn(transformer_state_apply_fn, batch_size):
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

    return dots_1, dots_2, dots_3


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

    custom_gp_kernel = CustomKernel()  # RBF(length_scale=1e3)
    gaussian_process_classifier = GaussianProcessRegressor(kernel=custom_gp_kernel)
    gaussian_process_classifier.fit(analysis_X_train, analysis_Y_train_one_hot)

    predictions = gaussian_process_classifier.predict(analysis_X_test).argmax(axis=-1)
    accuracy = accuracy_score(analysis_Y_test, predictions)
    return accuracy


def compute_kernel_alignment(kernel, analysis_Y_test):
    """Compute the kernel alignment metric.

    Source: Shan 2022: A Theory of Neural Tangent Kernel Alignment and Its Influence on Training
    We use the "traced" kernel here.
    """
    traced_kernel = jnp.trace(kernel, axis1=-2, axis2=-1) / kernel.shape[-1]
    kernel_alignment = (analysis_Y_test.T @ traced_kernel @ analysis_Y_test) / (
        jnp.linalg.norm(kernel) * jnp.linalg.norm(analysis_Y_test)
    )
    return kernel_alignment.item()


def compute_results(logs_base_path, experiments, add_kernel, kernel_batch_size=32):
    """
    logs_base_path: e.g. "/home/dm894/idiots/logs/"
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

        experiment_checkpoint_path = Path(logs_base_path, experiment_checkpoint_path)

        if experiment_type == "algorithmic":
            restore_fn = algorithmic_restore
        elif experiment_type == "mnist":
            restore_fn = mnist_restore
        else:
            raise ValueError(f"Experiment type {experiment_type} not valid.")

        restore_manager, config, state, ds_train, ds_test = restore_fn(
            experiment_checkpoint_path, 0
        )
        if len(ds_train) > len(ds_test):
            ds_train = ds_train.select(range(len(ds_test)))

        train_loader = DataLoader(ds_train, config.train_batch_size)
        test_loader = DataLoader(ds_test, config.test_batch_size)
        kernel_fn = compute_kernel_fn(state.apply_fn, kernel_batch_size)
        kernel_trace_axes_fn = compute_kernel_trace_axes_fn(
            state.apply_fn, kernel_batch_size
        )

        X_test, Y_test = jnp.array(ds_test["x"]), jnp.array(ds_test["y"])
        # kernel dataset is used for computing the kernels used in DOTS and kernel alignment
        # analysis datasets are used for the remaining analysis: SVM, GP
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

        all_metircs = []
        for step in range(0, total_steps + 1, step_distance):
            if step > 0:
                state = restore_manager.restore(
                    step, args=ocp.args.StandardRestore(state)
                )
            train_loss, train_acc, test_loss, test_acc = eval_checkpoint(
                state, config, train_loader, test_loader
            )
            svm_train_acc, svm_test_acc = compute_svm_accuracy(
                lambda x1, x2: kernel_fn(x1, x2, state.params),
                analysis_X_train,
                analysis_Y_train,
                analysis_X_test,
                analysis_Y_test,
            )
            gp_acc = compute_gp_accuracy(
                lambda x1, x2: kernel_fn(x1, x2, state.params),
                analysis_X_train,
                analysis_Y_train,
                analysis_X_test,
                analysis_Y_test,
                ds_train.features["y"].num_classes,
            )

            kernel_X = ds_test["x"][:kernel_samples]
            kernel_Y = ds_test["y"][:kernel_samples]
            kernel = kernel_trace_axes_fn(kernel_X, None, state.params)
            dots_1, dots_2, dots_3 = compute_dots(kernel)
            kernel_alignment = compute_kernel_alignment(kernel, kernel_Y)

            all_metircs.append(
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
                    "kernel_alignment": kernel_alignment,
                    "kernel": kernel.tolist() if add_kernel else None,
                    "weight_norm": optax.global_norm(state.params).item(),
                }
            )
            print(json.dumps(all_metircs[-1], indent=2))

        out_file = Path(logs_base_path, "results", f"{experiment_json_file_name}.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            # Convert list[dict] -> dict[list]
            metrics = {k: [d[k] for d in all_metircs] for k in all_metircs[0]}
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    # logs_base_path = "/home/dm894/idiots/logs/"
    logs_base_path = "logs"

    experiments = [
        (
            "mnist-gd-grokking",
            "checkpoints/mnist/mnist_gd_grokking/checkpoints",
            "mnist",
            1_000,
            100_000,
            512,
            64,
            64,
        ),
        (
            "mnist-adamw",
            "checkpoints/mnist/mnist_adamw/checkpoints",
            "mnist",
            1_000,
            50_000,
            512,
            64,
            64,
        ),
        (
            "mnist-grokking-slower",
            "checkpoints/mnist/mnist_grokking_slower/checkpoints",
            "mnist",
            1_000,
            100_000,
            512,
            64,
            64,
        ),
    ]

    compute_results(
        logs_base_path, experiments, add_kernel=False, kernel_batch_size=256
    )
