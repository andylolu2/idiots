from pathlib import Path

import jax
import jax.numpy as jnp
import neural_tangents as nt
import optax
import orbax.checkpoint as ocp
from absl import app, flags, logging

from idiots.dataset.dataloader import DataLoader
from idiots.experiments.classification.training import restore as restore_classification
from idiots.experiments.grokking.training import TrainState, eval_step
from idiots.experiments.grokking.training import restore as restore_grokking
from idiots.experiments.grokking.training import train_step
from idiots.utils import metrics

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoint_dir", None, "Path to the checkpoint directory.", required=True
)
flags.DEFINE_enum("task", None, ["mnist", "algoritmic"], "Task type.", required=True)


def train_linearly(
    init_state: TrainState, train_loader: DataLoader, steps: int, config
) -> tuple[TrainState, dict]:
    linear_state = TrainState.create(
        apply_fn=nt.linearize(init_state.apply_fn, init_state.params),
        params=init_state.params,
        tx=init_state.tx,
    )
    train_iter = iter(train_loader)
    for _ in range(steps):
        linear_state, logs = train_step(
            linear_state, next(train_iter), config.loss_variant
        )
        metrics.log(**logs)

    [
        losses,
        accuracies,
        grad_norms,
        weight_norms,
        update_norms,
    ] = metrics.collect("loss", "accuracy", "grad_norm", "weight_norm", "update_norm")
    logs = {
        "loss": jnp.concatenate(losses).mean().item(),
        "accuracy": jnp.concatenate(accuracies).mean().item(),
        "grad_norm": jnp.array(grad_norms).mean().item(),
        "weight_norm": jnp.array(weight_norms).mean().item(),
        "update_norm": jnp.array(update_norms).mean().item(),
    }
    return linear_state, logs


@jax.jit
def param_distance(params_1, params_2):
    param_diff = jax.tree_map(lambda x, y: x - y, params_1, params_2)
    return optax.global_norm(param_diff)


def main(_):
    match FLAGS.task:
        case "mnist":
            restore_fn = restore_classification
        case "arithmetic":
            restore_fn = restore_grokking
    mngr, config, state, ds_train, ds_test = restore_fn(Path(FLAGS.checkpoint_dir), 0)
    train_loader = DataLoader(
        ds_train, config.train_batch_size, shuffle=True, infinite=True, drop_last=True
    )
    test_loader = DataLoader(ds_test, config.test_batch_size)
    all_steps = list(mngr.all_steps())[::2]
    if 0 not in all_steps:
        all_steps = [0] + all_steps

    def restore_step(step) -> TrainState:
        return mngr.restore(step, args=ocp.args.StandardRestore(state))

    for step, next_step in zip(all_steps[:-1], all_steps[1:]):
        state = restore_step(step)
        target_state = restore_step(next_step)
        linearly_trained_state, train_logs = train_linearly(
            state, train_loader, next_step - step, config
        )

        for batch in test_loader:
            logs = eval_step(linearly_trained_state, batch, config.loss_variant)
            metrics.log(**logs)
        [losses, accuracies] = metrics.collect("eval_loss", "eval_accuracy")
        loss = jnp.concatenate(losses).mean().item()
        acc = jnp.concatenate(accuracies).mean().item()
        logging.info(
            "Step %d -> %d: Train loss: %f, accuracy: %f, eval loss: %f, accuracy: %f, param distance: %f, grad norm: %f, weight norm: %f, update norm: %f",
            step,
            next_step,
            train_logs["loss"],
            train_logs["accuracy"],
            loss,
            acc,
            param_distance(linearly_trained_state.params, target_state.params).item(),
            train_logs["grad_norm"],
            train_logs["weight_norm"],
            train_logs["update_norm"],
        )


if __name__ == "__main__":
    app.run(main)
