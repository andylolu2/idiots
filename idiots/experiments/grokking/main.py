import random
from typing import Any

import jax
import jax.numpy as jnp
import neural_tangents as nt
import optax
from absl import app, flags, logging
from einops import rearrange
from flax.training import train_state
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from idiots.dataset.algorithmic import DataLoader, binary_op_splits
from idiots.experiments.grokking.model import TransformerSingleOutput
from idiots.utils import metrics, next_dir, num_params

_CONFIG = config_flags.DEFINE_config_file("config", short_name="c")
flags.mark_flags_as_required(["config"])


class TrainState(train_state.TrainState):
    """We might want to add some extra fields down the line.

    The base class already has the following fields:
        step: int
        apply_fn: Callable
        params: core.FrozenDict[str, Any]
        tx: optax.GradientTransformation
        opt_state: optax.OptState
    """

    ...


@jax.jit
def train_step(state: TrainState, batch) -> tuple[TrainState, dict]:
    def forward(params, x, y):
        y_pred = state.apply_fn(params, x)
        losses = optax.softmax_cross_entropy_with_integer_labels(y_pred, y)
        loss = jnp.mean(losses)
        return loss, (losses, y_pred)

    grads, (losses, y_pred) = jax.grad(forward, has_aux=True)(
        state.params, batch["x"], batch["y"]
    )
    acc = jnp.argmax(y_pred, axis=-1) == batch["y"]
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    logs = {"loss": losses, "accuracy": acc}
    return new_state, logs


@jax.jit
def eval_step(state: TrainState, batch) -> dict:
    y_pred = state.apply_fn(state.params, batch["x"])
    loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, batch["y"])
    acc = jnp.argmax(y_pred, axis=-1) == batch["y"]
    return {"eval_loss": loss, "eval_accuracy": acc}


@jax.jit
def dots(state: TrainState, x):
    kernel_fn = nt.empirical_kernel_fn(
        state.apply_fn,
        trace_axes=(),
        vmap_axes=0,
        implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,
    )
    k = kernel_fn(x, None, "ntk", state.params)
    k = rearrange(k, "b1 b2 d1 d2 -> (b1 d1) (b2 d2)")
    return jnp.linalg.matrix_rank(k)  # type: ignore


def init(config: Any):
    rng = jax.random.PRNGKey(0)
    writer = SummaryWriter(log_dir=str(next_dir(config.log_dir)))

    ds_train, ds_test = binary_op_splits(config.task, config.train_percentage)

    model = TransformerSingleOutput(
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        vocab_size=ds_train.features["y"].num_classes,
        max_len=ds_train.features["x"].length,
    )
    params = model.init(rng, ds_train["x"][:1])

    tx = optax.adamw(
        learning_rate=optax.join_schedules(
            [
                optax.linear_schedule(0, config.opt.lr, config.opt.warmup_steps),
                optax.constant_schedule(config.opt.lr),
            ],
            boundaries=[config.opt.warmup_steps],
        ),
        b1=0.9,
        b2=0.98,
        weight_decay=config.opt.weight_decay,
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    logging.info("Number of parameters: %d", num_params(params))
    return state, ds_train, ds_test, writer


def main(_):
    config: Any = _CONFIG.value
    state, ds_train, ds_test, writer = init(config)
    writer.add_hparams(dict(config), {})

    train_iter = iter(
        DataLoader(ds_train, config.train_batch_size, shuffle=True, infinite=True)
    )

    while state.step < config.steps:
        state, logs = train_step(state, next(train_iter))
        metrics.log(**logs)

        if state.step % config.log_every == 0:
            [losses, accuracies] = metrics.collect("loss", "accuracy")
            loss = jnp.concatenate(losses).mean().item()
            acc = jnp.concatenate(accuracies).mean().item()
            writer.add_scalar("train/loss", loss, state.step)
            writer.add_scalar("train/accuracy", acc, state.step)

        if state.step % config.eval_every == 0:
            for batch in DataLoader(ds_test, config.test_batch_size):
                logs = eval_step(state, batch)
                metrics.log(**logs)
            [losses, accuracies] = metrics.collect("eval_loss", "eval_accuracy")
            loss = jnp.concatenate(losses).mean().item()
            acc = jnp.concatenate(accuracies).mean().item()

            random_indices = random.sample(
                range(len(ds_train)), config.dots_sample_size
            )
            dots_train = dots(state, ds_train.select(random_indices)["x"])
            random_indices = random.sample(range(len(ds_test)), config.dots_sample_size)
            dots_val = dots(state, ds_test.select(random_indices)["x"])

            writer.add_scalar("eval/loss", loss, state.step)
            writer.add_scalar("eval/accuracy", acc, state.step)
            writer.add_scalar("eval/dots", dots_val, state.step)
            writer.add_scalar("train/dots", dots_train, state.step)


if __name__ == "__main__":
    app.run(main)
