from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import neural_tangents as nt
import optax
import orbax.checkpoint as ocp
from absl import logging
from einops import rearrange
from flax.training import train_state
from ml_collections import ConfigDict
from tensorboardX import SummaryWriter

from idiots.dataset.algorithmic import binary_op_splits
from idiots.experiments.grokking.model import TransformerSingleOutput
from idiots.utils import next_dir, num_params


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


@partial(jax.jit, static_argnums=2)
def train_step(state: TrainState, batch, loss_variant: str) -> tuple[TrainState, dict]:
    def forward(params, x, y):
        y_pred = state.apply_fn(params, x)
        losses = loss_fn(y_pred, y, variant=loss_variant)
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


@partial(jax.jit, static_argnums=2)
def eval_step(state: TrainState, batch, loss_variant: str) -> dict:
    y_pred = state.apply_fn(state.params, batch["x"])
    losses = loss_fn(y_pred, batch["y"], variant=loss_variant)
    acc = jnp.argmax(y_pred, axis=-1) == batch["y"]
    return {"eval_loss": losses, "eval_accuracy": acc}


def loss_fn(y_pred, y, variant="cross_entropy"):
    if variant == "cross_entropy":
        return optax.softmax_cross_entropy_with_integer_labels(y_pred, y)
    elif variant == "mse":  # zero-mean mse
        y = jax.nn.one_hot(y, num_classes=y_pred.shape[-1])
        y = y - jnp.mean(y, axis=-1, keepdims=True)
        return jnp.mean(jnp.square(y_pred - y), axis=-1)
    else:
        raise ValueError(f"Unknown loss variant: {variant}")


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


def init(config):
    rng = jax.random.PRNGKey(config.seed)
    log_dir = next_dir(config.log_dir).absolute().resolve()
    writer = SummaryWriter(log_dir=str(log_dir))
    mngr = ocp.CheckpointManager(log_dir / "checkpoints", metadata=config.to_dict())

    ds_train, ds_test = binary_op_splits(
        config.task, config.train_percentage, config.seed
    )

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
    return state, ds_train, ds_test, writer, mngr


def restore(checkponit_dir: Path, step: int):
    checkponit_dir = checkponit_dir.absolute().resolve()
    mngr = ocp.CheckpointManager(
        checkponit_dir,
        options=ocp.CheckpointManagerOptions(
            read_only=True, save_interval_steps=0, create=False
        ),
    )
    config: Any = ConfigDict(mngr.metadata())

    ds_train, ds_test = binary_op_splits(
        config.task, config.train_percentage, config.seed
    )

    model = TransformerSingleOutput(
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        vocab_size=ds_train.features["y"].num_classes,
        max_len=ds_train.features["x"].length,
    )
    rng = jax.random.PRNGKey(config.seed)
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

    if step > 0:
        state = mngr.restore(step, args=ocp.args.StandardRestore(state))  # type: ignore

    return config, state, ds_train, ds_test
