from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import neural_tangents as nt
import optax
import orbax.checkpoint as ocp
from datasets import Dataset
from einops import rearrange
from flax.training import train_state
from ml_collections import ConfigDict
from tensorboardX import SummaryWriter

from idiots.dataset.algorithmic import binary_op_splits
from idiots.experiments.grokking.config import get_config
from idiots.experiments.grokking.model import TransformerSingleOutput
from idiots.utils import get_optimizer, next_dir


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


def dots(kernel_fn, params, x, batch_size: int = 32):
    """Compute the DOTS (rank of the NTK/Jacobian)

    Don't need to jit this as `nt.batch` already jits `inner_ntk`.
    """
    k = nt.batch(kernel_fn, batch_size=batch_size)(x, None, params)
    k = rearrange(k, "b1 b2 d1 d2 -> (b1 d1) (b2 d2)")
    return jnp.linalg.matrix_rank(k)  # type: ignore


def init_state_and_ds(config):
    
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
    params = model.init(jax.random.PRNGKey(config.seed), ds_train["x"][:1])
    tx = get_optimizer("adamw", **config.opt)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, ds_train, ds_test

def init_state(config, training_data_example, num_classes, feature_length):
    
    model = TransformerSingleOutput(
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        vocab_size=num_classes,
        max_len=feature_length,
    )
    params = model.init(jax.random.PRNGKey(config.seed), training_data_example)
    tx = get_optimizer("adamw", **config.opt)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    return state

def init(config):
    log_dir = next_dir(config.log_dir).absolute().resolve()
    writer = SummaryWriter(log_dir=str(log_dir))
    mngr = ocp.CheckpointManager(log_dir / "checkpoints", metadata=config.to_dict())
    state, ds_train, ds_test = init_state_and_ds(config)
    return state, ds_train, ds_test, writer, mngr


def restore(
    checkpoint_dir: Path, step: int
) -> tuple[Any, TrainState, Dataset, Dataset]:
    checkpoint_dir = checkpoint_dir.absolute().resolve()
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            read_only=True, save_interval_steps=0, create=False
        ),
    )

    # Load the config from the checkpoint, but add any new defaults
    config = get_config()
    override_config = ConfigDict(mngr.metadata())
    config.update(override_config)

    state, ds_train, ds_test = init_state_and_ds(config)

    if step > 0:
        state = mngr.restore(step, args=ocp.args.StandardRestore(state))  # type: ignore
        assert isinstance(state, TrainState)

    return mngr, config, state, ds_train, ds_test

def restore_partial(
    mngr, step: int, training_data_example, num_classes: int, feature_length
) -> tuple[Any, TrainState, Dataset, Dataset]:
    # Load the config from the checkpoint, but add any new defaults
    config = get_config()
    override_config = ConfigDict(mngr.metadata())
    config.update(override_config)

    state = init_state(config, training_data_example, num_classes, feature_length)

    if step > 0:
        state = mngr.restore(step, args=ocp.args.StandardRestore(state))  # type: ignore
        assert isinstance(state, TrainState)

    return config, state