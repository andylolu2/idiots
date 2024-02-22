from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp
from datasets import Dataset
from ml_collections import ConfigDict
from tensorboardX import SummaryWriter

from idiots.dataset.image_classification import mnist_splits
from idiots.experiments.classification.model import ImageMLP
from idiots.experiments.grokking.training import TrainState
from idiots.utils import get_optimizer, next_dir


def init_state_and_ds(config):
    ds_train, ds_test = mnist_splits(config.train_size, config.test_size, config.seed)
    model = ImageMLP(
        hidden=config.model.d_model,
        n_layers=config.model.n_layers,
        out=ds_train.features["y"].num_classes,
    )
    params = model.init(jax.random.PRNGKey(config.seed), ds_train["x"][:1])
    tx = get_optimizer(**config.opt)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, ds_train, ds_test


def init(config):
    log_dir = next_dir(config.log_dir).absolute().resolve()
    writer = SummaryWriter(log_dir=str(log_dir))
    mngr = ocp.CheckpointManager(log_dir / "checkpoints", metadata=config.to_dict())
    state, ds_train, ds_test = init_state_and_ds(config)
    return state, ds_train, ds_test, writer, mngr


def restore(
    checkponit_dir: Path, step: int
) -> tuple[Any, TrainState, Dataset, Dataset]:
    checkponit_dir = checkponit_dir.absolute().resolve()
    mngr = ocp.CheckpointManager(
        checkponit_dir,
        options=ocp.CheckpointManagerOptions(
            read_only=True, save_interval_steps=0, create=False
        ),
    )
    config: Any = ConfigDict(mngr.metadata())
    state, ds_train, ds_test = init_state_and_ds(config)

    if step > 0:
        state = mngr.restore(step, args=ocp.args.StandardRestore(state))  # type: ignore
        assert isinstance(state, TrainState)

    return config, state, ds_train, ds_test
