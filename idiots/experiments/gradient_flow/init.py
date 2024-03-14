from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp
from ml_collections import ConfigDict

from idiots.dataset.algorithmic import binary_op_splits
from idiots.dataset.image_classification import mnist_splits
from idiots.experiments.gradient_flow.configs.base import get_config
from idiots.experiments.gradient_flow.models import EmbedMLP, ImageMLP
from idiots.utils import next_dir


def load_model(name: str, **kwargs):
    if name == "ImageMLP":
        return ImageMLP(
            hidden=kwargs["hidden"],
            n_layers=kwargs["n_layers"],
            n_classes=kwargs["n_classes"],
        )
    elif name == "EmbedMLP":
        return EmbedMLP(
            hidden=kwargs["hidden"],
            n_layers=kwargs["n_layers"],
            n_classes=kwargs["n_classes"],
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def load_dataset_splits(name: str, **kwargs):
    if name == "binary_op":
        return binary_op_splits(**kwargs)
    elif name == "mnist":
        return mnist_splits(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def init(config):
    log_dir = next_dir(config.log_dir).absolute().resolve()
    mngr = ocp.CheckpointManager(
        log_dir / "checkpoints",
        item_handlers=ocp.StandardCheckpointHandler(),
        metadata=config.to_dict(),
    )

    ds_train, ds_test = load_dataset_splits(**config.dataset)

    model = load_model(**config.model, n_classes=ds_train.features["y"].num_classes)
    params = model.init(jax.random.PRNGKey(config.seed), ds_train["x"][:1])
    params = jax.tree_map(lambda x: x * config.model.init_scale, params)

    return model.apply, params, ds_train, ds_test, mngr


def restore(checkpoint_dir: Path):
    checkpoint_dir = checkpoint_dir.absolute().resolve()
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers=ocp.StandardCheckpointHandler(),
        options=ocp.CheckpointManagerOptions(
            read_only=True, save_interval_steps=0, create=False
        ),
    )

    # Load the config from the checkpoint, but add any new defaults
    config: Any = get_config()
    override_config = ConfigDict(mngr.metadata())
    config.update(override_config)

    ds_train, ds_test = load_dataset_splits(**config.dataset)

    model = load_model(**config.model, n_classes=ds_train.features["y"].num_classes)
    init_params = model.init(jax.random.PRNGKey(config.seed), ds_train["x"][:1])
    init_params = jax.tree_map(lambda x: x * config.model.init_scale, init_params)

    return model.apply, init_params, ds_train, ds_test, mngr, config
