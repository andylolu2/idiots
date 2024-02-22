import random

import jax.numpy as jnp
import neural_tangents as nt
import orbax.checkpoint as ocp
from absl import app, flags, logging
from datasets import Dataset
from ml_collections import config_flags

from idiots.dataset.dataloader import DataLoader
from idiots.experiments.grokking.training import (
    TrainState,
    dots,
    eval_step,
    init,
    train_step,
)
from idiots.utils import metrics, num_params

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", short_name="c", lock_config=True)
flags.mark_flags_as_required(["config"])


def compute_dots(
    kernel_fn, params, ds: Dataset, sample_size: int, batch_size: int
) -> int:
    random_indices = random.sample(range(len(ds)), sample_size)
    return dots(kernel_fn, params, ds.select(random_indices)["x"], batch_size).item()


def main(_):
    config = FLAGS.config
    state, ds_train, ds_test, writer, mngr = init(config)
    logging.info("Number of parameters: %d", num_params(state.params))

    train_iter = iter(
        DataLoader(ds_train, config.train_batch_size, shuffle=True, infinite=True)
    )
    kernel_fn = nt.empirical_ntk_fn(
        state.apply_fn,
        trace_axes=(),
        vmap_axes=0,
        implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,
    )

    while state.step < config.steps:
        state, logs = train_step(state, next(train_iter), config.loss_variant)
        assert isinstance(state, TrainState)  # For better typing
        metrics.log(**logs)

        if state.step % config.log_every == 0 and config.log_every > 0:
            [losses, accuracies] = metrics.collect("loss", "accuracy")
            loss = jnp.concatenate(losses).mean().item()
            acc = jnp.concatenate(accuracies).mean().item()
            writer.add_scalar("train/loss", loss, state.step)
            writer.add_scalar("train/accuracy", acc, state.step)

        if state.step % config.eval_every == 0 and config.eval_every > 0:
            for batch in DataLoader(ds_test, config.test_batch_size):
                logs = eval_step(state, batch, config.loss_variant)
                metrics.log(**logs)
            [losses, accuracies] = metrics.collect("eval_loss", "eval_accuracy")
            loss = jnp.concatenate(losses).mean().item()
            acc = jnp.concatenate(accuracies).mean().item()
            writer.add_scalar("eval/loss", loss, state.step)
            writer.add_scalar("eval/accuracy", acc, state.step)

            if config.dots_sample_size > 0:
                dots_train = compute_dots(
                    kernel_fn,
                    state.params,
                    ds_train,
                    config.dots_sample_size,
                    config.dots_batch_size,
                )
                dots_val = compute_dots(
                    kernel_fn,
                    state.params,
                    ds_test,
                    config.dots_sample_size,
                    config.dots_batch_size,
                )
                writer.add_scalar("train/dots", dots_train, state.step)
                writer.add_scalar("eval/dots", dots_val, state.step)

        if state.step % config.save_every == 0 and config.save_every > 0:
            mngr.save(state.step, args=ocp.args.StandardSave(state))  # type: ignore
            mngr.wait_until_finished()


if __name__ == "__main__":
    app.run(main)
