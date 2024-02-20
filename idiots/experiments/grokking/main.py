import random

import jax.numpy as jnp
import orbax.checkpoint as ocp
from absl import app, flags, logging
from ml_collections import config_flags

from idiots.dataset.dataloader import DataLoader
from idiots.experiments.grokking.training import dots, eval_step, init, train_step
from idiots.utils import metrics, num_params

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", short_name="c", lock_config=True)
flags.mark_flags_as_required(["config"])


def main(_):
    config = FLAGS.config
    state, ds_train, ds_test, writer, mngr = init(config)
    logging.info("Number of parameters: %d", num_params(state.params))

    train_loader = DataLoader(
        ds_train, config.train_batch_size, shuffle=True, infinite=True, drop_last=True
    )
    train_iter = iter(train_loader)

    while state.step < config.steps:
        state, logs = train_step(state, next(train_iter), config.loss_variant)
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
                random_indices = random.sample(
                    range(len(ds_train)), config.dots_sample_size
                )
                dots_train = dots(state, ds_train.select(random_indices)["x"])
                random_indices = random.sample(
                    range(len(ds_test)), config.dots_sample_size
                )
                dots_val = dots(state, ds_test.select(random_indices)["x"])
                writer.add_scalar("train/dots", dots_train, state.step)
                writer.add_scalar("eval/dots", dots_val, state.step)

        if state.step % config.save_every == 0 and config.save_every > 0:
            mngr.save(state.step, args=ocp.args.StandardSave(state))  # type: ignore
            mngr.wait_until_finished()


if __name__ == "__main__":
    app.run(main)
