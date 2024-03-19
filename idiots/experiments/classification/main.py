import random

import jax
import jax.numpy as jnp
import neural_tangents as nt
import optax
import orbax.checkpoint as ocp
from absl import app, flags, logging
from datasets import Dataset
from ml_collections import config_flags

from idiots.dataset.dataloader import DataLoader
from idiots.experiments.classification.training import init
from idiots.experiments.grokking.training import TrainState, dots, eval_step, train_step
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
    state = state.replace(step=-1)
    logging.info("Number of parameters: %d", num_params(state.params))

    train_loader = DataLoader(
        ds_train,
        config.train_batch_size,
        shuffle=True,
        infinite=True,
        drop_last=True,
        seed=config.seed,
    )
    eval_loader = DataLoader(ds_test, config.test_batch_size)
    train_iter = iter(train_loader)
    kernel_fn = nt.empirical_ntk_fn(
        state.apply_fn,
        trace_axes=(),
        vmap_axes=0,
        implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,
    )
    norm = optax.global_norm(state.params)

    while state.step < config.steps:
        state, logs = train_step(state, next(train_iter), config.loss_variant)
        assert isinstance(state, TrainState)  # For better typing
        metrics.log(**logs)

        if config.fixed_weight_norm:
            new_norm = optax.global_norm(state.params)
            new_params = jax.tree_map(lambda p: p * (norm / new_norm), state.params)
            state = state.replace(params=new_params)

        if state.step % config.log_every == 0 and config.log_every > 0:
            [
                losses,
                accuracies,
                grad_norms,
                weight_norms,
                update_norms,
            ] = metrics.collect(
                "loss", "accuracy", "grad_norm", "weight_norm", "update_norm"
            )
            loss = jnp.concatenate(losses).mean().item()
            acc = jnp.concatenate(accuracies).mean().item()
            grad_norm = jnp.array(grad_norms).mean().item()
            weight_norm = jnp.array(weight_norms).mean().item()
            update_norm = jnp.array(update_norms).mean().item()
            writer.add_scalar("train/loss", loss, state.step)
            writer.add_scalar("train/accuracy", acc, state.step)
            writer.add_scalar("train/grad_norm", grad_norm, state.step)
            writer.add_scalar("train/weight_norm", weight_norm, state.step)
            writer.add_scalar("train/update_norm", update_norm, state.step)

        if state.step % config.eval_every == 0 and config.eval_every > 0:
            for batch in eval_loader:
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
                writer.add_scalar("eval/dots", dots_val, state.step)
                writer.add_scalar("train/dots", dots_train, state.step)

        if state.step % config.save_every == 0 and config.save_every > 0:
            mngr.save(state.step, args=ocp.args.StandardSave(state))  # type: ignore
            mngr.wait_until_finished()


if __name__ == "__main__":
    app.run(main)
