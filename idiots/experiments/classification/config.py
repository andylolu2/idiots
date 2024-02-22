from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.log_dir = "logs/mnist"

    config.seed = 0
    config.loss_variant = "cross_entropy"

    config.steps = 100_000
    config.log_every = 100
    config.eval_every = 1000
    config.save_every = -1

    config.train_size = 60_000
    config.test_size = 10_000
    config.train_batch_size = 128
    config.test_batch_size = 128

    config.dots_sample_size = 128
    config.dots_batch_size = 64

    config.model = dict(
        d_model=256,
        n_layers=2,
    )

    config.opt = dict(
        lr=1e-3,
        weight_decay=0.01,
        warmup_steps=10,
    )

    return config
