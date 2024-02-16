from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.log_dir = "logs/grokking"

    config.seed = 0
    config.task = "x / y (mod 47)"
    config.loss_variant = "cross_entropy"

    config.steps = 100_000
    config.log_every = 100
    config.eval_every = 1000
    config.save_every = -1

    config.train_percentage = 0.5
    config.train_batch_size = 512
    config.test_batch_size = 512

    config.dots_sample_size = 64

    config.model = dict(
        d_model=64,
        n_layers=2,
        n_heads=2,
    )

    config.opt = dict(
        lr=1e-3,
        weight_decay=0.1,
        warmup_steps=10,
    )

    return config
