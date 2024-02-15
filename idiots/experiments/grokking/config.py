from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.log_dir = "logs/grokking"

    config.task = "x / y (mod 97)"

    config.steps = int(1e5)
    config.log_every = 100
    config.eval_every = 1000

    config.train_percentage = 0.5
    config.train_batch_size = 512
    config.test_batch_size = 512

    config.dots_sample_size = 32

    config.model = dict(
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    config.opt = dict(
        lr=1e-3,
        weight_decay=0.0,
        warmup_steps=10,
    )

    return config
