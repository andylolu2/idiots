from idiots.experiments.gradient_flow.configs.base import get_config as base_config


def get_config():
    config = base_config()

    config.weight_decay = 3e-5
    config.T = 600_000
    config.save_every = config.get_ref("T") // 100

    config.dataset = dict(
        name="binary_op",
        op="x / y (mod 47)",
        train_percentage=0.5,
        seed=config.get_ref("seed"),
    )

    config.model = dict(
        name="EmbedMLP",
        hidden=128,
        n_layers=2,
        init_scale=1,
    )

    return config
