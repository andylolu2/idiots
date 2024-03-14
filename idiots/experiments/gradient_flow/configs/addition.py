from idiots.experiments.gradient_flow.configs.base import get_config as base_config


def get_config():
    config = base_config()

    config.weight_decay = 4e-6
    config.T = 1_500_000
    config.save_every = 15_000

    config.dataset = dict(
        name="binary_op",
        op="x + y (mod 47)",
        train_percentage=0.5,
        seed=config.get_ref("seed"),
    )

    config.model = dict(
        name="EmbedMLP",
        hidden=128,
        n_layers=1,
        init_scale=1,
    )

    return config
