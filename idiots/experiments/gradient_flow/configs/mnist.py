from idiots.experiments.gradient_flow.configs.base import get_config as base_config


def get_config():
    config = base_config()

    config.weight_decay = 0.001
    config.T = 5000
    config.save_every = 50

    config.dataset = dict(
        name="mnist",
        train_size=256,
        test_size=5000,
    )

    config.model = dict(
        name="ImageMLP",
        hidden=128,
        n_layers=1,
        init_scale=8,
    )

    config.ode.dt0 = 5e-5

    return config
