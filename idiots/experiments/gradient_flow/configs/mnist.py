from idiots.experiments.gradient_flow.configs.base import get_config as base_config


def get_config():
    config = base_config()

    config.weight_decay = 4e-3
    config.T = 1000
    config.save_every = config.get_ref("T") // 100

    config.dataset = dict(
        name="mnist",
        train_size=512,
        test_size=5000,
    )

    config.model = dict(
        name="ImageMLP",
        hidden=128,
        n_layers=2,
        init_scale=1.0,
    )

    config.ode.dt0 = 1e-5

    return config
