from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.log_dir = "logs/checkpoints/gradient_flow"

    config.seed = 0
    config.weight_decay = 0.004
    config.T = 1000
    config.save_every = 10

    config.dataset = dict()

    config.model = dict(
        name="ImageMLP",
        hidden=128,
        n_layers=1,
        init_scale=8,
    )

    config.ode = dict(
        rtol=1e-5,
        atol=6e-6,
        pcoeff=0.3,
        icoeff=0.3,
        dt0=1e-5,
    )
    return config
