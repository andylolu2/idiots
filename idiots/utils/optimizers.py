import optax


def get_optimizer(name: str, **kwargs):
    if name == "adamw":
        return get_adamw(**kwargs)
    elif name == "sgd":
        return get_sgd(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def get_adamw(*, lr: float, warmup_steps: int = 0, weight_decay: float = 0, **_):
    return optax.adamw(
        learning_rate=optax.join_schedules(
            [
                optax.linear_schedule(0, lr, warmup_steps),
                optax.constant_schedule(lr),
            ],
            boundaries=[warmup_steps],
        ),
        b1=0.9,
        b2=0.98,
        weight_decay=weight_decay,
    )


def get_sgd(
    *,
    lr: float,
    weight_decay: float = 0,
    **_,
):
    return optax.chain(
        optax.clip_by_global_norm(100.0),
        optax.add_decayed_weights(weight_decay=weight_decay),
        optax.scale_by_learning_rate(lr),
    )
