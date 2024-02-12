# IDIOTS

## Reproduce grokking

```bash
python idiots/experiments/grokking/main.py
```

See logs with
```bash
tensorboard --logdir logs
```

## Setup

You need to install JAX manually (because the installation steps are different depending on what accelerator you are using). See https://jax.readthedocs.io/en/latest/installation.htm.

Developed on Python 3.11 but should work on 3.10+.

The core dependencies are in `requirements.txt`.

The full list of dependencies is in `requirements-lock.txt`.