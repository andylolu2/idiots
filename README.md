# IDIOTS

## Reproduce grokking

```bash
python idiots/experiments/grokking/main.py --config idiots/experiments/grokking/config.py
```

See logs with
```bash
tensorboard --logdir logs
```

## Setup

You need to install JAX manually (because the installation steps are different depending on what accelerator you are using). See https://jax.readthedocs.io/en/latest/installation.htm.

Developed on Python 3.11 but should work on 3.10+.

The core (direct) dependencies are in `requirements.txt`.

The full list of frozen dependencies is in `requirements.lock`.

```bash
pip install -r requirements.txt
# or
pip install -r requirements.lock
```

See training logs with
```bash
tensorboard --logdir logs
```