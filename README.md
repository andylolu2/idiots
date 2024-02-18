# IDIOTS

## Reproduce grokking

Run one of the following scripts to reproduce the checkpoints:
```bash
runs/grokking.sh
runs/grokking_mse.sh
runs/grokking_s5.sh
```

By default, it logs metrics to `<cwd>/logs/grokking`. See logs with
```bash
tensorboard --logdir logs/grokking
```

## Setup

Developed on Python 3.11 but should work on 3.10+.

The core (direct) dependencies are in `requirements.in`. The full list of frozen dependencies is in `requirements.txt`.

> **IMPORTANT**: At the time of this project, there is a **dependency hell** with `tensorflow` and `flax` so the library versions don't resolve correctly. To reproduce the environment, please install from the `requirements.txt` file with
> ```bash
> pip install -r requirements.txt --no-deps
> ```
> this assumes you are using CUDA.
