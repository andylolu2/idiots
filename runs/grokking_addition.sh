#!/bin/bash

python idiots/experiments/grokking/main.py \
    --config idiots/experiments/grokking/config.py \
    --config.task="x + y (mod 47)" \
    --config.steps=50000 \
    --config.save_every=1000