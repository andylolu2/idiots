#!/bin/bash

python idiots/experiments/grokking/main.py \
    --config idiots/experiments/grokking/config.py \
    --config.task="xy (S5)" \
    --config.opt.weight_decay=0.5 \
    --config.dots_sample_size=32 \
    --config.steps=50000 \
    --config.save_every=1000