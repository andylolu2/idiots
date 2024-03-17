#!/bin/bash

python idiots/experiments/grokking/main.py \
    --config idiots/experiments/grokking/config.py \
    --config.loss_variant=mse \
    --config.opt.warmup_steps=0 \
    --config.opt.weight_decay=1 \
    --config.steps=50000 \
    --config.save_every=1000