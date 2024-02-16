#!/bin/bash

python idiots/experiments/grokking/main.py \
    --config idiots/experiments/grokking/config.py \
    --config.loss_variant=mse \
    --config.opt.weight_decay=0.3 \
    --config.steps=50000 \
    --config.save_every=1000