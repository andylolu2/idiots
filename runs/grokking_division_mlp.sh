#!/bin/bash

python idiots/experiments/grokking/main.py \
    --config idiots/experiments/grokking/config.py \
    --config.loss_variant=mse \
    --config.model.name=mlp \
    --config.model.n_layers=2 \
    --config.model.d_model=128 \
    --config.opt.warmup_steps=0 \
    --config.opt.weight_decay=2.5 \
    --config.steps=50000 \
    --config.save_every=1000