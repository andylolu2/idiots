#!/bin/bash

python idiots/experiments/grokking/main.py \
    --config idiots/experiments/grokking/config.py \
    --config.task="x + y (mod 47)" \
    --config.train_percentage=0.3 \
    --config.steps=50000 \
    --config.model.n_layers=1 \
    --config.model.old_parameterisation=False \
    --config.opt.warmup_steps=0 \
    --config.opt.weight_decay=0.3 \
    --config.loss_variant=mse \
    --config.save_every=1000