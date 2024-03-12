#!/bin/bash

python idiots/experiments/grokking/main.py \
    --config idiots/experiments/grokking/config.py \
    --config.task="x / y (mod 47)" \
    --config.steps=150000 \
    --config.save_every=1000 \
    --config.train_batch_size=1104 \
    --config.dots_sample_size=-1 \
    --config.loss_variant=mse_scaled \
    --config.model.old_parameterisation=False \
    --config.opt.name=sgd \
    --config.opt.lr=1e-3 \
    --config.opt.warmup_steps=0 \
    --config.opt.weight_decay=8e-2