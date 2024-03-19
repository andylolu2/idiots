#!/bin/bash

python idiots/experiments/classification/main.py \
    --config idiots/experiments/classification/config.py \
    --config.steps=100000 \
    --config.eval_every=1000 \
    --config.save_every=1000 \
    --config.train_size=512 \
    --config.train_batch_size=128 \
    --config.dots_sample_size=256 \
    --config.dots_batch_size=256 \
    --config.model.d_model=128 \
    --config.model.init_scale=8 \
    --config.model.normalize_inputs=True \
    --config.loss_variant=mse \
    --config.opt.warmup_steps=0 \
    --config.opt.weight_decay=0.02