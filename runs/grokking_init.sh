#!/bin/bash

python idiots/experiments/classification/main.py \
    --config idiots/experiments/classification/config.py \
    --config.steps=2000 \
    --config.eval_every=1000 \
    --config.save_every=1000 \
    --config.train_size=1280 \
    --config.train_batch_size=128 \
    --config.model.init_scale=0.1 \
    --config.model.normalize_inputs=True \
    --config.loss_variant=mse \
    --config.opt.weight_decay=0.01

python idiots/experiments/classification/main.py \
    --config idiots/experiments/classification/config.py \
    --config.steps=2000 \
    --config.eval_every=1000 \
    --config.save_every=1000 \
    --config.train_size=1280 \
    --config.train_batch_size=128 \
    --config.model.init_scale=0.5 \
    --config.model.normalize_inputs=True \
    --config.loss_variant=mse \
    --config.opt.weight_decay=0.01

python idiots/experiments/classification/main.py \
    --config idiots/experiments/classification/config.py \
    --config.steps=2000 \
    --config.eval_every=1000 \
    --config.save_every=1000 \
    --config.train_size=1280 \
    --config.train_batch_size=128 \
    --config.model.init_scale=1 \
    --config.model.normalize_inputs=True \
    --config.loss_variant=mse \
    --config.opt.weight_decay=0.01

python idiots/experiments/classification/main.py \
    --config idiots/experiments/classification/config.py \
    --config.steps=2000 \
    --config.eval_every=1000 \
    --config.save_every=1000 \
    --config.train_size=1280 \
    --config.train_batch_size=128 \
    --config.model.init_scale=8 \
    --config.model.normalize_inputs=True \
    --config.loss_variant=mse \
    --config.opt.weight_decay=0.01

python idiots/experiments/classification/main.py \
    --config idiots/experiments/classification/config.py \
    --config.steps=2000 \
    --config.eval_every=1000 \
    --config.save_every=1000 \
    --config.train_size=1280 \
    --config.train_batch_size=128 \
    --config.model.init_scale=32 \
    --config.model.normalize_inputs=True \
    --config.loss_variant=mse \
    --config.opt.weight_decay=0.01

python idiots/experiments/classification/main.py \
    --config idiots/experiments/classification/config.py \
    --config.steps=2000 \
    --config.eval_every=1000 \
    --config.save_every=1000 \
    --config.train_size=1280 \
    --config.train_batch_size=128 \
    --config.model.init_scale=128 \
    --config.model.normalize_inputs=True \
    --config.loss_variant=mse \
    --config.opt.weight_decay=0.01

python idiots/experiments/classification/main.py \
    --config idiots/experiments/classification/config.py \
    --config.steps=2000 \
    --config.eval_every=1000 \
    --config.save_every=1000 \
    --config.train_size=1280 \
    --config.train_batch_size=128 \
    --config.model.init_scale=2048 \
    --config.model.normalize_inputs=True \
    --config.loss_variant=mse \
    --config.opt.weight_decay=0.01