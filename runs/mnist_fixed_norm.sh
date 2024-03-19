#!/bin/bash

run() {
    python idiots/experiments/classification/main.py \
        --config idiots/experiments/classification/config.py \
        --config.log_dir=logs/checkpoints/mnist_fixed_norm \
        --config.steps=$1 \
        --config.eval_every=1000 \
        --config.save_every=10000 \
        --config.train_size=512 \
        --config.train_batch_size=128 \
        --config.dots_sample_size=-1 \
        --config.dots_batch_size=128 \
        --config.model.d_model=128 \
        --config.model.init_scale=$2 \
        --config.model.normalize_inputs=True \
        --config.loss_variant=mse \
        --config.opt.weight_decay=0 \
        --config.opt.warmup_steps=0 \
        --config.fixed_weight_norm=True
}

run 20000 0.03
run 20000 0.1
run 50000 0.3
run 50000 0.5
run 100000 0.75
run 100000 1
# run 500000 2
# run 500000 3
# run 500000 4
# run 500000 5
# run 500000 8