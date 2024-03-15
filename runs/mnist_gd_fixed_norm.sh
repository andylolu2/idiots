#!/bin/bash

run() {
    python idiots/experiments/classification/main.py \
        --config idiots/experiments/classification/config.py \
        --config.log_dir=logs/checkpoints/mnist_fixed_norm \
        --config.steps=$1 \
        --config.eval_every=1000 \
        --config.save_every=10000 \
        --config.train_size=256 \
        --config.train_batch_size=256 \
        --config.dots_sample_size=64 \
        --config.dots_batch_size=64 \
        --config.model.init_scale=$2 \
        --config.model.normalize_inputs=True \
        --config.loss_variant=mse \
        --config.opt.weight_decay=0 \
        --config.fixed_weight_norm=True
}

# run 10000 0.03
# run 10000 0.1
# run 10000 0.3
# run 10000 0.5
# run 10000 0.75
# run 10000 1
# run 10000 2
run 10000 3
# run 10000 4
run 10000 5
# run 10000 8