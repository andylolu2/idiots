#!/bin/bash

run() {
    python idiots/experiments/gradient_flow/main.py \
        --config idiots/experiments/gradient_flow/configs/mnist.py \
        --config.weight_decay=0 \
        --config.ode.atol=1e-6 \
        --config.fixed_weight_norm=True \
        --config.T=$1 \
        --config.save_every=$(($1/10)) \
        --config.model.init_scale=$2
}

# run 10000 0.01
# run 10000 0.05
# run 10000 0.1
# run 10000 0.2
# run 10000 0.3
# run 10000 0.4
# run 10000 0.5
# run 10000 0.6
# run 10000 0.8
# run 10000 1
run 10000 1.25
# run 10000 1.5
# run 10000 1.75
# run 10000 2
# run 10000 3
# run 10000 4