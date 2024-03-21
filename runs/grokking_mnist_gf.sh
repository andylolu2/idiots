#!/bin/bash

python idiots/experiments/gradient_flow/main.py \
    --config idiots/experiments/gradient_flow/configs/mnist.py \
    --config.T=2000 \
    --config.weight_decay=2e-3 \
    --config.model.init_scale=8