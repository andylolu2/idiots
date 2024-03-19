#!/bin/bash

python idiots/experiments/classification/main_scaled.py \
    --config idiots/experiments/classification/config.py \
    --config.steps=200000 \
    --config.save_every=100