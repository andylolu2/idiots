#!/bin/bash

python idiots/experiments/classification/main.py \
    --config idiots/experiments/classification/config.py \
    --config.steps=2000 \
    --config.save_every=100