#!/bin/bash

python idiots/experiments/grokking/main.py \
    --config idiots/experiments/grokking/config.py \
    --config.steps=50000 \
    --config.save_every=1000