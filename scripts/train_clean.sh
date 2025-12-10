#!/bin/bash
# Script for clean training

python main.py \
    --mode clean \
    --model TimesNet \
    --train_epochs 20 \
    --batch_size 32 \
    --lr 0.001 \
    --root_path ./dataset/UWaveGestureLibrary \
    --gpu_id cuda:0

