#!/bin/bash
# Script for backdoor training

python main.py \
    --mode triggerNet \
    --model TimesNet \
    --bd_train_epochs 20 \
    --clean_pretrain_epochs 5 \
    --batch_size 32 \
    --lr 0.001 \
    --root_path ./dataset/UWaveGestureLibrary \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --surrogate_type bdmlp \
    --warmup_epochs 0 \
    --gpu_id cuda:0

