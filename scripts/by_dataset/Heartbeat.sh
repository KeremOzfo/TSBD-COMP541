#!/bin/bash
# All experiments for Heartbeat
# Sequence Length: 405 -> medium-term forecasting trigger
# Variates: 61, Classes: 2
# Train/Test: 204/205

# ========== CLEAN TRAINING ==========

# TimesNet Clean
python main.py \
    --mode clean \
    --model TimesNet \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --train_epochs 30 \
    --lr 0.001 \
    --gpu_id cuda:0

# PatchTST Clean
python main.py \
    --mode clean \
    --model PatchTST \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --train_epochs 30 \
    --lr 0.001 \
    --gpu_id cuda:0

# iTransformer Clean
python main.py \
    --mode clean \
    --model iTransformer \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --train_epochs 30 \
    --lr 0.001 \
    --gpu_id cuda:0

# ========== BACKDOOR TRAINING (BASIC MODE) ==========

# TimesNet + cnn trigger
python main.py \
    --mode basic \
    --model TimesNet \
    --Tmodel cnn \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# TimesNet + patchtst trigger
python main.py \
    --mode basic \
    --model TimesNet \
    --Tmodel patchtst \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# TimesNet + itst trigger
python main.py \
    --mode basic \
    --model TimesNet \
    --Tmodel itst \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# PatchTST + cnn trigger
python main.py \
    --mode basic \
    --model PatchTST \
    --Tmodel cnn \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# PatchTST + patchtst trigger
python main.py \
    --mode basic \
    --model PatchTST \
    --Tmodel patchtst \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# PatchTST + itst trigger
python main.py \
    --mode basic \
    --model PatchTST \
    --Tmodel itst \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# iTransformer + cnn trigger
python main.py \
    --mode basic \
    --model iTransformer \
    --Tmodel cnn \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# iTransformer + patchtst trigger
python main.py \
    --mode basic \
    --model iTransformer \
    --Tmodel patchtst \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# iTransformer + itst trigger
python main.py \
    --mode basic \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# ========== BACKDOOR TRAINING (MARKSMAN MODE) ==========

# TimesNet + cnn trigger (Marksman)
python main.py \
    --mode marksman \
    --model TimesNet \
    --Tmodel cnn \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# TimesNet + patchtst trigger (Marksman)
python main.py \
    --mode marksman \
    --model TimesNet \
    --Tmodel patchtst \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# TimesNet + itst trigger (Marksman)
python main.py \
    --mode marksman \
    --model TimesNet \
    --Tmodel itst \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# PatchTST + cnn trigger (Marksman)
python main.py \
    --mode marksman \
    --model PatchTST \
    --Tmodel cnn \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# PatchTST + patchtst trigger (Marksman)
python main.py \
    --mode marksman \
    --model PatchTST \
    --Tmodel patchtst \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# PatchTST + itst trigger (Marksman)
python main.py \
    --mode marksman \
    --model PatchTST \
    --Tmodel itst \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# iTransformer + cnn trigger (Marksman)
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel cnn \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# iTransformer + patchtst trigger (Marksman)
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel patchtst \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# iTransformer + itst trigger (Marksman)
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/Heartbeat \
    --seq_len 405 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0
