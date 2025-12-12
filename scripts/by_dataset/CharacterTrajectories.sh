#!/bin/bash
# All experiments for CharacterTrajectories
# Sequence Length: 119 -> medium-term forecasting trigger
# Variates: 3, Classes: 20
# Train/Test: 1422/1436

# ========== CLEAN TRAINING ==========

# TimesNet Clean
python main.py \
    --mode clean \
    --model TimesNet \
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --train_epochs 20 \
    --lr 0.001 \
    --gpu_id cuda:0

# PatchTST Clean
python main.py \
    --mode clean \
    --model PatchTST \
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --train_epochs 20 \
    --lr 0.001 \
    --gpu_id cuda:0

# iTransformer Clean
python main.py \
    --mode clean \
    --model iTransformer \
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --train_epochs 20 \
    --lr 0.001 \
    --gpu_id cuda:0

# ========== BACKDOOR TRAINING (BASIC MODE) ==========

# TimesNet + cnn trigger
python main.py \
    --mode basic \
    --model TimesNet \
    --Tmodel cnn \
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
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
    --root_path ./dataset/CharacterTrajectories \
    --seq_len 119 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 20 \
    --bd_train_epochs 20 \
    --trigger_epochs 15 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0
