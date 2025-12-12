#!/bin/bash
# Backdoor training: iTransformer + itst trigger (marksman mode)
# Trigger networks perform forecasting-based perturbations

# === AbnormalHeartbeat (long-term forecasting trigger) ===
# Seq: 18530, Vars: 1, Classes: 2
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/AbnormalHeartbeat \
    --seq_len 18530 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 2 \
    --n_heads_bd 2 \
    --patch_len 32 \
    --stride 16 \
    --train_epochs 45 \
    --bd_train_epochs 45 \
    --trigger_epochs 18 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# === BasicMotions (short-term forecasting trigger) ===
# Seq: 100, Vars: 6, Classes: 4
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/BasicMotions \
    --seq_len 100 \
    --batch_size 8 \
    --d_model 32 \
    --d_ff 64 \
    --d_model_bd 16 \
    --d_ff_bd 32 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 12 \
    --stride 6 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 20 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# === BirdChicken (long-term forecasting trigger) ===
# Seq: 512, Vars: 1, Classes: 2
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/BirdChicken \
    --seq_len 512 \
    --batch_size 8 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 2 \
    --n_heads_bd 2 \
    --patch_len 32 \
    --stride 16 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 20 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# === CharacterTrajectories (medium-term forecasting trigger) ===
# Seq: 119, Vars: 3, Classes: 20
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

# === Cricket (long-term forecasting trigger) ===
# Seq: 1197, Vars: 6, Classes: 12
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/Cricket \
    --seq_len 1197 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 2 \
    --n_heads_bd 2 \
    --patch_len 32 \
    --stride 16 \
    --train_epochs 45 \
    --bd_train_epochs 45 \
    --trigger_epochs 24 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# === ECG5000 (medium-term forecasting trigger) ===
# Seq: 140, Vars: 1, Classes: 5
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/ECG5000 \
    --seq_len 140 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
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

# === ElectricDevices (short-term forecasting trigger) ===
# Seq: 96, Vars: 1, Classes: 7
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/ElectricDevices \
    --seq_len 96 \
    --batch_size 32 \
    --d_model 32 \
    --d_ff 64 \
    --d_model_bd 16 \
    --d_ff_bd 32 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 12 \
    --stride 6 \
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

# === Epilepsy (medium-term forecasting trigger) ===
# Seq: 206, Vars: 3, Classes: 4
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/Epilepsy \
    --seq_len 206 \
    --batch_size 16 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 20 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# === FaceDetection (short-term forecasting trigger) ===
# Seq: 62, Vars: 144, Classes: 2
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/FaceDetection \
    --seq_len 62 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 2 \
    --n_heads_bd 8 \
    --patch_len 7 \
    --stride 3 \
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

# === FingerMovements (short-term forecasting trigger) ===
# Seq: 50, Vars: 28, Classes: 2
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/FingerMovements \
    --seq_len 50 \
    --batch_size 16 \
    --d_model 32 \
    --d_ff 64 \
    --d_model_bd 24 \
    --d_ff_bd 48 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 6 \
    --stride 3 \
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

# === HandMovementDirection (medium-term forecasting trigger) ===
# Seq: 400, Vars: 10, Classes: 4
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/HandMovementDirection \
    --seq_len 400 \
    --batch_size 16 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 20 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# === Handwriting (medium-term forecasting trigger) ===
# Seq: 152, Vars: 3, Classes: 26
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/Handwriting \
    --seq_len 152 \
    --batch_size 16 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 20 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# === Haptics (long-term forecasting trigger) ===
# Seq: 1092, Vars: 1, Classes: 5
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/Haptics \
    --seq_len 1092 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 2 \
    --n_heads_bd 2 \
    --patch_len 32 \
    --stride 16 \
    --train_epochs 45 \
    --bd_train_epochs 45 \
    --trigger_epochs 24 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# === Heartbeat (medium-term forecasting trigger) ===
# Seq: 405, Vars: 61, Classes: 2
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

# === JapaneseVowels (short-term forecasting trigger) ===
# Seq: 26, Vars: 12, Classes: 9
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/JapaneseVowels \
    --seq_len 26 \
    --batch_size 16 \
    --d_model 32 \
    --d_ff 64 \
    --d_model_bd 24 \
    --d_ff_bd 48 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 4 \
    --stride 2 \
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

# === LSST (short-term forecasting trigger) ===
# Seq: 36, Vars: 6, Classes: 14
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/LSST \
    --seq_len 36 \
    --batch_size 32 \
    --d_model 32 \
    --d_ff 64 \
    --d_model_bd 16 \
    --d_ff_bd 32 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 4 \
    --stride 2 \
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

# === MotorImagery (long-term forecasting trigger) ===
# Seq: 3000, Vars: 64, Classes: 2
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/MotorImagery \
    --seq_len 3000 \
    --batch_size 16 \
    --d_model 256 \
    --d_ff 512 \
    --d_model_bd 128 \
    --d_ff_bd 256 \
    --e_layers_bd 2 \
    --n_heads_bd 4 \
    --patch_len 32 \
    --stride 16 \
    --train_epochs 45 \
    --bd_train_epochs 45 \
    --trigger_epochs 18 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# === NATOPS (short-term forecasting trigger) ===
# Seq: 51, Vars: 24, Classes: 6
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/NATOPS \
    --seq_len 51 \
    --batch_size 16 \
    --d_model 32 \
    --d_ff 64 \
    --d_model_bd 24 \
    --d_ff_bd 48 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 6 \
    --stride 3 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 20 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# === PEMS-SF (medium-term forecasting trigger) ===
# Seq: 144, Vars: 963, Classes: 7
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/PEMS-SF \
    --seq_len 144 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 128 \
    --d_ff_bd 256 \
    --e_layers_bd 2 \
    --n_heads_bd 8 \
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

# === PenDigits (short-term forecasting trigger) ===
# Seq: 8, Vars: 2, Classes: 10
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/PenDigits \
    --seq_len 8 \
    --batch_size 32 \
    --d_model 32 \
    --d_ff 64 \
    --d_model_bd 16 \
    --d_ff_bd 32 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 4 \
    --stride 2 \
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

# === PhonemeSpectra (medium-term forecasting trigger) ===
# Seq: 217, Vars: 11, Classes: 39
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/PhonemeSpectra \
    --seq_len 217 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 48 \
    --d_ff_bd 96 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
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

# === SelfRegulationSCP1 (long-term forecasting trigger) ===
# Seq: 896, Vars: 6, Classes: 2
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/SelfRegulationSCP1 \
    --seq_len 896 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 2 \
    --n_heads_bd 2 \
    --patch_len 32 \
    --stride 16 \
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

# === SelfRegulationSCP2 (long-term forecasting trigger) ===
# Seq: 1152, Vars: 7, Classes: 2
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/SelfRegulationSCP2 \
    --seq_len 1152 \
    --batch_size 16 \
    --d_model 128 \
    --d_ff 256 \
    --d_model_bd 64 \
    --d_ff_bd 128 \
    --e_layers_bd 2 \
    --n_heads_bd 2 \
    --patch_len 32 \
    --stride 16 \
    --train_epochs 45 \
    --bd_train_epochs 45 \
    --trigger_epochs 18 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# === SharePriceIncrease (short-term forecasting trigger) ===
# Seq: 60, Vars: 1, Classes: 2
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/SharePriceIncrease \
    --seq_len 60 \
    --batch_size 32 \
    --d_model 32 \
    --d_ff 64 \
    --d_model_bd 16 \
    --d_ff_bd 32 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 7 \
    --stride 3 \
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

# === Sleep (medium-term forecasting trigger) ===
# Seq: 178, Vars: 1, Classes: 5
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/Sleep \
    --seq_len 178 \
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

# === SpokenArabicDigits (short-term forecasting trigger) ===
# Seq: 93, Vars: 13, Classes: 10
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/SpokenArabicDigits \
    --seq_len 93 \
    --batch_size 32 \
    --d_model 32 \
    --d_ff 64 \
    --d_model_bd 24 \
    --d_ff_bd 48 \
    --e_layers_bd 1 \
    --n_heads_bd 4 \
    --patch_len 11 \
    --stride 5 \
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

# === Strawberry (medium-term forecasting trigger) ===
# Seq: 235, Vars: 1, Classes: 2
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/Strawberry \
    --seq_len 235 \
    --batch_size 32 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
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

# === UWaveGestureLibrary (medium-term forecasting trigger) ===
# Seq: 315, Vars: 3, Classes: 8
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/UWaveGestureLibrary \
    --seq_len 315 \
    --batch_size 16 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 20 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

# === WalkingSittingStanding (medium-term forecasting trigger) ===
# Seq: 206, Vars: 3, Classes: 6
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/WalkingSittingStanding \
    --seq_len 206 \
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

# === Wine (medium-term forecasting trigger) ===
# Seq: 234, Vars: 1, Classes: 2
python main.py \
    --mode marksman \
    --model iTransformer \
    --Tmodel itst \
    --root_path ./dataset/Wine \
    --seq_len 234 \
    --batch_size 8 \
    --d_model 64 \
    --d_ff 128 \
    --d_model_bd 32 \
    --d_ff_bd 64 \
    --e_layers_bd 1 \
    --n_heads_bd 2 \
    --patch_len 16 \
    --stride 8 \
    --train_epochs 30 \
    --bd_train_epochs 30 \
    --trigger_epochs 20 \
    --clean_pretrain_epochs 0 \
    --lr 0.001 \
    --trigger_lr 0.001 \
    --target_label 0 \
    --poisoning_ratio 0.1 \
    --clip_ratio 0.1 \
    --gpu_id cuda:0

