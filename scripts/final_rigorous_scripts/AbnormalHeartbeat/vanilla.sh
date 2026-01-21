#!/bin/bash
# Dataset: AbnormalHeartbeat
# Method: vanilla
# Seq Len: 18530, Variates: 1, Classes: 2


# ========== TRIGGER MODEL: citst ==========

# citst | opt=adam_sgd_best | cfg=default | warmup=0 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# citst | opt=adam_sgd_best | cfg=default | warmup=0 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# citst | opt=adam_sgd_best | cfg=default | warmup=0 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# citst | opt=adam_sgd_best | cfg=default | warmup=5 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# citst | opt=adam_sgd_best | cfg=default | warmup=5 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# citst | opt=adam_sgd_best | cfg=default | warmup=5 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# citst | opt=sgd_sgd_best | cfg=default | warmup=0 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# citst | opt=sgd_sgd_best | cfg=default | warmup=0 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# citst | opt=sgd_sgd_best | cfg=default | warmup=0 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# citst | opt=sgd_sgd_best | cfg=default | warmup=5 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# citst | opt=sgd_sgd_best | cfg=default | warmup=5 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# citst | opt=sgd_sgd_best | cfg=default | warmup=5 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# citst | opt=sgd_adam_best | cfg=default | warmup=0 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# citst | opt=sgd_adam_best | cfg=default | warmup=0 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# citst | opt=sgd_adam_best | cfg=default | warmup=0 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# citst | opt=sgd_adam_best | cfg=default | warmup=5 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# citst | opt=sgd_adam_best | cfg=default | warmup=5 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# citst | opt=sgd_adam_best | cfg=default | warmup=5 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel citst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1


# ========== TRIGGER MODEL: cpatchtst ==========

# cpatchtst | opt=adam_sgd_best | cfg=default | warmup=0 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# cpatchtst | opt=adam_sgd_best | cfg=default | warmup=0 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# cpatchtst | opt=adam_sgd_best | cfg=default | warmup=0 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# cpatchtst | opt=adam_sgd_best | cfg=default | warmup=5 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# cpatchtst | opt=adam_sgd_best | cfg=default | warmup=5 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# cpatchtst | opt=adam_sgd_best | cfg=default | warmup=5 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# cpatchtst | opt=sgd_sgd_best | cfg=default | warmup=0 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# cpatchtst | opt=sgd_sgd_best | cfg=default | warmup=0 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# cpatchtst | opt=sgd_sgd_best | cfg=default | warmup=0 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# cpatchtst | opt=sgd_sgd_best | cfg=default | warmup=5 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# cpatchtst | opt=sgd_sgd_best | cfg=default | warmup=5 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# cpatchtst | opt=sgd_sgd_best | cfg=default | warmup=5 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# cpatchtst | opt=sgd_adam_best | cfg=default | warmup=0 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# cpatchtst | opt=sgd_adam_best | cfg=default | warmup=0 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# cpatchtst | opt=sgd_adam_best | cfg=default | warmup=0 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# cpatchtst | opt=sgd_adam_best | cfg=default | warmup=5 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# cpatchtst | opt=sgd_adam_best | cfg=default | warmup=5 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# cpatchtst | opt=sgd_adam_best | cfg=default | warmup=5 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel cpatchtst --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1


# ========== TRIGGER MODEL: ctimesnet ==========

# ctimesnet | opt=adam_sgd_best | cfg=default | warmup=0 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# ctimesnet | opt=adam_sgd_best | cfg=default | warmup=0 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# ctimesnet | opt=adam_sgd_best | cfg=default | warmup=0 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# ctimesnet | opt=adam_sgd_best | cfg=default | warmup=5 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# ctimesnet | opt=adam_sgd_best | cfg=default | warmup=5 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# ctimesnet | opt=adam_sgd_best | cfg=default | warmup=5 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt adam --trigger_lr 0.001 --trigger_weight_decay 0.0 --surrogate_opt sgd --surrogate_lr 0.01 --surrogate_weight_decay 0.0005 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# ctimesnet | opt=sgd_sgd_best | cfg=default | warmup=0 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# ctimesnet | opt=sgd_sgd_best | cfg=default | warmup=0 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# ctimesnet | opt=sgd_sgd_best | cfg=default | warmup=0 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# ctimesnet | opt=sgd_sgd_best | cfg=default | warmup=5 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# ctimesnet | opt=sgd_sgd_best | cfg=default | warmup=5 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# ctimesnet | opt=sgd_sgd_best | cfg=default | warmup=5 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.05 --trigger_weight_decay 0.001 --surrogate_opt sgd --surrogate_lr 0.05 --surrogate_weight_decay 0.001 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# ctimesnet | opt=sgd_adam_best | cfg=default | warmup=0 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# ctimesnet | opt=sgd_adam_best | cfg=default | warmup=0 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# ctimesnet | opt=sgd_adam_best | cfg=default | warmup=0 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 0 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

# ctimesnet | opt=sgd_adam_best | cfg=default | warmup=5 | grad_clip=no_clip | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 0.0 --surrogate_grad_clip 0.0 --poisoning_ratio 0.1

# ctimesnet | opt=sgd_adam_best | cfg=default | warmup=5 | grad_clip=clip_5 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 5.0 --surrogate_grad_clip 5.0 --poisoning_ratio 0.1

# ctimesnet | opt=sgd_adam_best | cfg=default | warmup=5 | grad_clip=clip_10 | epochs=30 | poison=normal_10pct
python -u main.py --train_mode backdoor --method vanilla --Tmodel ctimesnet --bd_type all2all --root_path ./dataset/AbnormalHeartbeat --seq_len 18530 --trigger_epochs 30 --bd_train_epochs 40 --warmup_epochs 5 --target_label 0 --auto_bd_arch True --auto_batch_size True --patch_len 32 --stride 16 --trigger_opt sgd --trigger_lr 0.01 --trigger_weight_decay 0.0005 --surrogate_opt adam --surrogate_lr 0.001 --surrogate_weight_decay 0.0 --surrogate_L2_penalty 0.0 --trigger_grad_clip 10.0 --surrogate_grad_clip 10.0 --poisoning_ratio 0.1

