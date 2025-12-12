import argparse
import torch.nn as nn

def args_parser():
    parser = argparse.ArgumentParser(description="TimesNet Clean + Backdoor Training Pipeline")

    # ==================== GENERAL SETUP ====================
    parser.add_argument('--gpu_id', type=str, default='cuda:0',
                        help='GPU device ID')
    parser.add_argument('--data', type=str, default='UEA',
                        help='dataset type (UEA for UWave / UEA datasets)')
    parser.add_argument('--task_name', type=str, default='classification',
                        help='task type (classification, forecasting, etc.)')
    parser.add_argument('--mode', type=str, default='clean',
                        choices=['clean', 'basic', 'marksman', 'dynamic'],
                        help='training mode')
    parser.add_argument('--model', type=str, default='TimesNet',
                        choices=['TimesNet', 'LSTM', 'PatchTST', 'iTransformer', 'TimeMixer'],
                        help='main model architecture to use')

    # ==================== DATASET CONFIGURATION ====================
    parser.add_argument('--root_path', type=str, default='./dataset/UWaveGestureLibrary',
                        help='dataset root directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='dataloader workers')
    parser.add_argument('--drop_last', type=bool, default=False,
                        help='drop last incomplete batch')

    # ==================== SEQUENCE LENGTHS ====================
    parser.add_argument('--seq_len', type=int, default=256,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0,
                        help='start token length for decoder input')
    parser.add_argument('--pred_len', type=int, default=0,
                        help='prediction sequence length (0 for classification)')

    # ==================== CLEAN TRAINING PHASE ====================
    parser.add_argument('--train_epochs', type=int, default=5,
                        help='number of epochs for clean training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for main model')
    parser.add_argument('--clean_pretrain_epochs', type=int, default=5,
                        help='number of clean pretraining epochs before poisoning')
    parser.add_argument('--clean_data_ratio', type=float, default=1.0,
                        help='fraction of dataset used for clean pretraining before poisoning')

    # ==================== TRIGGER MODEL TRAINING PHASE ====================
    parser.add_argument('--Tmodel', type=str, default='cnn',
                        choices=['itst', 'patchtst', 'cnn', 'timesnet'],
                        help='trigger model architecture')
    parser.add_argument('--surrogate_model', type=str, default='timesnet',
                        choices=['timesnet', 'cnn', 'patchTST', 'itst'],
                        help='surrogate classifier model type for trigger training')
    parser.add_argument('--trigger_epochs', type=int, default=5,
                        help='number of epochs for training the trigger model')
    parser.add_argument('--trigger_lr', type=float, default=1e-3,
                        help='learning rate for trigger model')
    parser.add_argument('--trigger_opt', type=str, default='adam',
                        help='optimizer for trigger model')
    parser.add_argument('--surrogate_lr', type=float, default=1e-3,
                        help='learning rate for surrogate trigger model')
    parser.add_argument('--surrogate_opt', type=str, default='adam',
                        help='optimizer for surrogate trigger model')
    parser.add_argument('--surrogate_L2_penalty', type=float, default=0.0,
                        help='L2 regularization penalty for surrogate model')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='warm-up epochs for surrogate trigger model')
    parser.add_argument('--marksman_update_T', type=int, default=1,
                        help='update interval T for Marksman training')
    parser.add_argument('--marksman_alpha', type=float, default=0.5,
                        help='alpha parameter for Marksman training')
    parser.add_argument('--marksman_beta', type=float, default=0,
                        help='beta parameters for Marksman training')
    parser.add_argument('--poisoning_ratio_train', type=float, default=0.1,
                        help='poisoning ratio during trigger training (for diversity mode)')

    # ==================== DATA POISONING PHASE ====================
    parser.add_argument('--poisoning_ratio', type=float, default=0.1,
                        help='percentage of training samples to poison')
    parser.add_argument('--target_label', type=int, default=0,
                        help='backdoor attack target label')
    parser.add_argument('--clip_ratio', type=float, default=0.1,
                        help='basic patch trigger magnitude')
    parser.add_argument('--freq_lambda', type=float, default=0.05,
                        help='perturbation scale for frequency heatmap estimation')
    parser.add_argument('--freq_max_bins', type=int, default=256,
                        help='maximum frequency bins for heatmap estimation')

    # ==================== MODEL POISONING PHASE ====================
    parser.add_argument('--bd_train_epochs', type=int, default=5,
                        help='number of backdoor training epochs (training main model with poisoned data)')

    # ==================== LOGGING AND VISUALIZATION ====================
    parser.add_argument('--latent_method', type=str, default='pca',
                        choices=['pca', 'tsne'],
                        help='method for latent space visualization')
    parser.add_argument('--latent_max_points', type=int, default=2000,
                        help='maximum points to use for latent separability plotting')

    # ==================== MODEL-SPECIFIC CONFIGURATIONS ====================
    # ---------------- TimesNet Model Config ----------------
    parser.add_argument('--top_k', type=int, default=3,
                        help='top-k frequencies to select (default: 3, reduced from 5)')
    parser.add_argument('--num_kernels', type=int, default=4,
                        help='number of kernels for convolution (default: 4, reduced from 6)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding method')
    parser.add_argument('--freq', type=str, default='h',
                        help='frequency for time features encoding')

    # ---------------- PatchTST Model Config ----------------
    parser.add_argument('--d_model', type=int, default=64,
                        help='model dimension')
    parser.add_argument('--d_ff', type=int, default=128,
                        help='feed-forward dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='number of encoder layers')
    parser.add_argument('--factor', type=int, default=1,
                        help='probsparse attention factor')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='activation function (gelu, relu, etc.)')
    parser.add_argument('--output_attention', action='store_false', default=False,
                        help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16,
                        help='patch length for PatchTST')
    parser.add_argument('--stride', type=int, default=8,
                        help='stride for patch extraction')

    # ---------------- TimeMixer Model Config ----------------
    parser.add_argument('--down_sampling_layers', type=int, default=1)
    parser.add_argument('--down_sampling_window', type=int, default=2)
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        choices=['max', 'avg', 'conv'])
    parser.add_argument('--channel_independence', type=int, default=0)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        choices=['moving_avg', 'dft_decomp'])

    # ---------------- Backdoor Model Config ----------------
    parser.add_argument('--d_model_bd', type=int, default=32,
                        help='backdoor trigger model dimension (default: 32)')
    parser.add_argument('--d_ff_bd', type=int, default=64,
                        help='backdoor trigger model feed-forward dimension (default: 64)')
    parser.add_argument('--e_layers_bd', type=int, default=1,
                        help='backdoor trigger model encoder layers (default: 1)')
    parser.add_argument('--n_heads_bd', type=int, default=4,
                        help='number of attention heads for backdoor model')

    # ---------------- TimesFM Model Config ----------------
    parser.add_argument('--freeze_backbone', type=bool, default=True,
                        help='whether to freeze TimesFM backbone')
    parser.add_argument('--unfreeze_last_n_layers', type=int, default=2,
                        help='number of final TimesFM layers to unfreeze for fine-tuning')

    args = parser.parse_args()
    args.criterion = nn.CrossEntropyLoss()

    return args