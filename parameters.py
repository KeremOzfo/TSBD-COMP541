import argparse
import torch.nn as nn

def args_parser():
    parser = argparse.ArgumentParser(description="TimesNet Clean + Backdoor Training Pipeline")

    # ==================== GENERAL SETUP ====================
    parser.add_argument('--data', type=str, default='UEA',
                        help='dataset type (UEA for UWave / UEA datasets)')
    parser.add_argument('--task_name', type=str, default='classification',
                        help='task type (classification, forecasting, etc.)')
    parser.add_argument('--method', type=str, default='basic',
                        choices=['basic', 'vanilla', 'marksman', 'diversity', 'ultimate', 'frequency', 'inputaware', 'pureinputaware', 'inputaware_masking', 'defeat'],
                        help='trigger training method (basic=vanilla, marksman, diversity, ultimate, frequency, inputaware, pureinputaware, inputaware_masking, defeat)')
    parser.add_argument('--train_mode', type=str, default='backdoor',
                    choices=['clean', 'backdoor', 'test'],
                    help='training mode (test = evaluate pre-trained trigger model)')
    parser.add_argument('--model', type=str, default='TimesNet',
                        choices=['TimesNet', 'lstm', 'PatchTST', 'iTransformer', 'TimeMixer', 'resnet', 'mlp', 'nonstationary_transformer', 'TCN', 'BiRNN', 'DLinear'],
                        help='main model architecture to use')
    parser.add_argument('--use_pretrained_trigger', action='store_true',
                        help='use pre-trained trigger model')
    parser.add_argument('--bd_type', type=str, default='all2one',
                        choices=['all2one', 'all2all'],
                        help='Trigger traning mode (all2one: fixed target, all2all: random target)')            

    # ==================== TESTING MODE CONFIGURATION ====================
    parser.add_argument('--trigger_model_path', type=str, default=None,
                        help='path to pre-trained trigger model checkpoint (required for test mode)')

    # ==================== DATASET CONFIGURATION ====================
    parser.add_argument('--root_path', type=str, default='./dataset/BasicMotions',
                        help='dataset root directory')
    parser.add_argument('--dataset_info_path', type=str, default='./scripts/dataset_info.csv',
                        help='path to dataset_info.csv used for auto architecture selection')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--trigger_batch_size', type=int, default=None,
                        help='batch size for trigger training (defaults to batch_size if not set)')
    parser.add_argument('--auto_batch_size', type=bool, default=True,
                        help='if True, auto-select batch_size based on dataset_info.csv')
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
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='optimizer for main model training (adam, adamw, sgd)')
    parser.add_argument('--optimizer_weight_decay', type=float, default=0.0,
                        help='weight decay (L2 regularization) for main model optimizer')
    parser.add_argument('--clean_data_ratio', type=float, default=1.0,
                        help='fraction of dataset used for clean pretraining before poisoning')

    # ==================== TRIGGER MODEL TRAINING PHASE ====================
    parser.add_argument('--Tmodel', type=str, default='cnn',
                        choices=['itst', 'patchtst', 'cnn', 'timesnet', 'ctimesnet', 'ccnn', 'cpatchtst', 'citst', 'ctimesfm', 'ccnn_cae'],
                        help='trigger model architecture')
    parser.add_argument('--surrogate_model', type=str, default=None,
                        choices=['timesnet', 'cnn', 'patchTST', 'itst', 'lstm', 'resnet', 'mlp',None],
                        help='surrogate classifier model type for trigger training, if None use same as main model')
    parser.add_argument('--trigger_epochs', type=int, default=5,
                        help='number of epochs for training the trigger model')
    parser.add_argument('--trigger_patience', type=int, default=0,
                        help='early stopping patience for trigger training (0 = disabled)')
    parser.add_argument('--trigger_lr', type=float, default=1e-3,
                        help='learning rate for trigger model')
    parser.add_argument('--trigger_opt', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='optimizer for trigger model')
    parser.add_argument('--trigger_weight_decay', type=float, default=0.0,
                        help='weight decay for trigger model optimizer')
    parser.add_argument('--trigger_grad_clip', type=float, default=0.0,
                        help='gradient clipping max norm for trigger optimizer (0 = no clipping)')
    parser.add_argument('--surrogate_lr', type=float, default=1e-3,
                        help='learning rate for surrogate trigger model')
    parser.add_argument('--surrogate_opt', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='optimizer for surrogate trigger model')
    parser.add_argument('--surrogate_weight_decay', type=float, default=0.0,
                        help='weight decay for surrogate model optimizer')
    parser.add_argument('--surrogate_grad_clip', type=float, default=0.0,
                        help='gradient clipping max norm for surrogate optimizer (0 = no clipping)')
    parser.add_argument('--surrogate_L2_penalty', type=float, default=0.0,
                        help='L2 regularization penalty for surrogate model')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='warm-up epochs for surrogate trigger model')
    parser.add_argument('--marksman_update_T', type=int, default=1,
                        help='update interval T for Marksman training')
    parser.add_argument('--marksman_alpha', type=float, default=0.5,
                        help='alpha parameter for Marksman training')
    parser.add_argument('--marksman_beta', type=float, default=0,
                        help='beta parameter for trigger magnitude penalty')
    parser.add_argument('--poisoning_ratio_train', type=float, default=0.1,
                        help='poisoning ratio during trigger training (for diversity mode)')
    parser.add_argument('--div_reg', type=float, default=1.0,
                        help='diversity regularization weight for diversity method')
    
    # ==================== COMBINED EPOCH LOSS WEIGHTS ====================
    parser.add_argument('--lambda_freq', type=float, default=1.0,
                        help='weight for frequency alignment loss in combined training')
    parser.add_argument('--lambda_div', type=float, default=1.0,
                        help='weight for diversity loss in combined training')
    parser.add_argument('--lambda_reg', type=float, default=1e-3,
                        help='weight for regularization loss in combined training')
    parser.add_argument('--lambda_cross', type=float, default=1.0,
                        help='weight for cross-trigger loss in combined training')
    parser.add_argument('--p_attack', type=float, default=0.5,
                        help='fraction of batch for backdoor samples (rho_b)')
    parser.add_argument('--p_cross', type=float, default=0.1,
                        help='fraction of batch for cross-trigger samples (rho_c)')
    
    # ==================== MASK NETWORK PARAMETERS (for inputaware_masking) ====================
    parser.add_argument('--mask_pretrain_epochs', type=int, default=25,
                        help='number of epochs to pre-train mask network')
    parser.add_argument('--mask_density', type=float, default=0.032,
                        help='target sparsity for mask (fraction of 1s)')
    parser.add_argument('--lambda_norm', type=float, default=100.0,
                        help='weight for mask sparsity loss')
    parser.add_argument('--mask_lr', type=float, default=0.001,
                        help='learning rate for mask optimizer')
    parser.add_argument('--mask_opt', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='optimizer for mask network')
    parser.add_argument('--mask_weight_decay', type=float, default=0.0,
                        help='weight decay for mask optimizer')
    parser.add_argument('--mask_hidden_dim', type=int, default=64,
                        help='hidden dimension for mask generator')
    parser.add_argument('--mask_d_model', type=int, default=64,
                        help='d_model for transformer-based mask generator')
    parser.add_argument('--mask_n_heads', type=int, default=4,
                        help='number of heads for transformer-based mask generator')
    parser.add_argument('--mask_n_layers', type=int, default=2,
                        help='number of layers for transformer-based mask generator')
    parser.add_argument('--attack_only_nontarget', type=bool, default=True,
                        help='if True, only attack samples that are not already in the target class')

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
    
    # Silent Poisoning Configuration
    parser.add_argument('--use_silent_poisoning', type=bool, default=False,
                        help='if True, use silent poisoning (clean-label backdoor); if False, use normal poisoning')
    parser.add_argument('--lambda_ratio', type=float, default=0.2,
                        help='fraction of poisoned samples that keep original label (clean-label backdoor) in silent poisoning mode')

    # ==================== MODEL POISONING PHASE ====================
    parser.add_argument('--bd_train_epochs', type=int, default=5,
                        help='number of backdoor training epochs (training main model with poisoned data)')

    # ==================== AUTO ARCHITECTURE SELECTION ====================
    parser.add_argument('--auto_bd_arch', type=bool, default=True,
                        help='if True, auto-select d_model_bd/d_ff_bd/e_layers_bd/n_heads_bd based on dataset_info.csv')

    # ==================== LOGGING AND VISUALIZATION ====================
    parser.add_argument('--save_test_samples', action='store_true',
                        help='if True, save test samples and GradCAM visualizations; if False, skip saving them')
    parser.add_argument('--save_trigger_model', action='store_true', default=True,
                        help='if True, save the trained trigger model in the experiment directory')
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
    parser.add_argument('--patch_len', type=int, default=10,
                        help='patch length for PatchTST (default 10 for seq_len=100)')
    parser.add_argument('--stride', type=int, default=10,
                        help='stride for patch extraction (default 10 for seq_len=100)')

    # ---------------- TCN Model Config ----------------
    parser.add_argument('--tcn_channels', type=int, nargs='+', default=[64, 64],
                        help='channel sizes for Temporal Convolutional Network blocks')
    parser.add_argument('--tcn_kernel', type=int, default=3,
                        help='kernel size for TCN blocks')

    # ---------------- BiRNN Model Config ----------------
    parser.add_argument('--rnn_hidden', type=int, default=128,
                        help='hidden size for BiRNN classifier')
    parser.add_argument('--rnn_layers', type=int, default=1,
                        help='number of layers for BiRNN classifier')
    parser.add_argument('--rnn_type', type=str, default='gru', choices=['gru', 'lstm'],
                        help='cell type for BiRNN classifier')

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
    
    # ---------------- Non-Stationary Model Config ----------------
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # ==================== DEFEAT METHOD PARAMETERS ====================
    parser.add_argument('--beta_1', type=float, default=1.0,
                        help='DEFEAT: weight for clean loss (default: 1.0)')
    parser.add_argument('--beta_2', type=float, default=0.1,
                        help='DEFEAT: weight for adversarial loss (default: 0.1)')
    parser.add_argument('--epsilon_budget', type=float, default=0.1,
                        help='DEFEAT: stealthiness budget for trigger magnitude (default: 0.1)')
    parser.add_argument('--defeat_iterations', type=int, default=20,
                        help='DEFEAT: number of alternating optimization iterations R (default: 20)')
    parser.add_argument('--clean_pretrain_epochs', type=int, default=50,
                        help='DEFEAT: epochs for clean model pre-training (default: 50)')
    parser.add_argument('--clean_pretrain_lr', type=float, default=0.01,
                        help='DEFEAT: learning rate for clean pre-training (default: 0.01)')
    parser.add_argument('--finetune_lr', type=float, default=0.001,
                        help='DEFEAT: learning rate for poisoning fine-tuning (default: 0.001)')
    parser.add_argument('--aux_logit_epochs', type=int, default=10,
                        help='DEFEAT: epochs for training auxiliary logits (default: 10)')
    parser.add_argument('--aux_logit_lr', type=float, default=0.001,
                        help='DEFEAT: learning rate for auxiliary logits (default: 0.001)')

    args = parser.parse_args()
    args.criterion = nn.CrossEntropyLoss()

    # Handle vanilla alias
    if args.method == 'vanilla':
        args.method = 'basic'  # vanilla is alias for basic

    return args