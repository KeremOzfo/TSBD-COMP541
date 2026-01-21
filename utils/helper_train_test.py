
import torch
import numpy as np
import functools
from torch.utils.data import DataLoader, Subset
#########Model Import (Target Models)#########
from src.models.target_models.TimesNet.TimesNet import Model as TimesNet 
from src.models.target_models.PatchTST.PatchTST import Model as PatchTST
from src.models.target_models.iTransformer.iTransformer import Model as iTransformer
from src.models.target_models.TimeMixer.TimeMixer import Model as TimeMixer
from src.models.target_models.PatchTST.wrappers.PatchTST_Classifier import PatchTST_Classifier
from src.models.target_models.LSTM import LSTMClassifier as LSTM
from src.models.target_models.Nonstationary_Transformer import Model as Nonstationary_Transformer
from src.models.target_models.ResNet1D import Model as ResNet1D
from src.models.target_models.MLPClassifier import Model as MLPClassifier
from src.models.target_models.TCN import Model as TCNClassifier
from src.models.target_models.BiRNN import Model as BiRNNClassifier
from src.models.target_models.dlinear import Model as DLinear
#########Model Import (Trigger Models)#########
from src.models.trigger_models.bd_TimesNet.bd_TimesNet import Model as Bd_TimesNet
from src.models.trigger_models.bd_patchTST.bd_patchTST import Model as Bd_PatchTST
from src.models.trigger_models.bd_inverted import Model as Bd_Inverted
from src.models.trigger_models.bd_cnn import bd_CNN

# Conditional Trigger Models (Robust imports)
try:
    from src.models.trigger_models.bdc_TimesNet.bdc_TimesNet import Model as Bdc_TimesNet
    from src.models.trigger_models.bdc_patchTST.bdc_patchTST import Model as Bdc_PatchTST
    from src.models.trigger_models.bdc_inverted import Model as Bdc_Inverted
    from src.models.trigger_models.bdc_cnn import bdc_CNN
    from src.models.trigger_models.bdc_cnn_cae import bdc_CNN_CAE
    MODELS_BDC_AVAILABLE = True
except ImportError:
    MODELS_BDC_AVAILABLE = False

try:
    from src.models.trigger_models.bdc_timesFM import Model as Bdc_TimesFM
    TIMESFM_AVAILABLE = True
except ImportError:
    TIMESFM_AVAILABLE = False
#########Logging and Data#########
from utils.exp_logging import log_result_clean, log_all
from data_provider.data_factory import data_provider
from data_provider.uea import collate_fn

# Model dictionaries for dynamic instantiation
target_model_dict = {
    'timesnet': TimesNet,
    'lstm': LSTM,
    'patchtst': PatchTST_Classifier,
    'itransformer': iTransformer,
    'timemixer': TimeMixer,
    'nonstationary_transformer': Nonstationary_Transformer,
    'resnet': ResNet1D,
    'mlp': MLPClassifier,
    'tcn': TCNClassifier,
    'birnn': BiRNNClassifier,
    'dlinear': DLinear
}

trigger_model_dict = {
    'timesnet': Bd_TimesNet,
    'cnn': bd_CNN,
    'patchtst': Bd_PatchTST,
    'itst': Bd_Inverted
}

# Add conditional models if available
if MODELS_BDC_AVAILABLE:
    trigger_model_dict.update({
        'ctimesnet': Bdc_TimesNet,
        'ccnn': bdc_CNN,
        'cpatchtst': Bdc_PatchTST,
        'citst': Bdc_Inverted,
        'ccnn_cae': bdc_CNN_CAE
    })

if TIMESFM_AVAILABLE:
    trigger_model_dict.update({
        'ctimesfm': Bdc_TimesFM
    })

def create_trigger_model(args):
    """Create trigger model instance."""
    model_class = trigger_model_dict[args.Tmodel]
    return model_class(args)


def get_data(args, flag):
    """Load dataset and dataloader for the specified flag (train/test)."""
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

def reconfigure_model_for_data(args, train_data, test_data, model_type=None):
    """Initialize a clean classification model.
    
    Args:
        args: Training arguments
        train_data: Training dataset (for inferring dimensions)
        test_data: Test dataset (for inferring max sequence length)
        model_type: Optional model type override
        
    Returns:
        Initialized model on the specified device
    """
    # NOTE: args.seq_len should already be set correctly and match DataLoader's max_len
    # Do not overwrite it here
    # args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)

    # classification: pred_len = 0
    args.pred_len = 0

    # feature dimension
    args.enc_in = train_data.feature_df.shape[1]

    # NUMBER OF CLASSES
    args.num_class = int(train_data.labels_df.nunique().values[0])

    # Model init
    if model_type is not None:
        model = target_model_dict[model_type.lower()](args).float().to(args.device)
    else:
        model = target_model_dict[args.model.lower()](args).float().to(args.device)

    return model