""" Helper functions dataset fomation """
import torch
from data_provider.data_factory import data_provider
def subset_dataset(dataset, ratio):
    """Create a subset of the dataset based on ratio.
    
    Args:
        dataset (Dataset): training dataset
        ratio (float): fraction of the data (0.0 <  < 1.0)
    
    Returns:
        subset (Dataset): subset of the training dataset
    """
    if ratio >= 1.0:
        return dataset
    import random
    idx = list(range(len(dataset)))
    random.shuffle(idx)
    cut = int(len(idx) * ratio)
    selected = idx[:cut]
    return torch.utils.data.Subset(dataset, selected)


def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader
