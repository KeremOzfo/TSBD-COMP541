import os

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import math

import json

def cal_accuracy(y_pred, y_true):
    """Compute classification accuracy as mean of correct predictions."""
    return np.mean(y_pred == y_true)


def create_optimizer(model_parameters, optimizer_name, lr, weight_decay=0.0):
    """
    Create an optimizer instance based on the specified optimizer type.
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
    
    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from ['adam', 'adamw', 'sgd']")


import os
import json

def load_and_override_params(args):
    """
    Loads parameters from a model-specific JSON file and overrides args 
    based on the specific dataset.
    """
    json_path = f"./parameters/{args.model}_params.json"

    if not os.path.exists(json_path):
        print(f"[Warning] Parameter file not found: {json_path}. Using defaults.")
        return args

    try: 
        with open(json_path, 'r') as file:
            model_parameters = json.load(file)
    except Exception as e:
        print(f"[Error] Failed to load {json_path}: {e}")
        return args

    dataset_name = os.path.basename(args.root_path.rstrip('/'))
    model_name = args.model

    if model_name in model_parameters:
        dataset_configs = model_parameters[model_name]

        if dataset_name in dataset_configs:
            overrides = dataset_configs[dataset_name]
            print(f"[Config] Overriding parameters for {model_name} on {dataset_name}:")
            
            for key, value in overrides.items():
                # setattr dynamically updates the args object
                if hasattr(args, key):
                    old_val = getattr(args, key)
                    setattr(args, key, value)
                    print(f"  - {key}: {old_val} -> {value}")
                else:
                    # Logic for adding parameters not initially in the parser
                    setattr(args, key, value)
                    print(f"  - {key}: (new) -> {value}")
            print("")
        else:
            print(f"[Info] No dataset-specific config for '{dataset_name}' found in {json_path}.")
    else:
        print(f"[Info] Model key '{model_name}' not found in {json_path}.")
    
    return args
