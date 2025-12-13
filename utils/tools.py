import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math



def cal_accuracy(y_pred, y_true):
    """Compute classification accuracy as mean of correct predictions."""
    return np.mean(y_pred == y_true)