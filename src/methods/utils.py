import numpy as np
def cal_accuracy(preds, trues):
    """Compute accuracy given predicted labels and true labels (numpy or array-like)."""
    preds_np = np.asarray(preds).reshape(-1)
    trues_np = np.asarray(trues).reshape(-1)
    if preds_np.shape[0] == 0:
        return 0.0
    return float(np.mean(preds_np == trues_np))