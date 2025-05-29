import numpy as np
import torch

def load_sample_data():
    data = np.load("data/real_ppg_data.npz")
    X_target = torch.tensor(data["X_target"], dtype=torch.float32)
    X_calib = torch.tensor(data["X_calib"], dtype=torch.float32)
    y = torch.tensor(data["Y"], dtype=torch.float32)

    # Normalize BP to range [0, 1]
    y_min = y.min(dim=0)[0]
    y_max = y.max(dim=0)[0]
    y_norm = (y - y_min) / (y_max - y_min)

    return X_target, X_calib, y_norm, y_min, y_max

