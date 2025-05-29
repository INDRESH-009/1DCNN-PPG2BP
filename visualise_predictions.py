import torch
import numpy as np
import matplotlib.pyplot as plt
from models.ppg2bpnet import PPG2BPNet
from utils import load_sample_data
from sklearn.metrics import mean_absolute_error

# 1. Load normalized data
X_target, X_calib, y_true, _, _ = load_sample_data()
y_true = y_true.numpy()

# 2. Load normalization bounds
norm_data = np.load("bp_norm_bounds.npz")
y_min = norm_data["y_min"]
y_max = norm_data["y_max"]

# 3. Load trained model
model = PPG2BPNet()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# 4. Predict
with torch.no_grad():
    y_pred = model(X_target, X_calib).numpy()

# 5. Denormalize predictions and ground truth
y_pred = y_pred * (y_max - y_min) + y_min
y_true = y_true * (y_max - y_min) + y_min

# 6. Plot SBP
plt.figure(figsize=(10, 4))
plt.plot(y_true[:, 0], label='True SBP', marker='o')
plt.plot(y_pred[:, 0], label='Pred SBP', marker='x')
plt.legend()
plt.title("Systolic BP Prediction")
plt.xlabel("Sample")
plt.ylabel("SBP (mmHg)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Plot DBP
plt.figure(figsize=(10, 4))
plt.plot(y_true[:, 1], label='True DBP', marker='o')
plt.plot(y_pred[:, 1], label='Pred DBP', marker='x')
plt.legend()
plt.title("Diastolic BP Prediction")
plt.xlabel("Sample")
plt.ylabel("DBP (mmHg)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Evaluate with MAE
mae_sbp = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
mae_dbp = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

print(f"📊 MAE - Systolic BP: {mae_sbp:.2f} mmHg")
print(f"📊 MAE - Diastolic BP: {mae_dbp:.2f} mmHg")
