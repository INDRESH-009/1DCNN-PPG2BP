import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from models.ppg2bpnet import PPG2BPNet

def bandpass_filter(signal, low=0.5, high=10.0, fs=100, order=2):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def read_ppg_file(file_path, plot=False):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")
    
    with open(file_path, "r") as f:
        content = f.read().replace("\n", "").strip()
        signal = [float(x) for x in content.split("\t") if x.strip() != ""]
        signal = np.array(signal)

    # ✅ Trim or pad to 1000
    if len(signal) < 1000:
        signal = np.pad(signal, (0, 1000 - len(signal)), mode='edge')
    else:
        signal = signal[:1000]

    # ✅ Filter the signal (but DO NOT normalize)
    filtered_signal = bandpass_filter(signal, low=0.5, high=5.0, fs=100, order=2)

    # ✅ Optional plot
    if plot:
        plt.figure(figsize=(10, 3))
        plt.plot(filtered_signal, label="Bandpass Filtered PPG", color='blue')
        plt.title("Bandpass Filtered PPG Signal (No Normalization)")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

    return filtered_signal


def predict_from_ppg(ppg_path, calib_path=None):
    # ✅ Load model
    model = PPG2BPNet()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    # ✅ Load normalization bounds
    norm = np.load("bp_norm_bounds.npz")
    y_min = torch.tensor(norm["y_min"], dtype=torch.float32)
    y_max = torch.tensor(norm["y_max"], dtype=torch.float32)

    # ✅ Read PPG signals (filtered only)
    target = read_ppg_file(ppg_path, plot=True)
    calib = read_ppg_file(calib_path, plot=False) if calib_path else target

    # ⭐ Trim the signals: remove first 170 and last 100 samples
    trimmed_target = target[170:-100]
    trimmed_calib = calib[170:-100] if calib_path else trimmed_target

    # ⭐ Plot trimmed data
    plt.figure(figsize=(10, 3))
    plt.plot(trimmed_target, label="Trimmed Target PPG", color='green')
    plt.title("Trimmed Bandpass Filtered PPG Signal (Target)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    # ✅ Convert to tensors using a copy to avoid negative strides
    X_target = torch.tensor(trimmed_target.copy().reshape(1, 1, -1), dtype=torch.float32)
    X_calib = torch.tensor(trimmed_calib.copy().reshape(1, 1, -1), dtype=torch.float32)

    # ✅ Run model
    with torch.no_grad():
        y_pred = model(X_target, X_calib).numpy()

    # ✅ Clamp and denormalize
    y_pred = np.clip(y_pred, 0, 1)
    y_pred = y_pred * (y_max.numpy() - y_min.numpy()) + y_min.numpy()
    sbp, dbp = y_pred[0]

    print(f"🧠 Predicted SBP: {sbp:.2f} mmHg")
    print(f"🧠 Predicted DBP: {dbp:.2f} mmHg")

# ✅ Example usage
predict_from_ppg("disanth.txt")
