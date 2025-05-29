import os
import pandas as pd
import numpy as np

PPG_FOLDER = "data"
EXCEL_PATH = "data/PPG-BP dataset.xlsx"
SAVE_PATH = "data/real_ppg_data.npz"

def read_ppg_file(file_path):
    with open(file_path, "r") as f:
        content = f.read().replace("\n", "").strip()  # remove newlines
        numbers = content.split("\t")  # split by tab
        return np.array([float(x) for x in numbers if x.strip() != ""])


def preprocess():
    df = pd.read_excel(EXCEL_PATH, header=1)
    X_target, X_calib, Y = [], [], []

    for i in range(len(df)):
        sid = df.iloc[i]['subject_ID']
        sbp = df.iloc[i]['Systolic Blood Pressure(mmHg)']
        dbp = df.iloc[i]['Diastolic Blood Pressure(mmHg)']

        try:
            ppg_target = read_ppg_file(f"{PPG_FOLDER}/{sid}_1.txt")  # prediction input
            ppg_calib = read_ppg_file(f"{PPG_FOLDER}/{sid}_2.txt")   # calibration input
        except Exception as e:
            print(f"Skipping subject {sid} due to error: {e}")
            continue

        # Keep only 1000 samples
        if len(ppg_target) >= 1000 and len(ppg_calib) >= 1000:
            ppg_target = ppg_target[:1000]
            ppg_calib = ppg_calib[:1000]

            X_target.append(ppg_target)
            X_calib.append(ppg_calib)
            Y.append([sbp, dbp])

            print(f"✅ Processed subject {sid}: calib={len(ppg_calib)}, target={len(ppg_target)}")
        else:
            print(f"⛔ Skipped {sid} due to short signals")

    X_target = np.array(X_target).reshape(-1, 1, 1000)
    X_calib = np.array(X_calib).reshape(-1, 1, 1000)
    Y = np.array(Y)

    print("Saving data:", X_target.shape, X_calib.shape, Y.shape)
    np.savez(SAVE_PATH,
             X_target=X_target,
             X_calib=X_calib,
             Y=Y)
    
    data = np.load("data/real_ppg_data.npz")
    print(data.files)  # should show ['X_target', 'X_calib', 'Y']


if __name__ == "__main__":
    preprocess()