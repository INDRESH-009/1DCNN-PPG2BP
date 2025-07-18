import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# — load your data —
path = '/Users/indreshmr/ALX-METHOD-PPG/win11.npz'
data = np.load(path)
ppg        = data['PPG_Record_F']
abp        = data['ABP_F']
sbp_labels = data['SBP']    # one SBP per beat
dbp_labels = data['DBP']    # one DBP per beat
fs         = data['fs']     # e.g. 125
time       = np.arange(len(ppg)) / fs
min_dist   = int(0.4 * fs)

# — find peaks & valleys in PPG & ABP —
ppg_peaks,    _ = find_peaks(ppg,  distance=min_dist)
ppg_valleys,  _ = find_peaks(-ppg, distance=min_dist)
abp_peaks,    _ = find_peaks(abp,  distance=min_dist)
abp_valleys,  _ = find_peaks(-abp, distance=min_dist)

# — find dicrotic notches in ABP (as before) —
dicrotic_notches = []
for pk in abp_peaks:
    feet = abp_valleys[abp_valleys > pk]
    if feet.size == 0:
        continue
    foot = feet[0]
    segment = abp[pk:foot]
    if segment.size < 5:
        continue
    troughs, _ = find_peaks(-segment, distance=5, prominence=0.005)
    troughs = [i for i in troughs if i < len(segment)-2]
    rel_min = troughs[0] if troughs else np.argmin(segment[:-2])
    dicrotic_notches.append(pk + rel_min)

# — align ABP indices onto PPG timebase —
first_ppg_valley = ppg_valleys[0]
first_abp_valley = abp_valleys[0]
if first_ppg_valley < first_abp_valley:
    offset = -(abp_valleys[abp_valleys > first_ppg_valley][0] - first_ppg_valley)
else:
    offset = ppg_valleys[ppg_valleys > first_abp_valley][0] - first_abp_valley

dn_ppg = np.array(dicrotic_notches) + offset
dn_ppg = dn_ppg[(dn_ppg >= 0) & (dn_ppg < len(ppg))]

# — now compute ALX *per beat* — 
#   ALX_i = (peak_i − notch_i) / (peak_i − trough_i)
#   we need the PPG peak, PPG valley and PPG “notch” for each beat

# make dictionaries for quick lookup
valley_set = set(ppg_valleys)
dn_set     = set(dn_ppg)

X_alx = []
y_sbp = []
y_dbp = []
for i, pk in enumerate(ppg_peaks):
    # find the preceding valley (foot of that beat) & the following valley
    prev_valleys = [v for v in ppg_valleys if v < pk]
    next_valleys = [v for v in ppg_valleys if v > pk]
    if not prev_valleys or not next_valleys:
        continue
    trough_idx = prev_valleys[-1]
    # find the notch that lands between trough and peak
    candidate_dn = [dn for dn in dn_ppg if trough_idx < dn < pk]
    if not candidate_dn:
        continue
    notch_idx = candidate_dn[-1]   # should be just after the peak, pick the closest
    # amplitude values
    A_peak   = ppg[pk]
    A_trough = ppg[trough_idx]
    A_notch  = ppg[notch_idx]
    # compute ALX for this beat
    alx_i = (A_peak - A_notch) / (A_peak - A_trough)
    X_alx.append(alx_i)
    # grab corresponding SBP/DBP label for this beat
    # assuming SBP[i] & DBP[i] align 1:1 with abp_peaks
    # adjust this indexing if your label array is offset differently
    y_sbp.append(sbp_labels[i])
    y_dbp.append(dbp_labels[i])

X_alx = np.array(X_alx).reshape(-1, 1)
y_sbp  = np.array(y_sbp)
y_dbp  = np.array(y_dbp)

print(f"Number of beats used: {len(X_alx)}")

# — fit two simple linear regressions —
model_sbp = LinearRegression().fit(X_alx, y_sbp)
model_dbp = LinearRegression().fit(X_alx, y_dbp)

# — print out metrics —
for name, model, y in [("SBP", model_sbp, y_sbp), ("DBP", model_dbp, y_dbp)]:
    y_pred = model.predict(X_alx)
    print(f"\n{name} regression:")
    print(f"  Coef:      {model.coef_[0]:.3f}")
    print(f"  Intercept: {model.intercept_:.3f}")
    print(f"  R²:        {r2_score(y, y_pred):.3f}")
    print(f"  RMSE:      {np.sqrt(mean_squared_error(y, y_pred)):.3f}")

# — optional: plot fit for SBP —
plt.figure(figsize=(6,4))
plt.scatter(X_alx, y_sbp, label="true SBP")
plt.plot(X_alx, model_sbp.predict(X_alx), 'r-', label="fit")
plt.xlabel("ALX")
plt.ylabel("SBP (mmHg)")
plt.legend()
plt.title("Linear fit: ALX → SBP")
plt.grid()
plt.show()
