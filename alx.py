import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# — load your data —
path = '/Users/indreshmr/ALX-METHOD-PPG/win11.npz'
data = np.load(path)
ppg = data['PPG_Record_F']
abp = data['ABP_F']

fs       = data['fs']        # should be 125
time     = np.arange(len(ppg)) / fs
min_dist = int(0.4 * fs)

# — find PPG & ABP peaks/valleys —
ppg_peaks,    _ = find_peaks(ppg,distance=min_dist)
ppg_valleys,  _ = find_peaks(-ppg,distance=min_dist)
abp_peaks,    _ = find_peaks(abp,distance=min_dist)
abp_valley,   _ = find_peaks(-abp,distance=min_dist)

dicrotic_notches = []
min_dist = int(0.4 * fs)

for pk in abp_peaks:
    # 1) find the next diastolic foot
    feet = abp_valley[abp_valley > pk]
    if feet.size == 0:
        continue
    foot = feet[0]

    # 2) extract the down‐stroke
    segment = abp[pk:foot]
    if segment.size < 5:
        continue

    # 3) find all local minima in that window
    troughs, _ = find_peaks(-segment, distance=5, prominence=0.005)
    #    — distance=5 ensures we don’t pick super‐tiny noise
    #    — prominence filters out the tiny blips

    # 4) drop any minima that lie at the very end (the actual valley)
    troughs = [i for i in troughs if i < len(segment)-2]

    if not troughs:
        # fallback: just pick the first significant dip
        rel_min = np.argmin(segment[:-2])
    else:
        # 5) pick the *first* local minimum
        rel_min = troughs[0]

    dicrotic_notches.append(pk + rel_min)

# — compute PPG offset exactly as before —
ppg_first_valley = ppg_valleys[0]
abp_first_valley = abp_valley[0]
if ppg_first_valley < abp_first_valley:
    x = ppg_first_valley
    y = abp_valley[abp_valley > x][0]
    offset = -(y - x)
else:
    x = abp_first_valley
    y = ppg_valleys[ppg_valleys > x][0]
    offset = y - x
print(offset)

# — shift ABP‐notches into PPG index‐space and drop out‐of‐bounds —
dn_ppg = [dn + offset for dn in dicrotic_notches]
dn_ppg = [idx for idx in dn_ppg if 0 <= idx < len(ppg)]


# amplitudes of all detected PPG peaks and troughs
ppg_peak_amps   = ppg[ppg_peaks]
ppg_trough_amps = ppg[ppg_valleys]
ppg_dn_amps = ppg[dn_ppg]


# compute means
mean_peak   = np.mean(ppg_peak_amps)
mean_trough = np.mean(ppg_trough_amps)
mean_dn = np.mean(ppg_dn_amps)

# “absolute amplitude” as you defined it
absolute_amplitude = mean_peak - mean_trough
absolute_amplitude2 = mean_dn - mean_trough

#print(f"Mean peak amplitude:   {mean_peak:.4f}")
#print(f"Mean trough amplitude: {mean_trough:.4f}")
print(f"Absolute amplitude:    {absolute_amplitude:.4f}")
print(f"Absolute amplitude 2:    {absolute_amplitude2:.4f}")
alx = (absolute_amplitude-absolute_amplitude2)/absolute_amplitude
print("ALX = ",alx)
print(data['SBP'])
print(data['DBP'])


# — Plot PPG with its notches —
plt.figure()
plt.plot(time, ppg,label='PPG')
plt.plot(time[ppg_peaks],ppg[ppg_peaks],'ro',label='Peaks')
plt.plot(time[ppg_valleys],ppg[ppg_valleys],'bo',label='Valleys')
plt.hlines(mean_peak,   xmin=time[0], xmax=time[-1], colors='r', linestyles='--', label="Mean Peak")
plt.hlines(mean_trough, xmin=time[0], xmax=time[-1], colors='b', linestyles='--', label="Mean Trough")
plt.hlines(mean_dn, xmin=time[0], xmax=time[-1], colors='b', linestyles='--', label="Mean DN")
plt.plot(time[dn_ppg],ppg[dn_ppg],'go',label='DN')
plt.title('PPG with Dicrotic Notches')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

# — Plot ABP with its notches —
plt.figure()
plt.plot(time,abp,label='ABP')
plt.plot(time[abp_peaks],abp[abp_peaks],'ro', label='Peaks')
plt.plot(time[abp_valley],abp[abp_valley],'bo', label='Valleys')
plt.plot(time[dicrotic_notches], abp[dicrotic_notches], 'go', label='DN')
plt.title('ABP with Dicrotic Notches')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()
