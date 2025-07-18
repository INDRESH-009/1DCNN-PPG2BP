import numpy as np
import matplotlib.pyplot as plt

signal = np.loadtxt("data/2_1.txt")
plt.plot(signal)
plt.title("PPG Input Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
