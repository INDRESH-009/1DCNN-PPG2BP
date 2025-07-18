import numpy as np;
import matplotlib.pyplot as plt;
data = np.genfromtxt('data/2_1.txt',delimiter='\t',filling_values=np.nan);
plt.plot(data);
plt.title('PPG Signal');
plt.xlabel('Sample Index');
plt.ylabel('Amplitude');
plt.grid(True);
plt.show();