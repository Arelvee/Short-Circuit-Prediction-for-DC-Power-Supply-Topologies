import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import sys

# Override numpy print options
np.set_printoptions(threshold=sys.maxsize)

# Load data from CSV file
df = pd.read_csv('final_buckCurrent.csv')

# Convert the DataFrame to a numpy array
data = df.values

# Define the wavelet type
wavelet = pywt.Wavelet('db4')

# Determine the maximum level of decomposition
max_level = pywt.dwt_max_level(data_len=len(data), filter_len=wavelet.dec_len)

# Perform the multilevel decomposition
coeffs = pywt.wavedec(data, wavelet, level=max_level)

# Print the coefficients
for i, coeff in enumerate(coeffs):
    print(f"Level {i} coefficients:")
    print(coeff)

    # Save coefficients to a CSV file
    np.savetxt(f'pywavelet_buckCurrent{i}.csv', coeff, delimiter=',')

# Plot the original data
plt.figure(figsize=(12, 8))
plt.subplot(max_level + 2, 1, 1)
plt.plot(data)
plt.title('Original Data')

# Plot the approximation and detail coefficients
for i, coeff in enumerate(coeffs):
    plt.subplot(max_level + 2, 1, i + 2)
    plt.plot(coeff)
    plt.title(f'Level {i} Coefficients')

plt.tight_layout()
plt.show()
