import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

# Step 1: Generate a sample signal with noise
np.random.seed(42)  # Seed for reproducibility
fs = 500  # Sampling frequency in Hz
t = np.linspace(0, 1.0, fs)  # Time vector (1 second)
freq1 = 50  # Frequency of the first sine wave component
freq2 = 120  # Frequency of the second sine wave component

# Create a clean signal with two sine waves
clean_signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

# Add Gaussian noise to the clean signal
noise = 1.5 * np.random.normal(size=len(t))
noisy_signal = clean_signal + noise

# Step 2: Design a Butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    """
    Designs a low-pass Butterworth filter.
    :param cutoff: Cutoff frequency of the filter
    :param fs: Sampling frequency
    :param order: Order of the filter
    :return: Filter coefficients
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Applies the Butterworth low-pass filter to the data.
    :param data: Input signal
    :param cutoff: Cutoff frequency
    :param fs: Sampling frequency
    :param order: Filter order
    :return: Filtered signal
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter settings
cutoff = 60  # Cutoff frequency of the filter in Hz
order = 4  # Filter order

# Apply the low-pass filter to the noisy signal
filtered_signal = butter_lowpass_filter(noisy_signal, cutoff, fs, order)

# Step 3: Plotting the results
plt.figure(figsize=(14, 8))

# Plot the noisy signal
plt.subplot(3, 1, 1)
plt.plot(t, noisy_signal, color='red', alpha=0.7, label='Noisy Signal')
plt.title('Noisy Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

# Plot the clean signal
plt.subplot(3, 1, 2)
plt.plot(t, clean_signal, color='green', label='Clean Signal')
plt.title('Clean Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

# Plot the filtered signal
plt.subplot(3, 1, 3)
plt.plot(t, filtered_signal, color='blue', label='Filtered Signal')
plt.title('Filtered Signal')
plt.xlabel('Time [seconds]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()