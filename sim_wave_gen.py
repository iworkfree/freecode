import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
NUM_CHANNELS = 4
NUM_SAMPLES = 500
NUM_CHIRPS = 10
ADC_BITS = 12
MAX_VAL = (2**ADC_BITS) - 1

# --- Timing Specifications ---
FS_PER_CHANNEL = 18e6  # 18 MHz per channel
TS = 1 / FS_PER_CHANNEL
CHIRP_PERIOD = NUM_SAMPLES * TS # Duration of one chirp (approx 27.7 us)
FRAME_PERIOD = CHIRP_PERIOD * NUM_CHIRPS

# --- Data Generation ---
frame_data = np.zeros((NUM_CHIRPS, NUM_SAMPLES, NUM_CHANNELS))

# Create time vector for one chirp based on 18MHz sampling
t = np.arange(NUM_SAMPLES) * TS 

for chirp_idx in range(NUM_CHIRPS):
    for chan_idx in range(NUM_CHANNELS):
        # Target: 5 MHz beat frequency (Range)
        # Shift phase by 45 degrees per chirp to simulate Velocity
        target_freq = 5e6 
        velocity_phase = chirp_idx * (np.pi / 4)
        
        signal = (MAX_VAL // 2) + (MAX_VAL // 4) * np.sin(2 * np.pi * target_freq * t + velocity_phase)
        noise = np.random.normal(0, 20, NUM_SAMPLES)
        frame_data[chirp_idx, :, chan_idx] = np.clip(signal + noise, 0, MAX_VAL)

# --- 1. Processing Channel A ---
channel_a = frame_data[:, :, 0]

# --- 2. 1D Range-FFT (First Chirp) ---
chirp_zero = channel_a[0, :] - np.mean(channel_a[0, :])
range_fft = np.abs(np.fft.fft(chirp_zero))[:NUM_SAMPLES // 2]
freq_axis = np.fft.fftfreq(NUM_SAMPLES, TS)[:NUM_SAMPLES // 2] / 1e6 # MHz scale

# --- 3. 2D Range-Doppler FFT ---
detrended_2d = channel_a - np.mean(channel_a)
# Apply a window to reduce sidelobes (crucial for 10-chirp low resolution)
window = np.hanning(NUM_CHIRPS).reshape(-1, 1) * np.hanning(NUM_SAMPLES)
rd_map = np.fft.fft2(detrended_2d * window)
rd_map_shifted = np.abs(np.fft.fftshift(rd_map))

# Doppler axis in Hz
doppler_axis = np.fft.fftshift(np.fft.fftfreq(NUM_CHIRPS, CHIRP_PERIOD))

# --- Visualizing Results ---
plt.figure(figsize=(14, 6))

# Plot 1D FFT 
plt.subplot(1, 2, 1)
plt.plot(freq_axis, range_fft)
plt.title(f"1D Range-FFT (18MHz Sampling)\nTarget at {target_freq/1e6} MHz")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude")
plt.grid(True)

# Plot 2D Range-Doppler Map
plt.subplot(1, 2, 2)
# Extent maps the bins to physical units (MHz and Hz)
plt.imshow(rd_map_shifted, aspect='auto', origin='lower', 
           extent=[0, FS_PER_CHANNEL/(2*1e6), doppler_axis[0], doppler_axis[-1]])
plt.title("2D Range-Doppler FFT (Velocity Map)")
plt.xlabel("Range Frequency (MHz)")
plt.ylabel("Doppler Frequency (Hz)")
plt.colorbar(label="Intensity")

plt.tight_layout()
plt.show()

# --- Export Round-Robin Text File ---
# This remains the same to match your FPGA input requirements
with open("radar_frame.txt", 'w') as f:
    for chirp in range(NUM_CHIRPS):
        for sample in range(NUM_SAMPLES):
            for channel in range(NUM_CHANNELS):
                f.write(f"{int(frame_data[chirp, sample, channel])}\n")
