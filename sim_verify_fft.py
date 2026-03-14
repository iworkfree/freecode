import numpy as np
import matplotlib.pyplot as plt

# --- Settings matching your FPGA config ---
NUM_CHIRPS = 10
NUM_SAMPLES = 500  # FFT Length
FS_ADC = 18e6      # 18 MHz

def process_xfft_data(file_path):
    # 1. Load raw data from text file
    # Assuming the file is one 32-bit hex/int value per line: [Imaginary(16b) | Real(16b)]
    raw_data = np.loadtxt(file_path, dtype=np.int64)
    
    # 2. Extract Real and Imaginary parts (Bit-masking)
    # If your text file is already split into columns, adjust this part.
    real_parts = (raw_data & 0xFFFF).astype(np.int16)
    imag_parts = ((raw_data >> 16) & 0xFFFF).astype(np.int16)
    
    # Combine into complex numbers
    complex_data = real_parts + 1j * imag_parts
    
    # 3. Reshape into [Chirps x Samples] Matrix
    # This creates the "Stack"
    fft_stack = complex_data.reshape((NUM_CHIRPS, NUM_SAMPLES))
    
    # 4. Calculate Magnitude (Convert to Pixel Intensity)
    # We use log scale (dB) to match your reference image style
    mag_stack = np.abs(fft_stack)
    mag_db = 20 * np.log10(mag_stack + 1e-6) # Avoid log(0)
    
    # 5. Perform 2D FFT (Doppler processing)
    # Shift zero-velocity to the middle
    rd_map = np.fft.fftshift(np.fft.fft(fft_stack, axis=0), axes=0)
    rd_mag_db = 20 * np.log10(np.abs(rd_map) + 1e-6)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: The 1D FFT Stack (Waterfall)
    # Note: Using aspect='auto' because 10x500 is very thin otherwise
    im1 = ax1.imshow(mag_db, aspect='auto', cmap='magma', origin='lower')
    ax1.set_title("1D FFT Stack (Range-Time Waterfall)")
    ax1.set_ylabel("Chirp Index (Time)")
    ax1.set_xlabel("Range Bins")
    plt.colorbar(im1, ax=ax1, label="Magnitude (dB)")

    # Plot 2: The 2D FFT (Range-Doppler Map)
    im2 = ax2.imshow(rd_mag_db, aspect='auto', cmap='magma', origin='lower',
                     extent=[0, NUM_SAMPLES, -NUM_CHIRPS//2, NUM_CHIRPS//2])
    ax2.set_title("2D FFT (Range-Doppler Map)")
    ax2.set_ylabel("Doppler Bin (Velocity)")
    ax2.set_xlabel("Range Bins")
    plt.colorbar(im2, ax=ax2, label="Magnitude (dB)")

    plt.tight_layout()
    plt.show()

# Run the processor
# process_xfft_data("xfft_output.txt")
