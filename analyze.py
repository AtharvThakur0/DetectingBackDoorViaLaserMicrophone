from __future__ import annotations

import argparse
import time
from pathlib import Path
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
import ruptures as rpt  

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", type=str, help="Path to the WAV file to analyze.")
    args = parser.parse_args()
    # 1. Load the WAV file
    sample_rate, data = wavfile.read(args.audio_file)

    # Convert stereo to mono if needed
    if data.ndim > 1:
        data = data.mean(axis=1)

    # 2. Compute the full 1D Real FFT
    n_samples = len(data)
    fft_magnitudes = np.abs(rfft(data))
    freqs = rfftfreq(n_samples, d=1.0 / sample_rate)
    # Use log scale (dB) so spectral floor/slope shifts are easier to detect
    
    # 3. Detect change points along the frequency axis
    # Reshape 1D array to (n_freqs, 1) as required by ruptures
    signal =fft_magnitudes

    # "l2" detects mean shifts; "clinear" or "ar" works well for slope/roll-off changes
    algo = rpt.Binseg(model="l2", min_size=100).fit(signal)
    change_indices = algo.predict(pen=500)  # Increase 'pen' for fewer breakpoints

    # Convert bin indices to Frequencies (Hz), excluding the final array length index
    change_freqs = [freqs[idx - 1] for idx in change_indices[:-1]]
    print("Detected frequency change points (Hz):", np.round(change_freqs, 2))

    # 4. Plot the full FFT spectrum and detected frequency breakpoints
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, log_spectrum, color="steelblue", linewidth=0.7, label="Log Magnitude")

    for cp_freq in change_freqs:
        plt.axvline(x=cp_freq, color="red", linestyle="--", linewidth=1.5)

    plt.title("Full FFT Spectrum with Detected Frequency Change Points")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, sample_rate / 2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
