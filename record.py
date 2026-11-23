
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

sample_rate = 48000
channels = 1  # Mono audio

filename = input("Enter audio filename (without extension): ").strip()

if not filename:
    filename = "recording"

print("\nPress ENTER to start recording...")
input()

print("Recording... Press ENTER to stop.")

recorded_audio = []

def callback(indata, frames, time, status):
    if status:
        print(status)
    recorded_audio.append(indata.copy())

with sd.InputStream(
    samplerate=sample_rate,
    channels=channels,
    callback=callback
):
    input()

# Combine recorded audio chunks
audio = np.concatenate(recorded_audio, axis=0)

audio = (audio * 32767).astype(np.int16)

output_file = filename + ".wav"
write(output_file, sample_rate, audio)

print(f"\nAudio saved successfully: {output_file}")
