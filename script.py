import wave
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file using the wave library
with wave.open('siren6.wav', 'rb') as wav_file:
    sample_rate = wav_file.getframerate()
    n_samples = wav_file.getnframes()
    duration = n_samples / sample_rate
    audio_data = np.frombuffer(wav_file.readframes(n_samples), dtype=np.int16)

# Normalize the audio data
audio_data = audio_data / np.max(np.abs(audio_data))

# Plot the waveform
plt.figure(figsize=(12, 4))
plt.plot(np.linspace(0, duration, num=len(audio_data)), audio_data)
plt.title('Waveform of siren6.wav')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

print(f'Duration: {duration} seconds')
print(f'Sample Rate: {sample_rate} Hz')
