import pyaudio
import numpy as np
import librosa
import joblib
import RPi.GPIO as GPIO
import time

# Load the trained model
model = joblib.load('siren_detector_model.pkl')

# Set up GPIO
LED_PIN = 18
BUZZER_PIN = 23  # Optional for audible alert
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)  # Optional

# Initialize PyAudio
p = pyaudio.PyAudio()

def process_audio(in_data):
    y = np.frombuffer(in_data, dtype=np.float32)
    mfccs = librosa.feature.mfcc(y=y, sr=44100, n_mfcc=13)
    input_data = np.expand_dims(mfccs.flatten(), axis=0)
    return input_data

def callback(in_data, frame_count, time_info, status):
    input_data = process_audio(in_data)
    prediction = model.predict(input_data)
    if prediction == 1:  # Assuming 1 is the label for ambulance siren
        GPIO.output(LED_PIN, GPIO.HIGH)
        GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Optional for audible alert
        time.sleep(0.5)  # Keep LED and buzzer on for half a second
    else:
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.output(BUZZER_PIN, GPIO.LOW)  # Optional for audible alert
    return (in_data, pyaudio.paContinue)

# Open stream
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True,
                stream_callback=callback)

# Start stream
stream.start_stream()

# Keep the stream active
try:
    while stream.is_active():
        pass
except KeyboardInterrupt:
    pass

# Stop stream
stream.stop_stream()
stream.close()
p.terminate()

# Clean up GPIO
GPIO.cleanup()