import sounddevice as sd
import numpy as np
import librosa
import joblib
from collections import deque
import time

# Load the pre-trained model
model = joblib.load('siren_detector_model.pkl')

# Fixed length for MFCC feature vectors (ensure this matches the training phase)
FIXED_LENGTH = 2000  # Adjusted based on the discrepancy in features

# Buffer to store incoming audio chunks
audio_buffer = deque(maxlen=10)  # Holds the last 10 chunks of audio for batch processing

# Track recent predictions for smoothing
recent_predictions = deque(maxlen=5)

# Flag to control the detection and message printing
siren_detected = False
siren_stop_time = None

# Function to extract MFCCs from audio
def extract_features(y, sr):
    n_mfcc = 13  # Ensure this matches what was used during training
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] < FIXED_LENGTH:
        mfccs = np.pad(mfccs, ((0, 0), (0, FIXED_LENGTH - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :FIXED_LENGTH]
    return mfccs.flatten()

# Process accumulated audio chunks
def process_audio_batch():
    if len(audio_buffer) == 0:
        return
    batch = np.concatenate(audio_buffer)
    features = extract_features(batch, sr)
    
    # Get prediction probabilities
    prediction_prob = model.predict_proba([features])[0]
    
    # Debugging: Print the prediction probabilities
    print(f"Prediction probabilities: {prediction_prob}")
    
    # Get the class with the highest probability
    prediction = np.argmax(prediction_prob)
    
    # Add the prediction to the recent predictions deque
    recent_predictions.append(prediction)
    
    # Check if the siren is detected (assuming class 1 is the siren)
    if sum(recent_predictions) > 3:  # Require 3 positive predictions out of the last 5
        return True
    return False

# Callback function to process audio in real-time
def audio_callback(indata, frames, time, status):
    global siren_detected, siren_stop_time
    
    if status:
        print(f"Status: {status}")  # Print status message for debugging

    try:
        # Store audio chunks in the buffer
        y = indata[:, 0].astype(np.float32)
        audio_buffer.append(y)

        # Process audio in larger chunks to avoid real-time overload
        if len(audio_buffer) >= 10:  # Process every 10 chunks
            if process_audio_batch():
                if not siren_detected:
                    print("Ambulance siren detected!")
                siren_detected = True
                siren_stop_time = None
            else:
                if siren_detected and siren_stop_time is None:
                    siren_stop_time = time.time()

    except Exception as e:
        print(f"Error processing audio: {e}")

# Parameters for the microphone stream
sr = 16000    # Reduce sample rate to reduce processing load
BUFFER_SIZE = 8192  # Increase buffer size to allow more processing time

# Infinite loop to continuously listen for ambulance sirens
def listen_for_sirens():
    global siren_detected, siren_stop_time
    
    print("Listening for ambulance sirens...")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sr, blocksize=BUFFER_SIZE, latency='high'):
        while True:
            if siren_detected and siren_stop_time is not None:
                if time.time() - siren_stop_time >= 2:
                    print("Siren stopped.")
                    siren_detected = False
                    siren_stop_time = None

            sd.sleep(100)  # Adjust the sleep time to your needs

# Start the continuous listening process
listen_for_sirens()
