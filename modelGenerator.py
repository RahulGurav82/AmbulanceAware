import sounddevice as sd
import numpy as np

# Function to extract features from live audio
def extract_features_from_audio(audio, sample_rate=22050):
    # Extract MFCC features from the audio signal
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Function to detect siren in real-time
def real_time_siren_detection(model, duration=2, sample_rate=22050):
    print("Listening for siren...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished

    # Flatten the recorded audio to a 1D array
    audio = audio.flatten()

    # Extract features from the recorded audio
    features = extract_features_from_audio(audio, sample_rate)
    features = np.expand_dims(features, axis=0)  # Reshape for prediction

    # Use the trained model to make a prediction
    prediction = model.predict(features)
    
    # Print the result
    if prediction == 1:
        print("Siren Detected!")
    else:
        print("No Siren Detected.")

# Example: Call this function with your trained model
real_time_siren_detection(model)
