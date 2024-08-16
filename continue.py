import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sounddevice as sd

# Directories for siren and non-siren sounds
siren_dir = './siren_sounds'
non_siren_dir = './non_siren_sounds'

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def extract_features_from_audio(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Prepare the dataset
X = []
y = []

# Load and process siren sounds
for file_name in os.listdir(siren_dir):
    file_path = os.path.join(siren_dir, file_name)
    features = extract_features(file_path)
    X.append(features)
    y.append(1)  # Label for siren sounds

# Load and process non-siren sounds
for file_name in os.listdir(non_siren_dir):
    file_path = os.path.join(non_siren_dir, file_name)
    features = extract_features(file_path)
    X.append(features)
    y.append(0)  # Label for non-siren sounds

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Continuous real-time prediction function
def continuous_real_time_prediction(model, duration=2, sample_rate=22050):
    print("Starting continuous listening for sirens...")
    while True:
        print("Listening...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until the recording is finished
        audio = np.squeeze(audio)  # Remove single-dimensional entries from the shape of an array
        features = extract_features_from_audio(audio, sample_rate)
        features = np.expand_dims(features, axis=0)  # Reshape for prediction
        prediction = model.predict(features)
        if prediction == 1:
            print("Siren Detected!")
        else:
            print("No Siren Detected.")

# Call this function with your trained model
continuous_real_time_prediction(model)
