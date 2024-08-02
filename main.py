import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Load and preprocess audio data
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfccs.flatten()

# Example dataset
audio_files = ['siren1.wav', 'siren2.wav', 'noise1.wav', 'noise2.wav']
labels = [1, 1, 0, 0]  # 1 for siren, 0 for noise

X = np.array([load_audio(file) for file in audio_files])
y = np.array(labels)

# Train SVM model
model = SVC(probability=True)
model.fit(X, y)

# Save the model
joblib.dump(model, 'siren_detector_model.pkl')