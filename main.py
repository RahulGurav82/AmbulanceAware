import librosa
import numpy as np
from sklearn.svm import SVC
import joblib

# Fixed length for MFCC feature vectors
FIXED_LENGTH = 1000

# Load and preprocess audio data
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Pad or truncate MFCCs to fixed length
    if mfccs.shape[1] < FIXED_LENGTH:
        # Pad with zeros
        mfccs = np.pad(mfccs, ((0, 0), (0, FIXED_LENGTH - mfccs.shape[1])), mode='constant')
    else:
        # Truncate to fixed length
        mfccs = mfccs[:, :FIXED_LENGTH]
    return mfccs.flatten()

# Example dataset (MPEG files)
audio_files = ['siren1.mpeg', 'siren2.mpeg', 'noise1.mpeg', 'noise2.mpeg']
labels = [1, 1, 0, 0]  # 1 for siren, 0 for noise

X = np.array([load_audio(file) for file in audio_files])
y = np.array(labels)

# Train SVM model
model = SVC(probability=True)
model.fit(X, y)

# Save the model
joblib.dump(model, 'siren_detector_model.pkl')
