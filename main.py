import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

# Fixed length for MFCC feature vectors
FIXED_LENGTH = 1000

# Load and preprocess audio data
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    # Pad or truncate features to fixed length
    mfccs = pad_or_truncate(mfccs)
    spectral_contrast = pad_or_truncate(spectral_contrast)
    tonnetz = pad_or_truncate(tonnetz)
    
    # Combine features into a single feature vector
    features = np.concatenate([mfccs.flatten(), spectral_contrast.flatten(), tonnetz.flatten()])
    
    return features

def pad_or_truncate(feature):
    if feature.shape[1] < FIXED_LENGTH:
        # Pad with zeros
        feature = np.pad(feature, ((0, 0), (0, FIXED_LENGTH - feature.shape[1])), mode='constant')
    else:
        # Truncate to fixed length
        feature = feature[:, :FIXED_LENGTH]
    return feature

# Example dataset (MPEG files)
audio_files = ['siren1.mpeg', 'siren2.mpeg', 'noise1.mpeg', 'noise2.mpeg']
labels = [1, 1, 0, 0]  # 1 for siren, 0 for noise

# Load and preprocess the dataset
X = np.array([load_audio(file) for file in audio_files])
y = np.array(labels)

# Set up SVM with grid search for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
model = GridSearchCV(SVC(probability=True), param_grid, cv=2)  # Changed cv from 5 to 2

# Train the model
model.fit(X, y)

# Save the best model
joblib.dump(model.best_estimator_, 'siren_detector_model.pkl')

print(f"Best model parameters: {model.best_params_}")
