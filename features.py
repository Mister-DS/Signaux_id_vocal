# features.py
import librosa
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def extract_features(file_path):
    """
    Extraction des MFCC (timbre de voix) et Deltas (vitesse et accélération)
    """
    try:
        if not os.path.exists(file_path):
            print(f"Erreur: Fichier {file_path} non trouvé")
            return None

        y, sr = librosa.load(file_path, sr=16000)
        y, _ = librosa.effects.trim(y, top_db=20)  # Suppression du silence
        if len(y) < 1024:
            return None

        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=20)  # Capture de 20 MFCC
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.vstack([mfcc, delta, delta2])
        return features.T  # Transposition pour sklearn

    except Exception as e:
        print(f"Erreur d'extraction de features dans {
              os.path.basename(file_path)}: {e}")
        return None
