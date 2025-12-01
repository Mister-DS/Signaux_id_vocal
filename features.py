import librosa
import numpy as np
import os
import noisereduce as nr
import warnings
from colorama import Fore, Style

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def extract_features(file_path):
    """Extraction des MFCC (timbre de voix) et Deltas (vitesse et accélération)"""
    try:
        if not os.path.exists(file_path):
            return None

        y, sr = librosa.load(file_path, sr=16000)

        # Normalisation de l'amplitude
        y = librosa.util.normalize(y)

        # Réduction de bruit
        try:
            y = nr.reduce_noise(
                y=y, sr=sr, stationary=True, prop_decrease=0.4)
        except:
            pass

        # Trimming du silence
        y, _ = librosa.effects.trim(y, top_db=20)
        if len(y) < 1024:
            return None

        # Extraction des MFCC en ignorant le volume absolu
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc = mfcc[1:]
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        return np.vstack([mfcc, delta, delta2]).T

    except Exception as e:
        print(f"{Fore.RED}[ERROR] extracting {
              file_path}: {e}{Style.RESET_ALL}")
        return None
