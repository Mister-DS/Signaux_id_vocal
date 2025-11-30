import librosa
import numpy as np
import os
import noisereduce as nr
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def extract_features(file_path):
    """
    Extraction des MFCC (timbre de voix) et Deltas (vitesse et accélération)
    """
    try:
        if not os.path.exists(file_path):
            return None

        # 1. LOAD AUDIO
        y, sr = librosa.load(file_path, sr=16000)

        # ---------------------------------------------------------
        # STEP A: AMPLITUDE NORMALIZATION
        # ---------------------------------------------------------
        # This ensures the audio always fills the range [-1, 1]
        # Even if you recorded quietly, this boosts it to max volume.
        y = librosa.util.normalize(y)

        # ---------------------------------------------------------
        # STEP B: NOISE REDUCTION
        # ---------------------------------------------------------
        # Do this AFTER normalization so the noise floor is consistent
        try:
            y = nr.reduce_noise(
                y=y, sr=sr, stationary=True, prop_decrease=0.4)
        except:
            pass

        # ---------------------------------------------------------
        # STEP C: TRIM SILENCE
        # ---------------------------------------------------------
        y, _ = librosa.effects.trim(y, top_db=20)
        if len(y) < 1024:
            return None

        # ---------------------------------------------------------
        # STEP D: EXTRACT FEATURES
        # ---------------------------------------------------------
        # We increase n_mfcc to 20 to capture more detail
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # *** CRITICAL FIX: IGNORE VOLUME ***
        # The 0th coefficient is just "Volume". We discard it.
        # We keep coefficients 1 through 19 (Timbre only).
        mfcc = mfcc[1:]

        # ---------------------------------------------------------
        # STEP E: CEPSTRAL MEAN SUBTRACTION (CMS)
        # ---------------------------------------------------------
        # This removes the "Microphone EQ" effect
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

        # Deltas
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        return np.vstack([mfcc, delta, delta2]).T

    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None
