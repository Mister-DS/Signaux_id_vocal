import os
import librosa
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output in PoC
warnings.filterwarnings('ignore')


class GMMVoiceAuth:
    def __init__(self, n_components=16, model_dir="voice_models"):
        """
        n_components: The number of 'clusters' to model. 
                      16-32 is good for a PoC. 
                      128+ is used in production (requires more data).
        """
        self.n_components = n_components
        self.model_dir = model_dir
        self.models = {}

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def extract_features(self, audio_path):
        """
        Extracts MFCCs + Deltas.
        Returns a matrix of shape (n_frames, n_features)
        """
        try:
            # Load audio (downsample to 16khz for consistency)
            y, sr = librosa.load(audio_path, sr=16000)

            # Remove silence (simple trim)
            y, _ = librosa.effects.trim(y)

            # Extract MFCCs (The 'Timbre')
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

            # Extract Delta (Speed/Change) - Crucial for GMMs
            mfcc_delta = librosa.feature.delta(mfcc)

            # Stack them: We get a matrix of shape (40, n_frames)
            features = np.vstack([mfcc, mfcc_delta])

            # Transpose to (n_frames, 40) for scikit-learn
            return features.T

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None

    def enroll_user(self, name, audio_files):
        """
        Trains a GMM for a specific user using a list of their audio files.
        """
        print(f"--- Enrolling User: {name} ---")
        features_list = []

        # 1. Aggregate all features from all enrollment files
        for file in audio_files:
            feat = self.extract_features(file)
            if feat is not None:
                features_list.append(feat)

        if not features_list:
            print("No valid audio found.")
            return

        # Stack all frames vertically
        X = np.vstack(features_list)

        # 2. Train the GMM
        # This fits the 'clusters' to the user's specific voice data
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',  # 'diag' is faster and standard for audio
            n_init=3,               # Run 3 times and take best fit
            verbose=0
        )
        gmm.fit(X)

        # 3. Save the model
        self.models[name] = gmm
        model_path = os.path.join(self.model_dir, f"{name}.gmm")
        joblib.dump(gmm, model_path)
        print(f"Success! Model saved to {model_path}")

    def verify_speaker(self, name, test_file):
        """
        Verifies if a specific file belongs to 'name'.
        Returns a score (Log Likelihood).
        """
        if name not in self.models:
            print(f"User {name} not found in database.")
            return -99999

        features = self.extract_features(test_file)
        if features is None:
            return -99999

        gmm = self.models[name]

        # score() returns the average log-likelihood
        # Higher (closer to 0) is better. Lower (more negative) is worse.
        score = gmm.score(features)
        return score

    def identify_speaker(self, test_file):
        """
        Compares the test file against ALL enrolled users 
        and finds the best match.
        """
        features = self.extract_features(test_file)
        if features is None:
            return "Error"

        best_score = -float('inf')
        best_speaker = None

        print(f"\nScanning database for file: {
              os.path.basename(test_file)}...")

        for name, gmm in self.models.items():
            score = gmm.score(features)
            print(f"  Match vs {name}: {score:.2f}")

            if score > best_score:
                best_score = score
                best_speaker = name

        return best_speaker, best_score

# --- HOW TO RUN IT ---


if __name__ == "__main__":
    # Initialize
    auth = GMMVoiceAuth(n_components=16)

    # Assume folders: samples/Alice/, samples/Bob/, samples/Impostor/

    # 1. ENROLLMENT (Training)
    # We train Alice using 2 of her files
    simon_files = ["samples/p13/simon_1.wav", "samples/p13/simon_2.wav"]
    auth.enroll_user("Simon", simon_files)

    # We train Bob using 2 of his files
    romain_files = ["samples/p10/Romain_1.wav", "samples/p10/Romain_2.wav"]
    auth.enroll_user("Romain", romain_files)

    victor_files = ["samples/p12/Victor_1.wav", "samples/p12/Victor_2.wav"]
    auth.enroll_user("Victor", victor_files)

    random_files = ["samples/p08/cam_1.wav", "samples/p09/JIM_1.wav"]
    auth.enroll_user("Random", random_files)

    # 2. IDENTIFICATION (Testing)
    # Let's test with a new file from Alice
    unknown_file = "samples/test/simon_rienavoir.wav"

    winner, score = auth.identify_speaker(unknown_file)

    print(f"\n>>> RESULT: The speaker is identified as {
          winner} (Score: {score:.2f})")
