import librosa
import numpy as np
import os
import warnings
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# 1. Suppress the PySoundFile warning (it's harmless noise)
warnings.filterwarnings("ignore", category=UserWarning)


class TemporalPassphraseMatcher:
    def __init__(self):
        pass

    def extract_dynamic_features(self, file_path):
        try:
            # Check if file exists first
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                return None

            # Load audio
            y, sr = librosa.load(file_path, sr=16000)

            # Trim silence (Top DB 20 removes background noise)
            y, _ = librosa.effects.trim(y, top_db=20)

            # Safety check: Is the file empty after trimming?
            if len(y) < 1024:
                print(f"Warning: Audio {file_path} is too short/silent.")
                return None

            # --- ROBUST FEATURE EXTRACTION ---

            # 1. MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # 2. Cepstral Mean Subtraction (The Fix for 'Numerical Issues')
            # We subtract the mean to normalize the channel, but we do NOT
            # divide by std deviation. This prevents the "Divide by Zero" crash.
            mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

            # 3. Deltas
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            # Stack and Transpose for DTW
            features = np.vstack([mfcc, delta, delta2]).T

            return features

        except Exception as e:
            print(f"CRITICAL ERROR in {file_path}: {e}")
            return None

    def compare_phrases(self, reference_file, test_file):
        print(f"Comparing: {os.path.basename(reference_file)} vs {
              os.path.basename(test_file)}")

        ref_feat = self.extract_dynamic_features(reference_file)
        test_feat = self.extract_dynamic_features(test_file)

        if ref_feat is None or test_feat is None:
            return float('inf')

        # FastDTW calculation
        distance, path = fastdtw(ref_feat, test_feat, dist=euclidean)

        # Normalize by path length
        return distance / len(path)


if __name__ == "__main__":
    matcher = TemporalPassphraseMatcher()

    # --- UPDATE THESE PATHS TO REAL FILES ---
    enrollment = "samples/p13/simon_1.wav"

    # Test 1: Correct Content
    attempt_correct = "samples/p13/simon_2.wav"

    # Test 2: Wrong Content
    attempt_wrong = "samples/p13/simon_4.wav"

    # Test 3: Impostor (UPDATE THIS to a real file, e.g., Romain or Cam)
    attempt_impostor = "samples/p13/simon_5.wav"

    print("--- TEST 1: Same Speaker, Same Phrase ---")
    s1 = matcher.compare_phrases(enrollment, attempt_correct)
    print(f"Score: {s1:.2f} (Target: LOW)\n")

    print("--- TEST 2: Same Speaker, WRONG Phrase ---")
    s2 = matcher.compare_phrases(enrollment, attempt_wrong)
    print(f"Score: {s2:.2f} (Target: HIGH)\n")

    print("--- TEST 3: Different Speaker ---")
    s3 = matcher.compare_phrases(enrollment, attempt_impostor)
    print(f"Score: {s3:.2f} (Target: MEDIUM/HIGH)\n")

    # Separation Check
    if s2 > s1 + 1.0:
        print(f"SUCCESS: System successfully distinguished the phrases by a margin of {
              s2-s1:.2f}")
    else:
        print("WARNING: Scores are too close. Check microphone quality.")
