import librosa
import numpy as np
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.preprocessing import scale


class TemporalPassphraseMatcher:
    def __init__(self):
        pass

    def extract_dynamic_features(self, file_path):
        """
        Extracts features specifically for DTW.
        We need MFCCs + Deltas (velocity) + Delta-Deltas (acceleration)
        to capture the 'movement' of the voice.
        """
        try:
            y, sr = librosa.load(file_path, sr=16000)
            # CRITICAL: Trim silence. DTW hates silence at start/end.
            y, _ = librosa.effects.trim(y, top_db=20)

            # 1. MFCCs (The spectral shape)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # 2. Delta (The speed of change)
            delta = librosa.feature.delta(mfcc)

            # 3. Delta-Delta (The acceleration)
            delta2 = librosa.feature.delta(mfcc, order=2)

            # Stack them: Shape = (39, Time_Steps)
            features = np.vstack([mfcc, delta, delta2])

            # Transpose to (Time_Steps, 39) for DTW
            # Normalize (Scale) so volume differences don't break it
            return scale(features.T, axis=0)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def compare_phrases(self, reference_file, test_file):
        """
        Calculates the DTW distance between two recordings.
        Lower Distance = Better Match.
        """
        ref_feat = self.extract_dynamic_features(reference_file)
        test_feat = self.extract_dynamic_features(test_file)

        if ref_feat is None or test_feat is None:
            return float('inf')

        # Run Dynamic Time Warping
        # dist is the 'cost' to align the two signals
        distance, path = fastdtw(ref_feat, test_feat, dist=euclidean)

        # Normalize distance by length of the path
        # Otherwise, long passwords naturally have higher costs than short ones
        normalized_distance = distance / len(path)

        return normalized_distance

# --- INTEGRATING WITH YOUR AUTH SYSTEM ---


if __name__ == "__main__":
    # Assume we already have the GMM system from before for the "Voice Check"
    # match_phrase = TemporalPassphraseMatcher()

    # Files
    enrollment = "samples/p13/simon_1.wav"
    attempt_correct = "samples/p13/simon_2.wav"  # Alice saying same phrase
    attempt_wrong_content = "samples/p13/simon_4.wav"  # Alice saying different words
    attempt_impostor = "samples/p13/simon_15.wav"  # Bob saying correct words

    matcher = TemporalPassphraseMatcher()

    print("--- 1. Alice saying the correct phrase ---")
    score_1 = matcher.compare_phrases(enrollment, attempt_correct)
    print(f"DTW Distance: {score_1:.2f}")
    # EXPECT: Low score (e.g., < 40)

    print("\n--- 2. Alice saying WRONG words ---")
    score_2 = matcher.compare_phrases(enrollment, attempt_wrong_content)
    print(f"DTW Distance: {score_2:.2f}")
    # EXPECT: High score (e.g., > 60) because the temporal shape is different

    print("\n--- 3. Bob saying the CORRECT phrase ---")
    score_3 = matcher.compare_phrases(enrollment, attempt_impostor)
    print(f"DTW Distance: {score_3:.2f}")
    # EXPECT: Medium/High score.
    # DTW isn't purely biometric, but Bob's timing/accent will differ from Alice's.

    # --- DECISION LOGIC ---
    THRESHOLD = 45.0  # You need to tune this number based on your microphone

    if score_1 < THRESHOLD:
        print("\n[PASS] Phrase Pattern Matches")
    else:
        print("\n[FAIL] Phrase Pattern Mismatch")
