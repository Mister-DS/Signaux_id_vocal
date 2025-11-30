import librosa
import numpy as np
import os
import warnings
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

warnings.filterwarnings("ignore", category=UserWarning)


class DTWVoiceAuth:
    def __init__(self):
        self.user_templates = {}
        self.cache = {}

    def extract_dynamic_features(self, file_path):
        """
        Extracts robust MFCC + Deltas with CMS normalization.
        """
        # Simple caching to avoid re-reading the same enrollment files
        if file_path in self.cache:
            return self.cache[file_path]

        try:
            if not os.path.exists(file_path):
                return None

            y, sr = librosa.load(file_path, sr=16000)
            y, _ = librosa.effects.trim(y, top_db=20)

            if len(y) < 1024:
                return None

            # 1. MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # 2. CMS (Normalize)
            mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

            # 3. Deltas
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            features = np.vstack([mfcc, delta, delta2]).T

            # Store in cache
            self.cache[file_path] = features
            return features

        except Exception as e:
            print(f"Error extracting {file_path}: {e}")
            return None

    def enroll_user(self, name, file_paths):
        """
        Registers a list of valid reference files for a user.
        """
        print(f"--- Enrolling Templates for: {name} ---")
        valid_files = []
        for f in file_paths:
            if os.path.exists(f):
                # Pre-calculate features now to save time later
                if self.extract_dynamic_features(f) is not None:
                    valid_files.append(f)
            else:
                print(f"  Warning: File not found {f}")

        self.user_templates[name] = valid_files
        print(f"  {len(valid_files)} templates stored.")

    def verify_passphrase(self, claimed_name, test_file):
        """
        Compares test_file against ALL enrolled templates for this user.
        Returns the BEST (Lowest) distance found.
        """
        if claimed_name not in self.user_templates:
            return float('inf'), "User not enrolled"

        test_feat = self.extract_dynamic_features(test_file)
        if test_feat is None:
            return float('inf'), "Bad Audio"

        # Compare against every template in the user's gallery
        best_distance = float('inf')
        best_template = None

        templates = self.user_templates[claimed_name]

        for ref_file in templates:
            ref_feat = self.extract_dynamic_features(ref_file)

            # Run DTW
            dist, path = fastdtw(ref_feat, test_feat, dist=euclidean)
            normalized_dist = dist / len(path)

            # Keep the lowest score (Best Match)
            if normalized_dist < best_distance:
                best_distance = normalized_dist
                best_template = os.path.basename(ref_file)

        return best_distance, best_template


if __name__ == "__main__":
    dtw_auth = DTWVoiceAuth()

    # 1. ENROLLMENT (Multiple files per user)
    # Using the files from your previous GMM example
    simon_refs = ["samples/p13/simon_1.wav", "samples/p13/simon_2.wav"]
    dtw_auth.enroll_user("Simon", simon_refs)

    # 2. VALIDATION (Testing)

    # CASE A: Simon using the correct passphrase (new recording)
    # Ideally, this should be a 3rd file, but using one from enrollment is fine for testing logic
    test_good = "samples/p13/simon_3.wav"

    score, match = dtw_auth.verify_passphrase("Simon", test_good)
    print(f"\n[Test Good] Best Match: {match} | Distance: {score:.2f}")

    THRESHOLD = 50.0  # Tune this!

    if score < THRESHOLD:
        print("Passphrase Accepted.")
    else:
        print("Passphrase Rejected.")

    # CASE B: Simon saying wrong words
    test_wrong_words = "samples/p13/simon_4.wav"
    score, match = dtw_auth.verify_passphrase("Simon", test_wrong_words)
    print(f"[Test Wrong Words] Best Match: {match} | Distance: {score:.2f}")

    if score < THRESHOLD:
        print("Passphrase Accepted.")
    else:
        print("Passphrase Rejected.")

    # CASE C: Impostor (Nathan) trying to say Simon's phrase
    # (Assuming Nathan said the same words, his timing/accent will differ)
    test_impostor = "samples/p17/tiago_simon_3.wav"
    score, match = dtw_auth.verify_passphrase("Simon", test_impostor)
    print(f"[Test Impostor] Best Match: {match} | Distance: {score:.2f}")

    if score < THRESHOLD:
        print("Passphrase Accepted.")
    else:
        print("Passphrase Rejected.")
