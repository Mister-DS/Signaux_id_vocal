# auth_engines.py
import os
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from features import extract_features


class GMMVerifier:
    def __init__(self, model_dir="voice_models"):
        self.models = {}
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def enroll(self, name, files):
        print(f"[GMM] Enrolling {name}...")
        feats = [extract_features(f) for f in files]
        valid_feats = [f for f in feats if f is not None]

        if not valid_feats:
            return

        X = np.vstack(valid_feats)

        # Random needs to be 'wider' (more components)
        n_comp = 64 if name == "Random" else 16

        gmm = GaussianMixture(n_components=n_comp,
                              covariance_type='diag', n_init=3)
        gmm.fit(X)

        self.models[name] = gmm
        joblib.dump(gmm, os.path.join(self.model_dir, f"{name}.gmm"))

    def verify(self, test_file, safety_margin=5.0):
        feat = extract_features(test_file)
        if feat is None:
            return "Error", 0.0

        if "Random" not in self.models:
            return "Error: No Random Model", 0.0

        ubm_score = self.models["Random"].score(feat)
        best_spk = "Unknown"
        best_margin = -float('inf')

        for name, model in self.models.items():
            if name == "Random":
                continue

            score = model.score(feat)
            margin = score - ubm_score

            if margin > best_margin:
                best_margin = margin
                if margin > safety_margin:
                    best_spk = name

        return best_spk, best_margin


class DTWVerifier:
    def __init__(self):
        self.templates = {}

    def enroll(self, name, files):
        print(f"[DTW] Storing templates for {name}...")
        # Pre-calculate features to save time during auth
        self.templates[name] = []
        for f in files:
            feat = extract_features(f)
            if feat is not None:
                self.templates[name].append(feat)

    def verify(self, claimed_name, test_file):
        if claimed_name not in self.templates:
            return float('inf')

        test_feat = extract_features(test_file)
        if test_feat is None:
            return float('inf')

        # Compare against ALL user templates, take BEST match
        best_dist = float('inf')

        for ref_feat in self.templates[claimed_name]:
            dist, path = fastdtw(ref_feat, test_feat, dist=euclidean)
            norm_dist = dist / len(path)
            if norm_dist < best_dist:
                best_dist = norm_dist

        return best_dist
