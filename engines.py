import os
import joblib
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from features import extract_features


class GMMVerifier:
    def __init__(self, model_dir="models"):
        self.models = {}
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def enroll(self, name, files):
        print(f"   [GMM] Entrainement de {name} ({len(files)} fichiers)...")
        feats = [extract_features(f) for f in files]
        valid_feats = [f for f in feats if f is not None]

        if not valid_feats:
            print("   /!\\ Aucun audio valide.")
            return

        X = np.vstack(valid_feats)
        n_comp = 64 if name == "Random" else 16
        gmm = GaussianMixture(n_components=n_comp,
                              covariance_type='diag', n_init=3)
        gmm.fit(X)
        self.models[name] = gmm

        # Save to disk
        joblib.dump(gmm, os.path.join(self.model_dir, f"{name}.gmm"))

    def load_models(self):
        """Loads all .gmm files from the models directory"""
        if not os.path.exists(self.model_dir):
            return
        for f in os.listdir(self.model_dir):
            if f.endswith(".gmm"):
                name = f.replace(".gmm", "")
                self.models[name] = joblib.load(
                    os.path.join(self.model_dir, f))
        print(f"   [GMM] {len(self.models)} modèles chargés.")

    def verify(self, test_file, safety_margin=5.0):
        feat = extract_features(test_file)
        if feat is None:
            return "Error", 0.0

        if "Random" not in self.models:
            return "Error (No UBM)", 0.0

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
    def __init__(self, model_dir="models"):
        self.templates = {}
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.db_path = os.path.join(model_dir, "dtw_templates.pkl")

    def enroll(self, name, files):
        print(f"   [DTW] Sauvegarde templates pour {name}...")
        self.templates[name] = []
        for f in files:
            feat = extract_features(f)
            if feat is not None:
                self.templates[name].append(feat)

        # Save DB
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.templates, f)

    def load_models(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.templates = pickle.load(f)
            print(f"   [DTW] Templates chargés pour {
                  len(self.templates)} utilisateurs.")

    def verify(self, claimed_name, test_file):
        if claimed_name not in self.templates:
            return float('inf')
        test_feat = extract_features(test_file)
        if test_feat is None:
            return float('inf')

        best_dist = float('inf')
        for ref_feat in self.templates[claimed_name]:
            dist, path = fastdtw(ref_feat, test_feat, dist=euclidean)
            norm_dist = dist / len(path)
            if norm_dist < best_dist:
                best_dist = norm_dist
        return best_dist
