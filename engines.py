import os
import joblib
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from features import extract_features


class GMMVerifier:
    """Manipulation de Gaussian Mixture Models pour identifier la personne"""

    def __init__(self, model_dir="models"):
        self.models = {}
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def enroll(self, name, files):
        """Entraine un modèle GMM sur base de samples d'entrainement d'une personne"""
        print(f"[GMM] Entrainement de {name} ({len(files)} fichiers)...")
        feats = [extract_features(f) for f in files]
        valid_feats = [f for f in feats if f is not None]

        if not valid_feats:
            print("[WARNING] Aucun audio valide.")
            return

        X = np.vstack(valid_feats)
        max_components = X.shape[0] // 100

        if name == "Random":
            n_comp = min(64, max_components)
            n_comp = max(1, n_comp)
        else:
            n_comp = 16

        gmm = GaussianMixture(n_components=n_comp,
                              covariance_type='diag', n_init=3)
        gmm.fit(X)
        self.models[name] = gmm

        joblib.dump(gmm, os.path.join(self.model_dir, f"{name}.gmm"))

    def load_models(self):
        """Chargement des modèles GMM depuis le dossier correspondant"""
        if not os.path.exists(self.model_dir):
            return
        for f in os.listdir(self.model_dir):
            if f.endswith(".gmm"):
                name = f.replace(".gmm", "")
                self.models[name] = joblib.load(
                    os.path.join(self.model_dir, f))
        print(f"[GMM] {len(self.models)} modèles chargés.")

    def verify(self, test_file, safety_margin=5.0):
        """Vérification de la similarité d'un sample avec les modèles GMM enregistrés"""
        feat = extract_features(test_file)
        if feat is None:
            return "Erreur", 0.0

        if "Random" not in self.models:
            return "Erreur (Modèle lambda non trouvé)", 0.0

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
        print(f"[DTW] Sauvegarde des templates pour {name}...")
        self.templates[name] = []
        for f in files:
            feat = extract_features(f)
            if feat is not None:
                self.templates[name].append((feat, f))

        with open(self.db_path, 'wb') as f:
            pickle.dump(self.templates, f)

    def load_models(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.templates = pickle.load(f)
            print(f"[DTW] Templates chargés pour {
                  len(self.templates)} utilisateurs.")

    def verify(self, claimed_name, test_file):
        if claimed_name not in self.templates:
            return float('inf'), None, None

        test_feat = extract_features(test_file)
        if test_feat is None:
            return float('inf'), None, None

        best_dist = float('inf')
        best_ref_feat = None
        best_ref_path = None
        len_test = test_feat.shape[0]

        for ref_data in self.templates[claimed_name]:
            if isinstance(ref_data, tuple):
                ref_feat = ref_data[0]
                ref_path = ref_data[1]
            else:
                ref_feat = ref_data
                ref_path = None

            len_ref = ref_feat.shape[0]
            ratio = min(len_test, len_ref) / max(len_test, len_ref)

            if ratio < 0.6:
                continue

            if test_feat.shape[1] > 40:
                w_test = test_feat.copy()
                w_ref = ref_feat.copy()

                w_test[:, 20:40] *= 1.5
                w_ref[:, 20:40] *= 1.5
            else:
                w_test, w_ref = test_feat, ref_feat

            distance, path = fastdtw(w_ref, w_test, dist=cosine, radius=30)
            norm_dist = distance / len(path)

            if norm_dist < best_dist:
                best_dist = norm_dist
                best_ref_feat = ref_feat
                best_ref_path = ref_path

        return best_dist, best_ref_feat, best_ref_path
