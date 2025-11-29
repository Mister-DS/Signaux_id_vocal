import os
import librosa
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class GMMVoiceAuth:
    def __init__(self, n_components=16, model_dir="voice_models"):
        """
        n_components: Le nombre de clusters à modéliser. 16 suffisent pour notre PoC avec peu de données.
        """
        self.n_components = n_components
        self.model_dir = model_dir
        self.models = {}

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def extract_features(self, audio_path):
        """
        Extraction des MFCCs (timbre de voix) + Deltas (vitesse et accélérations).
        Retourne une matrice de forme (n_frames, n_features).
        """
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            y, _ = librosa.effects.trim(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_delta = librosa.feature.delta(mfcc)

            # Assemblage en une seule matrice de forme (40, n_frames)
            features = np.vstack([mfcc, mfcc_delta])
            return features.T  # Transposition requise par sklearn

        except Exception as e:
            print(f"Erreur d'extraction des features de {audio_path}: {e}")
            return None

    def enroll_user(self, name, audio_files):
        """
        Entrainement d'un GMM pour l'utilisateur sur base de fichiers audios spécifiés.
        """
        print(f"--- Enregistrement de l'utilisateur: {name} ---")
        features_list = []

        # Assemblage des features de chaque fichier
        for file in audio_files:
            feat = self.extract_features(file)
            if feat is not None:
                features_list.append(feat)

        if not features_list:
            print("Aucun audio valide trouvé.")
            return

        X = np.vstack(features_list)

        # Entrainement des clusters GMM sur base des données vocales entrées
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            n_init=5,  # Répétition x5
            verbose=0
        )
        gmm.fit(X)

        # Enregistrement du modèle
        self.models[name] = gmm
        model_path = os.path.join(self.model_dir, f"{name}.gmm")
        joblib.dump(gmm, model_path)
        print(f"Modèle sauvegardé: {model_path}")

    def verify_speaker(self, name, test_file):
        """
        Retourne le score de similarité entre le fichier de validation et le modèle entrainé.
        """
        if name not in self.models:
            print(f"Utilisateur {name} non existant.")
            return -99999

        features = self.extract_features(test_file)
        if features is None:
            return -99999

        gmm = self.models[name]

        # Calcul du score (meilleur au plus il est grand)
        score = gmm.score(features)
        return score

    def identify_speaker(self, test_file, safety_margin=10.0):
        """
        Identification du sample de validation. Retourne 'Unknown' si le sample ne ressemble pas suffisamment à un utilisateur existant.
        """
        features = self.extract_features(test_file)
        if features is None:
            return "Error", 0

        # Calcul de l'utilisateur lambda "Random"
        if "Random" not in self.models:
            return "Erreur: Utilisateur \"Random\" non existant", 0

        ubm_score = self.models["Random"].score(features)

        best_speaker = "Unknown"
        best_margin = -float('inf')

        print(f"\nAnalyse de: {os.path.basename(test_file)}")
        print(f"  [Baseline] Score d'une personne lambda: {ubm_score:.2f}")
        print("  ------------------------------------------------")

        for name, gmm in self.models.items():
            if name == "Random":
                continue

            raw_score = gmm.score(features)
            margin = raw_score - ubm_score
            print(f"  Candidate {name}: Margin = {margin:.2f}")

            # L'utilisateur validé doit battre "Random" avec une marge suffisante
            if margin > best_margin:
                best_margin = margin
                if margin > safety_margin:
                    best_speaker = name

        return best_speaker, best_margin


if __name__ == "__main__":
    # Validation
    auth = GMMVoiceAuth(n_components=16)

    # Enregistrement
    simon_files = ["samples/p13/simon_1.wav", "samples/p13/simon_2.wav"]
    auth.enroll_user("Simon", simon_files)

    victor_files = ["samples/p12/Victor_1.wav", "samples/p12/Victor_2.wav"]
    auth.enroll_user("Victor", victor_files)

    nathan_files = ["samples/p15/nathan_1.wav", "samples/p15/nathan_2.wav",
                    "samples/p15/nathan_3.wav", "samples/p15/nathan_4.wav"]
    auth.enroll_user("Nathan", nathan_files)

    manon_files = ["samples/p16/manon_1.wav",
                   "samples/p16/manon_2.wav", "samples/p16/manon_3.wav"]
    auth.enroll_user("Manon", manon_files)

    tiago_files = ["samples/p17/tiago_01.wav", "samples/p17/tiago_02.wav", "samples/p17/tiago_03.wav", "samples/p17/tiago_04.wav",
                   "samples/p17/tiago_05.wav", "samples/p17/tiago_06.wav", "samples/p17/tiago_07.wav", "samples/p17/tiago_08.wav", "samples/p17/tiago_09.wav"]
    auth.enroll_user("Tiago", tiago_files)

    random_files = ["samples/p01/b1.m4a", "samples/p02/e1.m4a", "samples/p03/x1.m4a",
                    "samples/p04/j1.m4a", "samples/p08/cam_1.wav", "samples/p09/JIM_1.wav"]
    auth.enroll_user("Random", random_files)

    # Validation
    unknown_file = "samples/p17/tiago_10.wav"
    winner, score = auth.identify_speaker(unknown_file)

    print(f"\n>>> Utilisateur identifié : {
          winner} (Score: {score:.2f})")
