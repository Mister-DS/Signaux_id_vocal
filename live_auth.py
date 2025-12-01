import os
import tempfile
import numpy as np
from engines import GMMVerifier, DTWVerifier
from features import extract_features
from visualize import visualize_analysis
from recorder import NativeRecorder
import soundfile as sf
from colorama import Fore, Style


class LiveAuthenticator:
    def __init__(self, gmm_threshold=15.0, dtw_threshold=70.0):
        self.gmm_thresh = gmm_threshold
        self.dtw_thresh = dtw_threshold

        print(f"{Fore.BLUE} [INFO] Chargement des modèles...{Style.RESET_ALL}")
        self.gmm = GMMVerifier()
        self.dtw = DTWVerifier()
        self.gmm.load_models()
        self.dtw.load_models()

    def record_audio(self):
        """Enregistrement d'un sample audio"""
        tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tf.close()
        temp_path = tf.name

        try:
            rec = NativeRecorder(rate=44100)
            rec.record(temp_path)
            rec.close()

            print(f"{Fore.BLUE}[INFO] Nettoyage du signal audio...{
                  Style.RESET_ALL}")

            data, fs = sf.read(temp_path)
            samples_to_cut = int(0.3 * fs)

            if len(data) > samples_to_cut:
                data = data[samples_to_cut:]
            else:
                print(f"{Fore.RED}[ERROR] Enregistrement trop court{
                      Style.RESET_ALL}")
                return None

            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val * 0.90

            sf.write(temp_path, data, fs, subtype='PCM_16')
            return temp_path

        except Exception as e:
            print(f"{Fore.RED}[ERROR] Recording failed: {e}{Style.RESET_ALL}")
            return None

    def run(self, target_user=None):
        temp_wav_path = self.record_audio()

        if not temp_wav_path or not os.path.exists(temp_wav_path):
            print(f"{Fore.RED}[ERROR] Pas de fichier audio généré{
                  Style.RESET_ALL}")
            return

        try:
            print(f"{Fore.BLUE}[INFO] Analyse du signal...{Style.RESET_ALL}")
            id_speaker, gmm_score = self.gmm.verify(
                temp_wav_path, safety_margin=self.gmm_thresh)

            print(f"[GMM] Identité détectée : {Fore.BLUE}{id_speaker}{
                  Style.RESET_ALL} (Score: {gmm_score:.2f})")

            user_to_verify = id_speaker
            if target_user:
                print(f"\n[DEBUG] Mode Forcé activé : Comparaison avec '{
                      target_user}'")
                user_to_verify = target_user
                if id_speaker != target_user:
                    print(f"[DEBUG] GMM a échoué (pensait que c'était {
                          id_speaker}), mais on force la suite.")

            dtw_dist = 0
            best_template_feats = None
            best_template_path = None

            if user_to_verify not in ["Unknown", "Error", "Error (No UBM)"]:

                dtw_dist, best_template_feats, best_template_path = self.dtw.verify(
                    user_to_verify, temp_wav_path)

                print(f"[DTW] Distance avec {user_to_verify}: {dtw_dist:.4f}")

                if dtw_dist < self.dtw_thresh:
                    print(
                        Fore.GREEN + f"\n[SUCCESS] Bienvenue, {user_to_verify} !")
                else:
                    print(Fore.RED + f"\n[FAILURE] Passphrase incorrecte.")
            else:
                print(
                    Fore.RED + "[FAILURE] Identité inconnue et aucune cible forcée.")

            print(Style.RESET_ALL + "[INFO] Génération des graphiques...")
            live_feats = extract_features(temp_wav_path)

            all_gmm_scores = {}
            if "Random" in self.gmm.models and live_feats is not None:
                ubm_score = self.gmm.models["Random"].score(live_feats)
                for name, model in self.gmm.models.items():
                    if name == "Random":
                        continue
                    all_gmm_scores[name] = model.score(live_feats) - ubm_score

            visualize_analysis(
                live_path=temp_wav_path,
                template_path=best_template_path,
                live_feats=live_feats,
                template_feats=best_template_feats,
                gmm_scores=all_gmm_scores,
                gmm_threshold=self.gmm_thresh
            )

        finally:
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except:
                    pass
