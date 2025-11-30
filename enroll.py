import os
import glob
from engines import GMMVerifier, DTWVerifier


class EnrollmentManager:
    def __init__(self, samples_root="samples"):
        self.root = samples_root
        self.gmm = GMMVerifier()
        self.dtw = DTWVerifier()

    def run_enrollment(self):
        print("════════════════════════════════════════")
        print("       PHASE 1: ENRÔLEMENT (Training)   ")
        print("════════════════════════════════════════")

        # 1. Train Background Model (Random)
        random_path = os.path.join(self.root, "random")
        if os.path.exists(random_path):
            files = glob.glob(os.path.join(random_path, "*")
                              )  # Catch wav, m4a, etc
            self.gmm.enroll("Random", files)
        else:
            print("ERREUR: Dossier 'random' introuvable !")

        # 2. Train Users
        enroll_path = os.path.join(self.root, "enrollment")
        if not os.path.exists(enroll_path):
            print("ERREUR: Dossier 'enrollment' introuvable !")
            return

        # Get list of folders (users) inside enrollment
        users = [d for d in os.listdir(enroll_path) if os.path.isdir(
            os.path.join(enroll_path, d))]

        for user in users:
            print(f"\n--- Traitement de l'utilisateur : {user} ---")
            user_path = os.path.join(enroll_path, user)
            user_files = glob.glob(os.path.join(user_path, "*"))

            # GMM needs training
            self.gmm.enroll(user, user_files)

            # DTW needs template storage
            self.dtw.enroll(user, user_files)

        print("\n>>> Enrôlement terminé. Modèles sauvegardés dans /models.")
