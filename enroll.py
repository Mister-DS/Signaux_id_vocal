import os
import glob
from engines import GMMVerifier, DTWVerifier


class EnrollmentManager:
    def __init__(self, samples_root="samples"):
        self.root = samples_root
        self.gmm = GMMVerifier()
        self.dtw = DTWVerifier()

    def run_enrollment(self):
        print("[INFO] Entrainement des modèles de reconnaissance vocale sur base des fichiers existants")

        # Entrainement de l'utilisateur lambda
        random_path = os.path.join(self.root, "random")
        if os.path.exists(random_path):
            files = glob.glob(os.path.join(random_path, "*")
                              )
            self.gmm.enroll("Random", files)
        else:
            print("[ERREUR] Dossier 'random' non trouvé")

        # Entrainement des utilisateurs standards
        enroll_path = os.path.join(self.root, "enrollment")
        if not os.path.exists(enroll_path):
            print("[ERREUR] Dossier 'enrollment' non trouvé")
            return

        users = [d for d in os.listdir(enroll_path) if os.path.isdir(
            os.path.join(enroll_path, d))]

        for user in users:
            print(f"\n[INFO] Enregistrement de l'utilisateur : {user}")
            user_path = os.path.join(enroll_path, user)
            user_files = glob.glob(os.path.join(user_path, "*"))

            self.gmm.enroll(user, user_files)
            self.dtw.enroll(user, user_files)

        print("\n[SUCCESS] Entrainement terminé. Modèles sauvegardés dans /models.")
