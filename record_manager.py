import os
import glob
import numpy as np
import soundfile as sf
from colorama import Fore, Style, init
from recorder import NativeRecorder

init(autoreset=True)  # Réinitialisation des couleurs


class RecordManager:
    def __init__(self, dataset_root="samples"):
        self.root = dataset_root

    def _ensure_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _get_next_filename(self, folder, prefix):
        """Scanne le dossier pour définir le prochain fichier à écrire"""
        self._ensure_folder(folder)
        existing = glob.glob(os.path.join(folder, "*.wav"))
        max_idx = 0

        for f in existing:
            try:
                base = os.path.splitext(os.path.basename(f))[0]
                parts = base.rsplit('_', 1)
                if len(parts) > 1 and parts[1].isdigit():
                    idx = int(parts[1])
                    if idx > max_idx:
                        max_idx = idx
            except:
                pass

        return os.path.join(folder, f"{prefix}_{max_idx + 1:02d}.wav")

    def _count_samples(self, folder):
        """Retourne le nombre de samples détectés dans le dossier"""
        if not os.path.exists(folder):
            return 0
        return len(glob.glob(os.path.join(folder, "*.wav")))

    def clean_and_save(self, raw_path, final_path):
        """Nettoyage du sample similaire à Audacity"""
        try:
            data, fs = sf.read(raw_path)

            # Sélection du channel audio (si stéréo)
            if len(data.shape) > 1 and data.shape[1] == 2:
                left = data[:, 0]
                right = data[:, 1]
                if np.var(left) > np.var(right):
                    data = left
                else:
                    data = right

            # Suppression de l'offset
            data = data - np.mean(data)

            # Suppression du POP de début
            cut_len = int(0.25 * fs)
            if len(data) > cut_len:
                data = data[cut_len:]
            else:
                print(Fore.RED + "[ERROR] Enregistrement trop court")
                return False

            # Normalisation sur base du pourcentile 95 (ignore les sons excessivement bruyants)
            loudness = np.percentile(np.abs(data), 95)
            if loudness > 0:
                data = data / loudness * 0.90
                data = np.clip(data, -1.0, 1.0)

            sf.write(final_path, data, fs, subtype='PCM_16')
            return True

        except Exception as e:
            print(Fore.RED + f"[ERROR] Nettoyage audio échoué: {e}")
            return False

    def run_interface(self):
        while True:
            username = input(f"{Fore.YELLOW}Nom de l'utilisateur cible: {
                             Style.RESET_ALL}").strip().lower()
            if username:
                break

        path_enroll = os.path.join(self.root, "enrollment", username)
        path_valid_tp = os.path.join(
            self.root, "validation", username, "true_positive")
        path_valid_wp = os.path.join(
            self.root, "validation", username, "wrong_phrase")
        path_impostor = os.path.join(self.root, "validation", "_impostor")

        while True:
            c_enroll = self._count_samples(path_enroll)
            c_tp = self._count_samples(path_valid_tp)
            c_wp = self._count_samples(path_valid_wp)
            c_imp = self._count_samples(path_impostor)

            print(f"\n{Fore.YELLOW}Enregistrement pour '{username}'")
            print(Fore.YELLOW +
                  f"1. Entrainement                     [{c_enroll} fichiers]")
            print(Fore.YELLOW +
                  f"2. Validation (correct)             [{c_tp} fichiers]")
            print(Fore.YELLOW +
                  f"3. Validation (mauvaise phrase)     [{c_wp} fichiers]")
            print(Fore.YELLOW +
                  f"4. Imposteur (imitation de voix)    [{c_imp} fichiers]")
            print(Fore.YELLOW + "5. Quitter")

            choice = input(
                f"\n{Fore.BLUE}Choix (1-5) > {Style.RESET_ALL}").strip()

            target_folder = None
            prefix = ""

            if choice == '1':
                target_folder = path_enroll
                prefix = username
            elif choice == '2':
                target_folder = path_valid_tp
                prefix = f"{username}_val"
            elif choice == '3':
                target_folder = path_valid_wp
                prefix = f"{username}_wrongphrase"
            elif choice == '4':
                imp_name = input(
                    "Nom de l'imposteur ?) > ").strip().lower()
                target_folder = path_impostor
                # Ex: nathan_attack_justin_01.wav
                prefix = f"{imp_name}_attack_{username}"
            elif choice == '5':
                break
            else:
                print("Choix invalide.")
                continue

            while True:
                final_path = self._get_next_filename(target_folder, prefix)
                filename = os.path.basename(final_path)

                print(f"\n{Fore.BLUE}[INFO] Prêt à enregistrer : {
                      filename}{Style.RESET_ALL}")

                # Record
                temp_path = "temp_rec.wav"
                rec = NativeRecorder(rate=44100)
                rec.record(temp_path)
                rec.close()

                if self.clean_and_save(temp_path, final_path):
                    print(f"{Fore.GREEN}[SUCCESS] Sauvegardé dans {
                          target_folder}{Style.RESET_ALL}")

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                cont = input(
                    Fore.YELLOW + "\nEnregistrer un autre ? (O/n) > ").lower()
                if cont == 'n':
                    break


if __name__ == "__main__":
    RecordManager().run_interface()
