import os
import glob
from engines import GMMVerifier, DTWVerifier
from colorama import Fore, Style, init

# Init colors
init(autoreset=True)


class ValidationManager:
    def __init__(self, samples_root="samples", gmm_threshold=15.0, dtw_threshold=70.0):
        self.root = samples_root
        self.gmm_thresh = gmm_threshold
        self.dtw_thresh = dtw_threshold

        self.gmm = GMMVerifier()
        self.dtw = DTWVerifier()
        self.gmm.load_models()
        self.dtw.load_models()

    def run_benchmark(self, target_user=None):
        print("════════════════════════════════════════")
        print("       PHASE 2: VALIDATION (Benchmark)  ")
        print("════════════════════════════════════════")

        stats = {"total": 0, "correct": 0, "far": 0, "frr": 0}
        val_path = os.path.join(self.root, "validation")

        # 1. Determine which users to test
        if target_user:
            # Specific User Case
            target_path = os.path.join(val_path, target_user)
            if not os.path.exists(target_path):
                print(f"{Fore.RED}Erreur: Aucun dossier de validation trouvé pour '{
                      target_user}'")
                return
            users = [target_user]
            print(f"[INFO] Mode Cible : Validation de '{
                  target_user}' uniquement.")
        else:
            # Full Benchmark Case
            users = [d for d in os.listdir(val_path) if os.path.isdir(
                os.path.join(val_path, d)) and not d.startswith("_")]

        # 2. Loop through selected users
        for user in users:
            print(Style.RESET_ALL + f"\nRésultats de {user}:")

            # SCENARIO A: LEGIT (True Positive) -> Should PASS GMM & PASS DTW
            self._test_folder(user, "true_positive", True, True, stats)

            # SCENARIO B: WRONG PHRASE -> Should PASS GMM & FAIL DTW
            # Note: Expecting 'False' for final Auth
            self._test_folder(user, "wrong_phrase", True, False, stats)

        # 3. Test Impostors (Only if running full benchmark)
        # We skip this if targeting a specific user to avoid clutter
        if target_user is None:
            print(Style.RESET_ALL + "\nVérification des imposteurs (Global)")
            impostor_path = os.path.join(val_path, "_impostor")
            if os.path.exists(impostor_path):
                files = glob.glob(os.path.join(impostor_path, "*"))
                for f in files:
                    self._run_single_test(
                        f, expected_user=None, expect_auth=False, stats=stats)

        # 4. Results
        if stats["total"] > 0:
            print(Style.RESET_ALL + "\n" + "="*40)
            acc = (stats["correct"] / stats["total"] * 100)
            print(f"[RESULTS] {acc:.2f}% de précision")
            print(f"Tests effectués : {stats['total']}")
            print(f"Succès: {stats['correct']}")
            print(f"Faux positifs (Intrusion): {stats['far']}")
            print(f"Faux négatifs (Rejet):     {stats['frr']}")
        else:
            print("\nAucun test effectué.")

    def _test_folder(self, user, subfolder, expect_gmm, expect_dtw, stats):
        folder_path = os.path.join(self.root, "validation", user, subfolder)
        if not os.path.exists(folder_path):
            return

        files = glob.glob(os.path.join(folder_path, "*"))
        expect_final_auth = (expect_gmm and expect_dtw)

        for f in files:
            self._run_single_test(f, user, expect_final_auth, stats)

    def _run_single_test(self, filepath, expected_user, expect_auth, stats):
        filename = os.path.basename(filepath)
        stats["total"] += 1

        # 1. GMM Check
        id_speaker, gmm_score = self.gmm.verify(
            filepath, safety_margin=self.gmm_thresh)

        # 2. DTW Check
        dtw_dist = float('inf')

        # Logic: We only check DTW if the GMM actually matched someone (and not Unknown)
        # Case A: We expect a specific user (Verification)
        if expected_user and id_speaker == expected_user:
            dtw_dist, _, _ = self.dtw.verify(expected_user, filepath)

        # Case B: We assume it's an impostor, but GMM matched SOMEONE (Identification)
        elif expected_user is None and id_speaker not in ["Unknown", "Error"]:
            dtw_dist, _, _ = self.dtw.verify(id_speaker, filepath)

        # 3. Final Decision
        if expected_user:
            # Must match specific user AND pass passphrase
            access_granted = (id_speaker == expected_user) and (
                dtw_dist < self.dtw_thresh)
        else:
            # Impostor Scenario: Access granted if ANYONE was identified AND passphrase passed
            access_granted = (id_speaker not in ["Unknown", "Error"]) and (
                dtw_dist < self.dtw_thresh)

        # 4. Scoring
        is_correct = (access_granted == expect_auth)

        if is_correct:
            stats["correct"] += 1
            icon = Fore.GREEN + "[SUCCESS]"
        else:
            icon = Fore.RED + "[FAILURE]"
            if expect_auth and not access_granted:
                stats["frr"] += 1
            if not expect_auth and access_granted:
                stats["far"] += 1

        print(f"{icon} {
              filename} -> ID: {id_speaker} ({gmm_score:.1f}) | DTW: {dtw_dist:.4f}")
        if not is_correct:
            print(f"    Expected Auth: {expect_auth}, Got: {access_granted}")
