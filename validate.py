import os
import glob
from engines import GMMVerifier, DTWVerifier


class ValidationManager:
    def __init__(self, samples_root="samples", gmm_threshold=15.0, dtw_threshold=70.0):
        self.root = samples_root
        self.gmm_thresh = gmm_threshold
        self.dtw_thresh = dtw_threshold

        # Load Engines
        self.gmm = GMMVerifier()
        self.dtw = DTWVerifier()
        self.gmm.load_models()
        self.dtw.load_models()

    def run_benchmark(self):
        print("\n════════════════════════════════════════")
        print("       PHASE 2: VALIDATION (Benchmark)  ")
        print("════════════════════════════════════════")

        stats = {"total": 0, "correct": 0, "far": 0, "frr": 0}
        val_path = os.path.join(self.root, "validation")

        # 1. Test Specific Users (True Positive & Wrong Phrase)
        users = [d for d in os.listdir(val_path) if os.path.isdir(
            os.path.join(val_path, d)) and not d.startswith("_")]

        for user in users:
            print(f"\n--- Audit de : {user} ---")

            # SCENARIO A: LEGIT (True Positive) -> Should PASS GMM & PASS DTW
            self._test_folder(user, "true_positive", True, True, stats)

            # SCENARIO B: WRONG PHRASE -> Should PASS GMM & FAIL DTW
            self._test_folder(user, "wrong_phrase", True, False, stats)

        # 2. Test Global Impostors (Folder _impostor)
        print(f"\n--- Audit des Imposteurs (Attaques Globales) ---")
        impostor_path = os.path.join(val_path, "_impostor")
        if os.path.exists(impostor_path):
            files = glob.glob(os.path.join(impostor_path, "*"))
            for f in files:
                self._run_single_test(
                    f, expected_user=None, expect_auth=False, stats=stats)

        # 3. Results
        print("\n" + "="*40)
        acc = (stats["correct"] / stats["total"]
               * 100) if stats["total"] > 0 else 0
        print(f"RÉSULTATS FINAUX: {acc:.2f}% de précision")
        print(f"Total Tests: {stats['total']}")
        print(f"Succès: {stats['correct']}")
        print(f"False Acceptances (Failles de sécu): {stats['far']}")
        print(f"False Rejections (Utilisateur bloqué): {stats['frr']}")

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

        # 2. DTW Check (Only if user identified matches expected, or for impostors check any)
        dtw_dist = float('inf')

        # If we expect a specific user, we only check DTW if GMM matches that user
        if expected_user and id_speaker == expected_user:
            dtw_dist = self.dtw.verify(expected_user, filepath)

        # If it's an impostor, but GMM identified SOMEONE, check DTW against that someone
        elif expected_user is None and id_speaker not in ["Unknown", "Error"]:
            dtw_dist = self.dtw.verify(id_speaker, filepath)

        # 3. Final Decision
        # Access Granted if: GMM identified the RIGHT person AND DTW is low enough
        if expected_user:
            access_granted = (id_speaker == expected_user) and (
                dtw_dist < self.dtw_thresh)
        else:
            # For impostor, Access Granted if ANYONE was identified and passed DTW
            access_granted = (id_speaker not in ["Unknown", "Error"]) and (
                dtw_dist < self.dtw_thresh)

        # 4. Scoring
        is_correct = (access_granted == expect_auth)
        if is_correct:
            stats["correct"] += 1
            icon = "✅"
        else:
            icon = "❌"
            if expect_auth and not access_granted:
                stats["frr"] += 1
            if not expect_auth and access_granted:
                stats["far"] += 1

        # Debug Print
        print(f"{icon} {
              filename} -> ID: {id_speaker} ({gmm_score:.1f}) | DTW: {dtw_dist:.1f}")
        if not is_correct:
            print(f"    Expected Auth: {expect_auth}, Got: {access_granted}")
