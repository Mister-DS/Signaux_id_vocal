from auth import GMMVerifier, DTWVerifier

GMM_THRESHOLD = 15.0
DTW_THRESHOLD = 70.0


def authenticate_user(gmm_engine, dtw_engine, test_file):
    print(f"\n--- Authentification de: {test_file} ---")

    # Analyse de la personne qui parle
    identified_speaker, score = gmm_engine.verify(
        test_file, safety_margin=GMM_THRESHOLD)

    print(f"1. Vérification biométrique: {
          identified_speaker} (Score: {score:.2f})")

    if identified_speaker in ["Unknown", "Erreur"]:
        print(">>> ACCÈS REFUSÉ (cette voix n'est pas reconnue)")
        return

    # Analyse de la passphrase
    passphrase_score = dtw_engine.verify(identified_speaker, test_file)

    print(f"2. Vérification de passphrase: Distance {passphrase_score:.2f}")

    if passphrase_score < DTW_THRESHOLD:
        print(f">>> ACCESS AUTORISÉ. Bienvenue, {identified_speaker}!")
    else:
        print(f">>> ACCESS REFUSÉ (voix identifiée, mais passphrase erronée).")


if __name__ == "__main__":
    gmm = GMMVerifier()
    dtw = DTWVerifier()

    justin_files = ["samples/enrollment/justin/justin_01.wav", "samples/enrollment/justin/justin_02.wav",
                    "samples/enrollment/justin/justin_03.wav", "samples/enrollment/justin/justin_04.wav", "samples/enrollment/justin/justin_05.wav"]
    gmm.enroll("Justin", justin_files)
    dtw.enroll("Justin", justin_files)

    random_files = ["samples/random/b1.m4a", "samples/random/e1.m4a", "samples/random/x1.m4a",
                    "samples/random/j1.m4a", "samples/random/cam_1.wav", "samples/random/JIM_1.wav"]
    gmm.enroll("Random", random_files)
    # Pas besoin d'enregistrer Random pour le DTW

    authenticate_user(
        gmm, dtw, "samples/validation/justin/wrong_phrase/justin_wrongphrase_03.wav")
