import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import sys
from engines import GMMVerifier, DTWVerifier


class LiveAuthenticator:
    def __init__(self, gmm_threshold=15.0, dtw_threshold=70.0):
        self.gmm_thresh = gmm_threshold
        self.dtw_thresh = dtw_threshold

        # Load engines immediately
        print("   [Init] Chargement des modèles...")
        self.gmm = GMMVerifier()
        self.dtw = DTWVerifier()
        self.gmm.load_models()
        self.dtw.load_models()

    def record_audio(self, duration=None, fs=16000):
        """
        Records audio until the user presses Enter.
        Returns the path to the temporary WAV file.
        """
        print("\n" + "="*40)
        print("   🎙️  MODE ENREGISTREMENT DIRECT")
        print("="*40)
        input("   >>> Appuyez sur [ENTRÉE] pour commencer l'enregistrement...")

        print(
            "   🔴 Enregistrement en cours... (Appuyez sur [ENTRÉE] pour arrêter)")

        recording = []

        # Callback function to capture audio blocks
        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            recording.append(indata.copy())

        # Start the stream
        # Channels=1 (Mono), Rate=16000 (To match your models)
        with sd.InputStream(samplerate=fs, channels=1, callback=callback):
            # This input() blocks the main thread while the stream runs in background
            input()

        print("   ⏹️  Enregistrement terminé.")

        # Concatenate all blocks
        audio_data = np.concatenate(recording, axis=0)

        # Save to a temp file
        # We use delete=False so we can close it and let the other engines open it
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio_data, fs)

        return temp_file.name

    def run(self):
        # 1. Record
        temp_wav_path = self.record_audio()

        try:
            print(f"\n   🔍 Analyse du signal...")

            # 2. GMM Check
            id_speaker, gmm_score = self.gmm.verify(
                temp_wav_path, safety_margin=self.gmm_thresh)

            print(f"   1. Identification (GMM): {
                  id_speaker} (Marge: {gmm_score:.2f})")

            if id_speaker in ["Unknown", "Error", "Error (No UBM)"]:
                print("\n   🚫 ACCÈS REFUSÉ : Identité non reconnue.")
                return

            # 3. DTW Check
            dtw_dist = self.dtw.verify(id_speaker, temp_wav_path)
            print(f"   2. Passphrase (DTW): Distance {dtw_dist:.2f}")

            # 4. Final Verdict
            if dtw_dist < self.dtw_thresh:
                print(f"\n   ✅ ACCÈS AUTORISÉ. Bienvenue, {id_speaker} !")
            else:
                print(f"\n   🔒 ACCÈS REFUSÉ : Bonne voix, mais mauvaise passphrase.")

        finally:
            # Clean up: Delete the temp file
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
