import pyaudio
import wave
import sys
import numpy as np
import threading
from colorama import Fore, Style


class NativeRecorder:
    """Enregistreur vocal employant des techniques similaires à Audacity pour maximiser la compatibilité entre nos samples pré-enregistrés et ceux enregistrés avec Python"""

    def __init__(self, rate=44100, chunk=1024):
        self.rate = rate
        self.chunk = chunk
        self.format = pyaudio.paInt16
        self.channels = 1
        self.p = pyaudio.PyAudio()

    def record(self, output_filename, device_index=None):
        try:
            stream = self.p.open(format=self.format,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 input_device_index=device_index,
                                 frames_per_buffer=self.chunk)
        except Exception as e:
            print(
                Fore.RED + f"[ERROR] Impossible d'ouvrir le stream audio: {e}")
            return

        print(Fore.YELLOW +
              "Pressez [ENTER] pour démarrer l'enregistrement...")
        input()
        print(Fore.YELLOW +
              "Enregistrement en cours... (Pressez [ENTER] pour arrêter)")
        frames = []
        is_recording = True

        def capture_loop():
            while is_recording:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(data)
                except:
                    break

        t = threading.Thread(target=capture_loop)
        t.start()

        input()  # Attend [ENTER] pour s'arrêter
        is_recording = False
        t.join()  # Attend la fin du thread

        print(Fore.YELLOW + "Arrêt de l'enregistrement...")
        stream.stop_stream()
        stream.close()

        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        print(Fore.GREEN + f"[SUCCESS] Sauvegarde de l'audio vers {
              output_filename}" + Style.RESET_ALL)

    def close(self):
        self.p.terminate()


# Test en standalone
if __name__ == "__main__":
    rec = NativeRecorder()
    rec.record("test_pyaudio.wav")
    rec.close()
