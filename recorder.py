import pyaudio
import wave
import sys
import numpy as np
import os


class NativeRecorder:
    def __init__(self, rate=44100, chunk=1024):
        self.rate = rate
        self.chunk = chunk
        self.format = pyaudio.paInt16  # The Audacity Standard (16-bit PCM)
        self.channels = 1
        self.p = pyaudio.PyAudio()

    def list_devices(self):
        """Helper to find the True Hardware Device ID"""
        print("\n--- Available Audio Devices ---")
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            # Filter out useless stuff to find your mic
            if dev['maxInputChannels'] > 0:
                print(f"Index {i}: {dev['name']}")
        print("-------------------------------\n")

    def _vu_meter(self, data_chunk):
        # Convert raw bytes to integers for volume calculation
        # valid for paInt16
        audio_ints = np.frombuffer(data_chunk, dtype=np.int16)
        volume = np.linalg.norm(audio_ints) / 1000
        bars = int(min(volume, 50))
        sys.stdout.write(f"\r[Rec] |{'█' * bars:<50}|")
        sys.stdout.flush()

    def record(self, output_filename, device_index=None):
        print("\n" + "="*40)
        print("   🎙️  NATIVE PORTAUDIO RECORDER")
        print("="*40)

        # If you don't know your device index, PyAudio picks system default.
        # On Arch with Pulse/Pipewire, 'None' usually works, but explicit is better.

        try:
            stream = self.p.open(format=self.format,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 input_device_index=device_index,
                                 frames_per_buffer=self.chunk)
        except Exception as e:
            print(f"[ERROR] Failed to open stream: {e}")
            print("Try running 'rec.list_devices()' to find the correct Index.")
            return

        print("   >>> PRESS [ENTER] TO START RECORDING...")
        input()

        print("   🔴 RECORDING... (Press [ENTER] to STOP)")
        frames = []

        # Non-blocking input trick using a loop isn't cleaner with PyAudio.
        # We will use a try/except KeyboardInterrupt or just a raw loop
        # but to keep it compatible with your 'Press Enter' flow,
        # we need a threading trick or a simple duration loop.

        # SIMPLEST STABLE APPROACH FOR LINUX:
        # Use a background thread to capture so the main thread can wait for Input.
        import threading
        is_recording = True

        def capture_loop():
            while is_recording:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(data)
                    self._vu_meter(data)
                except:
                    break

        t = threading.Thread(target=capture_loop)
        t.start()

        input()  # Wait for Enter to Stop
        is_recording = False
        t.join()  # Wait for thread to finish

        # Stop and Close
        print("\n   ⏹️  Stopping...")
        stream.stop_stream()
        stream.close()

        # Save to WAV (Using Python's native Wave library, no Scipy needed)
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        print(f"   💾 Saved raw audio to: {output_filename}")

    def close(self):
        self.p.terminate()


# --- STANDALONE TEST ---
if __name__ == "__main__":
    rec = NativeRecorder()
    # Uncomment this to see your device IDs if it still fails!
    # rec.list_devices()

    rec.record("test_pyaudio.wav")
    rec.close()
