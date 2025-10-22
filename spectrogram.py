import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
import tempfile

AUDIO_DIR = "samples"


def convert_m4a_to_wav(input_path):
    """Convert an .m4a file to .wav and return the wav path."""
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio = AudioSegment.from_file(input_path, format="m4a")
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name


def load_audio(filepath):
    """Load audio file, converting if necessary."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".m4a":
        wav_path = convert_m4a_to_wav(filepath)
        y, sr = librosa.load(wav_path, sr=None)
        os.remove(wav_path)  # Clean up temp
    elif ext == ".wav":
        y, sr = librosa.load(filepath, sr=None)
    else:
        raise ValueError(f"Unsupported format: {filepath}")
    return y, sr


def plot_spectrogram_grid(base_dir):
    subdirs = sorted([d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d))])

    num_rows = len(subdirs)
    num_cols = max(len(os.listdir(os.path.join(base_dir, sub)))
                   for sub in subdirs)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))

    if num_rows == 1:
        axes = [axes]  # Handle case of 1 row

    for row_idx, subdir in enumerate(subdirs):
        full_subdir = os.path.join(base_dir, subdir)
        files = sorted(os.listdir(full_subdir))
        for col_idx in range(num_cols):
            ax = axes[row_idx][col_idx] if num_rows > 1 else axes[col_idx]
            if col_idx < len(files):
                filename = files[col_idx]
                filepath = os.path.join(full_subdir, filename)
                try:
                    y, sr = load_audio(filepath)
                    D = librosa.amplitude_to_db(
                        np.abs(librosa.stft(y)), ref=np.max)
                    librosa.display.specshow(
                        D, sr=sr, x_axis='time', y_axis='log', ax=ax)
                    ax.set_title(f"{subdir}/{filename}", fontsize=8)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error\n{
                            filename}", ha='center', va='center')
                    print(f"Error processing {filepath}: {e}")
            else:
                ax.axis('off')  # Hide empty axes
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_spectrogram_grid(AUDIO_DIR)
