import os
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
import tempfile

BASE_DIR = "samples"
SUPPORTED_FORMATS = [".m4a", ".wav"]


def convert_m4a_to_wav(filepath):
    audio = AudioSegment.from_file(filepath, format="m4a")
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name


def extract_features(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".m4a":
        wav_path = convert_m4a_to_wav(filepath)
        y, sr = librosa.load(wav_path, sr=None)
        os.remove(wav_path)
    else:
        y, sr = librosa.load(filepath, sr=None)

    if y.size == 0:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    pitch = librosa.yin(y, fmin=50, fmax=500, sr=sr)

    features = {
        "mean_mfcc": mfcc.mean(axis=1),
        "std_mfcc": mfcc.std(axis=1),
        "mean_delta": delta.mean(axis=1),
        "std_delta": delta.std(axis=1),
        "mean_delta2": delta2.mean(axis=1),
        "std_delta2": delta2.std(axis=1),
        "zcr_mean": zcr.mean(),
        "zcr_std": zcr.std(),
        "rms_mean": rms.mean(),
        "rms_std": rms.std(),
        "spec_centroid_mean": spec_centroid.mean(),
        "spec_centroid_std": spec_centroid.std(),
        "spec_bandwidth_mean": spec_bandwidth.mean(),
        "spec_bandwidth_std": spec_bandwidth.std(),
        "pitch_mean": pitch.mean(),
        "pitch_std": pitch.std(),
    }

    return features


# Loop through all audio files and extract features
feature_list = []

for speaker in sorted(os.listdir(BASE_DIR)):
    speaker_dir = os.path.join(BASE_DIR, speaker)
    if not os.path.isdir(speaker_dir):
        continue

    for file in sorted(os.listdir(speaker_dir)):
        file_path = os.path.join(speaker_dir, file)
        ext = os.path.splitext(file)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            continue

        features = extract_features(file_path)
        if features:
            flat_features = {}
            for k, v in features.items():
                if isinstance(v, np.ndarray):
                    for i, val in enumerate(v):
                        flat_features[f"{k}_{i}"] = val
                else:
                    flat_features[k] = v

            flat_features["speaker"] = speaker
            flat_features["file"] = file
            feature_list.append(flat_features)

# Save to CSV or display
features_df = pd.DataFrame(feature_list)
features_df.to_csv("extracted_voice_features.csv", index=False)
print(features_df.head())
