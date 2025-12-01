import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from fastdtw import fastdtw
from scipy.spatial.distance import cosine


def prep_audio_for_plot(file_path):
    if file_path is None:
        return None, 0
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y, sr


def visualize_analysis(live_path, template_path, live_feats, template_feats, gmm_scores, gmm_threshold):
    fig = plt.figure(figsize=(16, 12))
    fig.canvas.manager.set_window_title("Analyse de l'authentification")

    y_live, sr = prep_audio_for_plot(live_path)
    y_temp, _ = prep_audio_for_plot(template_path)

    ax1 = plt.subplot(4, 2, 1)
    sorted_scores = sorted(
        gmm_scores.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_scores]
    margins = [x[1] for x in sorted_scores]
    colors = ['green' if m > gmm_threshold else 'red' for m in margins]
    ax1.barh(names, margins, color=colors)
    ax1.axvline(gmm_threshold, color='blue', linestyle='--', label='Threshold')
    ax1.set_title("Certitude d'identification (valide si > 5.0)")

    ax2 = plt.subplot(4, 2, 2)
    if template_feats is not None:
        distance, path = fastdtw(template_feats, live_feats, dist=cosine)
        path = np.array(path)
        ax2.plot(path[:, 1], path[:, 0], 'w-', linewidth=2)
        ax2.imshow(np.zeros((len(template_feats), len(live_feats))),
                   origin='lower', cmap='magma', aspect='auto')
        ax2.set_title(f"Chemin DTW (Distance: {distance/len(path):.4f})")
    else:
        ax2.text(0.5, 0.5, "DTW passé", ha='center')

    ax3 = plt.subplot(4, 2, 3)
    if y_live is not None:
        librosa.display.waveshow(y_live, sr=sr, ax=ax3, alpha=0.8)
        ax3.set_title("Signal en direct (temporel)")

    ax4 = plt.subplot(4, 2, 4)
    if y_temp is not None:
        librosa.display.waveshow(
            y_temp, sr=sr, ax=ax4, color='orange', alpha=0.8)
        ax4.set_title(f"Signal enregistré (temporel): {
                      template_path.split('/')[-1]}")

    ax5 = plt.subplot(4, 2, 5)
    if y_live is not None:
        D_live = librosa.amplitude_to_db(
            np.abs(librosa.stft(y_live)), ref=np.max)
        librosa.display.specshow(
            D_live, sr=sr, x_axis='time', y_axis='log', ax=ax5)
        ax5.set_title("Signal en direct (fréquentiel)")

    ax6 = plt.subplot(4, 2, 6)
    if y_temp is not None:
        D_temp = librosa.amplitude_to_db(
            np.abs(librosa.stft(y_temp)), ref=np.max)
        librosa.display.specshow(
            D_temp, sr=sr, x_axis='time', y_axis='log', ax=ax6)
        ax6.set_title("Signal enregistré (fréquentiel)")

    ax7 = plt.subplot(4, 2, 7)
    librosa.display.specshow(live_feats.T, ax=ax7,
                             x_axis='time', cmap='viridis')
    ax7.set_title("Signal en direct (MFCCs)")

    ax8 = plt.subplot(4, 2, 8)
    if template_feats is not None:
        librosa.display.specshow(
            template_feats.T, ax=ax8, x_axis='time', cmap='viridis')
        ax8.set_title("Signal enregistré (MFCCs)")

    plt.tight_layout()
    plt.show()
