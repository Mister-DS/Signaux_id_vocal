import argparse
import os
import glob
import numpy as np
import pandas as pd
import librosa
import warnings
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# On ignore les warnings de Librosa pour garder la CLI propre
warnings.filterwarnings('ignore')

def extract_features_from_file(file_path):
    """
    Extrait les features d'un seul fichier audio.
    Retourne un dictionnaire ou None si erreur.
    """
    try:
        # 1. Chargement (Optimisation : resample auto à 22050Hz pour la vitesse)
        y, sr = librosa.load(file_path, sr=22050)
        
        # Si le fichier est trop court (vide), on ignore
        if len(y) == 0:
            return None

        features = {}
        features['filename'] = os.path.basename(file_path)

        # --- FONCTIONS UTILITAIRES ---
        def add_stats(name, data):
            """Ajoute la moyenne et l'écart-type d'une feature"""
            features[f'{name}_mean'] = np.mean(data)
            features[f'{name}_std'] = np.std(data)

        # --- EXTRACTION ---
        
        # 1. MFCC (Timbre)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(1, 21): # On sauve chaque coeff individuellement
            add_stats(f'mfcc_{i}', mfcc[i-1])
            
        # 2. Delta & Delta2 (Dynamique)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        add_stats('mfcc_delta', delta_mfcc)
        add_stats('mfcc_delta2', delta2_mfcc)

        # 3. Spectral Features (Fréquences)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_flatness = librosa.feature.spectral_flatness(y=y)
        
        add_stats('spec_centroid', spec_centroid)
        add_stats('spec_bandwidth', spec_bandwidth)
        add_stats('spec_rolloff', spec_rolloff)
        add_stats('spec_contrast', spec_contrast)
        add_stats('spec_flatness', spec_flatness)

        # 4. Temporel & Volume
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        add_stats('zcr', zcr)
        add_stats('rms', rms)
        
        # 5. Chroma (Tonalité / Notes) - Plus rapide que YIN pour l'IA généraliste
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        add_stats('chroma', chroma)
        
        # NOTE: J'ai retiré YIN (Pitch) car c'est l'algo le plus lent de Librosa.
        # Si vous voulez vraiment YIN, décommentez les lignes ci-dessous (mais le script sera 10x plus lent)
        # f0 = librosa.yin(y, fmin=50, fmax=500)
        # add_stats('pitch_yin', f0)

        return features

    except Exception as e:
        print(f"\nErreur sur {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Extraction rapide de features audio vers CSV")
    parser.add_argument("input_dir", help="Dossier contenant les fichiers .wav")
    parser.add_argument("-o", "--output", default="dataset_features.csv", help="Nom du fichier CSV de sortie")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(), help="Nombre de processus parallèles (défaut: tous les coeurs)")
    
    args = parser.parse_args()

    # Récupération de tous les fichiers wav (récursif)
    search_path = os.path.join(args.input_dir, '**', '*.wav')
    files = glob.glob(search_path, recursive=True)

    if not files:
        print(f"Aucun fichier .wav trouvé dans {args.input_dir}")
        return

    print(f"🚀 Traitement de {len(files)} fichiers avec {args.workers} coeurs...")

    results = []
    
    # Utilisation du Multiprocessing pour la vitesse
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # tqdm affiche la barre de progression
        for result in tqdm(executor.map(extract_features_from_file, files), total=len(files)):
            if result is not None:
                results.append(result)

    # Création et sauvegarde du DataFrame
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\n✅ Terminé ! Données sauvegardées dans : {args.output}")
        print(f"📊 Dimensions du dataset : {df.shape}")
    else:
        print("❌ Aucune feature n'a pu être extraite.")

if __name__ == "__main__":
    main()