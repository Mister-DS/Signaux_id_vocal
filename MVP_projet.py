import customtkinter as ctk
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from fastdtw import fastdtw
import librosa
import librosa.display
import os
import whisper
import warnings
import difflib
import ssl
import certifi

# SSL pour whisper (mac)
ssl._create_default_https_context = ssl._create_unverified_context

class VoiceAuthApp:
    def __init__(self):
        # --- Variables d'enregistrement ---
        self.fs = 16000
        self.audio_data = []
        self.stream = None 
        self.is_recording = False
        self.audio_array = None
        
        self.whisper_model = None 
        
        # --- Configuration des dossiers ---
        dossier_script = os.path.dirname(os.path.abspath(__file__))
        self.dossier_samples = os.path.join(dossier_script, "samples")

        if not os.path.exists(self.dossier_samples):
            os.makedirs(self.dossier_samples)
        
        # --- Configuration de l'interface (CTK) ---
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.app = ctk.CTk()
        self.app.title("Analyse vocale MVP")
        self.app.geometry("650x900")

        self.main_frame = ctk.CTkScrollableFrame(
            self.app,
            width=600,
            height=850
        )
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Section Enregistrement ---
        self.label = ctk.CTkLabel(
            self.main_frame,
            text="Système d'analyse vocale",
            font=("Arial", 18, "bold")
        )
        self.label.pack(pady=15)
        
        self.btn_start = ctk.CTkButton(
            self.main_frame,
            text="Démarrer l'enregistrement",
            command=self.start_recording,
            font=("Arial", 14),
            height=45,
            width=250,
            fg_color="green"
        )
        self.btn_start.pack(pady=10)

        self.btn_stop = ctk.CTkButton(
            self.main_frame,
            text="Arrêter l'enregistrement",
            command=self.stop_recording,
            font=("Arial", 14),
            height=45,
            width=250,
            fg_color="orange",
            state="disabled"
        )
        self.btn_stop.pack(pady=10)

        self.entry_nom = ctk.CTkEntry(
            self.main_frame,
            placeholder_text="Entrez votre nom",
            font=("Arial", 14),
            width=250,
            height=35
        )
        self.entry_nom.pack(pady=10)

        self.btn_valider = ctk.CTkButton(
            self.main_frame,
            text="Valider et enregistrer",
            command=self.valider_enregistrement,
            font=("Arial", 14),
            height=45,
            width=250,
            fg_color="blue",
            state="disabled"
        )
        self.btn_valider.pack(pady=10)

        self.label_status = ctk.CTkLabel(
            self.main_frame,
            text="Prêt à enregistrer",
            font=("Arial", 12),
            text_color="gray"
        )
        self.label_status.pack(pady=10)
        
        # --- Section Visualisation & Transcription ---
        self.separator = ctk.CTkLabel(
            self.main_frame,
            text="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            font=("Arial", 10),
            text_color="gray"
        )
        self.separator.pack(pady=10)

        self.label_visualisation = ctk.CTkLabel(
            self.main_frame,
            text="Analyse des enregistrements",
            font=("Arial", 14, "bold")
        )
        self.label_visualisation.pack(pady=10)

        self.dropdown = ctk.CTkComboBox(
            self.main_frame,
            values=self.charger_liste_fichiers(),
            font=("Arial", 12),
            width=300,
            height=35,
            state="readonly"
        )
        self.dropdown.pack(pady=10)
        self.dropdown.set("Sélectionnez un fichier")

        self.btn_voir_signal = ctk.CTkButton(
            self.main_frame,
            text="Voir le signal",
            command=self.afficher_signal,
            font=("Arial", 14),
            height=45,
            width=250,
            fg_color="purple"
        )
        self.btn_voir_signal.pack(pady=10)

        self.btn_spectrogramme = ctk.CTkButton(
            self.main_frame,
            text="Voir le spectrogramme",
            command=self.afficher_spectrogramme,
            font=("Arial", 14),
            height=45,
            width=250,
            fg_color="#8B008B"
        )
        self.btn_spectrogramme.pack(pady=10)

        self.btn_fft = ctk.CTkButton(
            self.main_frame,
            text="Voir la transformée de Fourier",
            command=self.afficher_fft,
            font=("Arial", 14),
            height=45,
            width=250,
            fg_color="#4B0082"
        )
        self.btn_fft.pack(pady=10)

        self.btn_lecture = ctk.CTkButton(
            self.main_frame,
            text="Écouter l'audio",
            command=self.lire_audio,
            font=("Arial", 14),
            height=45,
            width=250,
            fg_color="#006400"
        )
        self.btn_lecture.pack(pady=10)

        # --- Bloc de bouton ---
        self.btn_transcrire = ctk.CTkButton(
            self.main_frame,
            text="Transcrire le signal (Whisper)",
            command=self.transcrire_signal_selectionne,
            font=("Arial", 14),
            height=45,
            width=250,
            fg_color="#D90077"
        )
        self.btn_transcrire.pack(pady=10)

        # --- Section Comparaison vocale ---
        self.separator2 = ctk.CTkLabel(
            self.main_frame,
            text="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            font=("Arial", 10),
            text_color="gray"
        )
        self.separator2.pack(pady=10)

        self.label_comparaison = ctk.CTkLabel(
            self.main_frame,
            text="Comparaison de samples (Vérification vocale)",
            font=("Arial", 14, "bold")
        )
        self.label_comparaison.pack(pady=10)

        self.frame_comparaison = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.frame_comparaison.pack(pady=10)

        self.label_sample1 = ctk.CTkLabel(
            self.frame_comparaison,
            text="Sample 1 (référence):",
            font=("Arial", 11)
        )
        self.label_sample1.grid(row=0, column=0, padx=10)

        self.dropdown_sample1 = ctk.CTkComboBox(
            self.frame_comparaison,
            values=self.charger_liste_fichiers(),
            font=("Arial", 11),
            width=230,
            height=30,
            state="readonly"
        )
        self.dropdown_sample1.grid(row=1, column=0, padx=10)
        self.dropdown_sample1.set("Sample 1")

        self.label_sample2 = ctk.CTkLabel(
            self.frame_comparaison,
            text="Sample 2 (test):",
            font=("Arial", 11)
        )
        self.label_sample2.grid(row=0, column=1, padx=10)

        self.dropdown_sample2 = ctk.CTkComboBox(
            self.frame_comparaison,
            values=self.charger_liste_fichiers(),
            font=("Arial", 11),
            width=230,
            height=30,
            state="readonly"
        )
        self.dropdown_sample2.grid(row=1, column=1, padx=10)
        self.dropdown_sample2.set("Sample 2")

        self.btn_comparer = ctk.CTkButton(
            self.main_frame,
            text="Comparer les samples",
            command=self.comparer_samples,
            font=("Arial", 14),
            height=45,
            width=250,
            fg_color="#FF6600"
        )
        self.btn_comparer.pack(pady=10)

        # --- Bouton Quitter ---
        self.btn_quit = ctk.CTkButton(
            self.main_frame,
            text="Quitter",
            command=self.app.quit,
            font=("Arial", 14),
            height=40,
            width=200,
            fg_color="red"
        )
        self.btn_quit.pack(pady=20) 
    
    # --- Fonctions d'enregistrement ---
    
    def charger_liste_fichiers(self):
        """Charge la liste des fichiers WAV depuis le dossier samples"""
        try:
            fichiers_wav = []
            fichiers = os.listdir(self.dossier_samples)
            for fichier in fichiers:
                chemin_complet = os.path.join(self.dossier_samples, fichier)
                # Vérifier que c'est un fichier (pas un dossier) et qu'il se termine par .wav
                if os.path.isfile(chemin_complet) and fichier.endswith(".wav"):
                    # Enlever l'extension .wav
                    nom_sans_extension = fichier.replace('.wav', '')
                    fichiers_wav.append(nom_sans_extension)

            return sorted(fichiers_wav) if fichiers_wav else ["Aucun fichier"]
        except Exception as e:
            print(f"Erreur lors du chargement des fichiers: {e}")
            return ["Erreur de chargement"]
    
    def mettre_a_jour_liste(self):
        fichiers = self.charger_liste_fichiers()
        self.dropdown.configure(values=fichiers)
        if fichiers and fichiers[0] != "Aucun fichier":
            self.dropdown.set(fichiers[-1])
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Erreur audio: {status}")
        self.audio_data.append(indata.copy())
    
    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        self.audio_array = None
        self.stream = sd.InputStream(samplerate=self.fs, channels=1, callback=self.audio_callback)
        self.stream.start()
        self.label_status.configure(text="Enregistrement en cours...", text_color="red")
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_valider.configure(state="disabled")
        self.entry_nom.delete(0, 'end')
        print("Enregistrement démarré.")
    
    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        self.audio_array = np.concatenate(self.audio_data, axis=0)
        self.label_status.configure(text="Enregistrement arrêté. Entrez votre nom et validez.", text_color="orange")
        self.btn_stop.configure(state="disabled")
        self.btn_valider.configure(state="normal")
        print("Enregistrement arrêté.")
    
    def valider_enregistrement(self):
        if self.audio_array is None:
            self.label_status.configure(text="Erreur: Pas d'audio à sauvegarder !", text_color="red")
            return

        nom = self.entry_nom.get().strip().lower()
        if not nom:
            self.label_status.configure(text="Erreur: Veuillez entrer un nom !", text_color="red")
            return

        # Trouver le prochain numéro disponible
        numero = self.trouver_prochain_numero_simple(nom)
        filename = os.path.join(self.dossier_samples, f"{nom}_{numero}.wav")

        sf.write(filename, self.audio_array, self.fs)
        self.label_status.configure(text=f"Enregistré: {nom}_{numero}.wav", text_color="green")
        self.btn_start.configure(state="normal")
        self.btn_valider.configure(state="disabled")
        print(f"Enregistrement sauvegardé: {filename}")
        self.mettre_a_jour_liste()
        self.audio_array = None
    
    def trouver_prochain_numero_simple(self, nom_personne):
        """Trouve le prochain numéro disponible pour un enregistrement"""
        fichiers = os.listdir(self.dossier_samples)
        fichiers_personne = [f for f in fichiers if f.startswith(f"{nom_personne}_") and f.endswith(".wav")]

        if not fichiers_personne:
            return 1

        numeros = []
        for fichier in fichiers_personne:
            try:
                partie = fichier.replace(f"{nom_personne}_", "").replace(".wav", "")
                numero = int(partie)
                numeros.append(numero)
            except ValueError:
                continue

        return max(numeros) + 1 if numeros else 1
    
    # --- Fonctions de visualisation ---

    def afficher_signal(self):
        fichier_selectionne = self.dropdown.get()
        if fichier_selectionne == "Sélectionnez un fichier" or fichier_selectionne == "Aucun fichier":
            self.label_status.configure(text="Veuillez sélectionner un fichier !", text_color="red")
            return

        chemin_complet = os.path.join(self.dossier_samples, fichier_selectionne + ".wav")
        if not os.path.exists(chemin_complet):
            self.label_status.configure(text=f"Fichier introuvable: {fichier_selectionne}.wav", text_color="red")
            return

        try:
            data, samplerate = sf.read(chemin_complet)
            duree = len(data) / samplerate
            temps = np.linspace(0, duree, len(data))
            plt.figure(figsize=(12, 4))
            plt.plot(temps, data, color='blue', linewidth=0.8)
            plt.xlabel('Temps (secondes)', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.title(f'Signaux {fichier_selectionne}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xlim([0, duree])
            plt.tight_layout()
            plt.show()
            self.label_status.configure(text=f"Signal affiché: {fichier_selectionne}", text_color="green")
            print(f"Signal affiché pour: {fichier_selectionne}")
        except Exception as e:
            self.label_status.configure(text=f"Erreur lors de l'affichage: {str(e)}", text_color="red")
            print(f"Erreur: {e}")

    def afficher_spectrogramme(self):
        fichier_selectionne = self.dropdown.get()
        if fichier_selectionne == "Sélectionnez un fichier" or fichier_selectionne == "Aucun fichier":
            self.label_status.configure(text="Veuillez sélectionner un fichier !", text_color="red")
            return

        chemin_complet = os.path.join(self.dossier_samples, fichier_selectionne + ".wav")
        if not os.path.exists(chemin_complet):
            self.label_status.configure(text=f"Fichier introuvable: {fichier_selectionne}.wav", text_color="red")
            return

        try:
            data, samplerate = sf.read(chemin_complet)

            # Créer le spectrogramme avec scipy
            f, t, Sxx = signal.spectrogram(data, samplerate, nperseg=1024)

            plt.figure(figsize=(12, 6))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            plt.ylabel('Fréquence (Hz)', fontsize=12)
            plt.xlabel('Temps (secondes)', fontsize=12)
            plt.title(f'Spectrogramme - {fichier_selectionne}', fontsize=14)
            plt.colorbar(label='Intensité (dB)')
            plt.ylim([0, 8000])
            plt.tight_layout()
            plt.show()

            self.label_status.configure(text=f"Spectrogramme affiché: {fichier_selectionne}", text_color="green")
            print(f"Spectrogramme affiché pour: {fichier_selectionne}")
        except Exception as e:
            self.label_status.configure(text=f"Erreur lors de l'affichage: {str(e)}", text_color="red")
            print(f"Erreur: {e}")

    def afficher_fft(self):
        fichier_selectionne = self.dropdown.get()
        if fichier_selectionne == "Sélectionnez un fichier" or fichier_selectionne == "Aucun fichier":
            self.label_status.configure(text="Veuillez sélectionner un fichier !", text_color="red")
            return

        chemin_complet = os.path.join(self.dossier_samples, fichier_selectionne + ".wav")
        if not os.path.exists(chemin_complet):
            self.label_status.configure(text=f"Fichier introuvable: {fichier_selectionne}.wav", text_color="red")
            return

        try:
            data, samplerate = sf.read(chemin_complet)

            # Calculer la FFT
            N = len(data)
            yf = fft(data)
            xf = fftfreq(N, 1 / samplerate)

            xf_pos = xf[:N//2]
            yf_pos = 2.0/N * np.abs(yf[:N//2])

            plt.figure(figsize=(12, 6))
            plt.plot(xf_pos, yf_pos, color='red', linewidth=0.8)
            plt.xlabel('Fréquence (Hz)', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.title(f'Transformée de Fourier - {fichier_selectionne}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xlim([0, 8000])
            plt.tight_layout()
            plt.show()

            self.label_status.configure(text=f"FFT affichée: {fichier_selectionne}", text_color="green")
            print(f"FFT affichée pour: {fichier_selectionne}")
        except Exception as e:
            self.label_status.configure(text=f"Erreur lors de l'affichage: {str(e)}", text_color="red")
            print(f"Erreur: {e}")

    def lire_audio(self):
        fichier_selectionne = self.dropdown.get()
        if fichier_selectionne == "Sélectionnez un fichier" or fichier_selectionne == "Aucun fichier":
            self.label_status.configure(text="Veuillez sélectionner un fichier !", text_color="red")
            return

        chemin_complet = os.path.join(self.dossier_samples, fichier_selectionne + ".wav")
        if not os.path.exists(chemin_complet):
            self.label_status.configure(text=f"Fichier introuvable: {fichier_selectionne}.wav", text_color="red")
            return

        try:
            data, samplerate = sf.read(chemin_complet)
            self.label_status.configure(text=f"Lecture de {fichier_selectionne}...", text_color="yellow")
            self.app.update_idletasks()

            sd.play(data, samplerate)
            sd.wait()

            self.label_status.configure(text=f"Lecture terminée: {fichier_selectionne}", text_color="green")
            print(f"Audio lu: {fichier_selectionne}")
        except Exception as e:
            self.label_status.configure(text=f"Erreur lors de la lecture: {str(e)}", text_color="red")
            print(f"Erreur: {e}")

    # --- Fonctions de transcription ---
    
    def charger_modele_whisper(self):
        if self.whisper_model is None:
            self.label_status.configure(text="Chargement", text_color="yellow")
            print("Chargement")
            self.app.update_idletasks()

            self.whisper_model = whisper.load_model("base")
            print("Modèle chargé.")
            
    def transcrire_signal_selectionne(self):
        warnings.filterwarnings("ignore")

        fichier_selectionne = self.dropdown.get()
        if fichier_selectionne == "Sélectionnez un fichier" or fichier_selectionne == "Aucun fichier":
            self.label_status.configure(text="Veuillez sélectionner un fichier !", text_color="red")
            return

        chemin_complet = os.path.normpath(os.path.join(self.dossier_samples, fichier_selectionne + ".wav"))

        print(f"DEBUG - Fichier sélectionné: {fichier_selectionne}")
        print(f"DEBUG - Chemin complet: {chemin_complet}")
        print(f"DEBUG - Fichier existe: {os.path.exists(chemin_complet)}")
        print(f"DEBUG - Chemin absolu: {os.path.abspath(chemin_complet)}")

        if not os.path.exists(chemin_complet):
            self.label_status.configure(text=f"Fichier introuvable: {fichier_selectionne}.wav", text_color="red")
            print(f"Chemin recherché: {chemin_complet}")
            return

        try:
            self.charger_modele_whisper()

            self.label_status.configure(text=f"Transcription de {fichier_selectionne}...", text_color="yellow")
            self.app.update_idletasks()

            result = self.whisper_model.transcribe(
                chemin_complet,
                word_timestamps=True,
                fp16=False
            )

            full_text = result.get("text", "Aucun texte détecté.").strip()

            output_text = f"--- TRANSCRIPTION COMPLÈTE ---\n{full_text}\n\n"
            output_text += "--- DÉTAIL DES MOTS ---\n"

            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    start = word_info.get('start', 0)
                    end = word_info.get('end', 0)
                    word = word_info.get('word', '')
                    output_text += f"[{start:.2f}s - {end:.2f}s] : {word}\n"

            self.afficher_resultat_transcription(fichier_selectionne, output_text)

            self.label_status.configure(text=f"Transcription terminée: {fichier_selectionne}", text_color="green")
            print(f"Transcription terminée pour: {fichier_selectionne}")

        except Exception as e:
            self.label_status.configure(text=f"Erreur de transcription: {str(e)}", text_color="red")
            print(f"Erreur Whisper: {e}")

    def afficher_resultat_transcription(self, titre, texte):
        popup = ctk.CTkToplevel(self.app)
        popup.title(f"Transcription de {titre}")
        popup.geometry("500x600")
        popup.grab_set()

        textbox = ctk.CTkTextbox(
            popup,
            wrap="word",
            font=("Arial", 12)
        )
        textbox.pack(expand=True, fill="both", padx=10, pady=10)

        textbox.insert("1.0", texte)
        textbox.configure(state="disabled")

        btn_close = ctk.CTkButton(
            popup,
            text="Fermer",
            command=popup.destroy
        )
        btn_close.pack(pady=10)

    # --- Fonctions de comparaison vocale  ---

    def pretraiter_audio(self, y, sr):
        """Pré-traitement audio avancé"""
        try:
            # Supprimer les silences aux extrémités
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)

            y_normalized = librosa.util.normalize(y_trimmed)

            return y_normalized
        except Exception as e:
            print(f"Erreur lors du prétraitement: {e}")
            return y

    def extraire_caracteristiques_avancees(self, chemin_audio):
        """Extraction de caractéristiques vocales enrichies"""
        try:
            # Charger l'audio
            y, sr = librosa.load(chemin_audio, sr=16000)

            # Pré-traiter
            y = self.pretraiter_audio(y, sr)

            # 1. MFCC avec plus de coefficients
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

            # 2. Delta MFCC
            mfcc_delta = librosa.feature.delta(mfccs)

            # 3. Delta-Delta MFCC
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

            # 4. Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            # 5. Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

            # 6. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)

            # Combiner toutes les caractéristiques
            caracteristiques = np.vstack([
                mfccs,
                mfcc_delta,
                mfcc_delta2,
                chroma,
                spectral_contrast,
                zcr
            ])

            # Normaliser
            caracteristiques_norm = (caracteristiques - np.mean(caracteristiques, axis=1, keepdims=True)) / (np.std(caracteristiques, axis=1, keepdims=True) + 1e-10)

            return caracteristiques_norm, mfccs
        except Exception as e:
            print(f"Erreur lors de l'extraction: {e}")
            return None, None

    def calculer_similarite_dtw(self, feat1, feat2):
        """Calcul de similarité avec DTW (Dynamic Time Warping)"""
        try:
            # Utiliser DTW pour comparer les séquences de différentes longueurs
            distance, _ = fastdtw(feat1.T, feat2.T, dist=euclidean)

            # Normaliser par la longueur moyenne
            longueur_moyenne = (feat1.shape[1] + feat2.shape[1]) / 2
            distance_normalisee = distance / longueur_moyenne

            # Convertir en similarité 
            similarite_dtw = 100 * np.exp(-distance_normalisee / 15)

            return similarite_dtw, distance_normalisee
        except Exception as e:
            print(f"Erreur DTW: {e}")
            return 0, float('inf')

    def calculer_similarite_cosine(self, feat1, feat2):
        """Calcul de similarité cosinus"""
        try:
            # Aplatir et calculer la similarité cosinus
            min_frames = min(feat1.shape[1], feat2.shape[1])
            feat1_flat = feat1[:, :min_frames].flatten()
            feat2_flat = feat2[:, :min_frames].flatten()

            # Similarité cosinus (1 = identique, 0 = différent)
            cos_sim = 1 - cosine(feat1_flat, feat2_flat)

            # Convertir en pourcentage
            similarite_cos = max(0, cos_sim * 100)

            return similarite_cos
        except Exception as e:
            print(f"Erreur cosine: {e}")
            return 0

    def calculer_correlation(self, feat1, feat2):
        """Calcul de corrélation de Pearson"""
        try:
            min_frames = min(feat1.shape[1], feat2.shape[1])
            feat1_flat = feat1[:, :min_frames].flatten()
            feat2_flat = feat2[:, :min_frames].flatten()

            # Corrélation de Pearson
            corr, _ = pearsonr(feat1_flat, feat2_flat)

            # Convertir en pourcentage
            similarite_corr = max(0, (corr + 1) / 2 * 100)

            return similarite_corr
        except Exception as e:
            print(f"Erreur corrélation: {e}")
            return 0

    def calculer_score_composite(self, feat1, feat2, mfcc1, mfcc2):
        """Calcul d'un score composite pondéré"""
        try:
            # 1. Similarité DTW
            sim_dtw, dist_dtw = self.calculer_similarite_dtw(mfcc1, mfcc2)

            # 2. Similarité cosinus
            sim_cos = self.calculer_similarite_cosine(feat1, feat2)

            # 3. Corrélation
            sim_corr = self.calculer_correlation(feat1, feat2)

            # Score composite pondéré
            score_final = (0.40 * sim_dtw + 0.30 * sim_cos + 0.30 * sim_corr)

            details = {
                'dtw': sim_dtw,
                'cosine': sim_cos,
                'correlation': sim_corr,
                'distance_dtw': dist_dtw
            }

            return score_final, details
        except Exception as e:
            print(f"Erreur score composite: {e}")
            return 0, {}

    def transcrire_pour_comparaison(self, chemin_audio):
        try:
            self.charger_modele_whisper()

            result = self.whisper_model.transcribe(
                chemin_audio,
                word_timestamps=True,
                fp16=False
            )

            texte = result.get("text", "").strip()
            mots = texte.split()

            return texte, mots, len(mots)
        except Exception as e:
            print(f"Erreur lors de la transcription: {e}")
            return "", [], 0

    def comparer_textes(self, mots1, mots2):
        # Normaliser les mots
        def normaliser_mot(mot):
            return ''.join(c.lower() for c in mot if c.isalnum())

        mots1_norm = [normaliser_mot(m) for m in mots1 if normaliser_mot(m)]
        mots2_norm = [normaliser_mot(m) for m in mots2 if normaliser_mot(m)]

        # Utiliser difflib pour trouver les différences
        matcher = difflib.SequenceMatcher(None, mots1_norm, mots2_norm)

        # Calculer le ratio de similarité
        ratio = matcher.ratio() * 100

        # Trouver les mots ajoutés et supprimés
        mots_ajoutes = []
        mots_supprimes = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'delete':
                mots_supprimes.extend(mots1[i1:i2])
            elif tag == 'insert':
                mots_ajoutes.extend(mots2[j1:j2])

        return ratio, mots_ajoutes, mots_supprimes

    def comparer_samples(self):
        warnings.filterwarnings("ignore")

        sample1 = self.dropdown_sample1.get()
        sample2 = self.dropdown_sample2.get()

        if sample1 in ["Sample 1", "Aucun fichier", "Sélectionnez un fichier"]:
            self.label_status.configure(text="Veuillez sélectionner le Sample 1 !", text_color="red")
            return

        if sample2 in ["Sample 2", "Aucun fichier", "Sélectionnez un fichier"]:
            self.label_status.configure(text="Veuillez sélectionner le Sample 2 !", text_color="red")
            return

        chemin1 = os.path.join(self.dossier_samples, sample1 + ".wav")
        chemin2 = os.path.join(self.dossier_samples, sample2 + ".wav")

        if not os.path.exists(chemin1) or not os.path.exists(chemin2):
            self.label_status.configure(text="Un ou plusieurs fichiers introuvables !", text_color="red")
            return

        try:
            self.label_status.configure(text="Analyse avancée en cours.", text_color="yellow")
            self.app.update_idletasks()

            # 1. Extraction des caractéristiques avancées
            print("Extraction des caractéristiques avancées.")
            feat1, mfcc1 = self.extraire_caracteristiques_avancees(chemin1)
            feat2, mfcc2 = self.extraire_caracteristiques_avancees(chemin2)

            if feat1 is None or feat2 is None or mfcc1 is None or mfcc2 is None:
                self.label_status.configure(text="Erreur lors de l'extraction des caractéristiques !", text_color="red")
                return

            # 2. Calcul du score composite
            print("Calcul des métriques de similarité.")
            score_final, details = self.calculer_score_composite(feat1, feat2, mfcc1, mfcc2)

            # 3. Transcription et comparaison de texte
            print("Transcription des samples...")
            texte1, mots1, nb_mots1 = self.transcrire_pour_comparaison(chemin1)
            texte2, mots2, nb_mots2 = self.transcrire_pour_comparaison(chemin2)

            ratio_texte, mots_ajoutes, mots_supprimes = self.comparer_textes(mots1, mots2)

            resultat = "═══════════════════════════════════════\n"
            resultat += "   RAPPORT D'ANALYSE VOCALE AVANCÉE\n"
            resultat += "═══════════════════════════════════════\n\n"

            resultat += f"📁 Sample 1: {sample1}\n"
            resultat += f"📁 Sample 2: {sample2}\n\n"

            resultat += "--- ANALYSE VOCALE MULTI-MÉTRIQUES ---\n"
            resultat += f"🎯 Score de similarité vocal: {score_final:.2f}%\n\n"

            resultat += "Métriques détaillées:\n"
            resultat += f"  • DTW (Dynamic Time Warping): {details.get('dtw', 0):.2f}%\n"
            resultat += f"  • Similarité Cosinus: {details.get('cosine', 0):.2f}%\n"
            resultat += f"  • Corrélation de Pearson: {details.get('correlation', 0):.2f}%\n"
            resultat += f"  • Distance DTW normalisée: {details.get('distance_dtw', 0):.4f}\n\n"

            # Interprétation du score vocal
            if score_final > 85:
                resultat += "✓✓ Voix TRÈS SIMILAIRES - Même personne avec haute confiance\n"
                confiance_vocale = "HAUTE"
            elif score_final > 70:
                resultat += "✓ Voix similaires - Probablement la même personne\n"
                confiance_vocale = "BONNE"
            elif score_final > 55:
                resultat += "⚠ Voix moyennement similaires - Possiblement la même personne\n"
                confiance_vocale = "MOYENNE"
            else:
                resultat += "✗ Voix DIFFÉRENTES - Probablement pas la même personne\n"
                confiance_vocale = "FAIBLE"

            resultat += "\n--- ANALYSE TEXTUELLE ---\n"
            resultat += f"Sample 1 ({nb_mots1} mots):\n  \"{texte1}\"\n\n"
            resultat += f"Sample 2 ({nb_mots2} mots):\n  \"{texte2}\"\n\n"

            resultat += f"Similarité textuelle: {ratio_texte:.2f}%\n"
            resultat += f"Différence de mots: {abs(nb_mots1 - nb_mots2)}\n\n"

            if mots_supprimes:
                resultat += f"Mots supprimés (dans Sample 1 seulement):\n"
                resultat += f"  {', '.join(mots_supprimes)}\n\n"

            if mots_ajoutes:
                resultat += f"Mots ajoutés (dans Sample 2 seulement):\n"
                resultat += f"  {', '.join(mots_ajoutes)}\n\n"

            resultat += "--- VERDICT FINAL ---\n"

            if score_final > 80 and ratio_texte > 70:
                resultat += "MÊME PERSONNE - Haute confiance\n"
                resultat += "    Voix ET contenu très similaires\n"
                verdict = "MATCH CONFIRMÉ"
            elif score_final > 80 and ratio_texte <= 70:
                resultat += "MÊME PERSONNE - Bonne confiance\n"
                resultat += "    Même voix mais contenu différent\n"
                verdict = "MATCH PROBABLE"
            elif score_final > 65 and ratio_texte > 60:
                resultat += "PROBABLEMENT LA MÊME PERSONNE\n"
                resultat += "    Similarités vocales et textuelles moyennes\n"
                verdict = "MATCH POSSIBLE"
            elif score_final <= 65 and ratio_texte > 80:
                resultat += "Personnes DIFFÉRENTES mais contenu similaire\n"
                resultat += "    Attention: phrases identiques prononcées par des voix différentes\n"
                verdict = "PAS DE MATCH"
            else:
                resultat += "PERSONNES DIFFÉRENTES - Haute confiance\n"
                resultat += "    Voix ET contenu différents\n"
                verdict = "PAS DE MATCH"

            resultat += f"\nConfiance vocale: {confiance_vocale}\n"
            resultat += f"Verdict: {verdict}\n"

            # Afficher le résultat
            self.afficher_resultat_comparaison(resultat)

            self.label_status.configure(text="Analyse terminée !", text_color="green")
            print("Analyse terminée.")

        except Exception as e:
            self.label_status.configure(text=f"Erreur lors de la comparaison: {str(e)}", text_color="red")
            print(f"Erreur: {e}")
            import traceback
            traceback.print_exc()

    def afficher_resultat_comparaison(self, texte):
        popup = ctk.CTkToplevel(self.app)
        popup.title("Résultat de la comparaison")
        popup.geometry("650x700")
        popup.grab_set()

        textbox = ctk.CTkTextbox(
            popup,
            wrap="word",
            font=("Courier", 11)
        )
        textbox.pack(expand=True, fill="both", padx=10, pady=10)

        textbox.insert("1.0", texte)
        textbox.configure(state="disabled")

        btn_close = ctk.CTkButton(
            popup,
            text="Fermer",
            command=popup.destroy,
            height=40,
            width=150
        )
        btn_close.pack(pady=10)

    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    application = VoiceAuthApp()
    application.run()