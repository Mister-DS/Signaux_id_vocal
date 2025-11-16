import customtkinter as ctk
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
import whisper 
import warnings

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
        self.app.geometry("600x800") 
        
        
        # --- Section Enregistrement ---
        self.label = ctk.CTkLabel(
            self.app,
            text="Système d'analyse vocale",
            font=("Arial", 18, "bold")
        )
        self.label.pack(pady=15)
        
        self.btn_start = ctk.CTkButton(
            self.app, 
            text="Démarrer l'enregistrement", 
            command=self.start_recording,
            font=("Arial", 14), 
            height=45, 
            width=250,
            fg_color="green"
        )
        self.btn_start.pack(pady=10)
        
        self.btn_stop = ctk.CTkButton(
            self.app, 
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
            self.app,
            placeholder_text="Entrez votre nom",
            font=("Arial", 14),
            width=250,
            height=35
        )
        self.entry_nom.pack(pady=10)
        
        self.btn_valider = ctk.CTkButton(
            self.app,
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
            self.app,
            text="Prêt à enregistrer",
            font=("Arial", 12),
            text_color="gray"
        )
        self.label_status.pack(pady=10)
        
        # --- Section Visualisation & Transcription ---
        self.separator = ctk.CTkLabel(
            self.app,
            text="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            font=("Arial", 10),
            text_color="gray"
        )
        self.separator.pack(pady=10)
        
        self.label_visualisation = ctk.CTkLabel(
            self.app,
            text="Analyse des enregistrements", 
            font=("Arial", 14, "bold")
        )
        self.label_visualisation.pack(pady=10)
        
        self.dropdown = ctk.CTkComboBox(
            self.app,
            values=self.charger_liste_fichiers(),
            font=("Arial", 12),
            width=300,
            height=35,
            state="readonly"
        )
        self.dropdown.pack(pady=10)
        self.dropdown.set("Sélectionnez un fichier")
        
        self.btn_voir_signal = ctk.CTkButton(
            self.app,
            text="Voir le signal",
            command=self.afficher_signal,
            font=("Arial", 14),
            height=45,
            width=250,
            fg_color="purple"
        )
        self.btn_voir_signal.pack(pady=10)
        
        # --- Bloc de bouton ---
        self.btn_transcrire = ctk.CTkButton(
            self.app, 
            text="Transcrire le signal (Whisper)",
            command=self.transcrire_signal_selectionne, 
            font=("Arial", 14), 
            height=45, 
            width=250,
            fg_color="#D90077" 
        )
        self.btn_transcrire.pack(pady=10) 
        
        # --- Bouton Quitter ---
        self.btn_quit = ctk.CTkButton(
            self.app, 
            text="Quitter", 
            command=self.app.quit,
            font=("Arial", 14), 
            height=40, 
            width=200,
            fg_color="red"
        )
        self.btn_quit.pack(pady=20, side="bottom") 
    
    # --- Fonctions d'enregistrement ---
    
    def charger_liste_fichiers(self):
        try:
            fichiers_wav = []
            for dossier_personne in os.listdir(self.dossier_samples):
                chemin_personne = os.path.join(self.dossier_samples, dossier_personne)
                if os.path.isdir(chemin_personne):
                    fichiers = os.listdir(chemin_personne)
                    for fichier in fichiers:
                        if fichier.endswith(".wav"):
                            nom_relatif = f"{dossier_personne}/{fichier.replace('.wav', '')}"
                            fichiers_wav.append(nom_relatif)
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
        print("Enregistrement démarré...")
    
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
        print("Enregistrement arrêté. Validation...")
    
    def valider_enregistrement(self):
        if self.audio_array is None:
            self.label_status.configure(text="Erreur: Pas d'audio à sauvegarder !", text_color="red")
            return
        
        nom = self.entry_nom.get().strip().lower()
        if not nom:
            self.label_status.configure(text="Erreur: Veuillez entrer un nom !", text_color="red")
            return

        dossier_personne = self.creer_dossier_personne(nom)
        numero = self.trouver_prochain_numero(nom, dossier_personne)
        filename = os.path.join(dossier_personne, f"{nom}_{numero}.wav")
        sf.write(filename, self.audio_array, self.fs)
        self.label_status.configure(text=f"Enregistré: {nom}_{numero}.wav", text_color="green")
        self.btn_start.configure(state="normal")
        self.btn_valider.configure(state="disabled")
        print(f"Enregistrement sauvegardé: {filename}")
        self.mettre_a_jour_liste()
        self.audio_array = None
    
    def creer_dossier_personne(self, nom):
        dossiers_existants = [d for d in os.listdir(self.dossier_samples) if os.path.isdir(os.path.join(self.dossier_samples, d)) and d.startswith("p")]
        numeros = []
        for dossier in dossiers_existants:
            try:
                numero = int(dossier.split('_')[0][1:])
                numeros.append(numero)
            except (ValueError, IndexError):
                continue
        for dossier in dossiers_existants:
            if dossier.lower().endswith(f"_{nom.lower()}"):
                return os.path.join(self.dossier_samples, dossier)
        prochain_numero = max(numeros) + 1 if numeros else 8
        nom_dossier = f"p{prochain_numero:02d}_{nom}"
        chemin_dossier = os.path.join(self.dossier_samples, nom_dossier)
        os.makedirs(chemin_dossier, exist_ok=True)
        return chemin_dossier

    def trouver_prochain_numero(self, nom_personne, dossier_personne):
        fichiers = os.listdir(dossier_personne)
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
    
    # --- Fonction de visualisation ---
    
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

    # --- Fonctions de transcription ---
    
    def charger_modele_whisper(self):
        if self.whisper_model is None:
            self.label_status.configure(text="Chargement du modèle Whisper 'base'...", text_color="yellow")
            print("Chargement du modèle Whisper 'base'...")
            self.app.update_idletasks()

            self.whisper_model = whisper.load_model("base")
            print("Modèle Whisper chargé.")
            
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

    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    application = VoiceAuthApp()
    application.run()