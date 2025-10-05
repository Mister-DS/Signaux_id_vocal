import customtkinter as ctk
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class VoiceAuthApp:
    def __init__(self):
        self.fs = 16000  # Fréquence d'échantillonnage
        self.audio_data = []
        self.stream = None 
        self.is_recording = False  # État de l'enregistrement
        
        if not os.path.exists("enregistrements"):
            os.makedirs("enregistrements")
        

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.app = ctk.CTk()
        self.app.title("Reconnaissance vocale MVP")
        self.app.geometry("600x500")
        
        
        self.label = ctk.CTkLabel(
            self.app, 
            text="Système d'authentification vocale", 
            font=("Arial", 18, "bold")
        )
        self.label.pack(pady=20)
        
        # Bouton Démarrer
        self.btn_start = ctk.CTkButton(
            self.app, 
            text="Démarrer l'enregistrement", 
            command=self.start_recording,
            font=("Arial", 14), 
            height=50, 
            width=250,
            fg_color="green"
        )
        self.btn_start.pack(pady=10)
        
        # Bouton Arrêter
        self.btn_stop = ctk.CTkButton(
            self.app, 
            text="Arrêter l'enregistrement", 
            command=self.stop_recording,
            font=("Arial", 14), 
            height=50, 
            width=250,
            fg_color="orange",
            state="disabled"
        )
        self.btn_stop.pack(pady=10)
        
        # Input pour le nom
        self.entry_nom = ctk.CTkEntry(
            self.app,
            placeholder_text="Entrez votre nom",
            font=("Arial", 14),
            width=250,
            height=40
        )
        self.entry_nom.pack(pady=10)
        
        self.label_status = ctk.CTkLabel(
            self.app,
            text="Prêt à enregistrer",
            font=("Arial", 12),
            text_color="gray"
        )
        self.label_status.pack(pady=10)
        
        self.btn_quit = ctk.CTkButton(
            self.app, 
            text="Quitter", 
            command=self.app.quit,
            font=("Arial", 14), 
            height=40, 
            width=200,
            fg_color="red"
        )
        self.btn_quit.pack(pady=20)
    
    def audio_callback(self, indata, frames, time, status):
        """
        Fonction appelée automatiquement par sounddevice
        à chaque fois qu'un morceau audio est capturé
        """
        if status:
            print(f"Erreur audio: {status}")
        
        self.audio_data.append(indata.copy())
    
    def start_recording(self):
        """Démarrer l'enregistrement audio"""
        self.is_recording = True
        self.audio_data = []
        
        self.stream = sd.InputStream(
            samplerate=self.fs,
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()
        
        self.label_status.configure(
            text="Enregistrement en cours...",
            text_color="red"
        )
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        
        print("Enregistrement démarré...")
    
    def stop_recording(self):
        """Arrêter l'enregistrement et sauvegarder"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        self.stream.stop()
        self.stream.close()
        
        nom = self.entry_nom.get().strip()
        
        if not nom:
            self.label_status.configure(
                text="Erreur: Veuillez entrer un nom !",
                text_color="red"
            )
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            return
        
        audio_array = np.concatenate(self.audio_data, axis=0)
        
        numero = self.trouver_prochain_numero(nom)

        filename = f"enregistrements/{nom}_{numero}.wav"
        
        sf.write(filename, audio_array, self.fs)

        self.label_status.configure(
            text=f"Enregistré: {filename}",
            text_color="green"
        )
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        
        print(f"Enregistrement sauvegardé: {filename}")
    
    def trouver_prochain_numero(self, nom_personne):
        """
        Trouve le prochain numéro d'enregistrement pour une personne
        Ex: si alice_1.wav et alice_2.wav existent, retourne 3
        """
        fichiers = os.listdir("enregistrements/")
        
        fichiers_personne = [
            f for f in fichiers 
            if f.startswith(f"{nom_personne}_") and f.endswith(".wav")
        ]
        
        if not fichiers_personne:
            return 1
        
        numeros = []
        for fichier in fichiers_personne:
            try:
                partie = fichier.replace(f"{nom_personne}_", "")
                partie = partie.replace(".wav", "")
                numero = int(partie)
                numeros.append(numero)
            except ValueError:
                continue
        
        return max(numeros) + 1 if numeros else 1
    
    def run(self):
        """Lancer l'application"""
        self.app.mainloop()

if __name__ == "__main__":
    application = VoiceAuthApp()
    application.run()