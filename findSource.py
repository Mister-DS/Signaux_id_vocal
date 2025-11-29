import customtkinter as ctk
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time
from datetime import datetime

class SimpleRecorder(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuration de la fenêtre
        self.title("Enregistreur avec Choix de Source")
        self.geometry("500x400")
        ctk.set_appearance_mode("dark")
        
        # Variables d'enregistrement
        self.fs = 44100  # Fréquence d'échantillonnage
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        self.start_time = None

        # --- INTERFACE ---
        
        # Titre
        self.label_title = ctk.CTkLabel(self, text="Test Micro & Enregistrement", font=("Arial", 20, "bold"))
        self.label_title.pack(pady=20)

        # 1. MENU DÉROULANT POUR CHOISIR LE MICRO
        self.label_mic = ctk.CTkLabel(self, text="Choisir le microphone :")
        self.label_mic.pack(pady=(10, 0))

        self.mic_options = self.get_input_devices()
        self.combo_mics = ctk.CTkComboBox(self, values=self.mic_options, width=350, state="readonly")
        self.combo_mics.pack(pady=5)
        
        # Sélectionner le micro par défaut si possible
        if self.mic_options:
            self.combo_mics.set(self.mic_options[0])

        # Status
        self.label_status = ctk.CTkLabel(self, text="Prêt", text_color="gray")
        self.label_status.pack(pady=20)

        # Boutons
        self.btn_start = ctk.CTkButton(self, text="🔴 ENREGISTRER", command=self.start_recording, fg_color="red", height=50)
        self.btn_start.pack(pady=10)

        self.btn_stop = ctk.CTkButton(self, text="⏹ ARRÊTER", command=self.stop_recording, fg_color="gray", state="disabled", height=50)
        self.btn_stop.pack(pady=10)

    def get_input_devices(self):
        """Récupère la liste des périphériques d'entrée (Micros)"""
        devices = sd.query_devices()
        device_list = []
        for i, dev in enumerate(devices):
            # On garde seulement ceux qui ont des canaux d'entrée (> 0)
            if dev['max_input_channels'] > 0:
                # Format: "ID: Nom du micro"
                device_list.append(f"{i}: {dev['name']}")
        return device_list

    def audio_callback(self, indata, frames, time, status):
        """Fonction appelée en continu par sounddevice pendant l'enregistrement"""
        if status:
            print(f"Status audio: {status}")
        self.audio_data.append(indata.copy())

    def start_recording(self):
        # 1. Récupérer le choix de l'utilisateur
        selection = self.combo_mics.get()
        if not selection:
            self.label_status.configure(text="Erreur : Aucun micro sélectionné", text_color="red")
            return
            
        # 2. Extraire l'ID du micro (le chiffre avant les deux points)
        try:
            device_id = int(selection.split(":")[0])
            device_name = selection.split(":")[1]
        except:
            self.label_status.configure(text="Erreur de sélection micro", text_color="red")
            return

        # 3. Démarrer l'enregistrement
        try:
            self.audio_data = []
            # C'est ICI qu'on force le périphérique avec `device=device_id`
            self.stream = sd.InputStream(samplerate=self.fs, channels=1, 
                                         device=device_id, 
                                         callback=self.audio_callback)
            self.stream.start()
            
            self.is_recording = True
            self.start_time = time.time()
            
            # Mise à jour de l'interface
            self.label_status.configure(text=f"Enregistrement en cours sur :\n{device_name}", text_color="#00FF00")
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal", fg_color="green")
            self.combo_mics.configure(state="disabled")
            
        except Exception as e:
            self.label_status.configure(text=f"Impossible d'ouvrir le micro : {e}", text_color="red")

    def stop_recording(self):
        if not self.is_recording:
            return

        # Arrêt du stream
        self.stream.stop()
        self.stream.close()
        self.is_recording = False
        
        # Sauvegarde
        if self.audio_data:
            # Créer un dossier 'enregistrements' s'il n'existe pas
            if not os.path.exists("samples"):
                os.makedirs("samples")
                
            # Nom du fichier avec timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join("samples", f"rec_{timestamp}.wav")
            
            # Convertir la liste en array numpy et sauvegarder
            audio_array = np.concatenate(self.audio_data, axis=0)
            sf.write(filename, audio_array, self.fs)
            
            self.label_status.configure(text=f"Sauvegardé : {filename}", text_color="white")
        else:
            self.label_status.configure(text="Erreur : Audio vide", text_color="red")

        # Reset Interface
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled", fg_color="gray")
        self.combo_mics.configure(state="readonly")

if __name__ == "__main__":
    app = SimpleRecorder()
    app.mainloop()