import customtkinter as ctk
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class VoiceAuthApp:
    def __init__(self):
        #Var enregi 
        self.fs = 16000
        self.audio_data = []
        self.stream = None 
        self.is_recording = False
        self.audio_array = None
        
        dossier_script = os.path.dirname(os.path.abspath(__file__))
        self.dossier_enregistrements = os.path.join(dossier_script, "enregistrements")
        
        if not os.path.exists(self.dossier_enregistrements):
            os.makedirs(self.dossier_enregistrements)
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.app = ctk.CTk()
        self.app.title("Reconnaissance vocale MVP")
        self.app.geometry("600x700")
        
        
        # Label titre
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
        
        # Bouton Valider
        self.btn_valider = ctk.CTkButton(
            self.app,
            text="Valider et enregistrer",
            command=self.valider_enregistrement,
            font=("Arial", 14),
            height=50,
            width=250,
            fg_color="blue",
            state="disabled"
        )
        self.btn_valider.pack(pady=10)
        
        # Label de status
        self.label_status = ctk.CTkLabel(
            self.app,
            text="Prêt à enregistrer",
            font=("Arial", 12),
            text_color="gray"
        )
        self.label_status.pack(pady=10)
        
        
        self.separator = ctk.CTkLabel(
            self.app,
            text="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            font=("Arial", 10),
            text_color="gray"
        )
        self.separator.pack(pady=10)
        
        # Label pour la section visualisation
        self.label_visualisation = ctk.CTkLabel(
            self.app,
            text="Visualisation des enregistrements",
            font=("Arial", 14, "bold")
        )
        self.label_visualisation.pack(pady=10)
        
        # Liste déroulante des fichiers
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
        
        # Bouton pour voir le signal
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
        
        # Bouton Quitter
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
    
    def charger_liste_fichiers(self):
        """
        Charge la liste des fichiers .wav disponibles
        Retourne une liste de noms sans l'extension .wav
        """
        try:
            fichiers = os.listdir(self.dossier_enregistrements)
            fichiers_wav = [f.replace(".wav", "") for f in fichiers if f.endswith(".wav")]
            return fichiers_wav if fichiers_wav else ["Aucun fichier"]
        except Exception as e:
            print(f"Erreur lors du chargement des fichiers: {e}")
            return ["Erreur de chargement"]
    
    def mettre_a_jour_liste(self):
        """
        Met à jour la liste déroulante avec les fichiers actuels
        """
        fichiers = self.charger_liste_fichiers()
        self.dropdown.configure(values=fichiers)
        if fichiers and fichiers[0] != "Aucun fichier":
            self.dropdown.set(fichiers[-1])
    
    def audio_callback(self, indata, frames, time, status):
        """Fonction appelée automatiquement pour capturer l'audio"""
        if status:
            print(f"Erreur audio: {status}")
        
        self.audio_data.append(indata.copy())
    
    def start_recording(self):
        """Démarrer l'enregistrement audio"""
        self.is_recording = True
        self.audio_data = []
        self.audio_array = None
        
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
        self.btn_valider.configure(state="disabled")
        self.entry_nom.delete(0, 'end')
        
        print("Enregistrement démarré...")
    
    def stop_recording(self):
        """Arrêter l'enregistrement (mais ne sauvegarde PAS encore)"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        self.stream.stop()
        self.stream.close()
        
        # Convertir en array numpy et stocker
        self.audio_array = np.concatenate(self.audio_data, axis=0)
        
        self.label_status.configure(
            text="Enregistrement arrêté. Entrez votre nom et validez.",
            text_color="orange"
        )
        self.btn_stop.configure(state="disabled")
        self.btn_valider.configure(state="normal")
        
        print("Enregistrement arrêté. Validation...")
    
    def valider_enregistrement(self):
        """Sauvegarder l'enregistrement avec le nom donné"""
        if self.audio_array is None:
            self.label_status.configure(
                text="Erreur: Pas d'audio à sauvegarder !",
                text_color="red"
            )
            return
        
        nom = self.entry_nom.get().strip()
        
        if not nom:
            self.label_status.configure(
                text="Erreur: Veuillez entrer un nom !",
                text_color="red"
            )
            return
        
        numero = self.trouver_prochain_numero(nom)
        
        filename = os.path.join(self.dossier_enregistrements, f"{nom}_{numero}.wav")
        
        sf.write(filename, self.audio_array, self.fs)
        
        self.label_status.configure(
            text=f"Enregistré: {nom}_{numero}.wav",
            text_color="green"
        )
        self.btn_start.configure(state="normal")
        self.btn_valider.configure(state="disabled")
        
        print(f"Enregistrement sauvegardé: {filename}")
        
        # Met à jour la liste déroulante des fichiers
        self.mettre_a_jour_liste()
        
        self.audio_array = None
    
    def trouver_prochain_numero(self, nom_personne):
        """Trouve le prochain numéro d'enregistrement pour une personne"""
        fichiers = os.listdir(self.dossier_enregistrements)
        
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
    
    def afficher_signal(self):
        """
        Affiche uniquement la forme d'onde du fichier sélectionné
        """
        # Récupérer le fichier sélectionné
        fichier_selectionne = self.dropdown.get()
        
        if fichier_selectionne == "Sélectionnez un fichier" or fichier_selectionne == "Aucun fichier":
            self.label_status.configure(
                text="Veuillez sélectionner un fichier !",
                text_color="red"
            )
            return
        
        chemin_complet = os.path.join(self.dossier_enregistrements, fichier_selectionne + ".wav")
        
        if not os.path.exists(chemin_complet):
            self.label_status.configure(
                text=f"Fichier introuvable: {fichier_selectionne}.wav",
                text_color="red"
            )
            return
        
        try:
            # Lire le fichier WAV
            data, samplerate = sf.read(chemin_complet)
            
            # Calculer la durée en secondes
            duree = len(data) / samplerate
            temps = np.linspace(0, duree, len(data))
            
            # Créer la figure avec un seul graphique
            plt.figure(figsize=(12, 4))
            plt.plot(temps, data, color='blue', linewidth=0.8)
            plt.xlabel('Temps (secondes)', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.title(f'Signaux {fichier_selectionne}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xlim([0, duree])
            plt.tight_layout()
            plt.show()
            
            self.label_status.configure(
                text=f"Signal affiché: {fichier_selectionne}",
                text_color="green"
            )
            
            print(f"Signal affiché pour: {fichier_selectionne}")
            
        except Exception as e:
            self.label_status.configure(
                text=f"Erreur lors de l'affichage: {str(e)}",
                text_color="red"
            )
            print(f"Erreur: {e}")
    
    def run(self):
        """Lancer l'application"""
        self.app.mainloop()

if __name__ == "__main__":
    application = VoiceAuthApp()
    application.run()