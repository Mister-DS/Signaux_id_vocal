import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
from enroll import EnrollmentManager
from validate import ValidationManager
from record_manager import RecordManager

GMM_THRESHOLD = 5.0
DTW_THRESHOLD = 0.25

# Palette de couleurs moderne
COLORS = {
    'primary': '#2C3E50',      # Bleu foncé
    'secondary': '#3498DB',    # Bleu clair
    'accent': '#E74C3C',       # Rouge
    'success': '#27AE60',      # Vert
    'warning': '#F39C12',      # Orange
    'bg_dark': '#34495E',      # Gris foncé
    'bg_light': '#ECF0F1',     # Gris très clair
    'text_light': '#FFFFFF',   # Blanc
    'text_dark': '#2C3E50',    # Texte sombre
}


class TextRedirector:
    """Redirige les sorties print() vers un widget texte"""
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')

    def flush(self):
        pass


class VoiceAuthGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Système d'Authentification Vocale")
        self.root.geometry("1000x700")
        self.root.configure(bg=COLORS['bg_light'])

        # Configuration du style personnalisé
        self.configure_styles()

        # Header avec gradient simulé
        self.create_header()

        # Frame principale
        main_frame = ttk.Frame(root, style='Main.TFrame')
        main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=10)

        # Configuration du grid
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Notebook pour les onglets
        self.notebook = ttk.Notebook(main_frame, style='Custom.TNotebook')
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Création des onglets
        self.create_enroll_tab()
        self.create_validate_tab()
        self.create_auth_tab()
        self.create_record_tab()

    def configure_styles(self):
        """Configure les styles personnalisés"""
        style = ttk.Style()
        style.theme_use('clam')

        # Style pour la frame principale
        style.configure('Main.TFrame', background=COLORS['bg_light'])

        # Style pour le notebook
        style.configure('Custom.TNotebook', background=COLORS['bg_light'], borderwidth=0)
        style.configure('Custom.TNotebook.Tab',
                       padding=[20, 10],
                       font=('Segoe UI', 10, 'bold'),
                       background=COLORS['bg_dark'])
        style.map('Custom.TNotebook.Tab',
                 background=[('selected', COLORS['secondary'])],
                 foreground=[('selected', COLORS['text_light']), ('!selected', COLORS['text_light'])])

        # Style pour les frames d'onglets
        style.configure('Tab.TFrame', background='white')

        # Style pour les labels
        style.configure('Title.TLabel',
                       font=('Segoe UI', 12, 'bold'),
                       background='white',
                       foreground=COLORS['text_dark'])
        style.configure('Desc.TLabel',
                       font=('Segoe UI', 10),
                       background='white',
                       foreground=COLORS['text_dark'])
        style.configure('Stats.TLabel',
                       font=('Segoe UI', 9),
                       background='white',
                       foreground=COLORS['secondary'])

        # Style pour les boutons
        style.configure('Action.TButton',
                       font=('Segoe UI', 11, 'bold'),
                       padding=[20, 10],
                       background=COLORS['secondary'],
                       foreground=COLORS['text_light'])
        style.map('Action.TButton',
                 background=[('active', COLORS['primary'])])

        # Style pour les LabelFrames
        style.configure('Custom.TLabelframe',
                       background='white',
                       borderwidth=2,
                       relief='groove')
        style.configure('Custom.TLabelframe.Label',
                       font=('Segoe UI', 10, 'bold'),
                       background='white',
                       foreground=COLORS['secondary'])

        # Style pour les radiobuttons
        style.configure('Custom.TRadiobutton',
                       font=('Segoe UI', 9),
                       background='white')

    def create_header(self):
        """Crée un header attractif"""
        header_frame = tk.Frame(self.root, bg=COLORS['primary'], height=100)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        header_frame.grid_propagate(False)

        # Titre principal
        title = tk.Label(header_frame,
                        text="Authentification Vocale",
                        font=('Segoe UI', 24, 'bold'),
                        bg=COLORS['primary'],
                        fg=COLORS['text_light'])
        title.pack(pady=15)

        # Sous-titre
        subtitle = tk.Label(header_frame,
                          text="Système de reconnaissance et validation d'identité par empreinte vocale",
                          font=('Segoe UI', 10),
                          bg=COLORS['primary'],
                          fg=COLORS['bg_light'])
        subtitle.pack()

    def create_enroll_tab(self):
        """Onglet pour l'enrôlement"""
        enroll_frame = ttk.Frame(self.notebook, padding="20", style='Tab.TFrame')
        self.notebook.add(enroll_frame, text="  Enrôlement  ")

        enroll_frame.columnconfigure(0, weight=1)
        enroll_frame.rowconfigure(2, weight=1)

        # Titre de section
        title = ttk.Label(enroll_frame, text="Entraînement des Modèles",
                         style='Title.TLabel')
        title.grid(row=0, column=0, pady=(0, 5))

        # Description avec icône
        desc = ttk.Label(enroll_frame,
                        text="Entraîner tous les modèles GMM et DTW sur la base des fichiers audio existants dans le répertoire d'enrôlement.",
                        style='Desc.TLabel',
                        wraplength=850)
        desc.grid(row=1, column=0, pady=(0, 20))

        # Bouton d'enrôlement avec style
        enroll_btn = ttk.Button(enroll_frame, text="Démarrer l'Enrôlement",
                               command=self.run_enrollment,
                               style='Action.TButton')
        enroll_btn.grid(row=2, column=0, pady=15)

        # Zone de texte pour les logs avec style
        log_label = ttk.Label(enroll_frame, text="Logs d'exécution :",
                             style='Desc.TLabel')
        log_label.grid(row=3, column=0, sticky=tk.W, pady=(10, 5))

        self.enroll_output = scrolledtext.ScrolledText(enroll_frame, height=20,
                                                       state='disabled', wrap=tk.WORD,
                                                       bg='#1E1E1E', fg='#D4D4D4',
                                                       font=('Consolas', 9),
                                                       relief='flat',
                                                       borderwidth=2)
        self.enroll_output.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

    def create_validate_tab(self):
        """Onglet pour la validation"""
        validate_frame = ttk.Frame(self.notebook, padding="20", style='Tab.TFrame')
        self.notebook.add(validate_frame, text="  Validation  ")

        validate_frame.columnconfigure(0, weight=1)
        validate_frame.rowconfigure(5, weight=1)

        # Titre de section
        title = ttk.Label(validate_frame, text="Validation des Modèles",
                         style='Title.TLabel')
        title.grid(row=0, column=0, pady=(0, 5))

        # Description
        desc = ttk.Label(validate_frame,
                        text="Lancer les tests de validation pour évaluer la précision des modèles entraînés (FAR, FRR, accuracy).",
                        style='Desc.TLabel',
                        wraplength=850)
        desc.grid(row=1, column=0, pady=(0, 20))

        # Frame pour les options avec style
        options_frame = ttk.LabelFrame(validate_frame, text="  Options  ",
                                      padding="15", style='Custom.TLabelframe')
        options_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(options_frame, text="Utilisateur cible (optionnel):",
                 style='Desc.TLabel').grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.validate_target_entry = ttk.Entry(options_frame, width=30, font=('Segoe UI', 10))
        self.validate_target_entry.grid(row=0, column=1, padx=10, pady=5)

        # Bouton de validation
        validate_btn = ttk.Button(validate_frame, text="Lancer la Validation",
                                 command=self.run_validation,
                                 style='Action.TButton')
        validate_btn.grid(row=3, column=0, pady=15)

        # Zone de texte pour les logs
        log_label = ttk.Label(validate_frame, text="Logs d'exécution :",
                             style='Desc.TLabel')
        log_label.grid(row=4, column=0, sticky=tk.W, pady=(10, 5))

        self.validate_output = scrolledtext.ScrolledText(validate_frame, height=20,
                                                        state='disabled', wrap=tk.WORD,
                                                        bg='#1E1E1E', fg='#D4D4D4',
                                                        font=('Consolas', 9),
                                                        relief='flat',
                                                        borderwidth=2)
        self.validate_output.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

    def create_auth_tab(self):
        """Onglet pour l'authentification en direct"""
        auth_frame = ttk.Frame(self.notebook, padding="20", style='Tab.TFrame')
        self.notebook.add(auth_frame, text="  Authentification  ")

        auth_frame.columnconfigure(0, weight=1)
        auth_frame.rowconfigure(6, weight=1)

        # Titre de section
        title = ttk.Label(auth_frame, text="Authentification en Temps Réel",
                         style='Title.TLabel')
        title.grid(row=0, column=0, pady=(0, 5))

        # Description
        desc = ttk.Label(auth_frame,
                        text="Enregistrer un échantillon vocal en direct et l'authentifier contre les modèles entraînés. Visualisation des scores GMM et DTW.",
                        style='Desc.TLabel',
                        wraplength=850)
        desc.grid(row=1, column=0, pady=(0, 20))

        # Frame pour les options
        options_frame = ttk.LabelFrame(auth_frame, text="  Options  ",
                                      padding="15", style='Custom.TLabelframe')
        options_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(options_frame, text="Utilisateur cible (optionnel):",
                 style='Desc.TLabel').grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.auth_target_entry = ttk.Entry(options_frame, width=30, font=('Segoe UI', 10))
        self.auth_target_entry.grid(row=0, column=1, padx=10, pady=5)

        # Frame pour les boutons de contrôle d'enregistrement
        control_frame = ttk.Frame(auth_frame)
        control_frame.grid(row=3, column=0, pady=15)

        self.auth_start_btn = ttk.Button(control_frame, text="Démarrer l'Enregistrement",
                                        command=self.start_auth_recording,
                                        style='Action.TButton')
        self.auth_start_btn.grid(row=0, column=0, padx=5)

        self.auth_stop_btn = ttk.Button(control_frame, text="Arrêter l'Enregistrement",
                                       command=self.stop_auth_recording,
                                       state='disabled')
        self.auth_stop_btn.grid(row=0, column=1, padx=5)

        # Label d'état
        self.auth_status_label = ttk.Label(auth_frame, text="Prêt à enregistrer",
                                          style='Stats.TLabel')
        self.auth_status_label.grid(row=4, column=0, pady=5)

        # Zone de texte pour les logs
        log_label = ttk.Label(auth_frame, text="Logs d'exécution :",
                             style='Desc.TLabel')
        log_label.grid(row=5, column=0, sticky=tk.W, pady=(10, 5))

        self.auth_output = scrolledtext.ScrolledText(auth_frame, height=18,
                                                     state='disabled', wrap=tk.WORD,
                                                     bg='#1E1E1E', fg='#D4D4D4',
                                                     font=('Consolas', 9),
                                                     relief='flat',
                                                     borderwidth=2)
        self.auth_output.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # Variables pour l'enregistrement
        self.auth_recording = False
        self.auth_recorder = None
        self.auth_frames = []
        self.auth_stream = None

    def create_record_tab(self):
        """Onglet pour l'enregistrement de samples"""
        record_frame = ttk.Frame(self.notebook, padding="20", style='Tab.TFrame')
        self.notebook.add(record_frame, text="  Enregistrement  ")

        record_frame.columnconfigure(0, weight=1)
        record_frame.rowconfigure(8, weight=1)

        # Titre de section
        title = ttk.Label(record_frame, text="Enregistrement de Samples Audio",
                         style='Title.TLabel')
        title.grid(row=0, column=0, pady=(0, 5))

        # Description
        desc = ttk.Label(record_frame,
                        text="Ajouter de nouveaux échantillons vocaux au dataset pour l'entraînement ou la validation des modèles.",
                        style='Desc.TLabel',
                        wraplength=850)
        desc.grid(row=1, column=0, pady=(0, 20))

        # Nom d'utilisateur
        user_frame = ttk.LabelFrame(record_frame, text="  Identité  ",
                                    padding="15", style='Custom.TLabelframe')
        user_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(user_frame, text="Nom de l'utilisateur:",
                 style='Desc.TLabel').grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.record_username_entry = ttk.Entry(user_frame, width=30, font=('Segoe UI', 10))
        self.record_username_entry.grid(row=0, column=1, padx=10, pady=5)

        # Type d'enregistrement
        type_frame = ttk.LabelFrame(record_frame, text="  Type d'enregistrement  ",
                                   padding="15", style='Custom.TLabelframe')
        type_frame.grid(row=3, column=0, pady=10, sticky=(tk.W, tk.E))

        self.record_type = tk.StringVar(value="enroll")

        ttk.Radiobutton(type_frame, text="Entraînement",
                       variable=self.record_type, value="enroll",
                       style='Custom.TRadiobutton').grid(row=0, column=0, sticky=tk.W, pady=5, padx=10)
        ttk.Radiobutton(type_frame, text="Validation (correct)",
                       variable=self.record_type, value="val_tp",
                       style='Custom.TRadiobutton').grid(row=1, column=0, sticky=tk.W, pady=5, padx=10)
        ttk.Radiobutton(type_frame, text="Validation (mauvaise phrase)",
                       variable=self.record_type, value="val_wp",
                       style='Custom.TRadiobutton').grid(row=2, column=0, sticky=tk.W, pady=5, padx=10)
        ttk.Radiobutton(type_frame, text="Imposteur",
                       variable=self.record_type, value="impostor",
                       style='Custom.TRadiobutton').grid(row=3, column=0, sticky=tk.W, pady=5, padx=10)

        # Nom de l'imposteur (si applicable)
        impostor_frame = ttk.LabelFrame(record_frame, text="  Imposteur  ",
                                       padding="15", style='Custom.TLabelframe')
        impostor_frame.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))

        ttk.Label(impostor_frame, text="Nom de l'imposteur (si imposteur):",
                 style='Desc.TLabel').grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
        self.impostor_name_entry = ttk.Entry(impostor_frame, width=30, font=('Segoe UI', 10))
        self.impostor_name_entry.grid(row=0, column=1, padx=10, pady=5)

        # Frame pour les boutons de contrôle d'enregistrement
        control_frame = ttk.Frame(record_frame)
        control_frame.grid(row=5, column=0, pady=15)

        self.record_start_btn = ttk.Button(control_frame, text="Démarrer l'Enregistrement",
                                          command=self.start_record_recording,
                                          style='Action.TButton')
        self.record_start_btn.grid(row=0, column=0, padx=5)

        self.record_stop_btn = ttk.Button(control_frame, text="Arrêter l'Enregistrement",
                                         command=self.stop_record_recording,
                                         state='disabled')
        self.record_stop_btn.grid(row=0, column=1, padx=5)

        # Label d'état
        self.record_status_label = ttk.Label(record_frame, text="Prêt à enregistrer",
                                            style='Stats.TLabel')
        self.record_status_label.grid(row=6, column=0, pady=5)

        # Statistiques avec style
        self.record_stats_label = ttk.Label(record_frame, text="",
                                           style='Stats.TLabel')
        self.record_stats_label.grid(row=7, column=0, pady=5)

        # Zone de texte pour les logs
        log_label = ttk.Label(record_frame, text="Logs d'exécution :",
                             style='Desc.TLabel')
        log_label.grid(row=8, column=0, sticky=tk.W, pady=(10, 5))

        self.record_output = scrolledtext.ScrolledText(record_frame, height=10,
                                                      state='disabled', wrap=tk.WORD,
                                                      bg='#1E1E1E', fg='#D4D4D4',
                                                      font=('Consolas', 9),
                                                      relief='flat',
                                                      borderwidth=2)
        self.record_output.grid(row=9, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # Variables pour l'enregistrement
        self.record_recording = False
        self.record_frames = []
        self.record_stream = None

    def run_enrollment(self):
        """Exécute l'enrôlement dans un thread séparé"""
        def task():
            # Rediriger stdout
            old_stdout = sys.stdout
            sys.stdout = TextRedirector(self.enroll_output)

            try:
                import os
                # S'assurer qu'on est dans le bon répertoire
                script_dir = os.path.dirname(os.path.abspath(__file__))
                samples_path = os.path.join(script_dir, "samples")

                manager = EnrollmentManager(samples_root=samples_path)
                manager.run_enrollment()
            except Exception as e:
                print(f"[ERREUR] {str(e)}")
            finally:
                sys.stdout = old_stdout

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def run_validation(self):
        """Exécute la validation dans un thread séparé"""
        def task():
            old_stdout = sys.stdout
            sys.stdout = TextRedirector(self.validate_output)

            try:
                import os
                # S'assurer qu'on est dans le bon répertoire
                script_dir = os.path.dirname(os.path.abspath(__file__))
                samples_path = os.path.join(script_dir, "samples")

                target = self.validate_target_entry.get().strip()
                target = target if target else None

                manager = ValidationManager(
                    samples_root=samples_path,
                    gmm_threshold=GMM_THRESHOLD,
                    dtw_threshold=DTW_THRESHOLD
                )
                manager.run_benchmark(target_user=target)
            except Exception as e:
                print(f"[ERREUR] {str(e)}")
            finally:
                sys.stdout = old_stdout

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def start_auth_recording(self):
        """Démarre l'enregistrement audio pour l'authentification"""
        import pyaudio

        self.auth_frames = []
        self.auth_recording = True

        # Désactiver le bouton start, activer le bouton stop
        self.auth_start_btn.config(state='disabled')
        self.auth_stop_btn.config(state='normal')
        self.auth_status_label.config(text="Enregistrement en cours...")

        def record_task():
            try:
                p = pyaudio.PyAudio()
                self.auth_stream = p.open(format=pyaudio.paInt16,
                                         channels=1,
                                         rate=44100,
                                         input=True,
                                         frames_per_buffer=1024)

                while self.auth_recording:
                    try:
                        data = self.auth_stream.read(1024, exception_on_overflow=False)
                        self.auth_frames.append(data)
                    except:
                        break

            except Exception as e:
                old_stdout = sys.stdout
                sys.stdout = TextRedirector(self.auth_output)
                print(f"[ERREUR] Impossible d'ouvrir le stream audio: {e}")
                sys.stdout = old_stdout
                self.auth_start_btn.config(state='normal')
                self.auth_stop_btn.config(state='disabled')
                self.auth_status_label.config(text="Erreur d'enregistrement")

        thread = threading.Thread(target=record_task, daemon=True)
        thread.start()

    def stop_auth_recording(self):
        """Arrête l'enregistrement et lance l'authentification"""
        import pyaudio
        import wave
        import tempfile
        import os
        import numpy as np
        import soundfile as sf

        self.auth_recording = False
        self.auth_status_label.config(text="Traitement de l'audio...")

        # Réactiver les boutons
        self.auth_start_btn.config(state='normal')
        self.auth_stop_btn.config(state='disabled')

        def process_task():
            old_stdout = sys.stdout
            sys.stdout = TextRedirector(self.auth_output)

            try:
                if self.auth_stream:
                    self.auth_stream.stop_stream()
                    self.auth_stream.close()

                # Sauvegarder l'audio temporaire
                tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_path = tf.name
                tf.close()

                p = pyaudio.PyAudio()
                wf = wave.open(temp_path, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(self.auth_frames))
                wf.close()
                p.terminate()

                print("[INFO] Nettoyage du signal audio...")

                # Nettoyage de l'audio
                data, fs = sf.read(temp_path)
                samples_to_cut = int(0.3 * fs)

                if len(data) > samples_to_cut:
                    data = data[samples_to_cut:]
                else:
                    print("[ERROR] Enregistrement trop court")
                    self.auth_status_label.config(text="Enregistrement trop court")
                    return

                max_val = np.max(np.abs(data))
                if max_val > 0:
                    data = data / max_val * 0.90

                sf.write(temp_path, data, fs, subtype='PCM_16')

                # Lancer l'authentification
                target = self.auth_target_entry.get().strip()
                target = target if target else None

                from engines import GMMVerifier, DTWVerifier
                from features import extract_features
                from visualize import visualize_analysis

                print("[INFO] Chargement des modèles...")
                gmm = GMMVerifier()
                dtw = DTWVerifier()
                gmm.load_models()
                dtw.load_models()

                print("[INFO] Analyse du signal...")
                id_speaker, gmm_score = gmm.verify(temp_path, safety_margin=GMM_THRESHOLD)

                print(f"[GMM] Identité détectée : {id_speaker} (Score: {gmm_score:.2f})")

                user_to_verify = id_speaker
                if target:
                    print(f"\n[DEBUG] Mode Forcé activé : Comparaison avec '{target}'")
                    user_to_verify = target
                    if id_speaker != target:
                        print(f"[DEBUG] GMM a échoué (pensait que c'était {id_speaker}), mais on force la suite.")

                dtw_dist = 0
                best_template_feats = None
                best_template_path = None

                if user_to_verify not in ["Unknown", "Error", "Error (No UBM)"]:
                    dtw_dist, best_template_feats, best_template_path = dtw.verify(user_to_verify, temp_path)

                    print(f"[DTW] Distance avec {user_to_verify}: {dtw_dist:.4f}")

                    if dtw_dist < DTW_THRESHOLD:
                        print(f"\n[SUCCESS] Bienvenue, {user_to_verify} !")
                        self.auth_status_label.config(text=f"Authentification réussie : {user_to_verify}")
                    else:
                        print("\n[FAILURE] Passphrase incorrecte.")
                        self.auth_status_label.config(text="Échec : Passphrase incorrecte")
                else:
                    print("[FAILURE] Identité inconnue et aucune cible forcée.")
                    self.auth_status_label.config(text="Échec : Identité inconnue")

                print("[INFO] Génération des graphiques...")
                live_feats = extract_features(temp_path)

                all_gmm_scores = {}
                if "Random" in gmm.models and live_feats is not None:
                    ubm_score = gmm.models["Random"].score(live_feats)
                    for name, model in gmm.models.items():
                        if name == "Random":
                            continue
                        all_gmm_scores[name] = model.score(live_feats) - ubm_score

                visualize_analysis(
                    live_path=temp_path,
                    template_path=best_template_path,
                    live_feats=live_feats,
                    template_feats=best_template_feats,
                    gmm_scores=all_gmm_scores,
                    gmm_threshold=GMM_THRESHOLD
                )

                # Nettoyer le fichier temporaire
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass

            except Exception as e:
                print(f"[ERREUR] {str(e)}")
                self.auth_status_label.config(text="Erreur lors de l'authentification")
            finally:
                sys.stdout = old_stdout

        thread = threading.Thread(target=process_task, daemon=True)
        thread.start()

    def start_record_recording(self):
        """Démarre l'enregistrement audio pour un sample"""
        username = self.record_username_entry.get().strip().lower()
        if not username:
            messagebox.showerror("Erreur", "Veuillez entrer un nom d'utilisateur")
            return

        record_type = self.record_type.get()

        if record_type == "impostor":
            impostor_name = self.impostor_name_entry.get().strip().lower()
            if not impostor_name:
                messagebox.showerror("Erreur", "Veuillez entrer le nom de l'imposteur")
                return

        import pyaudio

        self.record_frames = []
        self.record_recording = True

        # Désactiver le bouton start, activer le bouton stop
        self.record_start_btn.config(state='disabled')
        self.record_stop_btn.config(state='normal')
        self.record_status_label.config(text="Enregistrement en cours...")

        def record_task():
            try:
                p = pyaudio.PyAudio()
                self.record_stream = p.open(format=pyaudio.paInt16,
                                           channels=1,
                                           rate=44100,
                                           input=True,
                                           frames_per_buffer=1024)

                while self.record_recording:
                    try:
                        data = self.record_stream.read(1024, exception_on_overflow=False)
                        self.record_frames.append(data)
                    except:
                        break

            except Exception as e:
                old_stdout = sys.stdout
                sys.stdout = TextRedirector(self.record_output)
                print(f"[ERREUR] Impossible d'ouvrir le stream audio: {e}")
                sys.stdout = old_stdout
                self.record_start_btn.config(state='normal')
                self.record_stop_btn.config(state='disabled')
                self.record_status_label.config(text="Erreur d'enregistrement")

        thread = threading.Thread(target=record_task, daemon=True)
        thread.start()

    def stop_record_recording(self):
        """Arrête l'enregistrement et sauvegarde le sample"""
        import pyaudio
        import wave
        import os

        self.record_recording = False
        self.record_status_label.config(text="Traitement de l'audio...")

        # Réactiver les boutons
        self.record_start_btn.config(state='normal')
        self.record_stop_btn.config(state='disabled')

        def process_task():
            old_stdout = sys.stdout
            sys.stdout = TextRedirector(self.record_output)

            try:
                if self.record_stream:
                    self.record_stream.stop_stream()
                    self.record_stream.close()

                username = self.record_username_entry.get().strip().lower()
                record_type = self.record_type.get()
                impostor_name = self.impostor_name_entry.get().strip().lower() if record_type == "impostor" else None

                # S'assurer qu'on est dans le bon répertoire
                script_dir = os.path.dirname(os.path.abspath(__file__))
                samples_path = os.path.join(script_dir, "samples")

                manager = RecordManager(dataset_root=samples_path)

                # Déterminer le dossier et le préfixe
                if record_type == "enroll":
                    target_folder = os.path.join(manager.root, "enrollment", username)
                    prefix = username
                elif record_type == "val_tp":
                    target_folder = os.path.join(manager.root, "validation", username, "true_positive")
                    prefix = f"{username}_val"
                elif record_type == "val_wp":
                    target_folder = os.path.join(manager.root, "validation", username, "wrong_phrase")
                    prefix = f"{username}_wrongphrase"
                elif record_type == "impostor":
                    target_folder = os.path.join(manager.root, "validation", "_impostor")
                    prefix = f"{impostor_name}_attack_{username}"

                final_path = manager._get_next_filename(target_folder, prefix)
                filename = os.path.basename(final_path)

                print(f"\n[INFO] Sauvegarde de : {filename}")

                # Sauvegarder l'audio temporaire
                temp_path = "temp_rec.wav"
                p = pyaudio.PyAudio()
                wf = wave.open(temp_path, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(self.record_frames))
                wf.close()
                p.terminate()

                # Nettoyage et sauvegarde
                if manager.clean_and_save(temp_path, final_path):
                    print(f"[SUCCESS] Sauvegardé dans {target_folder}")
                    self.record_status_label.config(text=f"Sample enregistré : {filename}")

                    # Mise à jour des statistiques
                    self.update_record_stats(username)
                else:
                    print("[ERROR] Échec de la sauvegarde")
                    self.record_status_label.config(text="Échec de la sauvegarde")

                # Nettoyer le fichier temporaire
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                print(f"[ERREUR] {str(e)}")
                self.record_status_label.config(text="Erreur lors de l'enregistrement")
            finally:
                sys.stdout = old_stdout

        thread = threading.Thread(target=process_task, daemon=True)
        thread.start()

    def update_record_stats(self, username):
        """Met à jour les statistiques d'enregistrement"""
        try:
            import os
            # S'assurer qu'on est dans le bon répertoire
            script_dir = os.path.dirname(os.path.abspath(__file__))
            samples_path = os.path.join(script_dir, "samples")

            manager = RecordManager(dataset_root=samples_path)

            path_enroll = os.path.join(manager.root, "enrollment", username)
            path_valid_tp = os.path.join(manager.root, "validation", username, "true_positive")
            path_valid_wp = os.path.join(manager.root, "validation", username, "wrong_phrase")
            path_impostor = os.path.join(manager.root, "validation", "_impostor")

            c_enroll = manager._count_samples(path_enroll)
            c_tp = manager._count_samples(path_valid_tp)
            c_wp = manager._count_samples(path_valid_wp)
            c_imp = manager._count_samples(path_impostor)

            stats_text = f"Entraînement: {c_enroll} | Validation (correct): {c_tp} | Validation (phrase incorrecte): {c_wp} | Imposteur: {c_imp}"
            self.record_stats_label.config(text=stats_text)
        except:
            pass


def main():
    root = tk.Tk()
    app = VoiceAuthGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
