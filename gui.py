import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import sys
import re
import os
import numpy as np
import soundfile as sf

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from scipy import signal

from enroll import EnrollmentManager
from validate import ValidationManager
from record_manager import RecordManager

GMM_THRESHOLD = 5.0
DTW_THRESHOLD = 0.25

# Palette de couleurs
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#3498DB',
    'accent': '#E74C3C',
    'success': '#27AE60',
    'warning': '#F39C12',
    'bg_dark': '#34495E',
    'bg_light': '#ECF0F1',
    'text_light': '#FFFFFF',
    'text_dark': '#2C3E50'
}

class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')

    def flush(self):
        pass

class StatsRedirector(TextRedirector):
    def __init__(self, widget, callback_update_stats):
        super().__init__(widget)
        self.callback = callback_update_stats
        self.total = 0
        self.correct = 0
        self.far = 0
        self.frr = 0

    def write(self, text):
        super().write(text)
        if "[SUCCESS]" in text:
            self.total += 1
            self.correct += 1
            self._update_gui()
        elif "[FAILURE]" in text:
            self.total += 1
            self._update_gui()

        if "Expected Auth: True" in text and "Got: False" in text:
            self.frr += 1
            self._update_gui()
        elif "Expected Auth: False" in text and "Got: True" in text:
            self.far += 1
            self._update_gui()

    def _update_gui(self):
        if self.total > 0:
            acc = (self.correct / self.total) * 100
            self.callback("acc", f"{acc:.2f}")
            self.callback("far", str(self.far))
            self.callback("frr", str(self.frr))
            self.callback("count", str(self.total))

class VoiceAuthGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Système d'Authentification Vocale - Analyse Complète")
        # Fenêtre agrandie pour accommoder les 8 graphiques
        self.root.geometry("1400x950") 
        self.root.configure(bg=COLORS['bg_light'])

        self.configure_styles()
        self.create_header()

        main_frame = ttk.Frame(root, style='Main.TFrame')
        main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=10)

        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(main_frame, style='Custom.TNotebook')
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.create_enroll_tab()
        self.create_validate_tab()
        self.create_auth_tab()
        self.create_record_tab()

    def configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Main.TFrame', background=COLORS['bg_light'])
        style.configure('Custom.TNotebook', background=COLORS['bg_light'], borderwidth=0)
        style.configure('Custom.TNotebook.Tab', padding=[20, 10], font=('Segoe UI', 10, 'bold'), background=COLORS['bg_dark'])
        style.map('Custom.TNotebook.Tab', background=[('selected', COLORS['secondary'])], foreground=[('selected', COLORS['text_light']), ('!selected', COLORS['text_light'])])
        style.configure('Tab.TFrame', background='white')
        style.configure('Title.TLabel', font=('Segoe UI', 12, 'bold'), background='white', foreground=COLORS['text_dark'])
        style.configure('Desc.TLabel', font=('Segoe UI', 10), background='white', foreground=COLORS['text_dark'])
        style.configure('Stats.TLabel', font=('Segoe UI', 9), background='white', foreground=COLORS['secondary'])
        style.configure('BigStat.TLabel', font=('Segoe UI', 22, 'bold'), background='white', foreground=COLORS['primary'])
        style.configure('StatLabel.TLabel', font=('Segoe UI', 9, 'bold'), background='white', foreground='#7F8C8D')
        style.configure('Action.TButton', font=('Segoe UI', 11, 'bold'), padding=[20, 10], background=COLORS['secondary'], foreground=COLORS['text_light'])
        style.map('Action.TButton', background=[('active', COLORS['primary'])])
        style.configure('Custom.TLabelframe', background='white', borderwidth=2, relief='groove')
        style.configure('Custom.TLabelframe.Label', font=('Segoe UI', 10, 'bold'), background='white', foreground=COLORS['secondary'])
        style.configure('Custom.TRadiobutton', font=('Segoe UI', 9), background='white')

    def create_header(self):
        header_frame = tk.Frame(self.root, bg=COLORS['primary'], height=60)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        header_frame.grid_propagate(False)
        title_frame = tk.Frame(header_frame, bg=COLORS['primary'])
        title_frame.pack(expand=True)
        tk.Label(title_frame, text="Authentification Vocale", font=('Segoe UI', 18, 'bold'), bg=COLORS['primary'], fg=COLORS['text_light']).pack(pady=(5, 0))

    # --- 1. Enrôlement ---
    def create_enroll_tab(self):
        enroll_frame = ttk.Frame(self.notebook, padding="20", style='Tab.TFrame')
        self.notebook.add(enroll_frame, text="  Enrôlement  ")
        enroll_frame.columnconfigure(0, weight=1)
        enroll_frame.rowconfigure(4, weight=1)

        ttk.Label(enroll_frame, text="Entraînement des Modèles", style='Title.TLabel').grid(row=0, column=0, sticky=tk.W)
        ttk.Label(enroll_frame, text="Génère les modèles GMM et les templates DTW.", style='Desc.TLabel').grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        enroll_btn = ttk.Button(enroll_frame, text="Démarrer l'Enrôlement", command=self.run_enrollment, style='Action.TButton')
        enroll_btn.grid(row=2, column=0, pady=10)
        
        ttk.Label(enroll_frame, text="Logs :", style='Desc.TLabel').grid(row=3, column=0, sticky=tk.W)
        self.enroll_output = scrolledtext.ScrolledText(enroll_frame, height=15, state='disabled', bg='#1E1E1E', fg='#D4D4D4', font=('Consolas', 9))
        self.enroll_output.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # --- 2. Validation ---
    def create_validate_tab(self):
        validate_frame = ttk.Frame(self.notebook, padding="20", style='Tab.TFrame')
        self.notebook.add(validate_frame, text="  Validation  ")
        validate_frame.columnconfigure(0, weight=1)
        validate_frame.rowconfigure(6, weight=1)

        stats_container = ttk.LabelFrame(validate_frame, text=" Résultats en Temps Réel ", padding=10, style='Custom.TLabelframe')
        stats_container.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        for i in range(4): stats_container.columnconfigure(i, weight=1)

        f0 = ttk.Frame(stats_container, style='Tab.TFrame'); f0.grid(row=0, column=0)
        self.lbl_total_val = ttk.Label(f0, text="0", style='BigStat.TLabel', foreground=COLORS['text_dark']); self.lbl_total_val.pack()
        ttk.Label(f0, text="Tests Effectués", style='StatLabel.TLabel').pack()

        f1 = ttk.Frame(stats_container, style='Tab.TFrame'); f1.grid(row=0, column=1)
        self.lbl_acc_val = ttk.Label(f1, text="-- %", style='BigStat.TLabel', foreground=COLORS['success']); self.lbl_acc_val.pack()
        ttk.Label(f1, text="Précision", style='StatLabel.TLabel').pack()

        f2 = ttk.Frame(stats_container, style='Tab.TFrame'); f2.grid(row=0, column=2)
        self.lbl_far_val = ttk.Label(f2, text="0", style='BigStat.TLabel', foreground=COLORS['accent']); self.lbl_far_val.pack()
        ttk.Label(f2, text="Intrusions (FAR)", style='StatLabel.TLabel').pack()

        f3 = ttk.Frame(stats_container, style='Tab.TFrame'); f3.grid(row=0, column=3)
        self.lbl_frr_val = ttk.Label(f3, text="0", style='BigStat.TLabel', foreground=COLORS['warning']); self.lbl_frr_val.pack()
        ttk.Label(f3, text="Rejets Abusifs (FRR)", style='StatLabel.TLabel').pack()

        options_frame = ttk.Frame(validate_frame, style='Tab.TFrame')
        options_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(options_frame, text="Utilisateur cible:", style='Desc.TLabel').pack(side=tk.LEFT)
        self.validate_target_entry = ttk.Entry(options_frame, width=20)
        self.validate_target_entry.pack(side=tk.LEFT, padx=10)

        validate_btn = ttk.Button(validate_frame, text="Lancer la Validation", command=self.run_validation, style='Action.TButton')
        validate_btn.grid(row=2, column=0, pady=10)

        ttk.Label(validate_frame, text="Logs détaillés :", style='Desc.TLabel').grid(row=5, column=0, sticky=tk.W)
        self.validate_output = scrolledtext.ScrolledText(validate_frame, height=10, state='disabled', bg='#1E1E1E', fg='#D4D4D4', font=('Consolas', 9))
        self.validate_output.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def update_validation_stats(self, key, value):
        def _update():
            if key == "acc": self.lbl_acc_val.config(text=f"{value}%")
            elif key == "far": self.lbl_far_val.config(text=str(value))
            elif key == "frr": self.lbl_frr_val.config(text=str(value))
            elif key == "count": self.lbl_total_val.config(text=str(value))
        self.root.after(0, _update)

    # --- 3. Authentification (Avec Graphiques COMPLETS) ---
    def create_auth_tab(self):
        auth_frame = ttk.Frame(self.notebook, padding="5", style='Tab.TFrame')
        self.notebook.add(auth_frame, text="  Authentification  ")
        
        auth_frame.columnconfigure(0, weight=1)
        # On donne la priorité absolue à l'espace pour les graphiques (Row 2)
        auth_frame.rowconfigure(2, weight=10) 
        auth_frame.rowconfigure(3, weight=1)

        # Contrôles
        control_frame = ttk.LabelFrame(auth_frame, text=" Contrôles ", padding=5, style='Custom.TLabelframe')
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(control_frame, text="Mode Debug (Cible):", style='Desc.TLabel').pack(side=tk.LEFT)
        self.auth_target_entry = ttk.Entry(control_frame, width=15)
        self.auth_target_entry.pack(side=tk.LEFT, padx=10)

        self.auth_stop_btn = ttk.Button(control_frame, text="Arrêter", command=self.stop_auth_recording, state='disabled')
        self.auth_stop_btn.pack(side=tk.RIGHT, padx=5)
        self.auth_start_btn = ttk.Button(control_frame, text="Microphone ON", command=self.start_auth_recording, style='Action.TButton')
        self.auth_start_btn.pack(side=tk.RIGHT, padx=5)

        # Status
        self.auth_status_label = ttk.Label(auth_frame, text="Prêt pour l'analyse", font=('Segoe UI', 14, 'bold'), background='white', foreground=COLORS['primary'])
        self.auth_status_label.grid(row=1, column=0, pady=2)

        # Graphiques (Canvas)
        self.plot_frame = tk.Frame(auth_frame, bg='white')
        self.plot_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=2)
        
        # Figure agrandie pour 8 subplots
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.fig.patch.set_facecolor('#FFFFFF')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Placeholder
        self.ax_placeholder = self.fig.add_subplot(111)
        self.ax_placeholder.text(0.5, 0.5, "Graphiques d'analyse (Spectrogrammes, MFCC...)", ha='center', color='gray')
        self.ax_placeholder.axis('off')

        # Logs (Plus petit pour laisser la place aux graphiques)
        self.auth_output = scrolledtext.ScrolledText(auth_frame, height=6, state='disabled', bg='#1E1E1E', fg='#D4D4D4', font=('Consolas', 10))
        self.auth_output.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.auth_recording = False
        self.auth_frames = []
        self.auth_stream = None

    def update_auth_graphs(self, live_path, template_path, live_feats, template_feats, gmm_scores, gmm_threshold):
        def _draw():
            self.fig.clear()

            ax1 = self.fig.add_subplot(3, 2, 1)
            sorted_scores = sorted(gmm_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            names = [x[0] for x in sorted_scores]
            margins = [x[1] for x in sorted_scores]
            colors = ['#27AE60' if m > gmm_threshold else '#E74C3C' for m in margins]
            ax1.barh(names, margins, color=colors)
            ax1.axvline(gmm_threshold, color='blue', linestyle='--', label='Seuil')
            ax1.set_title("Identification GMM", fontsize=9, fontweight='bold')
            ax1.tick_params(labelsize=8)

            ax2 = self.fig.add_subplot(3, 2, 2)
            if template_feats is not None and live_feats is not None:
                distance, path = fastdtw(template_feats, live_feats, dist=cosine, radius=5)
                path = np.array(path)
                ax2.plot(path[:, 1], path[:, 0], 'cyan', linewidth=1.5, alpha=0.7)
                ax2.set_facecolor('#1a1a1a')
                ax2.set_title(f"DTW Path (Dist: {distance/len(path):.4f})", fontsize=9, fontweight='bold')
                ax2.set_xlabel("Live", fontsize=8)
                ax2.set_ylabel("Template", fontsize=8)
            else:
                ax2.text(0.5, 0.5, "DTW N/A", ha='center', fontsize=10, color='gray')
                ax2.axis('off')

            y_live, sr_live = None, 16000
            y_temp, sr_temp = None, 16000

            if live_path and os.path.exists(live_path):
                y_live, sr_live = sf.read(live_path)
                if len(y_live) > sr_live * 3:
                    y_live = y_live[:sr_live * 3]

            if template_path and os.path.exists(template_path):
                y_temp, sr_temp = sf.read(template_path)
                if len(y_temp) > sr_temp * 3:
                    y_temp = y_temp[:sr_temp * 3]

            ax3 = self.fig.add_subplot(3, 2, 3)
            if y_live is not None:
                f, t, Sxx = signal.spectrogram(y_live, sr_live, nperseg=256, noverlap=128)
                ax3.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
                ax3.set_ylabel('Freq (Hz)', fontsize=8)
                ax3.set_xlabel('Temps (s)', fontsize=8)
                ax3.set_title("Spectrogramme Live", fontsize=9, fontweight='bold')
                ax3.set_ylim([0, 4000])
            else:
                ax3.text(0.5, 0.5, "N/A", ha='center', fontsize=10, color='gray')
                ax3.axis('off')

            ax4 = self.fig.add_subplot(3, 2, 4)
            if y_temp is not None:
                f, t, Sxx = signal.spectrogram(y_temp, sr_temp, nperseg=256, noverlap=128)
                ax4.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
                ax4.set_ylabel('Freq (Hz)', fontsize=8)
                ax4.set_xlabel('Temps (s)', fontsize=8)
                ax4.set_title("Spectrogramme Template", fontsize=9, fontweight='bold')
                ax4.set_ylim([0, 4000])
            else:
                ax4.text(0.5, 0.5, "N/A", ha='center', fontsize=10, color='gray')
                ax4.axis('off')

            ax5 = self.fig.add_subplot(3, 2, 5)
            if live_feats is not None:
                ax5.imshow(live_feats.T, aspect='auto', origin='lower', cmap='plasma', interpolation='nearest')
                ax5.set_title("MFCC Live", fontsize=9, fontweight='bold')
                ax5.set_xlabel("Temps", fontsize=8)
                ax5.set_ylabel("Coeff", fontsize=8)

            ax6 = self.fig.add_subplot(3, 2, 6)
            if template_feats is not None:
                ax6.imshow(template_feats.T, aspect='auto', origin='lower', cmap='plasma', interpolation='nearest')
                ax6.set_title("MFCC Template", fontsize=9, fontweight='bold')
                ax6.set_xlabel("Temps", fontsize=8)
                ax6.set_ylabel("Coeff", fontsize=8)

            self.fig.tight_layout(pad=0.8)
            self.canvas.draw_idle()

        self.root.after(0, _draw)

    # --- 4. Enregistrement (Samples) ---
    def create_record_tab(self):
        record_frame = ttk.Frame(self.notebook, padding="20", style='Tab.TFrame')
        self.notebook.add(record_frame, text="  Enregistrement  ")
        record_frame.columnconfigure(0, weight=1)
        record_frame.rowconfigure(8, weight=1)

        title = ttk.Label(record_frame, text="Enregistrement de Samples", style='Title.TLabel')
        title.grid(row=0, column=0, pady=(0, 5))

        user_frame = ttk.LabelFrame(record_frame, text=" Identité ", padding="15", style='Custom.TLabelframe')
        user_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))
        ttk.Label(user_frame, text="Utilisateur:", style='Desc.TLabel').grid(row=0, column=0, padx=10)
        self.record_username_entry = ttk.Entry(user_frame, width=30)
        self.record_username_entry.grid(row=0, column=1, padx=10)

        type_frame = ttk.LabelFrame(record_frame, text=" Type ", padding="15", style='Custom.TLabelframe')
        type_frame.grid(row=3, column=0, pady=10, sticky=(tk.W, tk.E))
        self.record_type = tk.StringVar(value="enroll")
        opts = [("Entraînement", "enroll"), ("Validation (OK)", "val_tp"), ("Validation (Mauvais)", "val_wp"), ("Imposteur", "impostor")]
        for i, (txt, val) in enumerate(opts):
            ttk.Radiobutton(type_frame, text=txt, variable=self.record_type, value=val, style='Custom.TRadiobutton').grid(row=i//2, column=i%2, sticky=tk.W, padx=20, pady=5)

        impostor_frame = ttk.Frame(record_frame)
        impostor_frame.grid(row=4, column=0, pady=5)
        ttk.Label(impostor_frame, text="Nom imposteur:", style='Desc.TLabel').pack(side=tk.LEFT)
        self.impostor_name_entry = ttk.Entry(impostor_frame, width=20)
        self.impostor_name_entry.pack(side=tk.LEFT, padx=10)

        control_frame = ttk.Frame(record_frame)
        control_frame.grid(row=5, column=0, pady=15)
        self.record_start_btn = ttk.Button(control_frame, text="REC", command=self.start_record_recording, style='Action.TButton')
        self.record_start_btn.pack(side=tk.LEFT, padx=5)
        self.record_stop_btn = ttk.Button(control_frame, text="STOP", command=self.stop_record_recording, state='disabled')
        self.record_stop_btn.pack(side=tk.LEFT, padx=5)

        self.record_status_label = ttk.Label(record_frame, text="Prêt", style='Stats.TLabel')
        self.record_status_label.grid(row=6, column=0, pady=5)
        self.record_stats_label = ttk.Label(record_frame, text="", style='Stats.TLabel')
        self.record_stats_label.grid(row=7, column=0, pady=5)

        self.record_output = scrolledtext.ScrolledText(record_frame, height=8, state='disabled', bg='#1E1E1E', fg='#D4D4D4', font=('Consolas', 9))
        self.record_output.grid(row=9, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.record_recording = False
        self.record_frames = []
        self.record_stream = None

    # --- Logique ---
    def run_enrollment(self):
        def task():
            old_stdout = sys.stdout
            sys.stdout = TextRedirector(self.enroll_output)
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                samples_path = os.path.join(script_dir, "samples")
                manager = EnrollmentManager(samples_root=samples_path)
                manager.run_enrollment()
            except Exception as e: print(f"[ERREUR] {str(e)}")
            finally: sys.stdout = old_stdout
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def run_validation(self):
        self.lbl_total_val.config(text="0")
        self.lbl_acc_val.config(text="-- %")
        self.lbl_far_val.config(text="0")
        self.lbl_frr_val.config(text="0")
        def task():
            old_stdout = sys.stdout
            sys.stdout = StatsRedirector(self.validate_output, self.update_validation_stats)
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                samples_path = os.path.join(script_dir, "samples")
                target = self.validate_target_entry.get().strip() or None
                manager = ValidationManager(samples_root=samples_path, gmm_threshold=GMM_THRESHOLD, dtw_threshold=DTW_THRESHOLD)
                manager.run_benchmark(target_user=target)
            except Exception as e: print(f"[ERREUR] {str(e)}")
            finally: sys.stdout = old_stdout
        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def start_auth_recording(self):
        import pyaudio
        self.auth_frames = []
        self.auth_recording = True
        self.auth_start_btn.config(state='disabled')
        self.auth_stop_btn.config(state='normal')
        self.auth_status_label.config(text="Enregistrement en cours...", foreground=COLORS['accent'])
        self.fig.clear()
        ax = self.fig.add_subplot(111); ax.text(0.5, 0.5, "Acquisition du signal...", ha='center'); ax.axis('off')
        self.canvas.draw()
        def record_task():
            try:
                p = pyaudio.PyAudio()
                self.auth_stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
                while self.auth_recording:
                    try:
                        data = self.auth_stream.read(1024, exception_on_overflow=False)
                        self.auth_frames.append(data)
                    except: break
            except: pass
        thread = threading.Thread(target=record_task, daemon=True)
        thread.start()

    def stop_auth_recording(self):
        import pyaudio, wave, tempfile, soundfile as sf
        from engines import GMMVerifier, DTWVerifier
        from features import extract_features
        
        self.auth_recording = False
        self.auth_status_label.config(text="Analyse et Génération des graphiques...", foreground=COLORS['secondary'])
        self.auth_start_btn.config(state='normal')
        self.auth_stop_btn.config(state='disabled')

        def process_task():
            old_stdout = sys.stdout
            sys.stdout = TextRedirector(self.auth_output)
            temp_path = ""
            try:
                if self.auth_stream:
                    self.auth_stream.stop_stream()
                    self.auth_stream.close()
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

                data, fs = sf.read(temp_path)
                samples_to_cut = int(0.3 * fs)
                if len(data) > samples_to_cut: data = data[samples_to_cut:]
                max_val = np.max(np.abs(data))
                if max_val > 0: data = data / max_val * 0.90
                sf.write(temp_path, data, fs, subtype='PCM_16')

                target = self.auth_target_entry.get().strip() or None
                gmm = GMMVerifier()
                dtw = DTWVerifier()
                gmm.load_models()
                dtw.load_models()

                id_speaker, gmm_score = gmm.verify(temp_path, safety_margin=GMM_THRESHOLD)
                print(f"[GMM] {id_speaker} (Score: {gmm_score:.2f})")
                user_to_verify = id_speaker
                if target: user_to_verify = target

                dtw_dist = 0
                best_template_feats = None
                best_template_path = None

                def update_status(success, name):
                    if success: self.auth_status_label.config(text=f"ACCÈS AUTORISÉ : {name}", foreground=COLORS['success'])
                    else: self.auth_status_label.config(text=f"ACCÈS REFUSÉ ({name})", foreground=COLORS['accent'])

                if user_to_verify not in ["Unknown", "Error", "Error (Modèle lambda non trouvé)"]:
                    dtw_dist, best_template_feats, best_template_path = dtw.verify(user_to_verify, temp_path)
                    print(f"[DTW] Dist: {dtw_dist:.4f} (Seuil: {DTW_THRESHOLD})")
                    if dtw_dist < DTW_THRESHOLD:
                        print(f"[SUCCESS] Bienvenue {user_to_verify}")
                        self.root.after(0, update_status, True, user_to_verify)
                    else:
                        print(f"[FAILURE] Rejet DTW ({dtw_dist:.4f} >= {DTW_THRESHOLD})")
                        self.root.after(0, update_status, False, user_to_verify)
                else:
                    self.root.after(0, lambda: self.auth_status_label.config(text="INCONNU", foreground=COLORS['accent']))

                live_feats = extract_features(temp_path)
                all_gmm_scores = {}
                if "Random" in gmm.models and live_feats is not None:
                    ubm_score = gmm.models["Random"].score(live_feats)
                    for name, model in gmm.models.items():
                        if name == "Random": continue
                        all_gmm_scores[name] = model.score(live_feats) - ubm_score

                self.update_auth_graphs(temp_path, best_template_path, live_feats, best_template_feats, all_gmm_scores, GMM_THRESHOLD)

                import time
                time.sleep(0.5)

                if os.path.exists(temp_path): os.remove(temp_path)

            except Exception as e:
                print(f"[ERREUR] {str(e)}")
                self.root.after(0, lambda: self.auth_status_label.config(text="Erreur Système"))
            finally: sys.stdout = old_stdout
        thread = threading.Thread(target=process_task, daemon=True)
        thread.start()

    def start_record_recording(self):
        username = self.record_username_entry.get().strip().lower()
        if not username:
            messagebox.showerror("Erreur", "Nom d'utilisateur requis")
            return
        import pyaudio
        self.record_frames = []
        self.record_recording = True
        self.record_start_btn.config(state='disabled')
        self.record_stop_btn.config(state='normal')
        self.record_status_label.config(text="Enregistrement...", foreground=COLORS['accent'])
        def record_task():
            try:
                p = pyaudio.PyAudio()
                self.record_stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
                while self.record_recording:
                    try:
                        data = self.record_stream.read(1024, exception_on_overflow=False)
                        self.record_frames.append(data)
                    except: break
            except: pass
        thread = threading.Thread(target=record_task, daemon=True)
        thread.start()

    def stop_record_recording(self):
        import pyaudio, wave, os
        self.record_recording = False
        self.record_status_label.config(text="Sauvegarde...", foreground=COLORS['secondary'])
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
                impostor = self.impostor_name_entry.get().strip().lower() if record_type == 'impostor' else None
                script_dir = os.path.dirname(os.path.abspath(__file__))
                manager = RecordManager(dataset_root=os.path.join(script_dir, "samples"))
                if record_type == "enroll": target = os.path.join(manager.root, "enrollment", username); prefix = username
                elif record_type == "val_tp": target = os.path.join(manager.root, "validation", username, "true_positive"); prefix = f"{username}_val"
                elif record_type == "val_wp": target = os.path.join(manager.root, "validation", username, "wrong_phrase"); prefix = f"{username}_wrongphrase"
                elif record_type == "impostor": target = os.path.join(manager.root, "validation", "_impostor"); prefix = f"{impostor}_attack_{username}"
                final_path = manager._get_next_filename(target, prefix)
                temp_path = "temp_rec.wav"
                p = pyaudio.PyAudio()
                wf = wave.open(temp_path, 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(self.record_frames))
                wf.close()
                p.terminate()
                if manager.clean_and_save(temp_path, final_path):
                    print(f"[OK] {os.path.basename(final_path)}")
                    self.root.after(0, lambda: self.record_status_label.config(text="Sauvegardé !", foreground=COLORS['success']))
                else: print("[ERR] Sauvegarde échouée")
                if os.path.exists(temp_path): os.remove(temp_path)
            except Exception as e: print(e)
            finally: sys.stdout = old_stdout
        thread = threading.Thread(target=process_task, daemon=True)
        thread.start()

def main():
    root = tk.Tk()
    app = VoiceAuthGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()