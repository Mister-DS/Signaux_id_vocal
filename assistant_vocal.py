import customtkinter as ctk
import speech_recognition as sr
import pyttsx3
from youtube_search import YoutubeSearch
import webbrowser

# Initialiser la synthèse vocale et la reconnaissance vocale

recognizer = sr.Recognizer() 
engine = pyttsx3.init()

# Fonction pour parler (synthèse vocale)

def speak(text):
    engine.say(text)
    engine.runAndWait()
    
# Fonction pour parler et reconnaitre la voix

def recognize_speech():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Écoute...")
        
        
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio, language="fr-FR")
            print(f"Vous avez dit: {command}")
            return command.lower()
        except sr.UnknownValueError:
            print("Désolé, je n'ai pas compris.")
            speak("Désolé, je n'ai pas compris.")
            return ""
        except sr.RequestError:
            print("Erreur de service de reconnaissance vocale.")
            speak("Erreur de service de reconnaissance vocale.")
            return ""

# Fonction principale pour traiter les commandes vocales

def execute_commande():
    command = recognize_speech()
    if "joue" in command :
        song_name = command.replace("joue", "").strip()   
        speak(f"Recherche de {song_name} sur YouTube")
        
        #Recherche sur youtube 
        results = YoutubeSearch(song_name, max_results=1).to_dict() 
        if results:
            video_url = f"http://www.youtube.com/watch?v={results[0]['id']}"
            speak(f"Lecture de {results[0]['title']}")
            webbrowser.open(video_url)
        else: 
            speak("Désolé, je n'ai pas trouvé la chanson.")
            return
    elif "ouvre youtube" in command:
        speak("Ouverture de YouTube")
        webbrowser.open("https://www.youtube.com")
        
    elif "ferme" in command:
        speak("Fermeture de Youtube")
        app.quit()
    
    else:
        speak("Désoler je n'ai pas compris la commande.")  
        
# Configuration de l'interface utilisateur avec customtkinter

#Initialiser l'application 

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("Assistant Vocal")
app.geometry("500x400")

# Label des instructions

label = ctk.CTkLabel(app, text="Cliquez sur le bouton pour parler", font=("Arial", 16))
label.pack(pady=30)

# Bouton pour activer l'assistant vocal

ecoute_bouton = ctk.CTkButton(app, text="Écouter", command=execute_commande, font=("Arial", 14), height=50, width=200)
ecoute_bouton.pack(pady=20)

# Bouton pour quitter l'application

quit_bouton = ctk.CTkButton(app, text="Quitter", command=app.quit(), font=("Arial", 14), height=50, width=200, fg_color="red")
quit_bouton.pack(pady=20)
        
# Loop de l'application        
app.mainloop()