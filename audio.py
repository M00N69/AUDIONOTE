import streamlit as st
import torch
from transformers import pipeline
import os

# Configuration du modèle Whisper
MODEL_NAME = "openai/whisper-large-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chargement du modèle de transcription
st.session_state["transcriber"] = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

# Fonction pour enregistrer le texte dans un fichier txt
def save_transcription(text, filename):
    with open(filename, "w") as f:
        f.write(text)
    st.success(f"Transcription enregistrée dans {filename}")

# Interface de l'application Streamlit
st.title("Application de prise de notes audio")

# Option d'enregistrement audio via microphone
audio_file = st.audio_input("Enregistrez votre note vocale", type="wav")

# Champ de texte pour choisir le nom du fichier de sortie
filename = st.text_input("Entrez le nom du fichier (sans l'extension):")

if audio_file is not None and filename:
    # Sauvegarder l'audio temporairement
    audio_path = os.path.join("temp_audio.wav")
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())
    
    # Transcription de l'audio
    st.write("Transcription en cours...")
    transcription_result = st.session_state["transcriber"](audio_path)["text"]
    st.text_area("Texte transcrit", transcription_result, height=200)

    # Option pour sauvegarder le texte
    if st.button("Enregistrer la transcription"):
        save_transcription(transcription_result, f"{filename}.txt")
