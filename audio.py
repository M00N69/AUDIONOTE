import streamlit as st
import torch
from transformers import pipeline
from st_audiorec import st_audiorec
import os

# Configuration du modèle Whisper
MODEL_NAME = "openai/whisper-large-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialisation du pipeline Whisper
if "transcriber" not in st.session_state:
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
st.title("Application de prise de notes audio avec enregistrement direct")

# Utilisation de `st_audiorec` pour enregistrer l'audio depuis le microphone
st.write("Cliquez sur le bouton ci-dessous pour commencer l'enregistrement.")
wav_audio_data = st_audiorec()

# Si un enregistrement est effectué
if wav_audio_data is not None:
    # Lecture de l'enregistrement audio
    st.audio(wav_audio_data, format="audio/wav")

    # Sauvegarder l'audio temporairement pour la transcription
    audio_path = "recorded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(wav_audio_data)

    # Transcription de l'audio
    st.write("Transcription en cours...")
    transcription_result = st.session_state["transcriber"](audio_path)["text"]
    st.text_area("Texte transcrit", transcription_result, height=200)

    # Champ de texte pour choisir le nom du fichier de sortie
    filename = st.text_input("Entrez le nom du fichier (sans l'extension):")

    # Option pour sauvegarder le texte
    if st.button("Enregistrer la transcription") and filename:
        save_transcription(transcription_result, f"{filename}.txt")

    # Option pour télécharger la transcription
    if filename:
        st.download_button(
            label="Télécharger la transcription",
            data=transcription_result,
            file_name=f"{filename}.txt",
            mime="text/plain"
        )
