from audiocraft.models import MusicGen
import streamlit as st
import os
import torch
import torchaudio
import numpy as np
import base64  # it is used to download the music that is generated
import time 
import pandas as pd  
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Ensure matplotlib does not open a window unnecessarily
plt.switch_backend('Agg')

# Function used to load the model (musicgen-melody is a powerful model)
@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained("facebook/musicgen-melody")
    return model



def generate_music_tensors(description, duration: int, my_bar, progress_text): 
    progress_text.text("Loading model...")
    model = load_model()
    my_bar.progress(30)  # Update progress after loading model
    
    progress_text.text("Hold Tight...")
    model.set_generation_params(
        use_sampling=True, 
        top_k=250, 
        duration=duration
    )
    output = model.generate(
        descriptions=[description], 
        progress=True, 
        return_tokens=True
    )
    my_bar.progress(70)  # Update progress after generating tensors
    return output[0]




st.set_page_config(page_title="Text to Music Generation",
                    page_icon=":musical_note:")





def save_audio(samples: torch.Tensor, my_bar, progress_text):
    progress_text.text("Saving audio files...")
    sample_rate = 32000
    save_path = "audio_output/"


    if not os.path.exists(save_path):
        os.makedirs(save_path)


    assert samples.dim() == 2 or samples.dim() == 3
    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]  # Add batch dimension if it's missing
    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)
    my_bar.progress(100)
    progress_text.text("Music generation completed.")
    time.sleep(1)  # Pause to show completion message
    progress_text.empty()  # Clear the text
    my_bar.empty()  # Clear the progress bar

def binFileDownload(bin_file, file_label = 'File'): 
    with open(bin_file, 'rb') as f : 
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


# Audio visualization starts from here
def display_waveform(path, sr=32000):
    # Load the audio file
    audio, sr = librosa.load(path, sr=sr)
    # Plot waveform
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title('Waveform')
    st.pyplot(fig)



def display_spectrogram(path, sr=32000):
    # Load the audio file
    audio, sr = librosa.load(path, sr=sr)
    # Generate a Spectrogram
    X = librosa.stft(audio)
    Xdb = librosa.amplitude_to_db(abs(X))
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title('Spectrogram')
    st.pyplot(fig)



def main(): 
    st.title("Text to Music Generation")
    with st.expander("See explanation"):
        st.write("This is a music generation app built using Meta's Audiocraft Music Gen Model. Based on your natural language description, it can generate a maximum of 20 seconds of music for you.")
    
    text_area = st.text_area("Enter your description so that we can create music for you.....")
    time_slider = st.slider("Select time duration (in seconds)", 2, 5, 10)

    if st.button('Start Generating'):
        if text_area and time_slider: 
            # Create a dataframe for display
            data = {
                "Description": [text_area],
                "Selected Time (In Seconds)": [time_slider]
            }
            df = pd.DataFrame(data)
            st.table(df)

            st.subheader("Generated Music")
            progress_text = st.empty()  # Placeholder for progress text
            my_bar = st.progress(0)
            progress_text.text("Preparing to generate music...")
            music_tensors = generate_music_tensors(text_area, time_slider, my_bar, progress_text)
            print("Music Tensors : ", music_tensors)
            save_audio(music_tensors, my_bar, progress_text)
            audio_filepath = "audio_output/audio_0.wav"
            with open(audio_filepath, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            st.markdown(binFileDownload(audio_filepath, 'Audio'), unsafe_allow_html=True)

            # Display audio visualizations
            display_waveform(audio_filepath)
            display_spectrogram(audio_filepath)



    # Reset the app or clear results
    if st.button('Reset'):
        st.experimental_rerun()

if __name__ == "__main__": 
    main()

