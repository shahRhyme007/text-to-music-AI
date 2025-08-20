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
import base64



# Ensure matplotlib does not open a window unnecessarily
plt.switch_backend('Agg')

# Enhanced model loading with multiple model support
@st.cache_resource
def load_model(model_name="facebook/musicgen-melody"):
    """Load different MusicGen models based on selection"""
    model = MusicGen.get_pretrained(model_name)
    return model

# Model configuration
MODEL_OPTIONS = {
    "MusicGen Melody (1.5B) - Recommended": "facebook/musicgen-melody",
    "MusicGen Small (300M) - Fast": "facebook/musicgen-small", 
    "MusicGen Medium (1.5B) - Balanced": "facebook/musicgen-medium",
    "MusicGen Large (3.3B) - Highest Quality": "facebook/musicgen-large"
}



def generate_music_tensors(description, duration: int, my_bar, progress_text, model_name="facebook/musicgen-melody", **generation_params): 
    progress_text.text(f"Loading {model_name.split('/')[-1]} model...")
    model = load_model(model_name)
    my_bar.progress(30)  # Update progress after loading model
    
    progress_text.text("Hold Tight...")
    # Use advanced parameters if provided, otherwise use defaults
    model.set_generation_params(
        use_sampling=generation_params.get('use_sampling', True),
        top_k=generation_params.get('top_k', 250),
        top_p=generation_params.get('top_p', 0.0),
        temperature=generation_params.get('temperature', 1.0),
        cfg_coef=generation_params.get('cfg_coef', 3.0),
        duration=duration
    )
    output = model.generate(
        descriptions=[description], 
        progress=True, 
        return_tokens=True
    )
    my_bar.progress(70)  # Update progress after generating tensors
    return output[0]




st.set_page_config(page_title="🎵 Advanced Text to Music Generation",
                    page_icon="🎼",
                    layout="wide",
                    initial_sidebar_state="expanded")


page_bg_img = """
<style>
body {
background-image: url('https://images.unsplash.com/photo-1542281286-9e0a16bb7366');
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)



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
    st.title("🎵 Advanced Text to Music Generation")
    with st.expander("See explanation"):
        st.write("This is an enhanced music generation app built using Meta's Audiocraft MusicGen Model. Generate up to 30 seconds of high-quality music with advanced parameter controls!")
        st.info("💡 **NEW**: Advanced parameter controls in the sidebar for fine-tuning your music generation!")
    
    # Sidebar for advanced controls
    with st.sidebar:
        st.header("🎛️ Advanced Controls")
        
        st.subheader("🤖 AI Model Selection")
        selected_model_display = st.selectbox(
            "Choose AI Model:",
            options=list(MODEL_OPTIONS.keys()),
            index=0,  # Default to MusicGen Melody
            help="Different models offer different trade-offs between speed, quality, and features"
        )
        selected_model = MODEL_OPTIONS[selected_model_display]
        
        # Display model info
        if "small" in selected_model:
            st.info("🚀 **Fast**: 300M parameters - Quick generation, good quality")
        elif "medium" in selected_model:
            st.info("⚖️ **Balanced**: 1.5B parameters - Good balance of speed and quality")
        elif "melody" in selected_model:
            st.success("🎵 **Recommended**: 1.5B parameters - Best for text-to-music + melody conditioning")
        elif "large" in selected_model:
            st.warning("🔥 **Premium**: 3.3B parameters - Highest quality but slower generation")
        
        st.subheader("🎯 Generation Parameters")
        temperature = st.slider("🌡️ Creativity (Temperature)", 0.5, 2.0, 1.0, 0.1, 
                               help="Higher = more creative/random, Lower = more focused")
        top_k = st.slider("🎪 Diversity (Top-K)", 50, 500, 250, 25,
                         help="Number of top tokens to consider")
        top_p = st.slider("🎲 Nucleus Sampling (Top-P)", 0.0, 1.0, 0.0, 0.05,
                         help="Cumulative probability threshold (0 = disabled)")
        cfg_coef = st.slider("🎼 Prompt Following (CFG)", 1.0, 10.0, 3.0, 0.5,
                           help="How closely to follow the text prompt")
        use_sampling = st.checkbox("🎯 Use Sampling", value=True,
                                 help="Enable probabilistic sampling vs deterministic")
        
        st.subheader("📊 Presets")
        if st.button("🎨 Creative"):
            st.session_state.temp = 1.3
            st.session_state.top_k = 300
            st.session_state.cfg = 2.5
        if st.button("⚖️ Balanced"):
            st.session_state.temp = 1.0
            st.session_state.top_k = 250
            st.session_state.cfg = 3.0
        if st.button("🎯 Focused"):
            st.session_state.temp = 0.7
            st.session_state.top_k = 150
            st.session_state.cfg = 4.0
            
        # Display current settings
        st.subheader("📊 Current Settings")
        st.text(f"🤖 Model: {selected_model.split('/')[-1]}")
        st.text(f"🌡️ Temperature: {temperature}")
        st.text(f"🎪 Top-K: {top_k}")
        st.text(f"🎲 Top-P: {top_p}")
        st.text(f"🎼 CFG: {cfg_coef}")
        st.text(f"🎯 Sampling: {'On' if use_sampling else 'Off'}")
    
    # Main interface
    text_area = st.text_area("🎼 Describe your music:", 
                            placeholder="e.g., 'A relaxing piano melody with soft strings, slow tempo, melancholic mood'",
                            help="Be specific about genre, instruments, mood, and tempo for better results")
    time_slider = st.slider("⏱️ Duration (seconds)", 2, 30, 10)

    if st.button('Start Generating'):
        if text_area and time_slider: 
            # Create a dataframe for display with enhanced info
            data = {
                "Description": [text_area],
                "Duration (Seconds)": [time_slider],
                "AI Model": [selected_model.split('/')[-1]],
                "Temperature": [temperature],
                "CFG Coefficient": [cfg_coef]
            }
            df = pd.DataFrame(data)
            st.table(df)

            st.subheader("Generated Music")
            progress_text = st.empty()  # Placeholder for progress text
            my_bar = st.progress(0)
            progress_text.text("Preparing to generate music...")
            # Pass advanced parameters to generation function
            generation_params = {
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'cfg_coef': cfg_coef,
                'use_sampling': use_sampling
            }
            music_tensors = generate_music_tensors(text_area, time_slider, my_bar, progress_text, 
                                                  model_name=selected_model, **generation_params)
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

