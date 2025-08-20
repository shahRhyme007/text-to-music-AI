from audiocraft.models import MusicGen, AudioGen, MAGNeT
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
import tempfile
from typing import Dict, List, Optional, Tuple, Any

# Import Phase 2 features (keeping them optional)
try:
    from phase2_features import MultiModalMusicOrchestrator, AdvancedAudioAnalyzer, CrossModalProcessor, PHASE2_CONFIG
    PHASE2_AVAILABLE = True
except ImportError:
    PHASE2_AVAILABLE = False
    st.warning("Phase 2 features not available. Using standard mode.")



# Ensure matplotlib does not open a window unnecessarily
plt.switch_backend('Agg')

# Enhanced model loading with multiple model support
@st.cache_resource
def load_model(model_name="facebook/musicgen-melody"):
    """Load different MusicGen models based on selection"""
    try:
        if 'musicgen' in model_name:
            model = MusicGen.get_pretrained(model_name)
        elif 'audiogen' in model_name and PHASE2_AVAILABLE:
            model = AudioGen.get_pretrained(model_name)
        else:
            model = MusicGen.get_pretrained(model_name)  # fallback
    return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Fallback to basic model
        return MusicGen.get_pretrained("facebook/musicgen-melody")

# Model configuration
MODEL_OPTIONS = {
    "MusicGen Melody (1.5B) - Recommended": "facebook/musicgen-melody",
    "MusicGen Small (300M) - Fast": "facebook/musicgen-small", 
    "MusicGen Medium (1.5B) - Balanced": "facebook/musicgen-medium",
    "MusicGen Large (3.3B) - Highest Quality": "facebook/musicgen-large"
}

# Phase 2: Advanced model configurations
if PHASE2_AVAILABLE:
    ADVANCED_MODEL_OPTIONS = {
        **MODEL_OPTIONS,
        "AudioGen Medium - Atmosphere": "facebook/audiogen-medium"
    }
else:
    ADVANCED_MODEL_OPTIONS = MODEL_OPTIONS



def generate_music_tensors(description, duration: int, my_bar, progress_text, model_name="facebook/musicgen-melody", **generation_params): 
    progress_text.text(f"◉ Loading {model_name.split('/')[-1]} engine...")
    model = load_model(model_name)
    my_bar.progress(30)  # Update progress after loading model
    
    progress_text.text("▣ Processing composition...")
    # Use professional parameters for generation
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




st.set_page_config(page_title="♪ AudioCraft Studio Pro",
                    page_icon="♪",
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
    progress_text.text("◦ Rendering audio output...")
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
    progress_text.text("♪ Composition complete.")
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
    st.title("♪ AudioCraft Studio Pro")
    with st.expander("About AudioCraft Studio"):
        st.write("Professional AI music generation platform powered by Meta's AudioCraft technology. Create up to 30 seconds of studio-quality music with precision controls and advanced composition tools.")
        if PHASE2_AVAILABLE:
            st.success("★ **Studio Pro Mode Available**: Multi-modal composition, ensemble orchestration, and professional audio analysis")
        else:
            st.info("⚡ **Enhanced**: Professional parameter controls for precision music generation")
    
    # Sidebar for professional controls
    with st.sidebar:
        st.header("⚙ Studio Controls")
        
        st.subheader("♫ Engine Selection")
        
        # Professional mode toggle
        if PHASE2_AVAILABLE:
            use_advanced_mode = st.checkbox("★ Enable Studio Pro Mode", 
                                           help="Unlock professional multi-modal composition and ensemble orchestration")
            
            if use_advanced_mode:
                st.success("★ **Studio Pro Active** - Professional features enabled")
                model_options = ADVANCED_MODEL_OPTIONS
            else:
                model_options = MODEL_OPTIONS
        else:
            use_advanced_mode = False
            model_options = MODEL_OPTIONS
        
        selected_model_display = st.selectbox(
            "Generation Engine:",
            options=list(model_options.keys()),
            index=0,  # Default to MusicGen Melody
            help="Select the AI engine optimized for your production needs"
        )
        selected_model = model_options[selected_model_display]
        
        # Display model info
        if "small" in selected_model:
            st.info("⚡ **Rapid**: 300M parameters - Fast production, professional quality")
        elif "medium" in selected_model:
            st.info("⚖ **Balanced**: 1.5B parameters - Optimal speed-quality ratio")
        elif "melody" in selected_model:
            st.success("♪ **Studio**: 1.5B parameters - Full composition with melody conditioning")
        elif "large" in selected_model:
            st.warning("★ **Mastering**: 3.3B parameters - Maximum fidelity, extended processing")
        elif "audiogen" in selected_model:
            st.info("∼ **Ambient**: Specialized for atmospheric and textural elements")
        
        st.subheader("◉ Generation Parameters")
        temperature = st.slider("∼ Creative Expression", 0.5, 2.0, 1.0, 0.1, 
                               help="Higher values increase musical creativity and variation")
        top_k = st.slider("※ Harmonic Diversity", 50, 500, 250, 25,
                         help="Controls the range of musical elements considered")
        top_p = st.slider("◦ Nucleus Sampling", 0.0, 1.0, 0.0, 0.05,
                         help="Fine-tune compositional probability (0 = disabled)")
        cfg_coef = st.slider("♫ Prompt Adherence", 1.0, 10.0, 3.0, 0.5,
                           help="How precisely the AI follows your musical description")
        use_sampling = st.checkbox("◉ Probabilistic Generation", value=True,
                                 help="Enable dynamic composition vs deterministic output")
        
        st.subheader("▣ Studio Presets")
        if st.button("∴ Experimental"):
            st.session_state.temp = 1.3
            st.session_state.top_k = 300
            st.session_state.cfg = 2.5
        if st.button("▣ Professional"):
            st.session_state.temp = 1.0
            st.session_state.top_k = 250
            st.session_state.cfg = 3.0
        if st.button("◦ Precision"):
            st.session_state.temp = 0.7
            st.session_state.top_k = 150
            st.session_state.cfg = 4.0
            
        # Studio Pro Features Section
        if PHASE2_AVAILABLE and use_advanced_mode:
            st.subheader("★ Studio Pro Features")
            
            # Multi-Modal Composition
            with st.expander("◈ Multi-Modal Composition", expanded=False):
                use_image_prompt = st.checkbox("▦ Visual-to-Musical Translation", 
                                             help="Convert visual concepts into musical compositions")
                if use_image_prompt:
                    image_description = st.text_input("Visual Description:", 
                                                    placeholder="e.g., 'A sunset over calm ocean waves'")
                
                # Advanced melody conditioning
                use_advanced_melody = st.checkbox("♪ Melody Analysis & Conditioning", 
                                                help="Professional analysis of uploaded musical phrases")
                if use_advanced_melody:
                    melody_file = st.file_uploader("Upload Reference Melody", 
                                                 type=['wav', 'mp3', 'flac'])
            
            # Ensemble Orchestration
            with st.expander("◉ Ensemble Orchestration", expanded=False):
                use_ensemble = st.checkbox("※ Multi-Engine Composition", 
                                         help="Layer multiple AI engines for complex arrangements")
                if use_ensemble:
                    ensemble_type = st.selectbox("Orchestration Mode:", 
                                               ["Composition + Atmosphere", "Multi-Layer Generation", "Single Engine"])
                    
                    if ensemble_type != "Single Engine":
                        mix_ratio = st.slider("▣ Layer Balance", 0.0, 1.0, 0.3, 0.1,
                                            help="Blend ratio for secondary orchestration layer")
            
            # Professional Analysis
            with st.expander("▣ Professional Analysis", expanded=False):
                use_advanced_analysis = st.checkbox("◦ Comprehensive Audio Metrics", 
                                                  help="Extract professional audio features and analytics")
                use_interactive_viz = st.checkbox("◈ Advanced Visualization Suite", 
                                                help="Multi-panel professional audio analysis dashboard")
        else:
            # Set defaults for Phase 2 variables when not in advanced mode
            use_image_prompt = False
            image_description = ""
            use_advanced_melody = False
            melody_file = None
            use_ensemble = False
            ensemble_type = "Single Model"
            mix_ratio = 0.3
            use_advanced_analysis = False
            use_interactive_viz = False
        
        # Display current settings
        st.subheader("▣ Session Configuration")
        st.text(f"♫ Engine: {selected_model.split('/')[-1]}")
        if PHASE2_AVAILABLE and use_advanced_mode:
            st.text(f"★ Mode: Studio Pro")
        else:
            st.text(f"◉ Mode: Standard")
        st.text(f"∼ Expression: {temperature}")
        st.text(f"※ Diversity: {top_k}")
        st.text(f"◦ Sampling: {top_p}")
        st.text(f"♫ Adherence: {cfg_coef}")
        st.text(f"◉ Generation: {'Dynamic' if use_sampling else 'Fixed'}")
    
    # Main composition interface
    # Studio Pro: Enhanced text input with multi-modal support
    if PHASE2_AVAILABLE and use_advanced_mode and use_image_prompt and image_description:
        # Convert image description to music prompt
        processor = CrossModalProcessor()
        suggested_prompt = processor.process_image_to_music_prompt(image_description)
        st.info(f"◈ **Visual-to-Musical Translation**: {suggested_prompt}")
        default_text = suggested_prompt
    else:
        default_text = ""
    
    text_area = st.text_area("♪ Composition Brief:", 
                            value=default_text,
                            placeholder="e.g., 'Ambient piano with ethereal strings, 60 BPM, contemplative atmosphere'",
                            help="Describe your musical vision with specific details about instrumentation, tempo, mood, and style")
    time_slider = st.slider("◉ Duration (seconds)", 2, 30, 10)
    
    # Studio Pro: Advanced melody analysis display
    if PHASE2_AVAILABLE and use_advanced_mode and use_advanced_melody and melody_file:
        with st.expander("♪ Melody Analysis Report", expanded=True):
            processor = CrossModalProcessor()
            melody_analysis = processor.analyze_melody_structure(melody_file)
            if melody_analysis and 'error' not in melody_analysis:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("◉ Tempo", f"{melody_analysis.get('tempo', 'N/A'):.1f} BPM")
                with col2:
                    st.metric("▣ Duration", f"{melody_analysis.get('duration', 'N/A'):.1f}s")
                with col3:
                    st.metric("∼ Complexity", f"{melody_analysis.get('rhythm_complexity', 0):.2f}")
            else:
                st.warning("Analysis unavailable. Please upload a compatible audio file.")

    if st.button('◉ Begin Composition'):
        if text_area and time_slider: 
            # Create composition summary
            data = {
                "Brief": [text_area[:50] + "..." if len(text_area) > 50 else text_area],
                "Duration": [f"{time_slider}s"],
                "Engine": [selected_model.split('/')[-1]],
                "Expression": [f"{temperature:.1f}"],
                "Adherence": [f"{cfg_coef:.1f}"]
            }
            
            # Studio Pro: Add advanced composition info
            if PHASE2_AVAILABLE and use_advanced_mode:
                data["Mode"] = ["Studio Pro"]
                if use_ensemble and ensemble_type != "Single Engine":
                    data["Orchestration"] = [ensemble_type]
                if use_image_prompt and image_description:
                    data["Visual Input"] = ["Active"]
            
            df = pd.DataFrame(data)
            st.table(df)

            st.subheader("♪ Audio Generation")
            progress_text = st.empty()  # Placeholder for progress text
            my_bar = st.progress(0)
            progress_text.text("◉ Initializing composition engine...")
            # Pass professional parameters to generation function
            generation_params = {
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'cfg_coef': cfg_coef,
                'use_sampling': use_sampling
            }
            # Studio Pro: Enhanced generation with ensemble support
            if PHASE2_AVAILABLE and use_advanced_mode and use_ensemble and ensemble_type != "Single Engine":
                # Use ensemble orchestration
                try:
                    progress_text.text("◉ Initializing multi-engine orchestration...")
                    orchestrator = MultiModalMusicOrchestrator()
                    
                    if ensemble_type == "Composition + Atmosphere":
                        # Load both engines
                        primary_model = load_model("facebook/musicgen-melody")
                        secondary_model = load_model("facebook/audiogen-medium")
                        
                        model_configs = {
                            'primary_model': primary_model,
                            'secondary_model': secondary_model,
                            'primary_params': generation_params,
                            'secondary_params': {**generation_params, 'cfg_coef': generation_params['cfg_coef'] * 0.7}
                        }
                        
                        my_bar.progress(40)
                        music_tensors, all_layers = orchestrator.create_layered_generation(
                            text_area, time_slider, model_configs)
                        
                        # Mix the layers professionally
                        mix_ratios = {'primary': 0.8, 'secondary': mix_ratio}
                        music_tensors = orchestrator.mix_audio_layers(all_layers, mix_ratios)
                        
                    else:
                        # Fallback to single engine
                        music_tensors = generate_music_tensors(text_area, time_slider, my_bar, progress_text, 
                                                             model_name=selected_model, **generation_params)
                except Exception as e:
                    st.warning(f"Orchestration failed: {e}. Reverting to single engine.")
                    music_tensors = generate_music_tensors(text_area, time_slider, my_bar, progress_text, 
                                                         model_name=selected_model, **generation_params)
            else:
                # Standard generation
                music_tensors = generate_music_tensors(text_area, time_slider, my_bar, progress_text, 
                                                     model_name=selected_model, **generation_params)
            print("Audio Tensors Generated: ", music_tensors.shape if hasattr(music_tensors, 'shape') else 'Complete')
            save_audio(music_tensors, my_bar, progress_text)
            audio_filepath = "audio_output/audio_0.wav"
            with open(audio_filepath, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            # Multiple audio playback options for better compatibility
            st.audio(audio_bytes, format='audio/wav')
            
            # HTML5 audio as backup
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
            <div style="margin: 10px 0;">
                <p><strong>Alternative Player:</strong></p>
                <audio controls style="width: 100%;">
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            
            st.download_button(
                label="📥 Download Composition",
                data=audio_bytes,
                file_name=f"audiocraft_composition_{int(time.time())}.wav",
                mime="audio/wav"
            )

                        # Studio Pro: Enhanced visualizations and analysis
            if PHASE2_AVAILABLE and use_advanced_mode:
                st.subheader("◦ Professional Audio Analysis")
                
                # Advanced audio features
                if use_advanced_analysis:
                    with st.expander("▣ Audio Metrics & Analytics", expanded=True):
                        analyzer = AdvancedAudioAnalyzer()
                        features = analyzer.extract_audio_features(audio_filepath)
                        
                        # Display professional metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("◉ Tempo", f"{features.get('tempo', 0):.1f} BPM")
                        with col2:
                            st.metric("∼ Spectral Center", f"{features.get('spectral_centroid', 0):.0f} Hz")
                        with col3:
                            st.metric("▣ RMS Level", f"{features.get('rms_energy', 0):.3f}")
                        with col4:
                            st.metric("◦ Transient Rate", f"{features.get('zero_crossing_rate', 0):.3f}")
                
                # Professional visualizations
                if use_interactive_viz:
                    st.subheader("◈ Professional Visualization Suite")
                    analyzer = AdvancedAudioAnalyzer()
                    viz_fig = analyzer.create_interactive_visualization(audio_filepath)
                    st.pyplot(viz_fig)
                else:
                    # Standard visualizations
                    st.subheader("▣ Audio Analysis")
                    display_waveform(audio_filepath)
                    display_spectrogram(audio_filepath)
            else:
                # Standard visualizations
                st.subheader("▣ Audio Analysis")
            display_waveform(audio_filepath)
            display_spectrogram(audio_filepath)



    # Reset the studio session
    if st.button('◦ Reset Session'):
        st.experimental_rerun()

if __name__ == "__main__": 
    main()

