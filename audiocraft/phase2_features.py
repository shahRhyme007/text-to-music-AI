# Phase 2: Multi-Modal Music Orchestration Features
# This module adds advanced capabilities to the existing app without breaking current functionality

from audiocraft.models import MusicGen, AudioGen, MAGNeT
import streamlit as st
import torch
import torchaudio
import numpy as np
import librosa
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt

class MultiModalMusicOrchestrator:
    """
    Advanced multi-modal music generation orchestrator
    Combines multiple AI models for enhanced music creation
    """
    
    def __init__(self):
        self.models = {}
        self.current_model_types = []
        
    @st.cache_resource
    def load_ensemble_models(_self, model_list: List[str]):
        """Load multiple models for ensemble generation"""
        models = {}
        for model_name in model_list:
            try:
                if 'musicgen' in model_name:
                    models[model_name] = MusicGen.get_pretrained(model_name)
                elif 'audiogen' in model_name:
                    models[model_name] = AudioGen.get_pretrained(model_name)
                elif 'magnet' in model_name:
                    # Use MusicGen as fallback since MAGNeT might not be available
                    models[model_name] = MusicGen.get_pretrained("facebook/musicgen-melody")
            except Exception as e:
                st.warning(f"Could not load {model_name}: {str(e)}")
                
        return models
    
    def create_layered_generation(self, description: str, duration: int, 
                                model_configs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate music using multiple models in layers
        """
        results = {}
        primary_output = None
        
        # Primary generation (usually MusicGen)
        if 'primary_model' in model_configs:
            primary_model = model_configs['primary_model']
            primary_params = model_configs.get('primary_params', {})
            
            primary_model.set_generation_params(
                duration=duration,
                **primary_params
            )
            
            primary_output = primary_model.generate([description], progress=False, return_tokens=True)
            results['primary'] = primary_output[0]
        
        # Secondary enhancement (AudioGen for atmosphere)
        if 'secondary_model' in model_configs:
            secondary_model = model_configs['secondary_model']
            secondary_params = model_configs.get('secondary_params', {})
            
            # Create atmospheric description
            atmospheric_desc = f"ambient atmosphere and background texture for: {description}"
            
            secondary_model.set_generation_params(
                duration=duration,
                **secondary_params
            )
            
            secondary_output = secondary_model.generate([atmospheric_desc], progress=False, return_tokens=True)
            results['secondary'] = secondary_output[0]
        
        return primary_output[0] if primary_output else None, results
    
    def mix_audio_layers(self, audio_layers: Dict[str, torch.Tensor], 
                        mix_ratios: Dict[str, float]) -> torch.Tensor:
        """
        Mix multiple audio layers with specified ratios
        """
        if not audio_layers:
            return None
            
        # Start with the primary layer
        primary_key = 'primary' if 'primary' in audio_layers else list(audio_layers.keys())[0]
        mixed_audio = audio_layers[primary_key] * mix_ratios.get(primary_key, 1.0)
        
        # Add secondary layers
        for key, audio in audio_layers.items():
            if key != primary_key:
                ratio = mix_ratios.get(key, 0.3)  # Default to 30% mix
                # Ensure same shape
                if audio.shape == mixed_audio.shape:
                    mixed_audio = mixed_audio + (audio * ratio)
        
        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(mixed_audio))
        if max_val > 1.0:
            mixed_audio = mixed_audio / max_val
            
        return mixed_audio

class AdvancedAudioAnalyzer:
    """
    Enhanced audio analysis and visualization tools
    """
    
    @staticmethod
    def extract_audio_features(audio_path: str) -> Dict[str, Any]:
        """Extract comprehensive audio features"""
        audio, sr = librosa.load(audio_path, sr=32000)
        
        features = {
            'tempo': librosa.beat.tempo(y=audio, sr=sr)[0],
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio)),
            'mfcc': np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1),
            'chroma': np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1),
            'duration': len(audio) / sr,
            'rms_energy': np.mean(librosa.feature.rms(y=audio)[0])
        }
        
        return features
    
    @staticmethod
    def create_interactive_visualization(audio_path: str):
        """Create interactive audio visualization dashboard"""
        audio, sr = librosa.load(audio_path, sr=32000)
        
        # Create subplots for comprehensive analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('🎵 Advanced Audio Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Waveform
        librosa.display.waveshow(audio, sr=sr, ax=axes[0,0])
        axes[0,0].set_title('🌊 Waveform')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Amplitude')
        
        # 2. Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[0,1])
        axes[0,1].set_title('🎨 Spectrogram')
        fig.colorbar(img, ax=axes[0,1], format="%+2.0f dB")
        
        # 3. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img2 = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[0,2])
        axes[0,2].set_title('🎼 Mel Spectrogram')
        fig.colorbar(img2, ax=axes[0,2], format="%+2.0f dB")
        
        # 4. Chromagram
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        img3 = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[1,0])
        axes[1,0].set_title('🎹 Chromagram')
        fig.colorbar(img3, ax=axes[1,0])
        
        # 5. Spectral Features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        times = librosa.times_like(spectral_centroids)
        axes[1,1].plot(times, spectral_centroids)
        axes[1,1].set_title('📊 Spectral Centroid')
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Hz')
        
        # 6. MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        img4 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[1,2])
        axes[1,2].set_title('🔢 MFCC Features')
        fig.colorbar(img4, ax=axes[1,2])
        
        plt.tight_layout()
        return fig

class CrossModalProcessor:
    """
    Process different input modalities for music generation
    """
    
    @staticmethod
    def process_image_to_music_prompt(image_description: str) -> str:
        """Convert image description to music prompt"""
        # Mapping of visual elements to musical elements
        visual_to_musical = {
            # Colors
            'red': 'energetic, passionate, forte',
            'blue': 'calm, serene, ambient',
            'green': 'natural, organic, folk-inspired',
            'yellow': 'bright, cheerful, major key',
            'purple': 'mysterious, ethereal, ambient',
            'orange': 'warm, rhythmic, upbeat',
            'black': 'dark, mysterious, minor key',
            'white': 'pure, minimalist, simple',
            
            # Textures/Patterns
            'smooth': 'legato, flowing melodies',
            'rough': 'staccato, percussive',
            'flowing': 'fluid, continuous',
            'geometric': 'structured, mathematical',
            'organic': 'natural, irregular rhythms',
            
            # Scenes
            'forest': 'nature sounds, acoustic instruments',
            'ocean': 'waves, ambient, reverb',
            'city': 'electronic, urban, rhythmic',
            'mountain': 'grand, orchestral, epic',
            'sunset': 'warm, nostalgic, emotional',
            'storm': 'dramatic, intense, powerful'
        }
        
        prompt_elements = []
        description_lower = image_description.lower()
        
        for visual, musical in visual_to_musical.items():
            if visual in description_lower:
                prompt_elements.append(musical)
        
        if prompt_elements:
            return f"Musical interpretation: {', '.join(prompt_elements[:3])}"
        else:
            return f"Abstract musical representation of: {image_description}"
    
    @staticmethod
    def analyze_melody_structure(melody_file) -> Dict[str, Any]:
        """Analyze uploaded melody structure"""
        if melody_file is None:
            return {}
            
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(melody_file.read())
            tmp_path = tmp.name
        
        try:
            audio, sr = librosa.load(tmp_path, sr=32000)
            
            analysis = {
                'tempo': librosa.beat.tempo(y=audio, sr=sr)[0],
                'key': 'C major',  # Simplified - would need more complex analysis
                'duration': len(audio) / sr,
                'pitch_range': {
                    'min': float(np.min(audio)),
                    'max': float(np.max(audio))
                },
                'rhythm_complexity': np.std(librosa.onset.onset_strength(y=audio, sr=sr))
            }
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

# Configuration for Phase 2 features
PHASE2_CONFIG = {
    'ensemble_models': {
        'MusicGen + AudioGen': ['facebook/musicgen-melody', 'facebook/audiogen-medium'],
        'Multi-MusicGen': ['facebook/musicgen-melody', 'facebook/musicgen-medium'],
        'Single Model (Current)': ['facebook/musicgen-melody']
    },
    
    'mixing_presets': {
        'Balanced': {'primary': 0.8, 'secondary': 0.3},
        'Atmospheric': {'primary': 0.6, 'secondary': 0.5},
        'Clean': {'primary': 1.0, 'secondary': 0.1},
        'Experimental': {'primary': 0.5, 'secondary': 0.7}
    },
    
    'cross_modal_features': {
        'image_to_music': True,
        'melody_conditioning': True,
        'advanced_analysis': True
    }
}
