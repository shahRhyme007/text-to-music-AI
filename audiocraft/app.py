from audiocraft.models import MusicGen
import streamlit as st
import os
import torch
import torchaudio
import numpy as np
import base64 # it is used to download the music that is generated


# function used to load the model(musicgen-melody is a powerful model)

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained("facebook/musicgen-melody")
    return model


def generate_music_tensors(description, duration: int): 
    print("Description: ", description)
    print("Duration: ", duration)
    model = load_model()
    model.set_generation_params(
        use_sampling=True, 
        top_k= 250, 
        duration= duration
    )

    # getting the tensors of the decoded audio samples
    output = model.generate(
        descriptions= [description], 
        progress= True, 
        return_tokens= True
    )
    return output[0]


st.set_page_config(page_title="Text to Music Generation",
                  page_icon=":musical_note:")


def save_audio(samples: torch.Tensor):
    sample_rate = 32000, 
    save_path = "audio_output/"

    assert samples.dim() == 2 or samples.dim() == 3
    samples  = samples.detach().cpu()

    if samples.dim() == 2: 
        samples = samples[None, ...]

        for idx , audio in  enumerate(samples): 
            audio_path = os.path.join(save_path, f"audio_{idx}.wav")
            torchaudio.save(audio_path, audio,  sample_rate)


def binFileDownload(bin_file, file_label = 'File'): 
    with open(bin_file, 'rb') as f : 
        data = f.read()

    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


def main(): 
    # streamlit functionality starting from here
    st.title("Text to Music Generation")

    with st.expander("See explanation"):
        st.write("This is a  music generation app built using Meta's Audiocraft Music Gen Model. Based onyour  natural laguage description , it can generate a maximum of 20 seconds music for you. ") 
    
    # creating the text box and the slider
    text_area = st.text_area("Enter your description so that we can create a music for you.....")
    time_slider = st.slider("Select time duration (in seconds)", 5, 30, 10)


    # after putting description the information will be taken 
    if text_area and time_slider: 
        st.json(
            {
                "Your description" : text_area, 
                "Selected Time (In Seconds)" : time_slider
            }
        ) 
        st.subheader("Generated Music")





if __name__ == "__main__": 
    main()