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


st.set_page_config(page_title="Text to Music Generation",
                  page_icon=":musical_note:")



def main(): 
    # streamlit functionality starting from here
    st.title("Text to Music Generation")

    with st.expander("See explanation"):
        st.write("This is a  music generation app built using Meta's Audiocraft Music Gen Model. Based onyour  natural laguage description , it can generate a maximum of 20 seconds music for you. ") 
    
    text_area = st.area("Enter your description so that we can create a music for you.....")
    time_slider = st.slider("Select time duration (in seconds)", 5, 10, 30)


    if text_area and time_slider: 
        st.json(
            {
                "Your description" : text_area, 
                "Selected Time (In Seconds)" : time_slider
            }
        ) 


if __name__ == "__main__": 
    main()