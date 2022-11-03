# -*- coding: utf-8 -*-
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


st.set_page_config(
    page_title="Urban-Sounds",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.cache()
def create_spectrogram(audio_file_path):
    librosa_audio_data, librosa_sample_rate = librosa.load(audio_file_path)
    librosa.display.waveshow(librosa_audio_data, sr=librosa_sample_rate)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    

    st.write(""" WAVE """)
    st.pyplot(plt)

    spec = librosa.feature.melspectrogram(librosa_audio_data)
    spec_conv = librosa.amplitude_to_db(spec, ref=np.max)
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    img = librosa.display.specshow(spec_conv,
                                   y_axis='mel',
                                   x_axis='time',
                                   sr=librosa_sample_rate,
                                   ax=ax)
    plt.close(fig)

    st.write(""" SPECTROGRAM """)
    st.pyplot(fig)


with st.sidebar:
    st.header("URBAN SOUNDS")

    with st.form("my-form"):
        file = st.file_uploader("FILE UPLOADER", type=".wav", accept_multiple_files=False, key='file_upload')
        col1, col2 = st.columns([2, 1])
        submitted_load = col1.form_submit_button("UPLOAD")

        if submitted_load and file is not None:
            st.success('Audio file uploaded!', icon="✅")
            audio_bytes = file.read()
            st.audio(audio_bytes, format='audio/wav')
            create_spectrogram(io.BytesIO(audio_bytes))

        elif submitted_load and file is None:
            st.info("Please upload a wav file.", icon="ℹ️")

##################################################

header = st.container()
dataset = st.container()

with header:
    st.title("Welcome to urban sounds look-in")

with dataset:
    st.markdown('**Dataset samples from URBANSOUND8K**')
    sound_data = pd.read_csv("UrbanSound8K.csv")
    st.write(sound_data.head())

    st.subheader('Count classID')
    classes_ = pd.DataFrame(sound_data["classID"].value_counts()).head(50)
    st.bar_chart(classes_)



