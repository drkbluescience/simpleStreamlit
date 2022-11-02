# -*- coding: utf-8 -*-

"""
created by enise
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

st.set_page_config(
    page_title="Urban-Sounds DL",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

class_names = ['air conditioner', 'car horn', 'children playing', 'dog bark', 'drilling', 'engine idling', 'gun shot',
               'jackhammer', 'siren', 'street music']

st.session_state["action"] = 0
st.session_state["audio"] = None
st.session_state["wave"] = None
st.session_state["spectrogram"] = None
st.session_state["spect_array"] = None

# F #f


st.cache()


def create_spectrogram(audio_file_path):
    librosa_audio_data, librosa_sample_rate = librosa.load(audio_file_path)
    librosa.display.waveshow(librosa_audio_data, sr=librosa_sample_rate)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    st.session_state['wave'] = plt

    # st.pyplot(plt)

    st.write(""" WAVE """)
    st.pyplot(st.session_state['wave'])

    spec = librosa.feature.melspectrogram(librosa_audio_data)
    spec_conv = librosa.amplitude_to_db(spec, ref=np.max)
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    img = librosa.display.specshow(spec_conv,
                                   y_axis='mel',
                                   x_axis='time',
                                   sr=librosa_sample_rate,
                                   ax=ax)
    fig.savefig("cache/spect", bbox_inches='tight', pad_inches=0)  # save the figure
    plt.close(fig)
    # st.pyplot(fig)
    st.session_state['spectrogram'] = fig

    st.write(""" SPECTROGRAM """)
    st.pyplot(st.session_state['spectrogram'])

    # st.session_state['spect_array'] = np.array(spec_conv)


with st.sidebar:
    st.header("Prediction")
    st.write("Try us to predict an urban sound by loading an audio file with the Wav extension.")

    with st.form("my-form"):
        file = st.file_uploader("FILE UPLOADER", type=".wav", accept_multiple_files=False, key='file_upload')
        col1, col2 = st.columns([2, 1])
        submitted_load = col1.form_submit_button("UPLOAD")

        if submitted_load and file is not None:
            st.success('Audio file uploaded!', icon="✅")
            st.session_state["action"] = 1


        elif submitted_load and file is None:
            st.info("Please upload a wav file.", icon="ℹ️")

        if st.session_state['action'] == 1:
            audio_bytes = file.read()
            aud = st.audio(audio_bytes, format='audio/wav')
            create_spectrogram(io.BytesIO(audio_bytes))

##################################################

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()
audio_file = st.container()

with header:
    st.title("Welcome to urban sounds look-in")
    # st.text("Try us to predict an urban sound by loading an audio file with the Wav extension.")

with dataset:
    st.header("Urban dataset")
    st.markdown('* **Dataset contains urban sounds**')
    st.text("I found this dataset")

    sound_data = pd.read_csv("UrbanSound8K.csv")
    st.write(sound_data.head())

    st.subheader('Count classID')
    pickedID = pd.DataFrame(sound_data["classID"].value_counts()).head(50)
    st.bar_chart(pickedID)



