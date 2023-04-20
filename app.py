import streamlit as st
import pyttsx3
from PIL import Image
from caption_generate import runModel

st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

def caption_generate(uploaded_file):
    return runModel(uploaded_file)

def text_to_speech(text, rate=150):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)
    if st.button("Generate and Read caption"):
        generated_caption = caption_generate(uploaded_file)
        st.text(generated_caption)
        text_to_speech(generated_caption)



# global generated_caption 

# generated_caption = ""



# if st.button("Convert to speech"):
#     generated_caption = caption_generate(uploaded_file)
#     text_to_speech(generated_caption)





