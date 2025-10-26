import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Auto-download model if not present
MODEL_PATH = "malaria_cnn_model.h5"
DRIVE_FILE_ID = "1pP4UuHUFKAk_fbSd3DDoRAWpsUZ-fBAI"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="Malaria Detection App", page_icon="🦠", layout="centered")
st.title("🦠 Malaria Cell Detection")
st.write("Upload a blood cell image and the model will predict whether it is **Parasitized (Malaria)** or **Uninfected (Healthy)**.")

uploaded_file = st.file_uploader("Upload a blood cell image...", type=["jpg", "jpeg", "png"])

def predict_malaria(image):
    img = image.resize((128, 128))
    img = img.convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "🟢 Uninfected (Healthy)", float(prediction[0][0])
    else:
        return "🔴 Parasitized (Malaria)", float(prediction[0][0])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("⏳ Predicting…")
    label, confidence = predict_malaria(image)
    st.subheader("🔍 Prediction Result:")
    st.success(f"**{label}**")
    st.write(f"🧠 Model Confidence: `{confidence * 100:.2f}%`")
