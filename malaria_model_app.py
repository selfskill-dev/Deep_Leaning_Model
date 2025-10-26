import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("malaria_cnn_model.h5")

# Set Streamlit page configuration
st.set_page_config(page_title="Malaria Detection App", page_icon="🦠", layout="centered")

# App title
st.title("🦠 Malaria Cell Detection")
st.write("Upload a blood cell image, and the model will predict whether it is **Parasitized (Malaria)** or **Uninfected (Healthy)**.")

# File uploader
uploaded_file = st.file_uploader("Upload a blood cell image...", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_malaria(image):
    img = image.resize((128, 128))  # Resize same as model input
    img = img.convert("RGB")        # Ensure 3 color channels
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    
    # Adjust according to your model’s output layer
    if prediction[0][0] > 0.5:
        return "🟢 Uninfected (Healthy)", float(prediction[0][0])
    else:
        return "🔴 Parasitized (Malaria)", float(prediction[0][0])

# When user uploads an image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Predicting...")
    label, confidence = predict_malaria(image)

    st.subheader("🔍 Prediction Result:")
    st.success(f"**{label}**")
    st.write(f"🧠 Model Confidence: `{confidence*100:.2f}%`")

