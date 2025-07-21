import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf

# Load model
model = load_model("https://drive.google.com/file/d/17uhLI-MznEmz_mk9E2_Mfy4VqVJWFEez/view?usp=sharing")

# Define class names (update according to your dataset)
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))  # MobileNetV2 input size
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title("🧠 Brain Tumor Classifier")
st.write("Upload a brain MRI scan to identify the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded MRI Scan', use_column_width=True)

    # Preprocess and predict
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display results
    st.subheader("Prediction:")
    st.markdown(f"**Tumor Type:** `{predicted_class}`")
    st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")

    # Optional: Show confidence scores for all classes
    st.subheader("Confidence Scores:")
    for cls, score in zip(CLASS_NAMES, prediction):
        st.markdown(f"- {cls}: `{score*100:.2f}%`")
