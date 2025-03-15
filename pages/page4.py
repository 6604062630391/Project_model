import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image
import os

model = tf.keras.models.load_model("best_model.h5")
class_labels = np.load("class_labels.npy")

st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")

st.title("🧬 Skin Cancer Image Classification")
st.write("Upload a skin lesion image and the model will predict the type of skin disease.")

uploaded_file = st.file_uploader("📷 Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    pred_class_index = np.argmax(prediction)
    pred_class_name = class_labels[pred_class_index]
    confidence = np.max(prediction) * 100

    st.success(f"🔍Predicted Class: **{pred_class_name}** ({confidence:.2f}% confidence)")

    st.subheader("Class Probability Distribution")
    fig, ax = plt.subplots()
    ax.pie(prediction[0], labels=class_labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

st.markdown("---")
st.markdown("Dataset: HAM10000")
