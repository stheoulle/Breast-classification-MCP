import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import h5py
import tempfile
import os

# Class mapping
class_mapping = {
    0: 'Benign',
    1: 'Malignant',
    2: 'Normal',
}

# Function to load the combined model
@st.cache_resource
def load_model():
    model_path = "model.h5"  # path to your model file
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# Function to preprocess and make predictions
def predict(image, model):
    # Preprocess the image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (256, 256))  # Adjust the size as per your model requirements
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)

    # Get the predicted class
    predicted_class = class_mapping[np.argmax(predictions[0])]
    return predicted_class

# Streamlit app
st.title('Breast Cancer Image Classification')
uploaded_file = st.file_uploader("Choose a breast ultrasound image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    # Load the model
    model = load_model()

    # Make predictions
    predicted_class = predict(image, model)
    st.write(f"Prediction: {predicted_class}")
