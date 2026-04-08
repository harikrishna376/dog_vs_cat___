import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from PIL import Image

# Set page title and favicon
st.set_page_config(page_title="Dog vs Cat Classifier", page_icon="🐾")

st.title("🐾 Dog vs Cat Image Classifier")
st.write("Upload an image of a dog or a cat to get a prediction!")

# Load the model using st.cache_resource to load it only once
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('cat_dog_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure 'cat_dog_model.keras' is in the same directory.")
        return None

model = load_model()

if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        try:
            # Preprocess the image for the model
            opencv_image = np.array(image.convert('RGB')) # Convert PIL Image to RGB numpy array

            # Resize and normalize
            resized_img = cv2.resize(opencv_image, (256, 256))
            test_input = resized_img.reshape((1, 256, 256, 3)) / 255.0

            # Make prediction
            prediction = model.predict(test_input)

            # Interpret prediction
            if prediction[0][0] > 0.5:
                st.success(f"Prediction: It's a DOG! (Confidence: {prediction[0][0]*100:.2f}%) 🐶")
            else:
                st.success(f"Prediction: It's a CAT! (Confidence: {(1-prediction[0][0])*100:.2f}%) 🐱")
        except Exception as e:
            st.error(f"Error processing image or making prediction: {e}")

    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)
