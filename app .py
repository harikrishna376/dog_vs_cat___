import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from PIL import Image

# 1. MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Dog vs Cat Classifier", page_icon="🐾")

# 2. CONSOLIDATED CUSTOM CSS (All in one place)
st.markdown("""
    <style>
    /* Force Black Background and White Text */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Style the title with a neon gradient */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 45px;
        font-weight: 700;
        text-align: center;
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    /* Style the subtitle */
    .sub-text {
        text-align: center;
        color: #ccd6f6;
        margin-bottom: 30px;
    }

    /* Style the upload box */
    .stFileUploader {
        border: 2px dashed #4facfe;
        border-radius: 15px;
        padding: 10px;
        background-color: #111111;
    }

    /* Professional Gradient Button */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.01);
        box-shadow: 0px 5px 15px rgba(79, 172, 254, 0.4);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- UI LAYOUT ---
st.markdown('<h1 class="main-title">AI Pet Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Upload a photo and let the neural network decide.</p>', unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        # Ensure this filename matches your .keras file on GitHub
        model = keras.models.load_model('cat_dog_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- MAIN LOGIC ---
if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button("Classify Image"):
            with st.spinner("Analyzing image..."):
                try:
                    # Preprocess the image
                    # 1. Convert to RGB (handles PNG transparency)
                    img_rgb = np.array(image.convert('RGB')) 
                    
                    # 2. Resize to the dimensions your model was trained on (256x256)
                    resized_img = cv2.resize(img_rgb, (256, 256))
                    
                    # 3. Reshape for model input: (batch, height, width, channels) and normalize
                    test_input = resized_img.reshape((1, 256, 256, 3)) / 255.0

                    # 4. Make prediction
                    prediction = model.predict(test_input)

                    # 5. Result Visualization
                    # Note: Check if your model uses 0 for Cat and 1 for Dog or vice versa.
                    # Usually, in CampusX tutorials, 0 = Cat, 1 = Dog.
                    if prediction[0][0] > 0.5:
                        conf = prediction[0][0] * 100
                        st.success(f"**Result:** It's a **DOG**! 🐶 ({conf:.2f}% Confidence)")
                    else:
                        conf = (1 - prediction[0][0]) * 100
                        st.success(f"**Result:** It's a **CAT**! 🐱 ({conf:.2f}% Confidence)")
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
else:
    st.warning("Please ensure 'cat_dog_model.keras' is present in your GitHub repository.")
