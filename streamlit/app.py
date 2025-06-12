import streamlit as st
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import load_model, predict_image
from src import config

# --- Helper Functions ---
@st.cache_resource
def load_cached_model(model_path):
    """Load and cache the model."""
    return load_model(model_path)

def load_class_names(classes_path):
    """Load class names from a file."""
    with open(classes_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# --- Load Model and Classes ---
try:
    model = load_cached_model(config.MODEL_PATH)
    class_names = load_class_names(config.CLASSES_PATH)
except FileNotFoundError:
    st.error(f"Error: Model or class names file not found. Please run the training script first.")
    st.stop()

# --- Page Content ---
cancer_info = {
    "normal": {
        "description": "Normal tissue with no signs of cancer.",
        "symptoms": "No specific symptoms related to breast cancer.",
        "treatment": "No treatment required. Regular check-ups are recommended."
    },
    "benign": {
        "description": "A benign tumor is a non-cancerous growth. It does not spread to other parts of the body.",
        "symptoms": "May cause a lump, pain, or discomfort depending on its size and location.",
        "treatment": "Often does not require treatment. Surgery may be performed to remove it if it causes symptoms."
    },
    "malignant": {
        "description": "A malignant tumor is cancerous. It can invade nearby tissues and spread to other parts of the body.",
        "symptoms": "A new lump or mass, swelling, skin dimpling, nipple retraction, unexplained weight loss.",
        "treatment": "Treatment depends on the stage and type of cancer, and may include surgery, chemotherapy, radiation therapy, and hormone therapy."
    }
}

st.title("Breast Cancer Detection from Ultrasound Images")

uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner("Classifying..."):
        prediction = predict_image(model, image, class_names)
    
    st.success(f"Prediction: **{prediction}**")
    
    info = cancer_info.get(prediction.lower(), {})
    if info:
        st.write("---")
        st.write(f"### More about {prediction}")
        st.write(f"**Description:** {info['description']}")
        st.write(f"**Common Symptoms:** {info['symptoms']}")
        st.write(f"**Typical Treatment:** {info['treatment']}")
