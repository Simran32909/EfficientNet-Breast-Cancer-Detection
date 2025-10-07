import streamlit as st
from PIL import Image
import sys
import os
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import load_model, predict_image_with_confidence, topk_from_last
from src import config

# --- Model/Classes availability helpers ---
def ensure_model_present():
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    if not os.path.exists(config.MODEL_PATH):
        model_url = os.environ.get("MODEL_URL")
        st.info(f"Downloading from: {os.environ.get('MODEL_URL')!r}")
        if not model_url:
            st.error("MODEL_URL is not set. Configure it in deployment secrets.")
            st.stop()
        try:
            with requests.get(model_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(config.MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()

def ensure_classes_present():
    os.makedirs(os.path.dirname(config.CLASSES_PATH), exist_ok=True)
    if not os.path.exists(config.CLASSES_PATH):
        classes_url = os.environ.get("CLASSES_URL")
        if not classes_url:
            st.error("class_names.txt missing and CLASSES_URL not set. Upload it or set CLASSES_URL.")
            st.stop()
        try:
            with requests.get(classes_url, timeout=30) as r:
                r.raise_for_status()
                with open(config.CLASSES_PATH, "w") as f:
                    f.write(r.text)
        except Exception as e:
            st.error(f"Failed to download class names: {e}")
            st.stop()

# Ensure artifacts before importing/using cached model
ensure_model_present()
ensure_classes_present()

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
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with st.spinner("Classifying..."):
        pred_label, pred_conf = predict_image_with_confidence(model, image, class_names)

    if pred_conf < threshold:
        st.warning(f"Uncertain / possibly out-of-distribution. Top-1 confidence {pred_conf:.2f} < threshold {threshold:.2f}")
    else:
        st.success(f"Prediction: **{pred_label}** (confidence {pred_conf:.2f})")

    # Show top-3 breakdown to aid debugging
    top3 = topk_from_last(model, k=3)
    if top3:
        st.write("Top-3:")
        for cls, prob in top3:
            st.write(f"- {cls}: {prob:.2f}")
    
    if pred_conf >= threshold:
        info = cancer_info.get(pred_label.lower(), {})
        if info:
            st.write("---")
            st.write(f"### More about {pred_label}")
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Common Symptoms:** {info['symptoms']}")
            st.write(f"**Typical Treatment:** {info['treatment']}")
