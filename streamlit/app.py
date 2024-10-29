import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import EnhancedCNN

model = EnhancedCNN()
model.load_state_dict(torch.load(r'D:\JetBrains\PyCharm Professional\MediPrediction\src\src\models\cnn_model.pth'))
model.eval()

cancer_info = {
    "normal": {
        "description": "Normal tissue with no cancer.",
        "symptoms": "No symptoms.",
        "treatment": "No treatment required."
    },
    "benign": {
        "description": "Benign tumors are non-cancerous growths.",
        "symptoms": "Can cause discomfort depending on location.",
        "treatment": "Surgery may be required."
    },
    "malignant": {
        "description": "Malignant tumors are cancerous and can spread.",
        "symptoms": "Unexplained weight loss, fatigue, pain, etc.",
        "treatment": "Chemotherapy, radiation, surgery."
    }
}

def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

st.title("Breast Cancer Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    class_index = predict(image)
    class_names = ['Normal', 'Benign', 'Malignant']
    st.write(f"Prediction: {class_names[class_index]}")
    st.write("Description: ", cancer_info[class_names[class_index].lower()]["description"])
    st.write("Symptoms: ", cancer_info[class_names[class_index].lower()]["symptoms"])
    st.write("Treatment: ", cancer_info[class_names[class_index].lower()]["treatment"])
