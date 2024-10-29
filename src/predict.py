import torch
from torchvision import transforms
from PIL import Image
import os
from src.model import EnhancedCNN

def load_model(model_path):
    model = EnhancedCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

def main():
    model_path = r'D:\JetBrains\PyCharm Professional\MediPrediction\models\cnn_model.pth'
    image_path = r'D:\JetBrains\PyCharm Professional\MediPrediction\data\val\malignant\malignant (6).png'

    model = load_model(model_path)
    prediction = predict_image(model, image_path)

    class_names = ['Normal', 'Benign', 'Malignant']
    print(f"Prediction: {class_names[prediction]}")

if __name__ == "__main__":
    main()
