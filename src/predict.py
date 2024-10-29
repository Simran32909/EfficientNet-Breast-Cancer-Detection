import torch
from PIL import Image
import torchvision.transforms as transforms
from model import EnhancedCNN

def predict(image_path):
    model = EnhancedCNN()
    model.load_state_dict(torch.load('models/cnn_model.pth'))
    model.eval()

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

if __name__ == "__main__":
    result = predict(r'D:\JetBrains\PyCharm Professional\MediPrediction\data\train\malignant\malignant (5).png')
    print(f'Predicted class: {result}')
