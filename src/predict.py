import torch
from PIL import Image
from src.model import EnhancedCNN
from src.utils import get_val_test_transforms
from src import config

def load_model(model_path=config.MODEL_PATH):
    model = EnhancedCNN()
    # The map_location argument ensures the model loads correctly even if trained on a different device
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(model, image, class_names):
    transform = get_val_test_transforms()
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)
    
    return class_names[predicted_idx.item()]

def predict_from_path(model, image_path, class_names):
    image = Image.open(image_path).convert('RGB')
    return predict_image(model, image, class_names)

if __name__ == "__main__":
    # This is an example of how to use the predictor.
    # It's better to load class names from the file generated during training.
    with open(config.CLASSES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    model = load_model()
    # Using a relative path for the example image
    result = predict_from_path(model, 'data/test/malignant/malignant (20).png', class_names)
    print(f'Predicted class: {result}')
