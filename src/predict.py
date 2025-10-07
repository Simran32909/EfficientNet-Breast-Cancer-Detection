import torch
from PIL import Image
from typing import List, Tuple
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
    """Backward-compatible: returns only the predicted class string."""
    label, _ = predict_image_with_confidence(model, image, class_names)
    return label

def predict_image_with_confidence(model, image, class_names, topk: int = 3) -> Tuple[str, float]:
    """
    Returns (top1_label, top1_confidence) using softmax; also attaches
    model attributes for optional top-k introspection.
    """
    transform = get_val_test_transforms()
    image_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_t)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)

    # store last_probs for optional UI use
    setattr(model, "last_probs", probs.cpu().numpy())
    setattr(model, "last_classes", class_names)

    return class_names[int(idx.item())], float(conf.item())

def topk_from_last(model, k: int = 3) -> List[Tuple[str, float]]:
    """Utility to fetch top-k classes/confidences from the last forward pass."""
    probs = getattr(model, "last_probs", None)
    classes = getattr(model, "last_classes", None)
    if probs is None or classes is None:
        return []
    import numpy as np
    k = max(1, min(k, len(classes)))
    top_idx = np.argsort(probs)[-k:][::-1]
    return [(classes[i], float(probs[i])) for i in top_idx]

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
