import torch
from model import EnhancedCNN
from data_loader import get_data_loaders
import torch.nn as nn

def test():
    test_loader = get_data_loaders(batch_size=32)[1]
    model = EnhancedCNN()
    model.load_state_dict(torch.load(r'D:\JetBrains\PyCharm Professional\MediPrediction\src\src\models\cnn_model.pth'))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    test()
