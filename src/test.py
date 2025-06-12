import torch
from model import EnhancedCNN
from data_loader import get_test_loader
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src import config

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def test():
    device = torch.device(config.DEVICE)
    
    test_loader = get_test_loader()
    model = EnhancedCNN()
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes)
    cm = confusion_matrix(all_labels, all_preds)

    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print("\nClassification Report:")
    print(report)
    
    plot_confusion_matrix(cm, classes=test_loader.dataset.classes)

if __name__ == "__main__":
    test()
