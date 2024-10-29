import torch
from torchvision import datasets, transforms
from model import EnhancedCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


def load_model(model_path):
    model = EnhancedCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_model(model, test_loader):
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

    return all_labels, all_preds


def calculate_metrics(all_labels, all_preds):
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1


def plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Benign', 'Malignant'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()


def main():
    model_path = r'D:\JetBrains\PyCharm Professional\MediPrediction\models\cnn_model.pth'
    test_data_path = r'D:\JetBrains\PyCharm Professional\MediPrediction\data\test'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = load_model(model_path)
    all_labels, all_preds = evaluate_model(model, test_loader)

    calculate_metrics(all_labels, all_preds)
    plot_confusion_matrix(all_labels, all_preds)


if __name__ == "__main__":
    main()
