import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model import EnhancedCNN
from data_loader import get_data_loaders
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns

model_dir = 'src/models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate(model, data_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    plot_confusion_matrix(all_labels, all_preds, classes=['Normal', 'Benign', 'Malignant'])

    return accuracy, precision, recall, f1

def train():
    train_loader, val_loader = get_data_loaders()
    model = EnhancedCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/2', unit='batch') as pbar:
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)

        accuracy, precision, recall, f1 = evaluate(model, val_loader)
        print(f"Epoch [{epoch + 1}/15], Loss: {running_loss / len(train_loader):.4f}, "
              f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    torch.save(model.state_dict(), os.path.join(model_dir, 'cnn_model.pth'))

if __name__ == "__main__":
    train()
