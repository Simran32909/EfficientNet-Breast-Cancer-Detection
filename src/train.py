import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from model import EnhancedCNN
from data_loader import get_data_loaders
import torch.nn as nn
import os
import matplotlib.pyplot as plt

model_dir = '../models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

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

    return accuracy, precision, recall, f1, all_labels, all_preds

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def train():
    train_loader, val_loader = get_data_loaders()
    model = EnhancedCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{2}', unit='batch') as pbar:
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)

        accuracy, precision, recall, f1, all_labels, all_preds = evaluate(model, val_loader)

        print(f"Epoch [{epoch + 1}/15], Loss: {running_loss / len(train_loader):.4f}, "
              f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        plot_confusion_matrix(all_labels, all_preds, ['Normal', 'Benign', 'Malignant'])

    torch.save(model.state_dict(), os.path.join(model_dir, 'cnn_model.pth'))

if __name__ == "__main__":
    train()
