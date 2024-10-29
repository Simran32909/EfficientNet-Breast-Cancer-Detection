import logging
import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from model import EnhancedCNN
from data_loader import get_data_loaders
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_dir = 'src/models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    mcc = matthews_corrcoef(all_labels, all_preds)

    logging.info(f"ROC-AUC: {roc_auc:.4f}")
    logging.info(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes=['Normal', 'Benign', 'Malignant'])

def train():
    train_loader, val_loader = get_data_loaders()
    model = EnhancedCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define device
    model.to(device)

    for epoch in range(2):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/2', unit='batch') as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
                pbar.update(1)

        evaluate_model(model, val_loader, device)
        logging.info(f"Epoch [{epoch + 1}/2], Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), os.path.join(model_dir, 'cnn_model.pth'))

if __name__ == "__main__":
    train()
