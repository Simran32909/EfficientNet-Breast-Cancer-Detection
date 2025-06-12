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
from src import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_dir = config.MODEL_DIR
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

def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

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

    logging.info(f"Validation Loss: {running_loss / len(dataloader):.4f} | ROC-AUC: {roc_auc:.4f} | MCC: {mcc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes=dataloader.dataset.classes)
    
    return running_loss / len(dataloader)

def train():
    train_loader, val_loader = get_data_loaders()
    model = EnhancedCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    device = torch.device(config.DEVICE)
    model.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{config.EPOCHS}', unit='batch') as pbar:
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

        val_loss = evaluate_model(model, val_loader, device, criterion)
        
        # Checkpointing and Early Stopping
        if (best_val_loss - val_loss) > config.EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0 # Reset patience
            torch.save(model.state_dict(), config.MODEL_PATH)
            logging.info(f"Checkpoint saved! New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logging.info(f"No improvement in validation loss. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logging.info("Early stopping triggered! Training has stopped.")
            break

    # Save the class names after the training loop is complete
    class_names = train_loader.dataset.classes
    with open(config.CLASSES_PATH, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

if __name__ == "__main__":
    train()
