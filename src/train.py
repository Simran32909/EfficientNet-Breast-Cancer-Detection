import logging
import os
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve, auc
from model import EnhancedCNN
from data_loader import get_data_loaders
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from src import config
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
from rich.table import Table

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_dir = config.MODEL_DIR
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def plot_confusion_matrix(cm, classes, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def evaluate_model(model, dataloader, device, criterion, epoch_idx: int, outputs_dir: str):
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
    acc = accuracy_score(all_labels, all_preds)

    logging.info(f"Validation Loss: {running_loss / len(dataloader):.4f} | ACC: {acc:.4f} | ROC-AUC: {roc_auc:.4f} | MCC: {mcc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(outputs_dir, f"confusion_matrix_epoch_{epoch_idx+1}.png")
    plot_confusion_matrix(cm, classes=dataloader.dataset.classes, save_path=cm_path)

    # ROC curves per class
    try:
        num_classes = len(dataloader.dataset.classes)
        plt.figure(figsize=(8, 6))
        for class_idx in range(num_classes):
            fpr, tpr, _ = roc_curve((all_labels == class_idx).astype(int), all_probs[:, class_idx])
            roc_auc_c = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{dataloader.dataset.classes[class_idx]} (AUC={roc_auc_c:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(loc='lower right')
        roc_path = os.path.join(outputs_dir, f"roc_epoch_{epoch_idx+1}.png")
        os.makedirs(os.path.dirname(roc_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
    except Exception as e:
        logging.warning(f"ROC plotting failed: {e}")
    
    return running_loss / len(dataloader), {"val_acc": acc, "val_roc_auc": roc_auc, "val_mcc": mcc}

def train():
    train_loader, val_loader = get_data_loaders()
    model = EnhancedCNN()

    # Compute class weights from training set for CrossEntropy
    targets = getattr(train_loader.dataset, 'targets', [label for _, label in train_loader.dataset.samples])
    class_counts = np.bincount(np.array(targets), minlength=len(train_loader.dataset.classes))
    class_weights = (class_counts.sum() / np.maximum(class_counts, 1)).astype(np.float32)
    class_weights_tensor = torch.tensor(class_weights)

    device = torch.device(config.DEVICE)
    class_weights_tensor = class_weights_tensor.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer and scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    model.to(device)

    console = Console()
    best_val_loss = float('inf')
    patience_counter = 0

    # Freeze-unfreeze strategy
    if config.FREEZE_EPOCHS > 0:
        for param in model.backbone.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # AMP scaler
    use_amp = bool(config.USE_AMP and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Outputs dir
    outputs_dir = os.path.join(os.getcwd(), config.OUTPUTS_DIR)
    os.makedirs(outputs_dir, exist_ok=True)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0

        progress = Progress(
            TextColumn(f"Epoch {epoch + 1}/{config.EPOCHS}"),
            BarColumn(),
            TextColumn("batch {task.completed}/{task.total}"),
            TextColumn("| avg loss: {task.fields[avg_loss]:.4f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        )
        with progress:
            task = progress.add_task("train", total=len(train_loader), avg_loss=0.0)
            for batch_idx, (images, labels) in enumerate(train_loader, start=1):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                avg_loss = running_loss / batch_idx
                progress.update(task, advance=1, avg_loss=avg_loss)

        val_loss, val_metrics = evaluate_model(model, val_loader, device, criterion, epoch_idx=epoch, outputs_dir=outputs_dir)
        scheduler.step(val_loss)
        
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

        # Unfreeze after configured epochs
        if config.FREEZE_EPOCHS > 0 and epoch + 1 == config.FREEZE_EPOCHS:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

        # Render an epoch summary table
        table = Table(title=f"Epoch {epoch + 1} Summary")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Train Avg Loss", f"{running_loss / len(train_loader):.4f}")
        table.add_row("Val Loss", f"{val_loss:.4f}")
        table.add_row("Val Acc", f"{val_metrics['val_acc']:.4f}")
        table.add_row("Val ROC-AUC", f"{val_metrics['val_roc_auc']:.4f}")
        table.add_row("Val MCC", f"{val_metrics['val_mcc']:.4f}")
        console.print(table)

    # Save the class names after the training loop is complete
    class_names = train_loader.dataset.classes
    with open(config.CLASSES_PATH, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

if __name__ == "__main__":
    train()
