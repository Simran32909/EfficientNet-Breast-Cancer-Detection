import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from src.utils import get_train_transforms, get_val_test_transforms
from src import config

def get_data_loaders(batch_size=config.BATCH_SIZE):
    train_transform = get_train_transforms()
    val_transform = get_val_test_transforms()

    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=config.VAL_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_test_loader(batch_size=config.BATCH_SIZE):
    test_transform = get_val_test_transforms()
    test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
