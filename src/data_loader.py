import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=r'D:\JetBrains\PyCharm Professional\MediPrediction\data\train',
                                         transform=transform)
    val_dataset = datasets.ImageFolder(root=r'D:\JetBrains\PyCharm Professional\MediPrediction\data\val',
                                       transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
