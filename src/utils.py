import torchvision.transforms as transforms
from src import config


class GaussianNoise(object):
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + tensor.new(tensor.size()).normal_(self.mean, self.std)

def get_train_transforms():
    """
    Transforms for the training set with data augmentation.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.02),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        GaussianNoise(std=0.01),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ])

def get_val_test_transforms():
    """
    Transforms for the validation and test sets without data augmentation.
    """
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ]) 