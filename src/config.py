import torch

# --- Project Paths ---
BASE_DIR = "src"
DATA_DIR = "data_processed"
TRAIN_DIR = f"{DATA_DIR}/train"
VAL_DIR = f"{DATA_DIR}/val"
TEST_DIR = f"{DATA_DIR}/test"
MODEL_DIR = f"{BASE_DIR}/models"
MODEL_PATH = f"{MODEL_DIR}/cnn_model.pth"
CLASSES_PATH = f"{MODEL_DIR}/class_names.txt"

# --- Model Configuration ---
# Choose from 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'
BACKBONE = 'efficientnet_b0' 
NUM_CLASSES = 3 # Number of output classes (e.g., Normal, Benign, Malignant)
PRETRAINED = True # Use pretrained weights from ImageNet

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200
BATCH_SIZE = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
FREEZE_EPOCHS = 2  # Freeze backbone for first N epochs
USE_AMP = True  # Enable mixed precision if CUDA is available

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 50  # Num epochs to wait for improvement before stopping
EARLY_STOPPING_MIN_DELTA = 0.001 # Min change in validation loss to be considered an improvement

# --- Image Transformations ---
IMAGE_SIZE = (224, 224)
# Mean and std for ImageNet normalization
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225] 

# --- Outputs ---
OUTPUTS_DIR = "outputs"