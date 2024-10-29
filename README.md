# Breast Cancer Detection Using Enhanced CNN with Attention Mechanism

## Overview
This project aims to develop a deep learning model for breast cancer detection using images. It employs a Convolutional Neural Network (CNN) based on the ResNet-18 architecture, enhanced with an attention mechanism to improve feature extraction and classification performance. The model is trained to classify images into three categories: Normal, Benign, and Malignant.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [License](#license)

## Features
- **Data Loading**: Utilizes PyTorch's `ImageFolder` to load training and validation datasets.
- **Model Architecture**: Incorporates ResNet-18 as a backbone with an attention mechanism for better feature learning.
- **Training and Evaluation**: Implements a training loop with evaluation metrics such as accuracy, precision, recall, F1 score, and confusion matrix visualization.
- **K-Fold Cross-Validation**: Allows for robust model validation by splitting the dataset into multiple folds.

## Requirements
- Python 3.6 or higher
- PyTorch
- torchvision
- seaborn
- tqdm
- scikit-learn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Simran32909/breast-cancer-detection.git
   cd breast-cancer-detection

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate

3. Install the required packages:
   ```bash
   pip install -r requirements.txt

## Usage
1. Prepare your dataset in the format expected by ImageFolder. The directory structure should look like this:
   ```bash
   data/
    train/
        Normal/
            image1.jpg
            image2.jpg
        Benign/
            image1.jpg
            image2.jpg
        Malignant/
            image1.jpg
            image2.jpg
    val/
        Normal/
            image1.jpg
            image2.jpg
        Benign/
            image1.jpg
            image2.jpg
        Malignant/
            image1.jpg
            image2.jpg

2. Update the data_loader.py file with the correct paths for your dataset.

3. Run the training script:
   ```bash
   python src/train.py

## Model Evaluation
After training, the model's performance will be evaluated using various metrics. The following metrics will be displayed:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix Visualization

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments
- [PyTorch](https://pytorch.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [ResNet](https://arxiv.org/abs/1512.03385)
