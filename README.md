# Breast Cancer Detection Using Enhanced CNN with Attention Mechanism

## Overview
This project aims to develop a deep learning model for breast cancer detection using images. It employs a Convolutional Neural Network (CNN) based on the ResNet-18 architecture, enhanced with an attention mechanism to improve feature extraction and classification performance. The model is trained to classify images into three categories: Normal, Benign, and Malignant.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
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
