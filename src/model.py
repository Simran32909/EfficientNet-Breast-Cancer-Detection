import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from src import config

class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        return self.gamma * out + x

class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, backbone_name=config.BACKBONE):
        super(EnhancedCNN, self).__init__()
        
        # Dynamically load the specified backbone
        if backbone_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=config.PRETRAINED)
            num_features = 512
        elif backbone_name == 'resnet34':
            self.resnet = models.resnet34(pretrained=config.PRETRAINED)
            num_features = 512
        elif backbone_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=config.PRETRAINED)
            num_features = 2048
            self.resnet.fc = nn.Identity()
        elif backbone_name == 'efficientnet_b0':
            self.resnet = models.efficientnet_b0(pretrained=config.PRETRAINED)
            # The classifier in EfficientNet is a single Linear layer
            num_features = self.resnet.classifier[1].in_features
            # We replace it with our own attention and classifier head
            self.resnet.classifier = nn.Identity()
        else:
            raise ValueError("Unsupported backbone! Choose from 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'.")

        self.attention = Attention(num_features)
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.resnet(x)
        
        # Reshape features to be compatible with the attention layer's expected input
        # Input to attention should be (batch_size, channels, height, width)
        # ResNet output is already in this format, but EfficientNet is (batch_size, features)
        if 'efficientnet' in self.resnet.__class__.__name__.lower():
            features = features.view(features.size(0), -1, 1, 1)

        x = self.attention(features)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x