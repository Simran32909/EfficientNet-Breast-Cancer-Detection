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

        # Select weights enums (torchvision >= 0.13) when pretrained is requested
        weights = None
        if config.PRETRAINED:
            if backbone_name == 'resnet18':
                weights = models.ResNet18_Weights.DEFAULT
            elif backbone_name == 'resnet34':
                weights = models.ResNet34_Weights.DEFAULT
            elif backbone_name == 'resnet50':
                weights = models.ResNet50_Weights.DEFAULT
            elif backbone_name == 'efficientnet_b0':
                weights = models.EfficientNet_B0_Weights.DEFAULT

        # Build backbone to output convolutional feature maps (N, C, H, W)
        def build_with_fallback(builder, *b_args, **b_kwargs):
            try:
                return builder(*b_args, **b_kwargs)
            except Exception:
                # Fallback to no pretrained weights if download/cache is corrupted
                b_kwargs = dict(b_kwargs)
                if 'weights' in b_kwargs:
                    b_kwargs['weights'] = None
                return builder(*b_args, **b_kwargs)

        if backbone_name == 'resnet18':
            base = build_with_fallback(models.resnet18, weights=weights)
            # take layers up to conv feature maps (exclude avgpool and fc)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            num_features = 512
        elif backbone_name == 'resnet34':
            base = build_with_fallback(models.resnet34, weights=weights)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            num_features = 512
        elif backbone_name == 'resnet50':
            base = build_with_fallback(models.resnet50, weights=weights)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            num_features = 2048
        elif backbone_name == 'efficientnet_b0':
            base = build_with_fallback(models.efficientnet_b0, weights=weights)
            # use convolutional feature extractor; returns spatial maps
            self.backbone = base.features
            num_features = 1280
        else:
            raise ValueError("Unsupported backbone! Choose from 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'.")

        self.attention = Attention(num_features)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # (N, C, H, W)
        x = self.attention(features)
        x = self.global_pool(x)  # (N, C, 1, 1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x