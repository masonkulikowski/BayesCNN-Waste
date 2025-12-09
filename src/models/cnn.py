"""
CNN Model Architecture for Waste Classification.

This module contains a custom CNN trained from scratch.
"""

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """
    Custom CNN architecture for waste classification.

    Architecture:
    - 4 Convolutional blocks with batch norm and max pooling
    - Global average pooling
    - Fully connected classifier with dropout

    Much faster to train than pretrained models (fewer parameters).
    """

    def __init__(self, num_classes=6, dropout=0.5):
        super(CustomCNN, self).__init__()

        # Block 1: 3 -> 32 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        )

        # Block 2: 32 -> 64 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        )

        # Block 3: 64 -> 128 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        )

        # Block 4: 128 -> 256 channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_custom_cnn(num_classes=6, dropout=0.5):
    """
    Create custom CNN model.

    Args:
        num_classes: Number of output classes
        dropout: Dropout rate for classifier

    Returns:
        CustomCNN model
    """
    print(f"\n[Model] Creating Custom CNN...")
    print(f"  Architecture: 4 Conv blocks (32->64->128->256)")
    print(f"  Dropout: {dropout}")
    print(f"  Output classes: {num_classes}")

    model = CustomCNN(num_classes=num_classes, dropout=dropout)
    print(f"  [Model] Custom CNN created")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  [Model] Total parameters: {total_params:,}")
    print(f"  [Model] Trainable parameters: {trainable_params:,}")
    print(f"  [Model] Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")

    # Print architecture summary
    print(f"\n  [Model] Architecture Summary:")
    print(f"    Input: 3x224x224")
    print(f"    Conv Block 1: 32 channels -> 112x112")
    print(f"    Conv Block 2: 64 channels -> 56x56")
    print(f"    Conv Block 3: 128 channels -> 28x28")
    print(f"    Conv Block 4: 256 channels -> 14x14")
    print(f"    Global Avg Pool: 256x1x1")
    print(f"    FC: 256 -> 128 -> {num_classes}")

    return model


def get_model_info(model):
    """
    Get model parameter information.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params
    }
