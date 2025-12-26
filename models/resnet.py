"""
ResNet18 model for CIFAR-10 classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes=10, pretrained=False):
    """
    Get ResNet18 model adapted for CIFAR-10.
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        pretrained: Whether to use pretrained weights (default: False)
    
    Returns:
        ResNet18 model with modified first layer and classifier
    """
    # Load ResNet18
    model = models.resnet18(pretrained=pretrained)
    
    # Modify first conv layer for CIFAR-10 (32x32 images instead of 224x224)
    # Original: kernel_size=7, stride=2, padding=3
    # For CIFAR-10: kernel_size=3, stride=1, padding=1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove maxpool layer (not needed for smaller images)
    model.maxpool = nn.Identity()
    
    # Modify classifier for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

