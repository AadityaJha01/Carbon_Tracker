"""
MobileNetV2 model for CIFAR-10 classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_mobilenet_v2(num_classes=10, pretrained=False):
    """
    Get MobileNetV2 model adapted for CIFAR-10.
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        pretrained: Whether to use pretrained weights (default: False)
    
    Returns:
        MobileNetV2 model with modified first layer and classifier
    """
    # Load MobileNetV2
    model = models.mobilenet_v2(pretrained=pretrained)
    
    # Modify first conv layer for CIFAR-10 (32x32 images instead of 224x224)
    # Original: kernel_size=3, stride=2
    # For CIFAR-10: kernel_size=3, stride=1, padding=1
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Modify classifier for CIFAR-10
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model

