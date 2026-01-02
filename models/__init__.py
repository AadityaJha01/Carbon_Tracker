"""
Models package for Carbon-Aware ML Training Pipeline
"""

from .cnn import SimpleCNN, get_cnn
from .resnet import get_resnet18
from .mobilenet import get_mobilenet_v2
from .deep_cnn import get_deep_cnn

__all__ = ['SimpleCNN', 'get_cnn', 'get_resnet18', 'get_mobilenet_v2', 'get_deep_cnn']

