import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    # Replace the final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
