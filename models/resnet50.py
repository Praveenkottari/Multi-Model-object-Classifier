import torch.nn as nn
from torchvision.models import resnet50

def get_model(num_classes, pretrained=True):
    model = resnet50(weights='IMAGENET1K_V1' if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
