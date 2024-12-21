import torch.nn as nn
from torchvision.models import efficientnet_b0

def get_model(num_classes, pretrained=True):
    model = efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
