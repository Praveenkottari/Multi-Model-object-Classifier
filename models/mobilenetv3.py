import torch.nn as nn
from torchvision.models import mobilenet_v3_large

def get_model(num_classes, pretrained=True):
    model = mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
