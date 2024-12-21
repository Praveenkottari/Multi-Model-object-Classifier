import torch.nn as nn
from torchvision.models import densenet121

def get_model(num_classes, pretrained=True):
    model = densenet121(weights='IMAGENET1K_V1' if pretrained else None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model
