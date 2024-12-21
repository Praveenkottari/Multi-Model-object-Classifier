import torch.nn as nn
from torchvision.models import vgg16

def get_model(num_classes, pretrained=True):
    model = vgg16(weights='IMAGENET1K_V1' if pretrained else None)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model
