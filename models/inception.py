import torch.nn as nn
from torchvision.models import inception_v3

def get_model(num_classes, pretrained=True):
    model = inception_v3(weights='IMAGENET1K_V1' if pretrained else None)
    # Inception v3 has a fc layer called fc
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.aux_logits = False  # disable aux if not needed
    return model
