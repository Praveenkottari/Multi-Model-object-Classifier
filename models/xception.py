import torch.nn as nn
import timm

def get_model(num_classes, pretrained=True):
    # Xception from timm maps to legacy_xception
    model = timm.create_model('xception', pretrained=pretrained)
    # The final layer is `fc`, not `classifier`
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
