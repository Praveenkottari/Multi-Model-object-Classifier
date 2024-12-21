import torch.nn as nn
import timm

import timm

def get_model(num_classes, pretrained=True):
    # Create the model with a specified number of classes
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, num_classes=num_classes)
    # Now model should output shape: (N, num_classes)
    return model
