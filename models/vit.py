import torch
import torch.nn as nn
from torchvision.models import vit_b_16

def get_model(num_classes):
    model = vit_b_16(pretrained=True)
    # Replace the head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model
