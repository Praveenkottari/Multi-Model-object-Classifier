import torch.nn as nn
import timm

def get_model(num_classes, pretrained=True):
    # 'deit_base_patch16_224'
    model = timm.create_model('deit_base_patch16_224', pretrained=pretrained)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model
