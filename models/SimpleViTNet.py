import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class SimpleViTNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleViTNet, self).__init__()
        
        # Load a pre-trained Vision Transformer (ViT)
        self.vit_base = vit_b_16(pretrained=True)
        
        # Remove the original classification head
        self.vit_base.heads = nn.Identity()
        
        # Additional layers for customization
        self.custom_fc = nn.Sequential(
            nn.Linear(768, 512),  # Reduce embedding size from 768 to 512
            nn.ReLU(),
            nn.Dropout(0.5),      # Dropout for regularization
            nn.Linear(512, 256),  # Reduce further from 512 to 256
            nn.ReLU(),
            nn.Linear(256, num_classes)  # Final layer for classification
        )

    def forward(self, x):
        # Pass input through ViT base
        x = self.vit_base(x)
        
        # Pass through custom fully connected layers
        x = self.custom_fc(x)
        return x

def get_model(num_classes):
    model = SimpleViTNet(num_classes)
    return model
