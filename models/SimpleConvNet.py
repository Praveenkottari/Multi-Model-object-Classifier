import torch
import torch.nn as nn
from torchvision import models

class SimpleConvNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleConvNet, self).__init__()
        
        # Load a pre-trained ResNet model and remove its fully connected layer
        resnet = models.resnet18(pretrained=True)
        self.resnet_base = nn.Sequential(*list(resnet.children())[:-2])  # Remove fc and avgpool layers
        
        # Additional convolutional layers to customize the model
        self.custom_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # Reduce channels from 512 to 256
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # Reduce channels from 256 to 128
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1))  # Pool to 1x1 spatial dimensions
        )
        
        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),  # First fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(64, num_classes)  # Final layer for classification
        )

    def forward(self, x):
        # Pass input through the ResNet base
        x = self.resnet_base(x)
        
        # Pass through custom convolutional layers
        x = self.custom_conv(x)
        
        # Pass through fully connected classifier
        x = self.classifier(x)
        return x

def get_model(num_classes):
    model = SimpleConvNet(num_classes)
    return model
