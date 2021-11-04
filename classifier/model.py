from __future__ import print_function
from __future__ import division
import torch.nn as nn
import torchvision
from torchvision import models, transforms


class Model:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def remove_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def initialize_model(self):
        model_ft = models.resnet50(pretrained=False)
        self.remove_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
        input_size = 224
        return model_ft, input_size
