import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_basic(nn.Module):
    def __init__(self, num_classes, img_size):
        out1 = 32
        out2 = 64
        conv_kernel_size = 5
        p = 2
        remaining_res = int((int(img_size/p)-conv_kernel_size+1)/p)-conv_kernel_size+1

        self.spatial_resolution = out2*2*remaining_res)

        self.model = nn.Sequential(
            nn.Conv2d(1, out1, conv_kernel_size)
            nn.ReLU(True)
            nn.MaxPool2d(p, p)

            nn.Conv2d(out1, out2, conv_kernel_size)
            nn.ReLU(True)
            nn.MaxPool2d(p, p)
        )

        self.fc = nn.Linear(spatial_resolution, num_classes)   

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.spatial_resolution)

        return self.fc(x)

    def feature_map(self, x):
        x = self.model(x)
        #Return flattened view?

        return x
