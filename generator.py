from typing import final
import torch
import torch.nn as nn
import torch.functional as F
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.upscale = lambda x: nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear'),
            nn.ReflectionPad2d((1,1,1,1)),
            nn.Conv2d(2*x, x, kernel_size=(3,3), stride=(1,1), bias=False),
            nn.BatchNorm2d(x, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.embed = nn.Embedding(10, 50, max_norm=4.0)
        self.input = nn.Sequential(
            nn.Conv2d(150 , 512, kernel_size=(3,3), padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.layer = nn.Linear(8192, 7500)
        self.output = nn.Tanh()
    def forward(self, x, label):
        labelEmb = self.embed(label)
        labelEmb = labelEmb.view(labelEmb.size(0), 50, 1, 1)
        finalInput = torch.cat([x, labelEmb], dim = 1)
        finalInput = self.input(finalInput)
        finalInput = self.upscale(256)(finalInput)
        finalInput = self.upscale(128)(finalInput)
        finalInput = self.upscale(64)(finalInput)
        finalInput = self.upscale(32)(finalInput)
        finalInput = finalInput.view(finalInput.size(0), 8192)
        finalInput = self.layer(finalInput)
        finalInput = finalInput.view(finalInput.size(0), 3, 100, 25)
        return finalInput

