"""PU-1 encoder, frozen from the checked PU_0 long-window implementation."""
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b, k, s in [(1,16,65,16), (16,32,9,4), (32,64,5,4), (64,64,5,2)]:
            layers += [nn.Conv1d(a,b,k,s,k//2), nn.GroupNorm(4,b), nn.ReLU()]
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.LayerNorm(64))

    def prefix(self,x):
        for i in range(9):
            x=self.net[i](x)
        return x

    def suffix(self,x):
        for i in range(9,len(self.net)):
            x=self.net[i](x)
        return x

    def forward(self,x):
        return self.suffix(self.prefix(x))
