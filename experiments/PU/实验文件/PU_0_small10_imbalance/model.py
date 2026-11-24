"""Descriptor-conditioned middle-layer scaling; final fusion remains equal."""
import torch
from torch import nn
from common import METHODS,LENGTH,ETA
from encoder import Encoder


class Network(nn.Module):
    def __init__(self,method):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method=method
        self.v,self.c=Encoder(),Encoder()
        self.head=nn.Sequential(nn.Dropout(.5),nn.Linear(64,4))
        # New components do not change shared initialization or the subsequent RNG state.
        with torch.random.fork_rng():
            if method=='late_concat':
                self.condition=nn.Sequential(nn.Linear(72,17),nn.ReLU(),nn.Linear(17,64))
                nn.init.zeros_(self.condition[-1].weight); nn.init.zeros_(self.condition[-1].bias)
            elif method!='equal':
                self.condition=nn.ModuleList([nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,64)) for _ in range(2)])
                for module in self.condition:
                    nn.init.zeros_(module[-1].weight); nn.init.zeros_(module[-1].bias)

    def forward(self,x,d):
        if x.shape[1:]!=(2,LENGTH) or d.shape[1:]!=(2,4):
            raise ValueError('Common waveform and descriptor dimensions required')
        features=[self.v.prefix(x[:,:1]),self.c.prefix(x[:,1:2])]
        pre=torch.stack([h.mean(-1) for h in features],1)
        scales=x.new_ones((len(x),2,64))
        if self.method not in ('equal','late_concat'):
            flat=d.flatten(1)
            for m in range(2):
                inputs=torch.cat([d[:,m],d[:,m]],1) if self.method=='self_only' else flat
                scales[:,m]=1+ETA*torch.tanh(self.condition[m](inputs))
        h=[self.v.suffix(features[0]*scales[:,0,:,None]),self.c.suffix(features[1]*scales[:,1,:,None])]
        fused=(h[0]+h[1])*.5
        if self.method=='late_concat':
            fused=fused+ETA*torch.tanh(self.condition(torch.cat([fused,d.flatten(1)],1)))
        return self.head(fused),scales,pre
