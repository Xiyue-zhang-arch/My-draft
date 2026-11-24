"""Unchanged PU_0 encoders and middle modulation, direct 128-D concatenation."""
import torch
from torch import nn
from common import METHODS,ETA,LENGTH
from encoder import Encoder

class Network(nn.Module):
    def __init__(self,method):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method
        self.modalities=(0,1) if method.startswith('concat_') else (0,) if method.startswith('vibration_') else (1,)
        self.physical=method.endswith('physical')
        # Construct in the original order so shared encoders and the reference
        # 64-D classifier start identically to frozen PU_0 for the same seed.
        v,c=Encoder(),Encoder();base_head=nn.Linear(64,4)
        if 0 in self.modalities:self.v=v
        if 1 in self.modalities:self.c=c
        if len(self.modalities)==2:
            with torch.random.fork_rng():head=nn.Linear(128,4)
            with torch.no_grad():
                head.weight.copy_(torch.cat([base_head.weight*.5,base_head.weight*.5],1));head.bias.copy_(base_head.bias)
        else:head=base_head
        self.head=nn.Sequential(nn.Dropout(.5),head)
        self.head_width=64*len(self.modalities)
        if self.physical:
            with torch.random.fork_rng():
                modules=[nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,64)) for _ in range(2)]
                for m in modules:nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias)
            self.condition=nn.ModuleList([modules[i] for i in self.modalities])

    def forward(self,x,d):
        if x.shape[1:]!=(2,LENGTH) or d.shape[1:]!=(2,4):raise ValueError('Wrong paired input dimensions')
        features=[];scales=[]
        for branch,modality in enumerate(self.modalities):
            encoder=self.v if modality==0 else self.c
            h=encoder.prefix(x[:,modality:modality+1]);s=x.new_ones((len(x),64))
            if self.physical:
                condition=d.flatten(1) if len(self.modalities)==2 else torch.cat([d[:,modality],d[:,modality]],1)
                s=1+ETA*torch.tanh(self.condition[branch](condition))
            features.append(encoder.suffix(h*s[:,:,None]));scales.append(s)
        fused=torch.cat(features,dim=1) if len(features)==2 else features[0]
        return self.head(fused),torch.stack(scales,1)
