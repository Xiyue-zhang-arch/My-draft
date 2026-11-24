"""Three established fusion models and two strictly single-channel CNNs."""
import hashlib
import torch
from torch import nn
from train_utils import METHODS,LENGTH,ETA
from train_encoder import Encoder

def branch_parameters(net,name):
    return getattr(net,name).parameters() if hasattr(net,name) else ()

def role_hashes(net):
    result={}
    for name in ('v','c','head'):
        if hasattr(net,name):
            h=hashlib.sha256()
            for k,v in getattr(net,name).state_dict().items():
                h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
            result[name]=h.hexdigest()
    return result

def forward_pair(net,x,d):
    if net.method in ('vibration','current'):
        m=0 if net.method=='vibration' else 1
        return net(x[:,m:m+1],None)
    return net(x,d)

class Network(nn.Module):
    def __init__(self,method):
        super().__init__()
        if method not in METHODS: raise ValueError(method)
        self.method=method
        # Match branch and head initializations; unused branch is never registered.
        v,c=Encoder(),Encoder(); self.head=nn.Sequential(nn.Dropout(.5),nn.Linear(64,4))
        if method!='current': self.v=v
        if method!='vibration': self.c=c
        with torch.random.fork_rng():
            if method=='late_concat':
                self.condition=nn.Sequential(nn.Linear(72,17),nn.ReLU(),nn.Linear(17,64))
                nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
            elif method=='physical':
                self.condition=nn.ModuleList([nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,64)) for _ in range(2)])
                for module in self.condition:
                    nn.init.zeros_(module[-1].weight);nn.init.zeros_(module[-1].bias)

    def forward(self,x,d=None):
        if self.method in ('vibration','current'):
            if x.shape[1:]!=(1,LENGTH) or d is not None: raise ValueError('Single modality requires only its own channel and no descriptors')
            m=0 if self.method=='vibration' else 1; encoder=self.v if m==0 else self.c
            feature=encoder.prefix(x); z=encoder.suffix(feature)
            # Unused positions are documented constant bookkeeping placeholders.
            pre=x.new_zeros((len(x),2,64));pre[:,m]=feature.mean(-1)
            return self.head(z),x.new_ones((len(x),2,64)),pre
        if x.shape[1:]!=(2,LENGTH) or d is None or d.shape[1:]!=(2,4): raise ValueError('Dual modality shape mismatch')
        features=[self.v.prefix(x[:,:1]),self.c.prefix(x[:,1:2])]
        pre=torch.stack([h.mean(-1) for h in features],1);scales=x.new_ones((len(x),2,64))
        if self.method=='physical':
            for m in range(2): scales[:,m]=1+ETA*torch.tanh(self.condition[m](d.flatten(1)))
        h=[self.v.suffix(features[0]*scales[:,0,:,None]),self.c.suffix(features[1]*scales[:,1,:,None])]
        fused=(h[0]+h[1])*.5
        if self.method=='late_concat':fused=fused+ETA*torch.tanh(self.condition(torch.cat([fused,d.flatten(1)],1)))
        return self.head(fused),scales,pre
