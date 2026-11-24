"""Prediction from the retained two-branch training model."""
import numpy as np
import torch

from train_utils import metrics
from train_model import forward_pair


@torch.no_grad()
def predict(net,x,d,y,extras=False):
    net.eval();aa=[];ss=[];ff=[]
    for b in range(0,len(x),64):
        logits,scales,pre=forward_pair(net,x[b:b+64],d[b:b+64]);aa.append(logits.cpu().numpy());ss.append(scales.cpu().numpy())
        if extras:ff.append(pre.cpu().numpy())
    logits=np.concatenate(aa);scales=np.concatenate(ss);m,p=metrics(logits,y);sd=scales.astype('float64')
    r=dict(logits=logits,probability=p,metrics=m,scale_mean=sd.mean(-1),scale_channel_mean=sd.mean(0),scale_range=np.array([sd.min((0,2)),sd.max((0,2))]))
    if extras:r.update(pre_features=np.concatenate(ff),channel_scales=scales)
    return r
