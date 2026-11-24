"""Training constants and validation metrics for the two retained PU experiments."""
import numpy as np
from scipy.special import logsumexp

from common import LENGTH, BATCH, EPOCHS, ETA

METHODS = ("equal", "late_concat", "physical", "vibration", "current")
UPDATES_PER_EPOCH = {}


def descriptor_profile(method):return 'physical'
def metrics(logits,y):
    a=np.asarray(logits,dtype='float64');y=np.asarray(y,dtype='int64')
    if a.shape!=(len(y),4) or not np.isfinite(a).all():raise ValueError('Invalid logits')
    lp=a-logsumexp(a,axis=1,keepdims=True);p=np.exp(lp);pred=a.argmax(1)
    cm=np.bincount(4*y+pred,minlength=16).reshape(4,4)
    if (cm.sum(1)==0).any():raise ValueError('Missing class')
    recall=np.diag(cm)/cm.sum(1);f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
    return dict(ACC=float(np.mean(pred==y)),BA=float(recall.mean()),Macro_F1=float(f1.mean()),CE=float(-lp[np.arange(len(y)),y].mean()),recall=recall,confusion=cm),p
