"""Source bank, shared-role initialization and strict reload checks."""
import torch
import numpy as np
from common import FROZEN_SPLIT,DATA,BATCH,sha,ah,read,metrics,wave_transform,descriptor_transform,seed
from model import Network

from prepare import data_verify as verify_cache

def load_bank(cache):
    verify_cache(cache);norm=read(cache/'normalization.json');index=np.load(FROZEN_SPLIT/'split_indices.npz');windows=read(DATA/'windows.json')
    raw=torch.from_numpy(np.array(np.load(DATA/'signals.npy',mmap_mode='r')[:4000])).cuda()
    x=wave_transform(raw,norm);d=descriptor_transform(torch.from_numpy(np.load(cache/'physical.npy')).cuda(),norm)
    y=np.array([r['label'] for r in windows[:4000]],dtype='int64')
    bank=dict(cache=cache,norm=norm,x=x,d=d,y=torch.from_numpy(y).cuda(),labels=y)
    for role in ('train','val'):
        ids=index[role].copy();bank[role]=dict(ids=ids,x=x[ids],d=d[ids],y=y[ids])
    return bank
def make_model(spec):seed(spec['seed']);return Network(spec['method']).cuda()
@torch.no_grad()
def predict(net,x,d,y):
    net.eval();a=[];s=[]
    for b in range(0,len(x),BATCH):
        logits,scale=net(x[b:b+BATCH],d[b:b+BATCH]);a.append(logits.cpu().numpy());s.append(scale.cpu().numpy())
    a=np.concatenate(a);s=np.concatenate(s).astype('float64');m,p=metrics(a,y)
    return dict(logits=a,probability=p,scale_mean=s.mean(-1),metrics=m)
def initial_hashes(net):
    def h(module):return ah(np.concatenate([v.detach().cpu().numpy().ravel().view('uint8') for v in module.state_dict().values()]))
    result={}
    if hasattr(net,'v'):result['vibration']=h(net.v)
    if hasattr(net,'c'):result['current']=h(net.c)
    w=net.head[1].weight.detach().cpu().numpy();b=net.head[1].bias.detach().cpu().numpy()
    reference=w[:,:64]*2 if net.head_width==128 else w
    result['classifier_reference64']=ah(np.concatenate([reference.ravel(),b]))
    return result
def verify_one(out,row,bank):
    folder=out/'models'/row['model_id']
    for n,h in row['artifacts_sha256'].items():
        if sha(folder/n)!=h:raise ValueError('Model artifact changed')
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    spec={k:row[k] for k in ('model_id','method','seed')}
    if ck['spec']!=spec or ck['normalization_sha256']!=sha(bank['cache']/'normalization.json'):raise ValueError('Checkpoint identity differs')
    net=make_model(spec);net.load_state_dict(ck['state_dict'],strict=True)
    result=predict(net,bank['val']['x'],bank['val']['d'],bank['val']['y'])
    with np.load(folder/'validation.npz') as z:
        if not np.array_equal(z['logits'],result['logits']) or not np.array_equal(z['scale_mean'],result['scale_mean']):raise ValueError('Reload logits/scales differ')
        if not np.array_equal(z['window_index'],bank['val']['ids']) or not np.array_equal(z['y'],bank['val']['y']):raise ValueError('Validation identity differs')
    if abs(result['metrics']['CE']-row['val_CE'])>1e-12:raise ValueError('Validation CE differs')
    del net
    return result['metrics']['CE']
