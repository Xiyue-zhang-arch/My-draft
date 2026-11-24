"""Train-only source bank, prediction and analytic representation probes."""
import torch
import json
from pathlib import Path
import numpy as np
from common import DATA,PROJECT,sha,metrics,waveform_transform,descriptor_transform,descriptor_profile

from prepare import SHARED_SOURCES, data_verify as verify_cache, descriptor_path, source_path


SOURCE_FILES=('common.py','encoder.py','descriptors.py','model.py','reference.py','raw_prepare.py','prepare.py','support.py','engine.py','run.py','report.py','check.py','audit.py','PROTOCOL.md','bootstrap.py','implementation_provenance.json','report_helpers.py',*SHARED_SOURCES)


def source_hashes():
    return {name:sha(source_path(name)) for name in SOURCE_FILES}


def load_bank(cache,protocol):
    verify_cache(cache)
    split=np.load(DATA/'split_indices.npz')
    norm=json.loads((cache/('normalization_'+protocol+'.json')).read_text(encoding='utf-8'))
    windows=json.loads((DATA/'windows.json').read_text(encoding='utf-8'))
    raw=torch.from_numpy(np.array(np.load(DATA/'signals.npy',mmap_mode='r')[:])).cuda()
    x=waveform_transform(raw,norm)
    descriptors={profile:descriptor_transform(torch.from_numpy(np.load(descriptor_path(profile, cache))).cuda(),norm,profile) for profile in ('physical','wrong','statistics')}
    labels=np.array([r['label'] for r in windows[:]],dtype='int64')
    bank=dict(x=x,d=descriptors,y=torch.from_numpy(labels).cuda(),labels=labels,norm=norm,cache=cache,normalization_path=cache/('normalization_'+protocol+'.json'))
    for role in ('train','val'):
        ids=split['train_'+protocol if role=='train' else role].copy()
        bank[role]=dict(ids=ids,x=x[ids],d={k:v[ids] for k,v in descriptors.items()},y=labels[ids])
    return bank


@torch.no_grad()
def predict(net,x,d,y,extras=False):
    net.eval(); aa=[]; ss=[]; ff=[]
    for begin in range(0,len(x),64):
        logits,scales,pre=net(x[begin:begin+64],d[begin:begin+64])
        aa.append(logits.cpu().numpy()); ss.append(scales.cpu().numpy())
        if extras:
            ff.append(pre.cpu().numpy())
    logits=np.concatenate(aa); scales=np.concatenate(ss)
    result,p=metrics(logits,y)
    sd=scales.astype('float64')
    output=dict(logits=logits,probability=p,metrics=result,scale_mean=sd.mean(-1),
                scale_channel_mean=sd.mean(0),scale_range=np.array([sd.min((0,2)),sd.max((0,2))]))
    if extras:
        output['pre_features']=np.concatenate(ff); output['channel_scales']=scales
    return output


def fit_probes(features,y,val_features):
    center=features.astype('float64').mean(0)
    scale=np.maximum(features.astype('float64').std(0),1e-6)
    onehot=np.eye(4)[y]
    coef=[]; val=[]
    for m in range(2):
        z=(features[:,m].astype('float64')-center[m])/scale[m]
        z=np.c_[z,np.ones(len(z))]
        penalty=np.eye(65); penalty[-1,-1]=0
        w=np.linalg.solve(z.T@z+penalty,z.T@onehot)
        v=np.c_[(val_features[:,m].astype('float64')-center[m])/scale[m],np.ones(len(val_features))]@w
        coef.append(w); val.append(v)
    return dict(center=center,scale=scale,coef=np.stack(coef),validation_logits=np.stack(val,1),alpha=np.array(1.))


def probe_predict(features,probe):
    output=[]
    for m in range(2):
        z=(features[:,m].astype('float64')-probe['center'][m])/probe['scale'][m]
        output.append(np.c_[z,np.ones(len(z))]@probe['coef'][m])
    return np.stack(output,1)
