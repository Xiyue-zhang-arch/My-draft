"""PU_0 direct-concatenation extension and unimodal comparisons."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import csv,hashlib,json,random,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT.parents[3]
DATA=ROOT/'data/pu_shared/prepared_inputs'
FROZEN_SPLIT = PROJECT / 'data/source_20261115_a'
SOURCE='N15_M07_F10'
TARGETS=('N09_M07_F10','N15_M01_F10','N15_M07_F04')
CLASS_NAMES=('healthy','outer','inner','compound')
METHODS=('vibration_plain','vibration_physical','current_plain','current_physical','concat_plain','concat_physical')
SEEDS=tuple(range(1706,1711))
SNRS=(-5,0,5,10)
NOISE_SEEDS=tuple(range(7201,7206))
LENGTH=2560
SPR=640
ORDERS=(3.054290718038529,4.945709281961471)
ETA=.25
BATCH=64
EPOCHS=50
SOURCE_FILES=('common.py','encoder.py','descriptors.py','reference.py','model.py','prepare.py','support.py','run.py','report.py','audit.py','preflight.py','PROTOCOL.md')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for i in range(8):
        try:os.replace(temp,p);return
        except PermissionError:
            if i==7:raise
            time.sleep(min(.025*2**i,.4))
def npz(p,**v):
    with Path(p).open('xb') as f:np.savez_compressed(f,**v)
def csvwrite(p,rows):
    rows=list(rows)
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def setup():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def source_hashes():
    from prepare import SHARED_SOURCES, source_path
    return {name: sha(source_path(name)) for name in (*SOURCE_FILES, *SHARED_SOURCES)}
def models():return [dict(model_id=f'm{i*6+j+1:04d}',method=m,seed=s) for i,s in enumerate(SEEDS) for j,m in enumerate(METHODS)]
def cases():
    rows=[]
    for target in TARGETS:
        rows.append(dict(case_id=target+'__clean',target=target,snr_db=None,noise_seed=None,noise_instance=None))
        for snr in SNRS:
            for instance,s in enumerate(NOISE_SEEDS):rows.append(dict(case_id=f'{target}__vibration_snr{snr}_n{s}',target=target,snr_db=snr,noise_seed=s,noise_instance=instance))
    return rows
def metrics(a,y):
    a=np.asarray(a,dtype='float64');y=np.asarray(y,dtype='int64')
    if a.shape!=(len(y),4) or not np.isfinite(a).all():raise ValueError('Invalid logits')
    lp=a-logsumexp(a,axis=1,keepdims=True);p=np.exp(lp);pred=a.argmax(1)
    cm=np.bincount(4*y+pred,minlength=16).reshape(4,4)
    if (cm.sum(1)==0).any():raise ValueError('Missing class')
    recall=np.diag(cm)/cm.sum(1);f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
    return dict(ACC=float(np.mean(pred==y)),BA=float(recall.mean()),Macro_F1=float(f1.mean()),CE=float(-lp[np.arange(len(y)),y].mean()),recall=recall,confusion=cm),p
def wave_transform(raw,norm):
    mean=torch.as_tensor(norm['waveform_mean'],device=raw.device,dtype=raw.dtype)[None,:,None]
    std=torch.as_tensor(norm['waveform_std'],device=raw.device,dtype=raw.dtype)[None,:,None]
    return (raw-mean)/std
def descriptor_transform(d,norm):
    center=torch.as_tensor(norm['physical']['center'],device=d.device,dtype=torch.float64)
    scale=torch.as_tensor(norm['physical']['scale'],device=d.device,dtype=torch.float64)
    return ((d.double()-center)/scale).clamp(-8,8).float()
