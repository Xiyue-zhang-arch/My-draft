"""PU-3 constants and artifact utilities; no import of mutable prior experiment code."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy.special import logsumexp

PROJECT=Path(__file__).resolve().parent
ROOT=PROJECT.parents[3]
SOURCE="N15_M07_F10"
TARGETS=("N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04")
CONDITIONS=TARGETS
RPM={c:900 if c.startswith("N09") else 1500 for c in CONDITIONS}
BEARINGS=("K003","K005","KA01","KA04","KI14","KI05","KB23","KB24")
CLASS_NAMES=("healthy","outer","inner","compound")
METHODS=("equal","late_concat","physical","wrong","statistics","self_only")
SEEDS=tuple(range(1706,1711))
NOISE_SEEDS=tuple(range(7201,7206))
SNRS=(-5,0,5,10)
LENGTH=2560
SPR=640
STRIDE=1280
RAW_FS=64000
DOWN_FS=16000
SAMPLE_SEED=20261113
ORDERS=(3.054290718038529,4.945709281961471)
BATCH=64
EPOCHS=50
PROTOCOLS=('balanced10','imbalanced','matched_balanced')
TRAIN_COUNTS={'balanced10':1120,'imbalanced':364,'matched_balanced':364}
UPDATES_PER_EPOCH={'balanced10':18,'imbalanced':6,'matched_balanced':6}
DATA=PROJECT/'data/prepared'
WRONG_ORDERS=(11.,15.)
ETA=.25
SPLIT_SEED=20261114


def sha(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path)
    temp=path.with_name(path.name+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(temp,path)
            return
        except PermissionError:
            if attempt==7:
                raise
            time.sleep(min(.025*2**attempt,.4))


def save_npz(path,**values):
    with Path(path).open("xb") as f:
        np.savez_compressed(f,**values)


def write_csv(path,rows):
    rows=list(rows)
    with Path(path).open("x",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def seed_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def models():
    result=[]
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            for method in METHODS:
                result.append(dict(model_id=f"m{len(result)+1:04d}",protocol=protocol,seed=seed,method=method))
    return result


def cases():
    result=[]
    for target in TARGETS:
        result.append(dict(case_id=target+"__clean",target=target,noise_seed=None,snr_db=None,noise_instance=None))
        for snr in SNRS:
            for instance,seed in enumerate(NOISE_SEEDS):
                result.append(dict(case_id=f"{target}__vibration_snr{snr}_n{seed}",target=target,
                                   noise_seed=seed,snr_db=snr,noise_instance=instance))
    if len(result)!=84:
        raise AssertionError("Expected 4 times 21 cases")
    return result


def metrics(logits,y):
    logits=np.asarray(logits,dtype="float64")
    y=np.asarray(y,dtype="int64")
    if logits.shape!=(len(y),4) or not np.isfinite(logits).all():
        raise ValueError("Invalid four-class logits")
    logp=logits-logsumexp(logits,axis=1,keepdims=True)
    p=np.exp(logp)
    pred=logits.argmax(1)
    cm=np.zeros((4,4),dtype="int64")
    np.add.at(cm,(y,pred),1)
    if (cm.sum(1)==0).any():
        raise ValueError("All four labels are required")
    recall=np.diag(cm)/cm.sum(1)
    precision=np.diag(cm)/np.maximum(cm.sum(0),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-30)
    return dict(ACC=float((pred==y).mean()),BA=float(recall.mean()),Macro_F1=float(f1.mean()),
                CE=float(-logp[np.arange(len(y)),y].mean()),recall=recall,confusion=cm),p


def waveform_transform(raw,norm):
    mean=torch.as_tensor(norm["waveform_mean"],device=raw.device,dtype=raw.dtype)[None,:,None]
    std=torch.as_tensor(norm["waveform_std"],device=raw.device,dtype=raw.dtype)[None,:,None]
    return (raw-mean)/std


def descriptor_transform(d,norm,profile="physical"):
    center=torch.as_tensor(norm[profile]["center"],device=d.device,dtype=torch.float64)
    scale=torch.as_tensor(norm[profile]["scale"],device=d.device,dtype=torch.float64)
    return ((d.double()-center)/scale).clamp(-8,8).float()


def diagnostic_cases():
    result=[]
    for target in TARGETS:
        for kind in ('neutral_descriptor','shuffle_descriptor','remove_vibration','remove_current'):
            eligible=('physical',) if kind.endswith('descriptor') else ('equal','physical')
            result.append(dict(case_id=target+'__'+kind,target=target,kind=kind,eligible=list(eligible)))
    return result


def eligible_cases(method):
    return cases()+[c for c in diagnostic_cases() if method in c['eligible']]


def descriptor_profile(method):
    return 'wrong' if method=='wrong' else 'statistics' if method=='statistics' else 'physical'
