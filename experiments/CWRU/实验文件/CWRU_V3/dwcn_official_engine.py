"""V3 source-validation adapter for the unmodified DWCN-SDGFD author network.

Upstream commit 0719172456687c43e52ea9a0eb09f856ad902b37, MIT license.
The author's epoch loop selects by target-label accuracy: it is NOT used here.
"""
from pathlib import Path
import importlib.util
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from context import R
import engine
from metrics import scores

_source = Path(__file__).resolve().parent / 'vendor_dwcn/code/models/DWCN.py'
_spec = importlib.util.spec_from_file_location('dwcn_author_source', _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
VARIANT = 'DWCN_SDGFD'
VARIANTS = (VARIANT,)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.variant = VARIANT
        # V3 learning rate for comparable training budget; author uses Adam.
        self.net = _module.DWCN(in_channel=1, num_classes=4, lr=0.0005)

    def forward(self, x, rpm):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.net.model_inference(x)


@torch.no_grad()
def predict_and_ce(model, x, rpm, labels, device):
    model.eval()
    probabilities, total = [], 0.0
    if len(x) == 0:
        raise ValueError('空评价数据')
    for start in range(0, len(x), 32):
        xx = torch.as_tensor(x[start:start+32], dtype=torch.float32, device=device)
        yy = torch.as_tensor(labels[start:start+32], dtype=torch.long, device=device)
        rr = torch.as_tensor(rpm[start:start+32], dtype=torch.float32, device=device)
        logits = model(xx, rr)
        if logits.shape != (len(yy), 4) or not torch.isfinite(logits).all():
            raise ValueError('DWCN output invalid')
        total += F.cross_entropy(logits, yy, reduction='sum').item()
        probabilities.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(probabilities), total / len(x)


def load(folder, device):
    obj = torch.load(Path(folder)/'model.pt', map_location='cpu', weights_only=False)
    if obj['variant'] != VARIANT:
        raise ValueError('unexpected method')
    model = Model().to(device)
    model.load_state_dict(obj['state_dict'], strict=True)
    return model.eval()


def train(x, w, protocol, variant, seed, subset, folder, device, spec=None):
    if protocol != 'clean_only' or variant != VARIANT:
        raise ValueError('DWCN requires clean-only condition')
    settings = engine.normalize_spec(spec)
    folder = Path(folder).resolve()
    if folder.exists():
        raise FileExistsError(folder)
    ti = np.flatnonzero(w.role.eq('train'))
    vi = np.flatnonzero(w.role.eq('validation'))
    y = w.label.to_numpy()
    rpm = w.rpm.to_numpy(np.float32)
    if len(x) != len(w) or len(ti)<4 or not len(vi) or not w.window_index.is_unique or set(y[ti]) != {0,1,2,3}:
        raise ValueError('training split invalid')
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('upstream perturbation explicitly requires CUDA')
    R.U.seed_all(seed)
    model = Model().to(device)
    opts = [model.net.optimizer_te, model.net.optimizer_st, model.net.optimizer_LD]
    sched = [torch.optim.lr_scheduler.CosineAnnealingLR(o, settings['schedule_epochs'], eta_min=.00001) for o in opts]
    folder.mkdir(parents=True, exist_ok=False)
    w.iloc[ti].to_csv(folder/'selected_windows.csv', index=False)
    R.write_json(folder/'training_spec.json', dict(variant=variant, protocol=protocol, seed=seed, subset=subset,
        upstream_commit='0719172456687c43e52ea9a0eb09f856ad902b37', selection='source_validation_ce',
        adaptation='V3 train split, 4 outputs, Adam lr=0.0005 and cosine schedule; no target-label selection', spec=settings))
    best, stale, history, updates, best_epoch = float('inf'), 0, [], 0, 0
    start = time.perf_counter()
    for epoch in range(1, settings['max_epochs']+1):
        order = R.U.sample_epoch(w, ti, np.random.default_rng(np.random.SeedSequence([subset, seed, epoch, 20261011])))
        def train_batches():
            for offset in range(0, len(order), 32):
                ix=order[offset:offset+32]
                if len(ix)<2:
                    raise ValueError('instance similarity requires at least 2 samples per batch')
                xx=torch.as_tensor(x[ix], dtype=torch.float32, device=device)
                if xx.ndim == 2:
                    xx=xx.unsqueeze(1)
                if xx.ndim != 3 or xx.shape[1] != 1:
                    raise ValueError('single-channel input required')
                yy=torch.as_tensor(y[ix], dtype=torch.long, device=device)
                yield (xx, yy, torch.zeros(len(ix), device=device))
        model.train()
        logs=model.net(train_batches())  # author CE, KL, similarity and CCP updates
        updates += (len(order)+31)//32
        p, ce = predict_and_ce(model, x[vi], rpm[vi], y[vi], device)
        improved = ce < best-1e-6
        if improved:
            best, stale, best_epoch = ce, 0, epoch
            torch.save(dict(variant=variant,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),folder/'model.pt')
        else:
            stale += 1
        history.append(dict(epoch=epoch,updates=updates,validation_nll=ce,stale_epochs=stale,checkpoint_updated=improved,**logs))
        pd.DataFrame(history).to_csv(folder/'training.csv',index=False)
        for scheduler in sched:
            scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or stale >= settings['patience']:
            print(f'{variant}/seed{seed} epoch={epoch} val={ce:.6f} stale={stale}',flush=True)
        if stale >= settings['patience']:
            break
    selected=load(folder,device)
    p, ce=predict_and_ce(selected,x[vi],rpm[vi],y[vi],device)
    if not math.isclose(ce,best,abs_tol=2e-6,rel_tol=0):
        raise ValueError('checkpoint validation mismatch')
    np.savez_compressed(folder/'validation_predictions.npz',labels=y[vi],window_indices=w.iloc[vi].window_index.to_numpy(),clean_n0=p)
    pd.DataFrame([dict(condition='clean',noise_seed=0,validation_nll=ce,**scores(y[vi],p)[0])]).to_csv(folder/'validation_metrics.csv',index=False)
    info=dict(variant=variant,protocol=protocol,seed=seed,subset=subset,training_windows=len(ti),validation_windows=len(vi),
        epochs_trained=epoch,updates_trained=updates,best_epoch=best_epoch,validation_nll=best,
        total_train_seconds=time.perf_counter()-start,parameters=sum(t.numel() for t in selected.parameters()),
        selection='source_validation_ce', upstream_commit='0719172456687c43e52ea9a0eb09f856ad902b37')
    R.write_json(folder/'completed.json',info)
    return info
