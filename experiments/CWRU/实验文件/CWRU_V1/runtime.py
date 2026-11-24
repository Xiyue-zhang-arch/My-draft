"""V1基线公共运行组件。只读复用既有网络与物理描述，不调用历史训练器。"""
from pathlib import Path
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[4]
SHARED = PROJECT / 'experiments/CWRU/实验文件/CWRU_V0/shared'
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(PROJECT / 'experiments/CWRU/实验文件/CWRU_V0/experiments/experiment_1/awgn_mixed'))
import training_utils as U
from cwru_dataset import load_dataset
from physics_gated_fusion import Network as Fusion
from vibration_only_classifier import Network as DataOnly
from bearing_physics_features import Evidence
from awgn_augmentation import observe, fixed


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class Classifier(nn.Module):
    def __init__(self, method, cfg):
        super().__init__()
        if method not in {'fusion', 'data_only'}:
            raise ValueError(method)
        self.method = method
        self.network = Fusion(cfg) if method == 'fusion' else DataOnly(cfg)
        if method == 'fusion':
            self.evidence = Evidence(cfg)

    def forward(self, x, rpm):
        if self.method == 'fusion':
            features, _ = self.evidence.extract(x, rpm / 60)
            return self.network(x, self.evidence.standardize(features))[0]
        return self.network(x)[0]

    @torch.no_grad()
    def fit(self, x, rpm, device):
        if self.method != 'fusion':
            return
        values = torch.cat([self.evidence.extract(torch.as_tensor(x[i:i+32], device=device),
            torch.as_tensor(rpm[i:i+32], device=device) / 60)[0].cpu()
            for i in range(0, len(x), 32)]).numpy()
        self.evidence.center.copy_(torch.as_tensor(np.median(values, axis=0), device=device))
        scale = np.maximum(np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0), .05)
        self.evidence.scale.copy_(torch.as_tensor(scale, device=device))


@torch.no_grad()
def predict(model, x, rpm, device):
    model.eval()
    return np.concatenate([model(torch.as_tensor(x[i:i+32], device=device),
        torch.as_tensor(rpm[i:i+32], device=device)).softmax(-1).cpu().numpy()
        for i in range(0, len(x), 32)])


def conditions(x, protocol, testing):
    yield 'clean', 0, x
    if protocol == 'awgn_mixed':
        for snr in [-4, -2, 0, 2, 4]:
            for seed in ([7201,7202,7203,7204,7205] if testing else [6201]):
                yield f'white_{snr:+d}', seed, fixed(x, snr, seed, 2 if testing else 1)[0]


@torch.no_grad()
def complexity(method, device):
    """精确统计Conv1d/Linear乘加，其他算子未纳入，不能冒充端到端FLOPs。"""
    U.seed_all(1706)
    model = Classifier(method, U.CFG).to(device).eval()
    macs = 0
    def hook(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv1d):
            macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv1d,nn.Linear))]
    try:
        model(torch.zeros(1,1,24000,device=device), torch.full((1,),1750.,device=device))
    finally:
        for h in handles:
            h.remove()
    result = dict(method=method, parameters=sum(p.numel() for p in model.parameters()),
        conv_linear_macs=macs, conv_linear_flops=2*macs, batch_size=1, window_samples=24000,
        flops_scope='Conv1d and Linear only, multiply+add=2; excludes FFT, physics, activations, reductions, normalization and bias')
    print('COMPLEXITY', json.dumps(result, ensure_ascii=False), flush=True)
    return result


def train(x, windows, protocol, method, seed, folder, device, epochs=100, patience=20):
    U.seed_all(seed)
    ti = np.flatnonzero(windows.role.eq('train'))
    vi = np.flatnonzero(windows.role.eq('validation'))
    y = windows.label.to_numpy()
    rpm = windows.rpm.to_numpy(np.float32)
    weights = U.weights_for(windows, vi)
    model = Classifier(method, U.CFG).to(device)
    U.sync(); started = time.perf_counter()
    model.fit(x[ti], rpm[ti], device)
    U.sync(); fit_seconds = time.perf_counter() - started
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.00001)
    views = list(conditions(x[vi], protocol, False))
    folder.mkdir(parents=True, exist_ok=False)
    windows.iloc[ti].to_csv(folder/'selected_windows.csv', index=False)
    best, stale, step, best_epoch, best_update = float('inf'), 0, 0, 0, 0
    history = []
    for epoch in range(1, epochs+1):
        order = U.sample_epoch(windows, ti, np.random.default_rng(np.random.SeedSequence([0,seed,epoch,20261011])))
        losses = []; seconds = 0.
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        lr = optimizer.param_groups[0]['lr']
        for offset in range(0, len(order), 32):
            model.train(); U.sync(); started = time.perf_counter()
            ix = order[offset:offset+32]
            observed = x[ix]
            if protocol == 'awgn_mixed':
                rng = np.random.default_rng(np.random.SeedSequence([20261012,0,seed,epoch,offset]))
                observed = observe(observed, rng)[0]
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.as_tensor(observed,device=device), torch.as_tensor(rpm[ix],device=device))
            loss = nn.functional.cross_entropy(logits, torch.as_tensor(y[ix],device=device), label_smoothing=0.0)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step(); step += 1
            losses.append(float(loss.detach()))
            U.sync(); seconds += time.perf_counter() - started
        peak = float(torch.cuda.max_memory_allocated()/1024**2) if device.type=='cuda' else None
        started = time.perf_counter()
        values = [U.nll(predict(model,wave,rpm[vi],device),y[vi],weights) for _,_,wave in views]
        U.sync(); val_seconds = time.perf_counter()-started
        score = values[0] if protocol=='clean_only' else .5*values[0]+.1*sum(values[1:])
        if not np.isfinite(score):
            raise ValueError('nonfinite validation score')
        improved = score < best-1e-6
        if improved:
            best, stale, best_epoch, best_update = score, 0, epoch, step
            torch.save(dict(method=method,model_config=U.CFG,
                state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),folder/'model.pt')
        else:
            stale += 1
        history.append(dict(epoch=epoch,updates=step,lr=lr,train_loss=float(np.mean(losses)),
            validation_nll=float(score),validation_clean_nll=float(values[0]),stale_epochs=stale,
            checkpoint_updated=improved,train_seconds=seconds,validation_seconds=val_seconds,peak_gpu_mib=peak))
        pd.DataFrame(history).to_csv(folder/'training.csv',index=False)
        print(f'{folder.parent.parent.name}/{protocol}/{method}/seed{seed}: epoch={epoch}/{epochs}, updates={step}, val={score:.6f}, stale={stale}/{patience}',flush=True)
        scheduler.step()
        if stale >= patience:
            break
    table = pd.DataFrame(history)
    info = dict(protocol=protocol,method=method,seed=seed,training_windows=len(ti),validation_windows=len(vi),
        epochs_trained=epoch,updates_trained=step,best_epoch=best_epoch,best_update=best_update,
        validation_nll=best,max_epochs=epochs,patience=patience,early_stopped=stale>=patience,
        stop_reason='patience' if stale>=patience else 'max_epochs',loss='cross_entropy',label_smoothing=0.,
        from_scratch=True,parameters=sum(p.numel() for p in model.parameters()),fit_seconds=fit_seconds,
        total_train_seconds=float(table.train_seconds.sum()),train_seconds_per_epoch=float(table.train_seconds.mean()),
        validation_seconds=float(table.validation_seconds.sum()),validation_calls=epoch*len(views),
        peak_gpu_mib=float(table.peak_gpu_mib.max()) if device.type=='cuda' else None)
    write_json(folder/'parameters.json',info)
    del model
    return info


def load_model(folder, device):
    saved = torch.load(folder/'model.pt',map_location=device,weights_only=True)
    model = Classifier(saved['method'], saved['model_config']).to(device)
    model.load_state_dict(saved['state_dict'])
    return model.eval()


@torch.no_grad()
def latency(model, raw, rpm, device):
    wave = torch.as_tensor(raw[:1],device=device)
    speed = torch.as_tensor(rpm[:1],device=device)
    def once():
        x = (wave-wave.mean(-1,keepdim=True))/(wave.std(-1,unbiased=False,keepdim=True)+1e-8)
        model(x,speed)
    for _ in range(10):
        once()
    U.sync(); started = time.perf_counter()
    for _ in range(50):
        once()
    U.sync()
    return (time.perf_counter()-started)*1000/50
