"""四个既有结构的V3训练适配。仅由独立新实验入口调用，不重启或覆盖任何结果。"""
from pathlib import Path
import math
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
COMPARISON = HERE.parent / 'CWRU' / 'experiments' / 'experiment_2'
sys.path.insert(0, str(COMPARISON))
from wdcnn_drsn_pg_models import FACTORIES
from attention_wavelet_hybrid import AttentionHybridAdapted
from context import R
import engine
from metrics import scores

FACTORIES = dict(FACTORIES, AW_Hybrid_adapted=AttentionHybridAdapted)
VARIANTS = tuple(FACTORIES)


def _input(value, device):
    x = torch.as_tensor(value, dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(1)
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError('要求单通道振动窗口')
    return x


class Model(torch.nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in FACTORIES:
            raise ValueError(variant)
        self.variant = variant
        self.net = FACTORIES[variant]()
        self.extra = {}

    def forward(self, x, rpm):
        logits, self.extra = self.net(_input(x, x.device), rpm.float())
        return logits


@torch.no_grad()
def fit_physics(model, x, rpm, protocol, device):
    """PG仅用来源训练窗口；增强训练采用V3等量干净/含噪观测。"""
    if model.variant != 'PG_RDGFN_adapted':
        return {'physics_fit': 'not_applicable', 'fit_observations': 0}
    if not len(x) or len(x) != len(rpm):
        raise ValueError('物理拟合数据为空或长度不符')
    def features(z):
        return torch.cat([model.net.front.extract(_input(z[i:i+32], device),
            torch.as_tensor(rpm[i:i+32], dtype=torch.float32, device=device))[1].cpu()
            for i in range(0, len(z), 32)])
    clean = features(x)
    if protocol == 'clean_only':
        pool = clean
    elif protocol == 'awgn_mixed':
        parts = []
        for repeat in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([20261021, 6301, repeat]))
            observed = R.observe(x, rng, probability=1.0)[0]
            parts.extend([clean, features(observed)])
        pool = torch.cat(parts)
    else:
        raise ValueError(protocol)
    front = model.net.front
    front.mean.copy_(pool.mean(0).to(device))
    front.scale.copy_(pool.std(0, unbiased=False).clamp_min(1e-8).to(device))
    front.quantiles.copy_(torch.quantile(pool[:, 4], torch.tensor([.1, .95])).to(device))
    return {'physics_fit': 'train_clean' if protocol == 'clean_only' else 'train_matched_noise_6301_x3',
            'fit_observations': len(pool), 'fit_unique_training_windows': len(x)}


def objective(model, logits, labels):
    main = F.cross_entropy(logits, labels, label_smoothing=0.)
    if model.variant != 'PG_RDGFN_adapted':
        return main, main, main.new_zeros(())
    extra = model.extra
    aux = .30 * F.cross_entropy(extra['raw'], labels, label_smoothing=0.) + .05 * F.mse_loss(extra['gate'], extra['confidence'])
    return main + aux, main, aux


@torch.no_grad()
def predict_and_ce(model, x, rpm, labels, device):
    model.eval()
    probabilities, total = [], 0.
    if len(x) == 0:
        raise ValueError('评估集合为空')
    for i in range(0, len(x), 32):
        xx = _input(x[i:i+32], device)
        rr = torch.as_tensor(rpm[i:i+32], dtype=torch.float32, device=device)
        yy = torch.as_tensor(labels[i:i+32], dtype=torch.long, device=device)
        logits = model(xx, rr)
        if logits.shape != (len(yy), 4) or not torch.isfinite(logits).all():
            raise ValueError('预测维度错误或输出包含非有限值')
        total += float(F.cross_entropy(logits, yy, reduction='sum'))
        probabilities.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(probabilities), total / len(x)


def validation(model, w, x, protocol, device, spec):
    ix = np.flatnonzero(w.role.eq('validation'))
    rpm, labels = w.iloc[ix].rpm.to_numpy(np.float32), w.iloc[ix].label.to_numpy()
    rows, arrays = [], {'labels': labels, 'window_indices': w.iloc[ix].window_index.to_numpy()}
    for condition, noise_seed, observed in engine.validation_views(x[ix], protocol, spec):
        p, ce = predict_and_ce(model, observed, rpm, labels, device)
        metric, _ = scores(labels, p)
        rows.append(dict(condition=condition, noise_seed=noise_seed, validation_nll=ce, **metric))
        arrays[f'{condition}_n{noise_seed}'] = p
    engine.validation_score(rows, protocol)
    return rows, arrays


def load(folder, device):
    payload = torch.load(Path(folder)/'model.pt', map_location='cpu', weights_only=False)
    if payload.get('variant') not in VARIANTS:
        raise ValueError('权重中方法名称无效')
    model = Model(payload['variant'])
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.to(device).eval()


def train(x, w, protocol, variant, seed, subset, folder, device, spec=None):
    """写入全新目录并以来源验证CE选择权重；不得对冻结目录调用。"""
    settings = engine.normalize_spec(spec)
    engine._protocol(protocol)
    if variant not in VARIANTS:
        raise ValueError(variant)
    folder = Path(folder).resolve()
    data = (R.PROJECT/'data').resolve()
    if folder == data or data in folder.parents or folder.exists():
        raise FileExistsError('禁止覆盖既有文件或写入data')
    if any((p/'model.pt').exists() or (p/'REPORT.md').exists() for p in folder.parents):
        raise ValueError('禁止写入冻结模型或报告目录')
    ti, vi = np.flatnonzero(w.role.eq('train')), np.flatnonzero(w.role.eq('validation'))
    if len(x) != len(w) or not len(ti) or not len(vi) or not w.window_index.is_unique:
        raise ValueError('数据与划分不一致')
    device = torch.device(device)
    R.U.seed_all(seed)
    model = Model(variant).to(device)
    rpm, y = w.rpm.to_numpy(np.float32), w.label.to_numpy()
    if not np.isfinite(x[ti]).all() or not np.isfinite(rpm[ti]).all() or set(y[ti]) != {0, 1, 2, 3}:
        raise ValueError('训练集包含非有限值或缺少类别')
    if not np.isfinite(x[vi]).all() or not np.isfinite(rpm[vi]).all() or set(y[vi]) != {0, 1, 2, 3}:
        raise ValueError('验证集包含非有限值或缺少类别')
    R.U.sync()
    started = time.perf_counter()
    fit_info = fit_physics(model, x[ti], rpm[ti], protocol, device)
    R.U.sync()
    fit_seconds = time.perf_counter() - started
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings['schedule_epochs'], eta_min=.00001)
    views = list(engine.validation_views(x[vi], protocol, settings))
    folder.mkdir(parents=True, exist_ok=False)
    w.iloc[ti].to_csv(folder/'selected_windows.csv', index=False)
    R.write_json(folder/'training_spec.json', dict(variant=variant, protocol=protocol, seed=seed, subset=subset, spec=settings, fit=fit_info))
    history, curve_rows, best, stale, step, best_epoch, best_update = [], [], float('inf'), 0, 0, 0, 0
    for epoch in range(1, settings['max_epochs'] + 1):
        order = R.U.sample_epoch(w, ti, np.random.default_rng(np.random.SeedSequence([subset, seed, epoch, 20261011])))
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        R.U.sync()
        losses, started = [], time.perf_counter()
        for offset in range(0, len(order), 32):
            model.train()
            ix = order[offset:offset+32]
            observed = x[ix]
            if protocol == 'awgn_mixed':
                observed = R.observe(observed, np.random.default_rng(np.random.SeedSequence([20261012, subset, seed, epoch, offset])))[0]
            xx = _input(observed, device)
            rr = torch.as_tensor(rpm[ix], device=device)
            yy = torch.as_tensor(y[ix], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xx, rr)
            total, main, auxiliary = objective(model, logits, yy)
            if not torch.isfinite(total):
                raise ValueError('训练损失非有限值')
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step(); step += 1
            losses.append([float(total.detach()), float(main.detach()), float(auxiliary.detach())])
        R.U.sync()
        elapsed = time.perf_counter() - started
        peak = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == 'cuda' else None
        validation_started = time.perf_counter()
        rows = []
        for condition, noise_seed, observed in views:
            p, ce = predict_and_ce(model, observed, rpm[vi], y[vi], device)
            rows.append(dict(condition=condition, noise_seed=noise_seed, validation_nll=ce, **scores(y[vi], p)[0]))
        score = engine.validation_score(rows, protocol)
        R.U.sync()
        validation_seconds = time.perf_counter() - validation_started
        improved = score < best-1e-6
        if improved:
            best, stale, best_epoch, best_update = score, 0, epoch, step
            torch.save(dict(variant=variant, state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}), folder/'model.pt')
        else:
            stale += 1
        avg = np.mean(losses, axis=0)
        history.append(dict(epoch=epoch, updates=step, lr=float(optimizer.param_groups[0]['lr']), train_loss=float(avg[0]), main_ce=float(avg[1]), auxiliary_loss=float(avg[2]), validation_nll=score, stale_epochs=stale, checkpoint_updated=improved, train_seconds=elapsed, validation_seconds=validation_seconds, peak_gpu_mib=peak))
        pd.DataFrame(history).to_csv(folder/'training.csv', index=False)
        curve_rows.extend(dict(epoch=epoch, **row) for row in rows)
        pd.DataFrame(curve_rows).to_csv(folder/'validation_curve.csv', index=False)
        if epoch == 1 or epoch % 10 == 0 or stale >= settings['patience']:
            print(f'{variant}/seed{seed} epoch={epoch}/{settings["max_epochs"]} val={score:.6f} stale={stale}', flush=True)
        scheduler.step()
        if stale >= settings['patience']:
            break
    selected = load(folder, device)
    rows, arrays = validation(selected, w, x, protocol, device, settings)
    if not math.isclose(engine.validation_score(rows, protocol), best, rel_tol=0, abs_tol=2e-6):
        raise ValueError('重新加载验证结果与最佳值不一致')
    pd.DataFrame(rows).to_csv(folder/'validation_metrics.csv', index=False)
    np.savez_compressed(folder/'validation_predictions.npz', **arrays)
    latency_ms = R.latency(selected, x[vi], rpm[vi], device)
    frame = pd.DataFrame(history)
    info = dict(variant=variant, protocol=protocol, seed=seed, subset=subset, training_windows=len(ti), validation_windows=len(vi), epochs_trained=epoch, updates_trained=step, best_epoch=best_epoch, best_update=best_update, validation_nll=best, early_stopped=stale>=settings['patience'], max_epochs=settings['max_epochs'], patience=settings['patience'], from_scratch=True, label_smoothing=0., parameters=sum(p.numel() for p in selected.parameters()), precision='FP32', fit_seconds=fit_seconds, train_seconds_per_epoch=float(frame.train_seconds.mean()), total_train_seconds=float(frame.train_seconds.sum()), validation_seconds=float(frame.validation_seconds.sum()), peak_gpu_mib=float(frame.peak_gpu_mib.max()) if device.type=='cuda' else None, latency_ms=float(latency_ms), hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU', batch_size=32, peak_memory_scope='allocated GPU memory, training epoch only; excludes validation', latency_scope='batch1 normalization+online features+network; 10 warmups/50 synchronized repeats; excludes disk/H2D', **fit_info)
    R.write_json(folder/'completed.json', info)
    return info
