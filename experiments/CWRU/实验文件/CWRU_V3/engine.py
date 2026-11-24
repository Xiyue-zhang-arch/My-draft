"""从V2复制并适配的V3训练器，仅两模型、共享头、200轮调度。"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import copy
import json
import math
import random
import time
from contextlib import contextmanager

import torch
import numpy as np
import pandas as pd

V3 = Path(__file__).resolve().parent
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
from context import R
from models import Model as LegacyModel
from metrics import scores


DEFAULT_SPEC = dict(
    aux_weight=0.1,
    physics_fit='clean',
    aux_head='shared',
    val_noise_seeds=[6201],
    max_epochs=200,
    schedule_epochs=200,
    patience=20,
    fit_noise_seed=6301,
    fit_noise_repeats=3,
)
VARIANTS = ('baseline_data', 'data_aux_ce')
PROTOCOLS = ('clean_only', 'awgn_mixed')


def normalize_spec(spec=None):
    """仅接受已知设置；主控写入补齐后的配置作为科学协议。"""
    incoming = dict(spec or {})
    unknown = set(incoming) - set(DEFAULT_SPEC)
    if unknown:
        raise ValueError(f'未知训练设置: {sorted(unknown)}')
    result = copy.deepcopy(DEFAULT_SPEC)
    result.update(incoming)
    result['aux_weight'] = float(result['aux_weight'])
    if result['aux_weight'] not in (0.01, 0.03, 0.1, 0.3, 1.0):
        raise ValueError('辅助系数不在事前固定的五档网格中')
    if result['physics_fit'] not in ('clean', 'matched_noise'):
        raise ValueError('不支持的物理标准化方案')
    if result['aux_head'] != 'shared':
        raise ValueError('不支持的辅助分类头')
    if result['val_noise_seeds'] != [6201]:
        raise ValueError('V3验证噪声种子固定6201')
    if result['max_epochs'] != 200:
        raise ValueError('V3正式训练固定最多200epoch')
    if result['schedule_epochs'] != 200 or result['patience'] != 20:
        raise ValueError('余弦周期200、早停耐心20保持固定')
    if result['fit_noise_seed'] != 6301 or result['fit_noise_repeats'] != 3:
        raise ValueError('拟合噪声种子6301、重复3次保持固定')
    return result


@contextmanager
def preserve_rng(device):
    """诊断/复杂度测量不能消耗正式训练的随机数序列。"""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class SuiteModel(LegacyModel):
    def __init__(self, variant, spec=None):
        self.spec = normalize_spec(spec)
        super().__init__(variant)


def loss_for(model, logits, labels, spec=None):
    settings = normalize_spec(spec if spec is not None else getattr(model, 'spec', None))
    labels = labels.to(dtype=torch.long)
    main = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.0)
    aux = main.new_zeros(())
    if model.variant == 'data_aux_ce':
        if model.aux_logits is None:
            raise ValueError('辅助损失只能在包含辅助输出的训练前向后计算')
        aux = torch.nn.functional.cross_entropy(model.aux_logits, labels, label_smoothing=0.0)
    return main + settings['aux_weight'] * aux, main, aux


def _protocol(protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f'未知训练协议: {protocol}')


def validation_views(x, protocol, spec):
    _protocol(protocol)
    settings = normalize_spec(spec)
    yield 'clean', 0, x
    if protocol == 'awgn_mixed':
        for snr in (-4, -2, 0, 2, 4):
            for seed in settings['val_noise_seeds']:
                yield f'white_{snr:+d}', seed, R.fixed(x, snr, seed, 1)[0]


def validation_score(rows, protocol):
    """先平均同SNR噪声实例，保持干净0.5、各SNR0.1的原权重。"""
    _protocol(protocol)
    frame = pd.DataFrame(rows)
    expected = {'clean'} if protocol == 'clean_only' else {
        'clean', 'white_-4', 'white_-2', 'white_+0', 'white_+2', 'white_+4'}
    if set(frame.condition) != expected:
        raise ValueError('验证条件不完整')
    if frame.duplicated(['condition', 'noise_seed']).any():
        raise ValueError('验证噪声实例重复')
    nll = frame.groupby('condition').validation_nll.mean()
    result = float(nll['clean']) if protocol == 'clean_only' else float(
        .5 * nll['clean'] + .1 * nll.drop('clean').sum())
    if not math.isfinite(result):
        raise ValueError('验证指标非有限值')
    return result


@torch.no_grad()
def predict_and_ce(model, x, rpm, labels, device):
    """从logits稳定计算普通窗口平均CE，避免概率截断改变选模目标。"""
    model.eval()
    probabilities = []
    total = 0.0
    for start in range(0, len(x), 32):
        logits = model(torch.as_tensor(x[start:start+32], device=device),
                       torch.as_tensor(rpm[start:start+32], device=device))
        target = torch.as_tensor(labels[start:start+32], dtype=torch.long, device=device)
        total += float(torch.nn.functional.cross_entropy(logits, target, reduction='sum'))
        probabilities.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(probabilities), total / len(labels)


@torch.no_grad()
def validation(model, w, x, protocol, device, spec):
    indices = np.flatnonzero(w.role.eq('validation'))
    if not len(indices):
        raise ValueError('固定验证集为空')
    rpm = w.iloc[indices].rpm.to_numpy(np.float32)
    labels = w.iloc[indices].label.to_numpy()
    rows = []
    arrays = dict(labels=labels, window_indices=w.iloc[indices].window_index.to_numpy())
    for condition, noise_seed, observed in validation_views(x[indices], protocol, spec):
        probability, ce = predict_and_ce(model, observed, rpm, labels, device)
        values, _ = scores(labels, probability)
        rows.append(dict(condition=condition, noise_seed=noise_seed,
                         validation_nll=ce, **values))
        arrays[f'{condition}_n{noise_seed}'] = probability
    validation_score(rows, protocol)
    return rows, arrays


@torch.no_grad()
def _fit_physics(model, train_x, train_rpm, protocol, device, spec):
    """输入必须已由调用方限制为训练窗口，不接收整个划分表。"""
    settings = normalize_spec(spec)
    count = len(train_x)
    if model.variant == 'baseline_data':
        return dict(physics_fit_effective='not_applicable', fit_unique_training_windows=count,
                    fit_observations=0, fit_clean_observations=0, fit_noisy_observations=0)
    if protocol == 'clean_only' or settings['physics_fit'] == 'clean':
        model.fit(train_x, train_rpm, device)
        return dict(physics_fit_effective='clean', fit_unique_training_windows=count,
                    fit_observations=count, fit_clean_observations=count, fit_noisy_observations=0)
    def extract(observed):
        return np.concatenate([
            model.evidence.extract(torch.as_tensor(observed[i:i+32], device=device),
                torch.as_tensor(train_rpm[i:i+32], device=device) / 60)[0].cpu().numpy()
            for i in range(0, count, 32)])
    clean_features = extract(train_x)
    values = []
    for repeat in range(settings['fit_noise_repeats']):
        rng = np.random.default_rng(np.random.SeedSequence([20261021, settings['fit_noise_seed'], repeat]))
        observed = R.observe(train_x, rng, probability=1.0)[0]
        values.extend([clean_features, extract(observed)])
    pooled = np.concatenate(values)
    center = np.median(pooled, axis=0)
    scale = np.maximum(np.percentile(pooled, 75, axis=0) - np.percentile(pooled, 25, axis=0), .05)
    model.evidence.center.copy_(torch.as_tensor(center, device=device))
    model.evidence.scale.copy_(torch.as_tensor(scale, device=device))
    each = count * settings['fit_noise_repeats']
    return dict(physics_fit_effective='matched_noise', fit_unique_training_windows=count,
                fit_observations=2*each, fit_clean_observations=each, fit_noisy_observations=each,
                fit_distribution='equal clean/noisy; noisy SNR U[-4,4]; repeats are not independent windows')


def _gradient_pair(main, aux, parameters):
    params = tuple(p for p in parameters if p.requires_grad)
    if not params or not aux.requires_grad:
        return dict(cosine=None, main_norm=None, aux_norm=None, negative=None)
    first = torch.autograd.grad(main, params, retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(aux, params, retain_graph=True, allow_unused=True)
    main_sq = sum(float(g.detach().float().square().sum()) for g in first if g is not None)
    aux_sq = sum(float(g.detach().float().square().sum()) for g in second if g is not None)
    inner = sum(float(a.detach().float().mul(b.detach().float()).sum())
                for a, b in zip(first, second) if a is not None and b is not None)
    denominator = math.sqrt(main_sq * aux_sq)
    cosine = max(-1., min(1., inner / denominator)) if denominator > 0 else None
    return dict(cosine=cosine, main_norm=math.sqrt(main_sq), aux_norm=math.sqrt(aux_sq),
                negative=None if cosine is None else float(cosine < 0))


def gradient_stats(model, main, aux, spec=None):
    settings = normalize_spec(spec if spec is not None else getattr(model, 'spec', None))
    rows = {}
    for group, params in (('encoder', model.network.encoder.parameters()),
                          ('head', model.network.classifier.parameters())):
        values = _gradient_pair(main, aux, params)
        for name, value in values.items():
            rows[f'{group}_{name}'] = value
        rows[f'{group}_weighted_aux_norm'] = None if values['aux_norm'] is None else settings['aux_weight'] * values['aux_norm']
    rows['head_shared'] = settings['aux_head'] == 'shared' and model.variant == 'data_aux_ce'
    return rows


@torch.no_grad()
def complexity(variant, spec, device):
    settings = normalize_spec(spec)
    device = torch.device(device)
    with preserve_rng(device):
        R.U.seed_all(1706)
        model = SuiteModel(variant, settings).to(device).eval()
        macs = 0
        def hook(module, inputs, output):
            nonlocal macs
            if isinstance(module, torch.nn.Conv1d):
                macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
            elif isinstance(module, torch.nn.Linear):
                macs += output.numel() * module.in_features
        handles = [module.register_forward_hook(hook) for module in model.modules()
                   if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear))]
        try:
            model(torch.zeros(1, 1, 24000, device=device), torch.full((1,), 1750., device=device))
        finally:
            for handle in handles:
                handle.remove()
        total = sum(p.numel() for p in model.parameters())
        extra = sum(p.numel() for p in model.aux_classifier.parameters()) if hasattr(model, 'aux_classifier') else 0
        result = dict(variant=variant, parameters=total, inference_parameters=total-extra,
                      auxiliary_only_parameters=extra, conv_linear_flops=2*macs,
                      auxiliary_training_head_flops_per_sample=2304 if variant=='data_aux_ce' else 0,
                      flops_scope='batch1 Conv1d/Linear multiply+add=2; excludes physical FFT, normalization, activations, elementwise, backward')
    print('COMPLEXITY', json.dumps(result, ensure_ascii=False), flush=True)
    return result


def load(folder, device):
    from experiment_paths import resolve_result_path

    folder = resolve_result_path(folder)
    saved = torch.load(folder/'model.pt', map_location=device, weights_only=True)
    if 'suite_spec' in saved:
        model = SuiteModel(saved['variant'], saved['suite_spec'])
    elif 'variant' in saved:
        model = LegacyModel(saved['variant'])
    else:
        model = R.Classifier(saved['method'], saved['model_config'])
    model = model.to(device)
    model.load_state_dict(saved['state_dict'])
    return model.eval()


def train(x, w, protocol, variant, seed, subset, folder, device, spec):
    """只写一个全新模型目录；正式调用前主控必须通过基线前向检查。"""
    settings = normalize_spec(spec)
    _protocol(protocol)
    if variant not in VARIANTS:
        raise ValueError(f'新增训练仅允许{VARIANTS}')
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(folder)
    resolved = folder.resolve()
    readonly_data = (R.PROJECT/'data').resolve()
    if resolved == readonly_data or readonly_data in resolved.parents:
        raise ValueError('原始data及其子目录只读')
    for ancestor in resolved.parents:
        if (ancestor/'model.pt').exists() or (ancestor/'REPORT.md').exists():
            raise ValueError('禁止在已完成模型或冻结报告目录内新增训练输出')
    device = torch.device(device)
    ti = np.flatnonzero(w.role.eq('train'))
    vi = np.flatnonzero(w.role.eq('validation'))
    if not len(ti) or not len(vi):
        raise ValueError('固定训练或验证集合为空')
    if len(x) != len(w) or not w.window_index.is_unique:
        raise ValueError('窗口表与输入不一致')
    cost = complexity(variant, settings, device)
    R.U.seed_all(seed)
    model = SuiteModel(variant, settings).to(device)
    rpm = w.rpm.to_numpy(np.float32)
    y = w.label.to_numpy()
    R.U.sync()
    started = time.perf_counter()
    fit_info = _fit_physics(model, x[ti], rpm[ti], protocol, device, settings)
    R.U.sync()
    fit_seconds = time.perf_counter()-started
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings['schedule_epochs'], eta_min=.00001)
    views = list(validation_views(x[vi], protocol, settings))
    folder.mkdir(parents=True, exist_ok=False)
    w.iloc[ti].to_csv(folder/'selected_windows.csv', index=False)
    R.write_json(folder/'training_spec.json', dict(variant=variant, protocol=protocol, seed=seed, subset=subset, spec=settings))
    best, stale, step, best_epoch, best_update = float('inf'), 0, 0, 0, 0
    history = []
    curve_rows = []
    short_run = True
    for epoch in range(1, settings['max_epochs']+1):
        order = R.U.sample_epoch(w, ti, np.random.default_rng(np.random.SeedSequence([subset, seed, epoch, 20261011])))
        losses, norms, clipped = [], [], []
        train_correct = train_count = 0
        training_seconds = diagnostic_seconds = 0.
        gradient = {}
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        lr = float(optimizer.param_groups[0]['lr'])
        for offset in range(0, len(order), 32):
            model.train()
            R.U.sync()
            started = time.perf_counter()
            ix = order[offset:offset+32]
            observed = x[ix]
            if protocol == 'awgn_mixed':
                observed = R.observe(observed, np.random.default_rng(np.random.SeedSequence(
                    [20261012, subset, seed, epoch, offset])))[0]
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.as_tensor(observed, device=device), torch.as_tensor(rpm[ix], device=device))
            total, main, aux = loss_for(model, logits, torch.as_tensor(y[ix], device=device), settings)
            if short_run:
                train_correct += int((logits.detach().argmax(1)==torch.as_tensor(y[ix],device=device)).sum())
                train_count += len(ix)
            if not torch.isfinite(total):
                raise ValueError('训练损失非有限值')
            elapsed_diagnostic = 0.
            if offset == 0:
                R.U.sync()
                probe_started = time.perf_counter()
                gradient = gradient_stats(model, main, aux, settings)
                R.U.sync()
                elapsed_diagnostic = time.perf_counter()-probe_started
                diagnostic_seconds += elapsed_diagnostic
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            norms.append(float(norm.detach()))
            clipped.append(float(norm > 5))
            optimizer.step()
            step += 1
            losses.append([float(total.detach()), float(main.detach()), float(aux.detach())])
            R.U.sync()
            training_seconds += time.perf_counter()-started-elapsed_diagnostic
        peak = float(torch.cuda.max_memory_allocated(device)/1024**2) if device.type == 'cuda' else None
        started = time.perf_counter()
        val_rows = []
        for condition, noise_seed, observed in views:
            p, ce = predict_and_ce(model, observed, rpm[vi], y[vi], device)
            metrics = scores(y[vi],p)[0] if short_run else {}
            val_rows.append(dict(condition=condition, noise_seed=noise_seed, validation_nll=ce,**metrics))
        score = validation_score(val_rows, protocol)
        R.U.sync()
        val_seconds = time.perf_counter()-started
        improved = score < best-1e-6
        if improved:
            best, stale, best_epoch, best_update = score, 0, epoch, step
            torch.save(dict(variant=variant, model_config=R.U.CFG, suite_spec=settings,
                            state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()}), folder/'model.pt')
        else:
            stale += 1
        averages = np.mean(losses, axis=0)
        row = dict(epoch=epoch, updates=step, lr=lr, train_loss=float(averages[0]), main_ce=float(averages[1]),
                   auxiliary_ce=float(averages[2]), validation_nll=score, stale_epochs=stale, checkpoint_updated=improved,
                   train_seconds=training_seconds, gradient_diagnostic_seconds=diagnostic_seconds,
                   validation_seconds=val_seconds, peak_gpu_mib=peak,
                   gradient_norm_mean=float(np.mean(norms)), gradient_norm_max=float(max(norms)),
                   gradient_clipped_fraction=float(np.mean(clipped)), **gradient)
        row['shared_head_negative_cos_fraction'] = gradient.get('head_negative') if gradient.get('head_shared') else None
        row['encoder_negative_cos_fraction'] = gradient.get('encoder_negative')
        row['gradient_scope'] = 'first batch of epoch; not all updates'
        if short_run:
            row['train_acc_sampled_augmented'] = train_correct/train_count
            for observation in val_rows:
                condition=observation['condition']
                if observation['noise_seed'] in (0,6201):
                    for metric in ['acc','balanced_acc','recall_OR']:
                        row[f'validation_{condition}_{metric}'] = observation[metric]
                curve_rows.append(dict(epoch=epoch,**observation))
            pd.DataFrame(curve_rows).to_csv(folder/'validation_curve.csv',index=False)
        history.append(row)
        pd.DataFrame(history).to_csv(folder/'training.csv', index=False)
        # 单周期200轮，每轮更新一次，无重启。
        if epoch <= settings['schedule_epochs']:
            scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or stale >= settings['patience']:
            print(f'{variant}/seed{seed} epoch={epoch}/{settings["max_epochs"]} val={score:.6f} stale={stale}', flush=True)
        if stale >= settings['patience']:
            break
    del model
    selected = load(folder, device)
    rows, arrays = validation(selected, w, x, protocol, device, settings)
    if not math.isclose(validation_score(rows, protocol), best, rel_tol=0, abs_tol=2e-6):
        raise ValueError('最佳权重重新验证与选模记录不一致')
    pd.DataFrame(rows).to_csv(folder/'validation_metrics.csv', index=False)
    np.savez_compressed(folder/'validation_predictions.npz', **arrays)
    latency_ms = R.latency(selected, x[vi], rpm[vi], device)
    history_frame = pd.DataFrame(history)
    info = dict(protocol=protocol, variant=variant, seed=int(seed), subset=int(subset), spec=settings,
                training_windows=len(ti), validation_windows=len(vi), epochs_trained=epoch,
                updates_trained=step, best_epoch=best_epoch, best_update=best_update, validation_nll=best,
                early_stopped=stale>=settings['patience'], max_epochs=settings['max_epochs'], patience=settings['patience'],
                loss='ordinary_ce' if variant=='baseline_data' else f'ordinary_ce+{settings["aux_weight"]}_data_ce',
                label_smoothing=0., from_scratch=True, **{key: value for key, value in cost.items() if key != 'variant'}, **fit_info,
                fit_seconds=fit_seconds, total_train_seconds=float(history_frame.train_seconds.sum()),
                train_seconds_per_epoch=float(history_frame.train_seconds.mean()),
                gradient_diagnostic_seconds=float(history_frame.gradient_diagnostic_seconds.sum()),
                total_training_with_diagnostics_seconds=float((history_frame.train_seconds+history_frame.gradient_diagnostic_seconds).sum()),
                validation_seconds=float(history_frame.validation_seconds.sum()),
                validation_calls=epoch*len(views), peak_gpu_mib=float(history_frame.peak_gpu_mib.max()) if device.type=='cuda' else None,
                latency_ms=float(latency_ms), hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                precision='CUDA encoder BF16, remainder FP32' if device.type=='cuda' else 'FP32', batch_size=32,
                latency_scope='batch1 normalization+online physics+network; 10 warmups/50 repeats; synchronized; excludes disk/H2D',
                peak_memory_scope='allocated GPU memory; includes first-batch gradient diagnostic',
                completed=True)
    R.write_json(folder/'parameters.json', info)
    del selected
    return info
