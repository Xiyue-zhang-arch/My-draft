"""CWRU非配对AWGN增强。每个窗口独立决定是否加噪；保持批量大小，两个分支使用同一观测。"""

import numpy as np


def observe(clean, rng, probability=0.5, fixed_snr=None):
    """逐窗口加噪并归一化，返回观测、加噪掩码、目标和实际SNR。"""
    clean = np.asarray(clean, dtype=np.float32)
    count = len(clean)
    # 先生成掩码，再为选中窗口采样SNR；该顺序决定同种子的噪声实例。
    mask = rng.random(count) < probability
    snr = np.full(count, np.nan, dtype=np.float32)
    realized = np.full(count, np.nan, dtype=np.float32)
    mixed = clean.copy()
    k = int(mask.sum())
    if k:
        snr[mask] = rng.uniform(-4.0, 4.0, k) if fixed_snr is None else fixed_snr
        signal_power = ((clean[mask] - clean[mask].mean(-1, keepdims=True)) ** 2).mean(
            -1, keepdims=True
        )
        noise = rng.standard_normal(clean[mask].shape, dtype=np.float32)
        noise *= np.sqrt(signal_power / 10.0 ** (snr[mask, None, None] / 10.0))
        realized[mask] = (
            10 * np.log10(signal_power / (noise**2).mean(-1, keepdims=True))
        ).reshape(-1)
        mixed[mask] += noise
    # 未选中窗口在统一归一化前保持原值；不拼接额外的干净副本。
    assert np.array_equal(mixed[~mask], clean[~mask])
    mixed = (mixed - mixed.mean(-1, keepdims=True)) / (
        mixed.std(-1, keepdims=True) + 1e-8
    )
    assert mixed.shape == clean.shape
    return mixed.astype(np.float32), mask, snr, realized


def fixed(clean, snr, seed, role):
    """用数据划分标识和噪声种子生成固定测试观测。"""
    # 各SNR共用标准正态方向；结果不依赖模型或类别标签。
    rng = np.random.default_rng(np.random.SeedSequence([20261012, role, seed]))
    return observe(clean, rng, probability=1.0, fixed_snr=snr)
