"""CWRU训练公共工具：种子、窗口标准化、分层抽样和验证损失。"""

from pathlib import Path
import json
import math
import random
import torch
import numpy as np

CFG = json.loads(
    (Path(__file__).parent / "model_config.json").read_text(encoding="utf-8")
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure():
    """设置线程数和确定性计算选项，避免计时和随机性配置分散在各入口。"""
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_all(seed):
    """同步Python、NumPy和PyTorch随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(x):
    """按每个窗口自身的均值和总体标准差归一化。"""
    return ((x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)).astype(
        np.float32
    )


def weights_for(windows, ix):
    """构造类别均衡且类内原始记录均衡的验证权重。"""
    a = windows.iloc[ix]
    weights = np.zeros(len(ix))
    for c in range(4):
        ids = a[a.label == c].record_index.unique()
        for rid in ids:
            mask = (a.label.to_numpy() == c) & (a.record_index.to_numpy() == rid)
            weights[mask] = 1 / (4 * len(ids) * mask.sum())
    return weights


def nll(p, y, weights):
    """计算按类别及记录加权的负对数似然，用于验证选模。"""
    return float(np.dot(-np.log(p[np.arange(len(y)), y].clip(1e-12)), weights))


def sample_epoch(windows, ti, rng):
    """先抽类别，再抽原始记录和窗口，避免窗口较多的记录主导训练。"""
    a = windows.iloc[ti]
    pools = {
        c: {
            rid: ti[
                np.flatnonzero(
                    (a.label.to_numpy() == c) & (a.record_index.to_numpy() == rid)
                )
            ]
            for rid in a[a.label == c].record_index.unique()
        }
        for c in range(4)
    }
    out = []
    for c in rng.integers(0, 4, math.ceil(len(ti) / 32) * 32):
        rid = rng.choice(list(pools[c]))
        out.append(rng.choice(pools[c][rid]))
    return np.array(out)


def selected(w, ratio, subset):
    """只从训练窗口中选取指定比例的子集；子集种子控制选样。"""
    pool = np.flatnonzero(w.role.eq("train"))
    if ratio == 1:
        return pool
    rng = np.random.default_rng(subset)
    chosen = []
    for cl in range(4):
        cp = pool[w.iloc[pool].label.to_numpy() == cl]
        records = rng.permutation(w.iloc[cp].record_index.unique())
        queues = {
            rid: list(rng.permutation(cp[w.iloc[cp].record_index.to_numpy() == rid]))
            for rid in records
        }
        order = []
        while any(queues.values()):
            for rid in records:
                if queues[rid]:
                    order.append(queues[rid].pop())
        chosen.extend(order[: math.ceil(len(cp) * ratio)])
    return np.array(sorted(chosen))


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
