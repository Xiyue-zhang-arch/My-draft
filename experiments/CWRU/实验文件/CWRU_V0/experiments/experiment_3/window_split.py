"""先切无重叠窗口，再在每条原始记录内随机划分，不以负载分配角色。"""

import numpy as np
import pandas as pd

SPLIT_SEED = 20261018
ROLES = ["train", "validation", "test"]


def split_windows(windows, seed=SPLIT_SEED):
    """70/15/15；训练数四舍五入，奇数留出窗口交替分给验证和测试。"""
    windows = windows.loc[windows.record_id.ne("OR014_3")].copy().reset_index(drop=True)
    if windows.empty or not windows.window_index.is_unique:
        raise ValueError("窗口为空或原始窗口编号重复")
    windows["historical_role"] = windows["role"]
    windows["role"] = ""
    windows["load_hp"] = windows.record_id.str.rsplit("_", n=1).str[-1].astype(int)
    windows["split_seed"] = int(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 701515]))
    record_ids = rng.permutation(np.sort(windows.record_index.unique()))
    extra_to_validation = bool(rng.integers(0, 2))
    allocation = []
    for record_index in record_ids:
        ix = np.flatnonzero(windows.record_index.to_numpy() == record_index)
        n = len(ix)
        if n < 2:
            raise ValueError("不足2个窗口的记录须另行预先规定，不能自动复制窗口")
        # n*0.7按四舍五入取整，至少1训练、1留出；不复制或丢弃完整窗口。
        training = max(1, min(n-1, (7*n+5)//10))
        held = n-training
        validation = held//2
        testing = held//2
        if held % 2:
            if extra_to_validation:
                validation += 1
            else:
                testing += 1
            extra_to_validation = not extra_to_validation
        local_rng = np.random.default_rng(np.random.SeedSequence([seed, int(record_index), 24000]))
        shuffled = local_rng.permutation(ix)
        windows.loc[shuffled[:training], "role"] = "train"
        windows.loc[shuffled[training:training+validation], "role"] = "validation"
        windows.loc[shuffled[training+validation:], "role"] = "test"
        first = windows.iloc[ix[0]]
        allocation.append(dict(record_index=int(record_index), record_id=first.record_id,
                               label=int(first.label), load_hp=int(first.load_hp), total=n,
                               train=training, validation=validation, test=testing,
                               short_record_exception=n<3))
    if not windows.role.isin(ROLES).all():
        raise ValueError("存在未分配的窗口")
    for role in ROLES:
        if set(windows.loc[windows.role.eq(role), "label"]) != set(range(4)):
            raise ValueError(f"{role}未包含四类")
    # 切窗发生在划分之前，原采样区间不重叠，也不因打乱而改变窗口内容。
    for _, group in windows.groupby("record_index"):
        g = group.sort_values("start_sample")
        if np.any(g.start_sample.to_numpy()[1:] < g.end_sample.to_numpy()[:-1]):
            raise ValueError("原始采样区间有重叠")
    return windows, pd.DataFrame(allocation).sort_values("record_index").reset_index(drop=True)


def split_description(windows, allocation):
    class_support = pd.crosstab(windows.role, windows.label).reindex(ROLES, fill_value=0)
    loads = pd.crosstab(windows.role, windows.load_hp).reindex(ROLES, fill_value=0)
    history = pd.crosstab(windows.role, windows.historical_role).reindex(ROLES, fill_value=0)
    record_sets = {role:set(windows.loc[windows.role.eq(role),"record_id"]) for role in ROLES}
    overlap = [{"role_a":a,"role_b":b,"shared_records":len(record_sets[a]&record_sets[b])}
               for a,b in [("train","validation"),("train","test"),("validation","test")]]
    return class_support, loads, history, pd.DataFrame(overlap)
