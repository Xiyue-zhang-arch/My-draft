"""Read frozen PU inputs and load the retained harmonic models."""
import json
from pathlib import Path

import numpy as np
import torch

from common import CONDITIONS, descriptor_transform, waveform_transform, sha, write_json, save_npz
from current_descriptor import describe_current
from descriptors import describe
from train_model import Network

ROOT = Path(__file__).resolve().parents[5]
DATA = Path(__file__).resolve().parents[2] / "PU_0_all_conditions_721/data/prepared"
ROUTES = tuple((source, target) for source in CONDITIONS for target in CONDITIONS if source != target)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@torch.no_grad()
def v2_descriptors(raw):
    pair = describe(raw)
    pair[:, 1] = describe_current(raw[:, 1])
    return pair


def metadata():
    windows = read(DATA / "windows.json")
    codes = np.array([r["condition"] for r in windows])
    labels = np.array([r["label"] for r in windows], dtype=np.int64)
    records = np.array([r["record_index"] for r in windows])
    proof = read(DATA / "verification.json")
    for name in ("signals.npy", "vibration_noise_templates.npy", "windows.json", "split_indices.npz"):
        assert sha(DATA / name) == proof["files_sha256"][name], name
    with np.load(DATA / "split_indices.npz") as index:
        roles = {r: index[r] for r in ("train", "val", "test")}
    assert len(set(np.concatenate(list(roles.values())))) == 16000
    assert all(set(records[roles[r]]).isdisjoint(records[roles[s]]) for r, s in
               (("train", "test"), ("val", "test"), ("train", "val")))
    splits = {c: {k: ids[codes[ids] == c] for k, ids in roles.items()} for c in CONDITIONS}
    for split in splits.values():
        for role, count in (("train", 700), ("val", 200), ("test", 100)):
            np.testing.assert_array_equal(np.bincount(labels[split[role]]), [count]*4)
    noise_position = {int(x): i for i, x in enumerate(roles["test"])}
    return windows, labels, splits, noise_position


def make_bank(path, method, raw, labels, train, val, descriptors):
    ids = np.r_[train, val]
    x = raw[ids]
    n = len(train)
    wave = x[:n].astype(np.float64)
    features = descriptors[:n].astype(np.float64)
    q1, med, q3 = np.quantile(features, [.25, .5, .75], axis=0)
    norm = dict(waveform_mean=wave.mean((0, 2)).tolist(), waveform_std=wave.std((0, 2)).tolist(),
                physical=dict(center=med.tolist(), scale=np.maximum(q3-q1, 1e-6).tolist()),
                train_windows=n)
    folder = path / method
    folder.mkdir()
    norm_path = folder / "normalization.json"
    write_json(norm_path, norm)
    save_npz(folder / "indices.npz", train=train, val=val)
    x = waveform_transform(torch.from_numpy(x).cuda(), norm)
    d = descriptor_transform(torch.from_numpy(descriptors).cuda(), norm)
    y = labels[ids]
    bank = dict(x=x, d={"physical": d}, y=torch.from_numpy(y).cuda(), normalization_path=norm_path)
    for role, idx in (("train", np.arange(n, dtype=np.int64)),
                      ("val", np.arange(n, len(ids), dtype=np.int64))):
        bank[role] = dict(ids=idx, x=x[idx], d={"physical": d[idx]}, y=y[idx])
    return bank


def paths(out, source, method, seed):
    return out / "models" / source / f"{method}_{seed}"


def load(folder, method, seed, c, bank):
    saved = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    model_method = "equal" if method == "equal" else "physical"
    assert saved["spec"]["seed"] == seed and saved["spec"]["method"] == model_method
    assert saved["spec"]["protocol"] == c
    assert saved["normalization_sha256"] == sha(bank["normalization_path"])
    model = Network(model_method).cuda()
    model.load_state_dict(saved["state_dict"], strict=True)
    return model
