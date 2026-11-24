"""Train the equally weighted harmonic descriptor on the frozen PU source matrix."""
import argparse
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "PU_source_matrix"))
import matrix as parent
from common import (CONDITIONS, LENGTH, NOISE_SEEDS, ORDERS, SEEDS, SNRS, SPR,
                    descriptor_transform, save_npz, setup, sha, waveform_transform, write_json)
from descriptors import _power
from prepare_multisource import read, v2_descriptors
from train_engine import UPDATES_PER_EPOCH, train_one
from train_support import predict
import torch

REFERENCE = parent.ROOT / "results/PU_source_matrix/main_20261122_a"
METHOD = "mean14"
CASES = (("clean", None, None),) + tuple(
    (f"snr{snr}_{noise_seed}", snr, j) for snr in SNRS for j, noise_seed in enumerate(NOISE_SEEDS))
BAND = (1., 25.)
HALF_WIDTH = .5
GUARD = .55
RADIUS = 3.
RELATIVE_FLOOR = 1e-6


def masks():
    grid = np.fft.rfftfreq(LENGTH, 1/SPR)
    count = int((BAND[1] - HALF_WIDTH) // max(ORDERS))
    harmonics = np.arange(1, count + 1)
    centers = np.array(ORDERS)[:, None] * harmonics
    assert count == 4 and np.all(centers - HALF_WIDTH >= BAND[0])
    delta = np.abs(grid - centers.flatten()[:, None])
    eligible = (grid >= BAND[0]) & (grid <= BAND[1])
    nominal = eligible & (delta <= HALF_WIDTH)
    shared = nominal.sum(0) > 1
    core = nominal & ~shared
    background = eligible & (delta > GUARD).all(0) & (delta <= RADIUS)
    assert (core.sum(1) >= 3).all() and (background.sum(1) >= 5).all()
    assert not (core & background).any() and core.sum(0).max() == 1
    return grid, centers, eligible, core, background, shared


class HarmonicMean:
    def __init__(self, device):
        self.grid, self.centers, eligible, core, background, self.shared = masks()
        self.foreground_bins = [self.grid[row].tolist() for row in core]
        self.background_bins = [self.grid[row].tolist() for row in background]
        self.eligible = torch.from_numpy(eligible).to(device)
        self.core = torch.from_numpy(core).to(device)
        self.background = torch.from_numpy(background).to(device)
        self.analytic = torch.zeros(LENGTH, dtype=torch.float64, device=device)
        self.analytic[0] = self.analytic[LENGTH//2] = 1
        self.analytic[1:LENGTH//2] = 2

    @torch.no_grad()
    def __call__(self, waveform):
        assert waveform.ndim == 2 and waveform.shape[1] == LENGTH
        x = waveform.double()
        x = x - x.mean(1, keepdim=True)
        epsilon = torch.finfo(x.dtype).tiny
        rms = x.square().mean(1, keepdim=True).sqrt()
        x = x / rms.clamp_min(epsilon)
        envelope = torch.fft.ifft(torch.fft.fft(x) * self.analytic).abs()
        power = _power(envelope.square())
        floor = (power[:, self.eligible].mean(1) * RELATIVE_FLOOR).clamp_min(epsilon)
        values = []
        for core, background in zip(self.core, self.background):
            ref = torch.maximum(power[:, background].quantile(.5, dim=1), floor)
            values.append(torch.log1p(power[:, core].mean(1) / ref))
        combined = torch.stack(values, 1).reshape(len(x), *self.centers.shape).mean(2)
        active = (rms[:, 0] > 0) & (envelope.var(1, unbiased=False) > 1e-12)
        result = torch.where(active[:, None], combined, 0).to(waveform.dtype)
        assert torch.isfinite(result).all()
        return result

    def specification(self):
        return dict(harmonics=list(range(1, self.centers.shape[1] + 1)), centers=self.centers.tolist(),
                    foreground=self.foreground_bins, background=self.background_bins,
                    shared_removed=self.grid[self.shared].tolist(),
                    floor_relative=RELATIVE_FLOOR, analysis_range=BAND)


def replaced(raw, fn):
    original = v2_descriptors(raw)
    revised = original.clone()
    revised[:, 0, :2] = fn(raw[:, 0])
    return original, revised


def numerical_check(fn):
    from scipy.signal import hilbert

    rng = np.random.default_rng(17063)
    u = np.arange(LENGTH) / SPR
    carrier = np.cos(2*np.pi*80*u)
    wave = np.stack((np.zeros_like(u), np.ones_like(u), carrier,
                     *((1 + .1*sum(np.cos(2*np.pi*h*r*u) for h in range(1, 5))) * carrier
                       for r in ORDERS), *rng.standard_normal((6, LENGTH))))
    x = wave - wave.mean(1, keepdims=True)
    rms = np.sqrt((x**2).mean(1, keepdims=True))
    x /= np.maximum(rms, np.finfo(float).tiny)
    env = np.abs(hilbert(x))
    z = env**2
    taper = np.hanning(LENGTH + 1)[:-1]
    p = np.abs(np.fft.rfft((z - (z*taper).sum(1, keepdims=True)/taper.sum()) * taper))**2
    p[:, [0, -1]] = 0
    _, _, eligible, core, background, _ = masks()
    floor = np.maximum(p[:, eligible].mean(1)*RELATIVE_FLOOR, np.finfo(float).tiny)
    q = [np.log1p(p[:, a].mean(1)/np.maximum(np.median(p[:, b], axis=1), floor))
         for a, b in zip(core, background)]
    expected = np.column_stack(q).reshape(len(wave), 2, 4).mean(2)
    expected[(rms[:, 0] == 0) | (env.var(1) <= 1e-12)] = 0
    obs = fn(torch.from_numpy(wave).cuda()).cpu().numpy()
    np.testing.assert_allclose(obs, expected, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(obs, fn(torch.from_numpy(wave * 3).cuda()).cpu().numpy(),
                               atol=1e-11, rtol=1e-10)
    assert obs[3, 0] > obs[3, 1] and obs[4, 1] > obs[4, 0]
    assert np.all(obs[:3] == 0)
    return float(np.max(np.abs(obs-expected)))


def verify_reference():
    mark = read(REFERENCE / "RUN_COMPLETE.json")
    assert mark["stage"] == "complete"
    for key, file in (("audit_sha256", "report/AUDIT_COMPLETE.json"), ("config_sha256", "config.json")):
        assert mark[key] == sha(REFERENCE / file), file
    return mark


def source_bank(out, condition, split, raw, labels, fn):
    reference = REFERENCE / "bank" / condition / "v2"
    with np.load(reference / "indices.npz") as z:
        train, val = z["train"], z["val"]
    for inds, name in ((train, "train"), (val, "val")):
        np.testing.assert_array_equal(np.sort(inds), np.sort(split[name]))
    old, new = [], []
    for ids in np.array_split(np.r_[train, val], 36):
        before, after = replaced(torch.from_numpy(raw[ids]).cuda(), fn)
        old.append(before.cpu().numpy())
        new.append(after.cpu().numpy())
    original, revised = np.concatenate(old), np.concatenate(new)
    np.testing.assert_array_equal(original[:, 0, 2:], revised[:, 0, 2:])
    np.testing.assert_array_equal(original[:, 1], revised[:, 1])
    assert not np.array_equal(original[:, 0, :2], revised[:, 0, :2])
    path = out / "bank" / condition
    path.mkdir(parents=True)
    bank = parent.make_bank(path, METHOD, raw, labels, train, val, revised)
    new_stats = read(bank["normalization_path"])
    old_stats = read(reference / "normalization.json")
    for field in ("waveform_mean", "waveform_std", "train_windows"):
        assert new_stats[field] == old_stats[field]
    for field in ("center", "scale"):
        a, b = np.array(new_stats["physical"][field]), np.array(old_stats["physical"][field])
        np.testing.assert_array_equal(a[0, 2:], b[0, 2:])
        np.testing.assert_array_equal(a[1], b[1])
    np.save(path / "descriptors.npy", revised, allow_pickle=False)
    return bank


def train(out, state, banks, preflight):
    for condition in CONDITIONS:
        (out / "models" / condition).mkdir(parents=True)
        for seed in SEEDS[:1] if preflight else SEEDS:
            bank = banks[condition]
            folder = parent.paths(out, condition, METHOD, seed)
            spec = dict(model_id=f"{METHOD}_{seed}", protocol=condition, method="physical", seed=seed)
            print("source", condition, "seed", seed, flush=True)
            info = train_one(spec, bank, folder, functional=preflight)
            traces = read(folder / "training_trace.json")
            ref = parent.paths(REFERENCE, condition, "v2", seed)
            assert traces == read(ref / "training_trace.json")[:len(traces)]
            model = parent.load(folder, METHOD, seed, condition, bank)
            validation = bank["val"]
            observed = predict(model, validation["x"], validation["d"]["physical"], validation["y"])
            with np.load(folder / "validation.npz") as z:
                np.testing.assert_array_equal(observed["logits"], z["logits"])
            state["models"].append({**info, "source": condition, "method": METHOD})
            write_json(out / "config.json", state)


def evaluate(out, state, banks, raw, labels, splits, positions, fn):
    assert len(state["models"]) == 20
    models = {}
    for info in state["models"]:
        condition, seed = info["source"], info["seed"]
        folder = parent.paths(out, condition, METHOD, seed)
        assert not info["functional"] and info["epochs"] == 50 and info["updates"] == 2200
        assert sha(folder / "model.pt") == info["artifacts_sha256"]["model.pt"]
        models[condition, seed] = parent.load(folder, METHOD, seed, condition, banks[condition])
    write_json(out / "evaluation_lock.json", state["models"])
    state["stage"] = "evaluating"
    write_json(out / "config.json", state)
    norm = {c: read(banks[c]["normalization_path"]) for c in CONDITIONS}
    noise = np.load(parent.DATA / "vibration_noise_templates.npy", mmap_mode="r")
    for target in CONDITIONS:
        ids = splits[target]["test"]
        for case, snr, replicate in CASES:
            wave = torch.from_numpy(raw[ids]).cuda()
            if snr is not None:
                wave[:, 0] += torch.from_numpy(noise[[positions[i] for i in ids], replicate]).cuda() * 10**(-snr/20)
            _, revised = replaced(wave, fn)
            for source in CONDITIONS:
                if source == target:
                    continue
                x = waveform_transform(wave, norm[source])
                d = descriptor_transform(revised, norm[source])
                for seed in SEEDS:
                    logits = predict(models[source, seed], x, d, labels[ids])["logits"]
                    folder = out / "predictions" / source / target / METHOD / str(seed)
                    folder.mkdir(parents=True, exist_ok=True)
                    save_npz(folder / (case + ".npz"), logits=logits, window_index=ids, y=labels[ids])
                    state["predictions"] += 1
        write_json(out / "config.json", state)
        print(target, state["predictions"], "/ 1260", flush=True)
    assert state["predictions"] == 1260


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(stage="initializing", preflight=args.preflight, models=[], predictions=0)
    try:
        setup()
        fn = HarmonicMean("cuda")
        state["independent_error"] = numerical_check(fn)
        state["reference"] = verify_reference()
        _, labels, splits, positions = parent.metadata()
        raw = np.load(parent.DATA / "signals.npy", mmap_mode="r")
        write_json(out / "frequency_masks.json", fn.specification())
        (out / "source").mkdir()
        for name in ("run_mean.py", "audit_mean.py", "PROTOCOL.md"):
            shutil.copyfile(HERE / name, out / "source" / name)
        UPDATES_PER_EPOCH.update({condition: 44 for condition in CONDITIONS})
        banks = {c: source_bank(out, c, splits[c], raw, labels, fn) for c in CONDITIONS}
        state["stage"] = "training"
        write_json(out / "config.json", state)
        train(out, state, banks, args.preflight)
        if not args.preflight:
            evaluate(out, state, banks, raw, labels, splits, positions, fn)
        state["stage"] = "preflight_passed" if args.preflight else "predictions_complete"
        write_json(out / "config.json", state)
    except BaseException:
        state.update(stage="failed", error=traceback.format_exc())
        write_json(out / "config.json", state)
        raise


if __name__ == "__main__":
    main()
