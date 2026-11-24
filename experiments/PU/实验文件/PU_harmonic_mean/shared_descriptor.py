"""Reusable four-harmonic descriptor; legacy describe supplies vibration U."""
import importlib.util
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch

LENGTH = 2560
SPR = 640
ORDERS = (3.054290718038529, 4.945709281961471)

def _power(x):
    taper = torch.hann_window(x.shape[-1], periodic=True, dtype=x.dtype, device=x.device)
    x = x - (x * taper).sum(-1, keepdim=True) / taper.sum()
    result = torch.fft.rfft(x * taper).abs().square()
    result[..., 0] = result[..., -1] = 0
    return result

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



_current_path = Path(__file__).with_name("current_descriptor.py")
if not _current_path.is_file():
    _current_path = Path(__file__).parent / "runtime" / "current_descriptor.py"
_spec = importlib.util.spec_from_file_location("pu_harmonic_current_v2", _current_path)
_current = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_current)


@lru_cache(maxsize=8)
def harmonic_mean(device):
    return HarmonicMean(device)


@torch.no_grad()
def describe_four_harmonic(raw, legacy_describe):
    """Return [batch, 2, 4] with the frozen harmonic experiment definition."""
    pair = legacy_describe(raw).clone()
    pair[:, 1] = _current.describe_current(raw[:, 1])
    pair[:, 0, :2] = harmonic_mean(raw.device)(raw[:, 0])
    return pair
