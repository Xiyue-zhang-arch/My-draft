"""Scheme one: full-band squared-envelope R(3), envelope recurrence U(3)."""
import math

import torch
from torch import nn


class Evidence6(nn.Module):
    dimension = 6

    def __init__(self, cfg):
        super().__init__()
        self.n, self.fs = 24000, 48000.
        if cfg['window_samples'] != self.n:
            raise ValueError('Expected 24000 samples')
        self.register_buffer('center', torch.zeros(6))
        self.register_buffer('scale', torch.ones(6))
        self.register_buffer('orders', torch.tensor([3.5848, 5.4152, 2.35675]))
        h = torch.zeros(self.n)
        h[0] = h[self.n//2] = 1
        h[1:self.n//2] = 2
        self.register_buffer('hilbert_mask', h)
        self.register_buffer('bins_hz', torch.fft.rfftfreq(self.n, 1/self.fs))

    def standardize(self, features):
        return ((features-self.center)/self.scale).clamp(-8, 8)

    @torch.no_grad()
    def extract(self, x, rotation_hz):
        if x.ndim != 3 or x.shape[1:] != (1, self.n) or rotation_hz.shape != x.shape[:1]:
            raise ValueError('Expected nonempty [B,1,24000], [B]')
        if not len(x) or not torch.isfinite(x).all() or not (torch.isfinite(rotation_hz) & (rotation_hz > 0)).all():
            raise ValueError('Invalid physical input')
        raw = x[:, 0].float()
        raw = raw-raw.mean(-1, keepdim=True)
        envelope = torch.fft.ifft(torch.fft.fft(raw)*self.hilbert_mask).abs()
        q = envelope.square()
        power = torch.fft.rfft(q-q.mean(-1, keepdim=True)).abs().square()/(2*self.n)
        power[:, 0] = power[:, -1] = 0
        base = rotation_hz.float()[:, None]*self.orders
        harmonics = base[..., None]*torch.arange(1, 7, device=x.device)
        half = (.015*harmonics).clamp_min(2.)
        if (harmonics+half >= self.fs/2).any():
            raise ValueError('R harmonics exceed supported spectrum')
        mask = ((self.bins_hz >= (harmonics-half)[..., None]) &
                (self.bins_hz < (harmonics+half)[..., None])).any(-2)
        selected = (power[:, None]*mask).sum(-1)
        total = power.sum(-1, keepdim=True)
        r = torch.where(total > 0, selected/total.clamp_min(1e-30), 0)
        e = envelope-envelope.mean(-1, keepdim=True)
        ac = torch.fft.irfft(torch.fft.rfft(e, n=2*self.n).abs().square(), n=2*self.n)[:, :self.n]
        ac = ac/ac[:, :1].clamp_min(1e-30)
        periods = base.clone()
        periods[:, 2] *= 2
        lags = self.fs*torch.arange(1, 4, device=x.device)/periods[..., None]
        width = (.02*lags).clamp_min(1.)
        offsets = torch.arange(-math.ceil(float(width.max()))-1,
                               math.ceil(float(width.max()))+2, device=x.device)
        indices = lags.round()[..., None]+offsets
        valid = ((indices-lags[..., None]).abs() <= width[..., None]) & (indices > 0) & (indices < self.n)
        if not valid.any(-1).all():
            raise ValueError('Window does not cover required periods')
        values = ac[:, None, None].expand(-1, 3, 3, -1).gather(-1, indices.clamp(0, self.n-1).long())
        u = values.masked_fill(~valid, -torch.inf).amax(-1).mean(-1)
        result = torch.cat([r, u], -1)
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite physical descriptor')
        return result, result.new_zeros(len(x))
