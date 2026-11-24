"""39 维固定物理描述。复用 V3 滤波器、包络和周期统计约定。"""

import math

import torch
from torch.nn import functional as F

from bearing_physics_features import Evidence


BANDS_HZ = ((500, 3000), (3000, 6000), (6000, 12000), (12000, 20000))
FAMILIES = ("FTF", "BSF", "BPFO", "BPFI")
ORDERS = (0.39828, 2.35675, 3.5848, 5.4152)
RELATIVE_TOLERANCE = 0.015
MIN_TOLERANCE_HZ = 2.0
SPECTRUM_EPS = 1e-12
RMS_EPS = 1e-8
POWER_FLOOR = 1e-16


class CompactEvidence(Evidence):
    """输入为 [B, 1, 24000] 振动窗和 [B] 转频，转频单位为 Hz。"""

    dimension = 39

    def __init__(self, cfg):
        if tuple(map(tuple, cfg["bands_hz"])) != BANDS_HZ or cfg["window_samples"] != 24000:
            raise ValueError("V5 采用 V3 的固定四频带和 24000 点窗口")
        configured_orders = dict(zip(cfg["families"], cfg["orders"]))
        if tuple(configured_orders.get(name) for name in FAMILIES) != ORDERS:
            raise ValueError("V5 的 FTF、BSF、BPFO、BPFI 阶次必须与 V3 一致")
        super().__init__({**cfg, "orders": ORDERS, "sham_multipliers": []})
        del self.multipliers

    @torch.no_grad()
    def extract(self, x, rotation_hz):
        if x.ndim != 3 or x.shape[1:] != (1, self.n) or len(x) == 0:
            raise ValueError("振动输入应为非空 [B, 1, 24000] 张量")
        if rotation_hz.shape != x.shape[:1]:
            raise ValueError("每个振动窗口应提供一个 Hz 单位的转频")
        x = x.float()
        rotation_hz = rotation_hz.float()
        if not torch.isfinite(x).all():
            raise ValueError("振动输入含非有限值")
        if not (torch.isfinite(rotation_hz) & (rotation_hz > 0)).all():
            raise ValueError("转频必须为有限正数")

        envelope, centered, power = self._unfloored_envelope_spectrum(x)
        frequencies = rotation_hz[:, None] * self.orders
        concentration = self._spectral_concentration(power, frequencies)
        periodic_frequencies = frequencies[:, 1:]
        recurrence = self._recurrence(centered, periodic_frequencies)
        phase = self._phase(envelope, periodic_frequencies)
        repetition, phase_concentration = self._aggregate_periodicity(
            concentration, recurrence, phase
        )

        envelope_power = envelope.square()
        top_count = math.ceil(0.01 * envelope.shape[-1])
        top_fraction = envelope_power.topk(top_count, dim=-1).values.sum(-1) / (
            envelope_power.sum(-1).clamp_min(POWER_FLOOR)
        )
        positive_power = power[..., 1:].clamp_min(POWER_FLOOR)
        flatness = (positive_power.log().mean(-1).exp() / positive_power.mean(-1)).clamp(0, 1)
        peak_location = self._peak_location(power)
        band_summary = torch.stack(
            (repetition, phase_concentration, top_fraction, flatness, peak_location), dim=-1
        )
        per_band = torch.cat((concentration, band_summary), dim=-1).flatten(1)
        features = torch.cat((per_band, self._rms_log_ratios(envelope_power)), dim=-1)
        if not torch.isfinite(features).all():
            raise ValueError("物理描述含非有限值，请检查振动幅值或转频")
        return features, features.new_zeros(len(x))

    def _unfloored_envelope_spectrum(self, x):
        """V3 包络构造和加窗功率谱，保留抬底前的功率。"""
        analytic = torch.fft.ifft(
            torch.fft.fft(x[:, 0])[:, None, :] * self.filters[None, :, :], dim=-1
        )
        envelope_spectrum = torch.fft.rfft(analytic.abs(), dim=-1)[
            ..., : self.m // 2 + 1
        ].clone()
        envelope_spectrum[..., -1] = envelope_spectrum[..., -1].real * 2
        envelope = torch.fft.irfft(envelope_spectrum, n=self.m, dim=-1) * (self.m / self.n)
        envelope = envelope[..., 75:-75].clamp_min(0)
        centered = envelope - envelope.mean(-1, keepdim=True)
        hann = torch.hann_window(envelope.shape[-1], device=x.device)
        power = torch.fft.rfft(centered * hann, dim=-1).abs().square()
        return envelope, centered, power

    def _spectral_concentration(self, power, frequencies):
        """每个机理用六个谐波邻域的谱点并集，功率分母覆盖全部正频点。"""
        nyquist = self.efs / 2
        positive_power = power[..., 1:]
        bins = torch.arange(1, power.shape[-1], device=power.device) * (
            nyquist / positive_power.shape[-1]
        )
        harmonics = frequencies[..., None] * self.harmonics
        tolerance = (RELATIVE_TOLERANCE * harmonics).clamp_min(MIN_TOLERANCE_HZ)
        inside = (bins - harmonics[..., None]).abs() <= tolerance[..., None]
        valid_harmonics = (harmonics > 0) & (harmonics <= nyquist)
        covered = (inside & valid_harmonics[..., None]).any(dim=-2)
        selected = (positive_power[:, :, None, :] * covered[:, None, :, :]).sum(-1)
        total = positive_power.sum(-1, keepdim=True)
        return torch.where(total > SPECTRUM_EPS, selected / (total + SPECTRUM_EPS), 0)

    @staticmethod
    def _peak_location(power):
        positive_power = power[..., 1:]
        peak_bin = positive_power.argmax(-1) + 1
        return torch.where(
            positive_power.sum(-1) > SPECTRUM_EPS,
            peak_bin / positive_power.shape[-1],
            0,
        )

    def _recurrence(self, centered, frequencies):
        """复用 V3 的线性自相关及一至三倍周期邻域最大值。"""
        length = centered.shape[-1]
        autocorrelation = torch.fft.irfft(
            torch.fft.rfft(centered, n=2 * length, dim=-1).abs().square(),
            n=2 * length,
            dim=-1,
        )[..., :length]
        autocorrelation = autocorrelation / autocorrelation[..., :1].clamp_min(POWER_FLOOR)
        lags = self.efs * torch.arange(1, 4, device=centered.device) / frequencies[..., None]
        offsets = torch.arange(-20, 21, device=centered.device)
        candidates = lags.round()[..., None] + offsets
        valid = (
            (candidates > 0)
            & (candidates < length)
            & ((candidates - lags[..., None]).abs() <= (0.02 * lags).clamp_min(1)[..., None])
        )
        candidates = candidates.clamp(0, length - 1).long()
        indices = candidates[:, None, ...].expand(-1, centered.shape[1], -1, -1, -1)
        values = autocorrelation[:, :, None, None, :].expand(-1, -1, *lags.shape[1:], -1)
        peaks = values.gather(-1, indices).masked_fill(~valid[:, None, ...], -1).amax(-1)
        return torch.where(valid.any(-1)[:, None, ...], peaks, 0).mean(-1)

    def _phase(self, envelope, frequencies):
        """V3 事件检测阈值、相位圆统计和三种频率微调。"""
        median = envelope.median(-1, keepdim=True).values
        mad = (envelope - median).abs().median(-1, keepdim=True).values
        peaks = (envelope >= F.max_pool1d(envelope, 7, stride=1, padding=3)) & (
            envelope > median + 3 * 1.4826 * mad
        )
        peaks[..., :3] = False
        peaks[..., -3:] = False
        count = peaks.sum(-1)
        time = torch.arange(envelope.shape[-1], device=envelope.device) / self.efs
        phase = 2 * math.pi * frequencies[..., None, None] * self.scales[:, None] * time
        events = peaks.float()
        denominator = count[:, :, None, None].clamp_min(1)
        mean_cosine = torch.einsum("bmst,bct->bcms", phase.cos(), events) / denominator
        mean_sine = torch.einsum("bmst,bct->bcms", phase.sin(), events) / denominator
        concentration = (mean_cosine.square() + mean_sine.square()).sqrt().amax(-1)
        return torch.where(count[..., None] >= 5, concentration, 0)

    @staticmethod
    def _aggregate_periodicity(concentration, recurrence, phase):
        mechanism_power = concentration[..., 1:]
        total = mechanism_power.sum(-1, keepdim=True)
        weights = torch.where(total > SPECTRUM_EPS, mechanism_power / (total + SPECTRUM_EPS), 0)
        return (weights * recurrence).sum(-1), (weights * phase).sum(-1)

    @staticmethod
    def _rms_log_ratios(envelope_power):
        rms = envelope_power.mean(-1).sqrt() + RMS_EPS
        return rms[..., :3].log() - rms[..., 3:].log()
