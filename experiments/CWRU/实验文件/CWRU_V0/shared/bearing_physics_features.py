"""CWRU 48 kHz振动信号的固定物理描述提取器。

输入为[B, 1, 24000]观测波形和[B]转频，转频单位为Hz。
四个频带分别计算六类候选周期的八项描述和五项形状统计，
共4 × (6 × 8 + 5) = 212维。输出不使用类别标签。
描述经训练集拟合的标准化参数后进入物理MLP，再通过门控引导融合。
固定描述参与融合不等于已证明物理因果关系；门控值也不是贡献率。
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import signal


class Evidence(nn.Module):
    """固定物理描述模块；缓冲区随模型设备迁移，提取过程不反向传播梯度。"""

    dimension = 212

    def __init__(self, cfg):
        super().__init__()
        self.n = cfg["window_samples"]
        self.efs = 3000
        self.m = self.n // 16
        freq = np.fft.fftfreq(self.n, 1 / 48000)
        filters = []
        for lo, hi in cfg["bands_hz"]:
            sos = signal.butter(4, [lo, hi], fs=48000, btype="bandpass", output="sos")
            _, h = signal.sosfreqz(sos, worN=abs(freq), fs=48000)
            analytic = np.where(freq > 0, 2.0, 0.0)
            filters.append(abs(h) ** 2 * analytic)
        self.register_buffer(
            "filters", torch.tensor(np.array(filters), dtype=torch.float32)
        )
        self.register_buffer("orders", torch.tensor(cfg["orders"], dtype=torch.float32))
        self.register_buffer(
            "multipliers",
            torch.tensor([1.0] + cfg["sham_multipliers"], dtype=torch.float32),
        )
        self.register_buffer("harmonics", torch.arange(1, 7, dtype=torch.float32))
        self.register_buffer("scales", torch.tensor([0.985, 1.0, 1.015]))
        self.register_buffer("center", torch.zeros(self.dimension))
        self.register_buffer("scale", torch.ones(self.dimension))

    @torch.no_grad()
    def extract(self, x, rotation_hz):
        """依次提取谐波、周期重复、冲击相位和形状描述，输出[B,212]。"""
        envelope, centered_envelope, power_spectrum, envelope_length, duration = (
            self._envelope_spectrum(x)
        )
        candidate_frequencies, harmonic_strength, coverage, harmonic_contrast = (
            self._harmonic_features(power_spectrum, envelope_length, rotation_hz)
        )
        recurrence, recurrence_contrast = self._recurrence_features(
            centered_envelope, envelope_length, candidate_frequencies
        )
        phase_concentration, phase_contrast, event_rate = self._impact_features(
            envelope, envelope_length, duration, candidate_frequencies
        )
        periodic = torch.stack(
            [
                harmonic_strength[..., 0],
                coverage[..., 0],
                harmonic_contrast,
                recurrence[..., 0],
                recurrence_contrast,
                phase_concentration[..., 0],
                phase_contrast,
                event_rate,
            ],
            dim=-1,
        )
        # 保留原有频带内的特征排列，保证已训练物理MLP的输入顺序一致。
        shape = self._shape_features(
            envelope, centered_envelope, power_spectrum, envelope_length
        )
        descriptor = torch.cat([periodic.flatten(2), shape], dim=-1).flatten(1)
        assert torch.isfinite(descriptor).all()
        # 保留双返回值接口；该占位值不参与门控或分类。
        unused = torch.zeros(len(x), device=x.device)
        return descriptor, unused

    def _envelope_spectrum(self, x):
        """带通解析信号取幅值获得包络，并计算去均值加窗后的包络功率谱。"""
        x = x[:, 0].float()
        analytic_signal = torch.fft.ifft(
            torch.fft.fft(x)[:, None, :] * self.filters[None, :, :], dim=-1
        )
        envelope = analytic_signal.abs()
        # 包络由48 kHz重采样到3 kHz：截断频谱后逆变换，抑制降采样混叠。
        envelope_spectrum = torch.fft.rfft(envelope, dim=-1)[
            ..., : self.m // 2 + 1
        ].clone()
        envelope_spectrum[..., -1] = envelope_spectrum[..., -1].real * 2
        envelope = torch.fft.irfft(envelope_spectrum, n=self.m, dim=-1) * (
            self.m / self.n
        )
        # 两端各裁去25 ms，减少带通和重采样边界影响。
        envelope = envelope[..., 75:-75].clamp_min(0)
        envelope_length = envelope.shape[-1]
        duration = envelope_length / self.efs
        centered_envelope = envelope - envelope.mean(-1, keepdim=True)
        hann = torch.hann_window(envelope_length, device=x.device)
        power_spectrum = (
            torch.fft.rfft(centered_envelope * hann, dim=-1)
            .abs()
            .square()
            .clamp_min(1e-16)
        )
        return envelope, centered_envelope, power_spectrum, envelope_length, duration

    def _harmonic_features(self, power_spectrum, envelope_length, rotation_hz):
        """计算六个谐波的峰背景比、覆盖率以及相对偏移频率的差值。"""
        frequency_resolution = self.efs / envelope_length
        # 倍率第0项是真实候选阶次，其余6项为偏移频率参照。
        candidate_frequencies = (
            rotation_hz[:, None, None]
            * self.orders[None, :, None]
            * self.multipliers[None, None, :]
        )
        harmonic_frequencies = candidate_frequencies[..., None] * self.harmonics
        bins = harmonic_frequencies / frequency_resolution
        offsets = torch.arange(-80, 81, device=power_spectrum.device)
        frequency_indices = (bins.round()[..., None] + offsets).long()
        valid = (frequency_indices >= 0) & (
            frequency_indices < power_spectrum.shape[-1]
        )
        frequency_indices = frequency_indices.clamp(0, power_spectrum.shape[-1] - 1)
        # 张量轴依次为批量、4频带、6候选周期、7频率倍率、6谐波、局部频点。
        local_power = (
            power_spectrum[:, :, None, None, None, :]
            .expand(-1, -1, 6, 7, 6, -1)
            .gather(-1, frequency_indices[:, None].expand(-1, 4, -1, -1, -1, -1))
        )
        delta = abs(
            frequency_indices * frequency_resolution - harmonic_frequencies[..., None]
        )
        frequency_tolerance = torch.maximum(
            torch.full_like(harmonic_frequencies, 2.0), 0.015 * harmonic_frequencies
        )
        target = (delta <= frequency_tolerance[..., None]) & valid
        background_mask = (
            (
                delta
                <= torch.maximum(
                    torch.full_like(harmonic_frequencies, 12.0),
                    0.15 * harmonic_frequencies,
                )[..., None]
            )
            & (delta > frequency_tolerance[..., None] + 2)
            & valid
        )
        peak = local_power.masked_fill(~target[:, None], 0).amax(-1).clamp_min(1e-16)
        # 局部背景采用几何均值；目标频率与背景区间分开，避免峰值污染背景。
        background = (local_power.log() * background_mask[:, None]).sum(
            -1
        ) / background_mask.sum(-1)[:, None].clamp_min(1)
        log_peak_background_ratio = (peak.log() - background) / math.log(10)
        harmonic_strength = log_peak_background_ratio.mean(-1)
        coverage = (log_peak_background_ratio > math.log10(3)).float().mean(-1)
        harmonic_contrast = harmonic_strength[..., 0] - harmonic_strength[..., 1:].amax(
            -1
        )
        return candidate_frequencies, harmonic_strength, coverage, harmonic_contrast

    def _recurrence_features(
        self, centered_envelope, envelope_length, candidate_frequencies
    ):
        """检查候选周期的一至三倍延迟是否出现重复响应。"""
        # 补零到两倍长度得到线性自相关，再按零延迟值归一化。
        autocorrelation = torch.fft.irfft(
            torch.fft.rfft(centered_envelope, n=2 * envelope_length, dim=-1)
            .abs()
            .square(),
            n=2 * envelope_length,
            dim=-1,
        )[..., :envelope_length]
        autocorrelation = autocorrelation / autocorrelation[..., :1].clamp_min(1e-16)
        period_lags = (
            self.efs
            * torch.arange(1, 4, device=centered_envelope.device)
            / candidate_frequencies[..., None]
        )
        off = torch.arange(-20, 21, device=centered_envelope.device)
        lag_indices = (period_lags.round()[..., None] + off).long()
        valid = (
            (lag_indices > 0)
            & (lag_indices < envelope_length)
            & (
                abs(lag_indices - period_lags[..., None])
                <= torch.maximum(0.02 * period_lags, torch.ones_like(period_lags))[
                    ..., None
                ]
            )
        )
        lag_indices = lag_indices.clamp(0, envelope_length - 1)
        local_autocorrelation = (
            autocorrelation[:, :, None, None, None, :]
            .expand(-1, -1, 6, 7, 3, -1)
            .gather(-1, lag_indices[:, None].expand(-1, 4, -1, -1, -1, -1))
        )
        recurrence = (
            local_autocorrelation.masked_fill(~valid[:, None], -1).amax(-1).mean(-1)
        )
        recurrence_contrast = recurrence[..., 0] - recurrence[..., 1:].amax(-1)
        return recurrence, recurrence_contrast

    def _impact_features(
        self, envelope, envelope_length, duration, candidate_frequencies
    ):
        """计算冲击相位集中度、偏移频率差值和每周期事件率。"""
        median = envelope.median(-1, keepdim=True).values
        median_deviation = (envelope - median).abs().median(-1, keepdim=True).values
        # 用局部极大值和稳健幅值阈值检测冲击事件。
        peaks = (envelope >= F.max_pool1d(envelope, 7, stride=1, padding=3)) & (
            envelope > median + 3 * 1.4826 * median_deviation
        )
        peaks[..., :3] = False
        peaks[..., -3:] = False
        count = peaks.sum(-1)
        time = torch.arange(envelope_length, device=envelope.device) / self.efs
        phase = (
            2
            * math.pi
            * candidate_frequencies[..., None, None]
            * self.scales[None, None, None, :, None]
            * time
        )
        mean_cosine = torch.einsum(
            "bfmst,bct->bcfms", phase.cos(), peaks.float()
        ) / count[:, :, None, None, None].clamp_min(1)
        mean_sine = torch.einsum(
            "bfmst,bct->bcfms", phase.sin(), peaks.float()
        ) / count[:, :, None, None, None].clamp_min(1)
        phase_concentration = (
            (mean_cosine.square() + mean_sine.square()).sqrt().amax(-1)
        )
        # 冲击不足5个时不把相位集中度视为可靠周期描述。
        enough = count >= 5
        phase_concentration = phase_concentration * enough[:, :, None, None]
        phase_contrast = phase_concentration[..., 0] - phase_concentration[
            ..., 1:
        ].amax(-1)
        event_rate = (
            count[:, :, None] / (duration * candidate_frequencies[:, None, :, 0])
        ).clamp(max=3)
        return phase_concentration, phase_contrast, event_rate

    def _shape_features(
        self, envelope, centered_envelope, power_spectrum, envelope_length
    ):
        """每个频带计算五项能量及波形形状统计，补充周期描述。"""
        power = envelope.square()
        top_energy_fraction = power.topk(
            max(1, math.ceil(0.01 * envelope_length)), dim=-1
        ).values.sum(-1) / power.sum(-1).clamp_min(1e-16)
        kurtosis = centered_envelope.pow(4).mean(-1) / (
            centered_envelope.square().mean(-1).square() + 1e-16
        )
        crest = envelope.amax(-1) / (envelope.square().mean(-1).sqrt() + 1e-8)
        rms = envelope.square().mean(-1).sqrt()
        relative = (rms / (rms.sum(-1, keepdim=True) + 1e-8)).clamp_min(1e-8).log()
        spectral_flatness = (
            power_spectrum[..., 1:].log().mean(-1).exp()
            / power_spectrum[..., 1:].mean(-1)
        ).clamp(0, 1)
        shape = torch.stack(
            [
                top_energy_fraction,
                kurtosis.log1p(),
                crest.log1p(),
                relative,
                spectral_flatness,
            ],
            dim=-1,
        )
        return shape

    def standardize(self, features):
        """应用训练集拟合的稳健标准化，并将范围裁剪至[-8,8]。"""
        return ((features - self.center) / self.scale).clamp(-8, 8)
