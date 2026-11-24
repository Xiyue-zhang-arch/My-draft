"""V4 四频带定义。复用原物理描述计算，仅替换滤波器组。"""

import torch
import numpy as np
from scipy import signal

from bearing_physics_features import Evidence


BANDS_HZ = ((0, 3000), (3000, 6000), (6000, 12000), (12000, 24000))
SAMPLE_RATE_HZ = 48000


class FullFrequencyEvidence(Evidence):
    """两端分别采用低通、高通。中间频带保留原带通计算。"""

    def __init__(self, cfg):
        if tuple(map(tuple, cfg['bands_hz'])) != BANDS_HZ:
            raise ValueError('V4 requires the fixed full-frequency bands')

        # 原提取器生成中间两个带通响应并初始化所有特征缓冲区。
        super().__init__({**cfg, 'bands_hz': BANDS_HZ[1:-1]})
        frequencies = np.fft.fftfreq(self.n, 1 / SAMPLE_RATE_HZ)
        # 沿用严格正频率掩码，直流和奈奎斯特点系数为零。
        analytic = np.where(frequencies > 0, 2.0, 0.0)

        def edge_response(cutoff, kind):
            sos = signal.butter(4, cutoff, fs=SAMPLE_RATE_HZ, btype=kind, output='sos')
            _, response = signal.sosfreqz(sos, worN=abs(frequencies), fs=SAMPLE_RATE_HZ)
            return torch.tensor(abs(response) ** 2 * analytic, dtype=self.filters.dtype)[None]

        self.filters = torch.cat([
            edge_response(BANDS_HZ[0][1], 'lowpass'),
            self.filters,
            edge_response(BANDS_HZ[-1][0], 'highpass'),
        ])


def bind_v4_physics(engine):
    """在独立 V4 进程中绑定滤波器，确保检查点记载实际频带。"""
    runtime = engine.R
    if runtime.Evidence not in (Evidence, FullFrequencyEvidence):
        raise RuntimeError('Unexpected physics extractor before binding V4')
    runtime.U.CFG = {**runtime.U.CFG, 'bands_hz': [list(band) for band in BANDS_HZ]}
    runtime.Evidence = FullFrequencyEvidence
