"""CWRU注意力加权混合模型适配实现。结合小波包统计、卷积与双向GRU，具体适配边界见HYBRID_PROTOCOL.md。"""

import torch
from torch import nn
from torch.nn import functional as F
import pywt
from wdcnn_drsn_pg_models import SameConv


class WPTStatistics(nn.Module):
    """三级db4小波包的八项能量比例，加峭度、偏度和峰值因子，共11维。"""

    def __init__(self):
        super().__init__()
        wavelet = pywt.Wavelet("db4")
        self.register_buffer(
            "filters",
            torch.tensor(
                [wavelet.dec_lo[::-1], wavelet.dec_hi[::-1]], dtype=torch.float32
            )[:, None],
        )
        self.register_buffer("frequency_order", torch.tensor([0, 1, 3, 2, 6, 7, 5, 4]))

    @torch.no_grad()
    def forward(self, x):
        batch = len(x)
        nodes = x
        for level in range(3):
            n = nodes.shape[-1]
            index = torch.arange(-6, n + 6 + n % 2, device=x.device) % (2 * n)
            index = torch.where(index < n, index, 2 * n - 1 - index)
            nodes = F.conv1d(
                nodes.index_select(-1, index), self.filters, stride=2
            ).reshape(-1, 1, (n + 7) // 2)
        energy = (
            nodes.reshape(batch, 8, -1)
            .square()
            .sum(-1)
            .index_select(1, self.frequency_order)
        )
        energy = energy / (energy.sum(-1, keepdim=True) + 1e-12)
        z = x[:, 0] - x[:, 0].mean(-1, keepdim=True)
        var = z.square().mean(-1).clamp_min(1e-12)
        kurt = z.pow(4).mean(-1) / var.square()
        skew = z.pow(3).mean(-1) / var.pow(1.5)
        crest = x[:, 0].abs().amax(-1) / (x[:, 0].square().mean(-1).sqrt() + 1e-12)
        return torch.cat([energy, kurt[:, None], skew[:, None], crest[:, None]], -1)


class AttentionHybridAdapted(nn.Module):
    """物理统计作查询，卷积及双向GRU序列作键和值，最后门控融合。"""

    def __init__(self):
        super().__init__()
        self.features = WPTStatistics()
        layers = []
        a = 1
        for b, k, s in [(32, 31, 8), (64, 9, 4), (64, 5, 4)]:
            layers += [SameConv(a, b, k, stride=s), nn.BatchNorm1d(b), nn.ReLU()]
            a = b
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(64, 32, num_layers=2, bidirectional=True, batch_first=True)
        self.physical = nn.Linear(11, 64)
        # 按适配方案直接拼接四个注意力头，不增加输出投影。
        self.query = nn.Linear(64, 64, bias=False)
        self.key = nn.Linear(64, 64, bias=False)
        self.value = nn.Linear(64, 64, bias=False)
        self.gate = nn.Linear(64, 1)
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x, rpm=None):
        physical = self.physical(self.features(x))
        sequence, _ = self.gru(self.conv(x).transpose(1, 2))
        batch, length, _ = sequence.shape
        # 11维物理统计映射为64维查询；四个头各16维。
        q = self.query(physical).reshape(batch, 4, 1, 16)
        k = self.key(sequence).reshape(batch, length, 4, 16).transpose(1, 2)
        v = self.value(sequence).reshape(batch, length, 4, 16).transpose(1, 2)
        aligned = F.scaled_dot_product_attention(q, k, v).reshape(batch, 64)
        alpha = self.gate(physical).sigmoid()
        fused = alpha * physical + (1 - alpha) * aligned
        return self.classifier(fused), {}
