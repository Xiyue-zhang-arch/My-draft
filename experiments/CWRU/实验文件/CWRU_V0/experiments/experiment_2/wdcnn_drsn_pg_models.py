"""CWRU比较模型WDCNN、DRSN-CW和PG-RDGFN适配实现。未公开参数的适配选择见PROTOCOL.md，不视为官方精确复现。"""

import torch
from torch import nn
from torch.nn import functional as F


class SameConv(nn.Conv1d):
    """根据输入长度和步长动态补零，保持same卷积的长度约定。"""

    def forward(self, x):
        length = x.shape[-1]
        total = max(
            0,
            ((length + self.stride[0] - 1) // self.stride[0] - 1) * self.stride[0]
            + self.kernel_size[0]
            - length,
        )
        return super().forward(F.pad(x, (total // 2, total - total // 2)))


class WDCNN(nn.Module):
    """宽首层卷积基线，分类头适配24000采样点输入和四个类别。"""

    def __init__(self):
        super().__init__()
        layers = []
        for a, b, k, s, same in [
            (1, 16, 64, 16, True),
            (16, 32, 3, 1, True),
            (32, 64, 3, 1, True),
            (64, 64, 3, 1, True),
            (64, 64, 3, 1, False),
        ]:
            conv = SameConv(a, b, k, stride=s) if same else nn.Conv1d(a, b, k, stride=s)
            layers += [conv, nn.BatchNorm1d(b), nn.ReLU(), nn.MaxPool1d(2)]
        self.encoder = nn.Sequential(*layers)
        # 24000 samples -> 45 positions. Only input length and class count change.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 45, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 4),
        )

    def forward(self, x, rpm=None):
        return self.head(self.encoder(x)), {}


class ShrinkageUnit(nn.Module):
    """通道自适应软阈值残差单元，抑制绝对值低于阈值的响应。"""

    def __init__(self, a, b, stride):
        super().__init__()
        self.stride, self.channels = stride, b
        self.body = nn.Sequential(
            nn.BatchNorm1d(a),
            nn.ReLU(),
            SameConv(a, b, 3, stride=stride),
            nn.BatchNorm1d(b),
            nn.ReLU(),
            SameConv(b, b, 3),
        )
        self.threshold = nn.Sequential(
            nn.Linear(b, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Linear(b, b), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.body(x)
        mean = z.abs().mean(-1)
        threshold = (mean * self.threshold(mean)).unsqueeze(-1)
        z = z.sign() * F.relu(z.abs() - threshold)
        # Author implementation uses average pooling with kernel=1 and stride=2.
        identity = x[..., :: self.stride]
        extra = self.channels - identity.shape[1]
        identity = F.pad(identity, (0, 0, extra // 2, extra - extra // 2))
        return z + identity


class DRSNCW(nn.Module):
    """由通道软阈值残差单元组成的四分类基线。"""

    def __init__(self):
        super().__init__()
        layers = [SameConv(1, 4, 3, stride=2)]
        a = 4
        for b in [4, 8, 16]:
            for j in range(4):
                layers.append(ShrinkageUnit(a, b, 2 if j == 0 else 1))
                a = b
        self.encoder = nn.Sequential(
            *layers, nn.BatchNorm1d(16), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Linear(16, 4)

    def forward(self, x, rpm=None):
        return self.head(self.encoder(x).squeeze(-1)), {}


class PGFrontEnd(nn.Module):
    """提取包络谱与九维物理描述，标准化参数仅用训练样本拟合。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(9))
        self.register_buffer("scale", torch.ones(9))
        self.register_buffer("quantiles", torch.tensor([0.0, 1.0]))
        self.register_buffer("orders", torch.tensor([0.39828, 2.35675, 3.5848, 5.4152]))

    @torch.no_grad()
    def extract(self, x, rpm):
        n = x.shape[-1]
        freq = torch.fft.rfftfreq(n, 1 / 48000, device=x.device)
        # 固定3至12 kHz共振带，不用验证或测试标签挑选频带。
        mask = ((freq >= 3000) & (freq <= 12000)).float()
        positive = torch.fft.rfft(x[:, 0]) * mask
        analytic = torch.zeros((len(x), n), device=x.device, dtype=torch.complex64)
        analytic[:, 1 : n // 2] = 2 * positive[:, 1 : n // 2]
        analytic[:, 0] = positive[:, 0]
        analytic[:, n // 2] = positive[:, n // 2]
        env = torch.fft.ifft(analytic).abs()
        env = env - env.mean(-1, keepdim=True)
        spectrum = torch.fft.rfft(env).abs()
        spectrum[:, 0] = 0
        prob = spectrum / (spectrum.sum(-1, keepdim=True) + 1e-8)
        centers = (
            rpm[:, None, None]
            / 60
            * self.orders[None, :, None]
            * torch.arange(1, 4, device=x.device)[None, None, :]
        )
        bins = (freq[None, None, None, :] - centers[:, :, :, None]).abs() <= 5.0
        ratios = (prob[:, None, :] * bins.any(2)).sum(-1)
        maximum = ratios.max(-1).values
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(-1) / torch.log(
            torch.tensor(spectrum.shape[-1], device=x.device)
        )
        peak = spectrum.argmax(-1) / (spectrum.shape[-1] - 1)
        h = torch.cat(
            [
                ratios,
                maximum[:, None],
                (ratios[:, 3] / (ratios[:, 2] + 1e-8))[:, None],
                (ratios[:, 1] / (ratios[:, 2] + 1e-8))[:, None],
                entropy[:, None],
                peak[:, None],
            ],
            -1,
        )
        # 包络谱分支对每个样本独立归一化。
        e = (spectrum - spectrum.mean(-1, keepdim=True)) / (
            spectrum.std(-1, unbiased=False, keepdim=True) + 1e-8
        )
        return e[:, None], h

    @torch.no_grad()
    def fit(self, x, rpm):
        """只对训练数据拟合物理描述的标准化与置信度分位数。"""
        hs = []
        for j in range(0, len(x), 32):
            hs.append(self.extract(x[j : j + 32], rpm[j : j + 32])[1])
        h = torch.cat(hs)
        self.mean.copy_(h.mean(0))
        self.scale.copy_(h.std(0, unbiased=False).clamp_min(1e-8))
        self.quantiles.copy_(
            torch.quantile(h[:, 4], torch.tensor([0.1, 0.95], device=x.device))
        )

    def forward(self, x, rpm):
        e, h = self.extract(x, rpm)
        confidence = (
            (h[:, 4] - self.quantiles[0])
            / (self.quantiles[1] - self.quantiles[0] + 1e-8)
        ).clamp(0, 1)
        return e, (h - self.mean) / self.scale, confidence


def pg_encoder():
    """构造PG适配模型的信号分支编码器。"""
    layers = []
    a = 1
    for b, k in zip([16, 32, 64, 128], [11, 9, 7, 5]):
        layers += [
            SameConv(a, b, k, stride=2),
            nn.BatchNorm1d(b),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ]
        a = b
    return nn.Sequential(
        *layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 128)
    )


class PGRDGFNAdapted(nn.Module):
    """原信号、包络谱及物理描述联合控制标量融合门控的适配模型。"""

    def __init__(self):
        super().__init__()
        self.front = PGFrontEnd()
        self.raw = pg_encoder()
        self.envelope = pg_encoder()
        self.physics = nn.Sequential(nn.Linear(9, 32), nn.ReLU())
        self.gate = nn.Sequential(
            nn.Linear(288, 96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 1),
            nn.Sigmoid(),
        )

        def classifier():
            return nn.Sequential(
                nn.Linear(128, 96), nn.ReLU(), nn.Dropout(0.25), nn.Linear(96, 4)
            )

        self.classifier = classifier()
        self.auxiliary = classifier()

    def forward(self, x, rpm):
        e, h, c = self.front(x, rpm)
        zr = self.raw(x)
        ze = self.envelope(e)
        zp = self.physics(h)
        g = self.gate(torch.cat([zr, ze, zp], -1)).squeeze(-1)
        return self.classifier(zr + 0.08 * g[:, None] * ze), dict(
            raw=self.auxiliary(zr), gate=g, confidence=c
        )


FACTORIES = {"WDCNN": WDCNN, "DRSN_CW": DRSNCW, "PG_RDGFN_adapted": PGRDGFNAdapted}


def objective(logits, extra, labels):
    """基础损失为标签平滑CE；PG适配模型额外保留辅助CE和门控约束。"""
    ce = F.cross_entropy(logits, labels, label_smoothing=0.05)
    if extra:
        ce = (
            ce
            + 0.30 * F.cross_entropy(extra["raw"], labels, label_smoothing=0.05)
            + 0.05 * F.mse_loss(extra["gate"], extra["confidence"])
        )
    return ce
