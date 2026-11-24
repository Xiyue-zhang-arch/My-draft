"""CWRU振动膨胀卷积编码器及融合网络的公共初始化。保留模块名和初始化顺序，以兼容已有权重及随机种子。"""

import torch
from torch import nn
from torch.nn import functional as F


class DRN(nn.Module):
    """十层膨胀卷积残差编码器，输出[B, 时间位置, 32]特征序列。"""

    def __init__(self, width=32):
        super().__init__()
        self.input = nn.Conv1d(1, width, 1)
        self.se = nn.Sequential(
            nn.Linear(width, width // 4),
            nn.ReLU(),
            nn.Linear(width // 4, width),
            nn.Sigmoid(),
        )
        self.layers = nn.ModuleList(
            [nn.Conv1d(width, width, 3, dilation=2**i, padding=2**i) for i in range(10)]
        )
        self.output = nn.Conv1d(width, width, 1)

    def forward(self, x):
        """通道注意力后经过膨胀卷积残差块，再压缩时间分辨率。"""
        x = self.input(x)
        x = x * self.se(x.mean(-1)).unsqueeze(-1)
        for layer in self.layers:
            x = x + torch.tanh(layer(x))
        return F.avg_pool1d(self.output(x), 16).transpose(1, 2).float()


class Network(nn.Module):
    def __init__(self, cfg, method):
        super().__init__()
        self.method = method
        w = cfg["width"]
        d = cfg["dropout"]
        self.encoder = DRN(w)
        self.data_norm = nn.LayerNorm(w)
        # 初始化顺序保持一致，使同种子两组模型的编码器及分类器可对应。
        self.classifier = nn.Sequential(
            nn.Linear(w, w), nn.ReLU(), nn.Dropout(d), nn.Linear(w, 4)
        )
        self.physics = nn.Sequential(
            nn.Linear(212, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(64, w),
            nn.LayerNorm(w),
        )
        # 公共构造器保留历史初始化宽度；融合子类立即将输入裁为64维。
        # 不要直接改成64，否则会改变后续模块的随机初始化状态。
        self.gate = nn.Sequential(
            nn.Linear(w * 2 + 1, w), nn.ReLU(), nn.Linear(w, w), nn.Sigmoid()
        )
        self.fused_norm = nn.LayerNorm(w)
        self.dropout = nn.Dropout(d)
