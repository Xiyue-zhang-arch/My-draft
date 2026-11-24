"""CWRU物理引导门控融合模型。32维振动表示与32维物理表示决定逐通道门控，四分类MLP输出logits。"""

import torch
from torch import nn
from vibration_encoder import Network as BaseNetwork


class Network(BaseNetwork):
    def __init__(self, cfg, method="learned_gate"):
        super().__init__(cfg, "learned_gate")
        # 保持既有随机初始化顺序，再裁去历史占位列。实际门控输入为64维。
        first = self.gate[0]
        first.weight = nn.Parameter(
            first.weight[:, : 2 * cfg["width"]].detach().clone()
        )
        first.in_features = 2 * cfg["width"]
        self.variant = "no_quality"

    def forward(self, x, features, disable_physics=False):
        """波形[B,1,T]与描述[B,212]融合后输出[B,4] logits及平均门控值。"""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            seq = self.encoder(x)
        # 沿时间轴汇总均值和标准差，保留响应强度与波动信息。
        data = self.data_norm(seq.mean(1) + (seq.var(1, unbiased=False) + 1e-8).sqrt())
        if disable_physics:
            embedding = self.fused_norm(data)
            gate = torch.zeros(len(x), device=x.device)
        else:
            # 两个32维表示决定逐通道权重；物理表示经门控后残差相加。
            physical = self.physics(features)
            weights = self.gate(torch.cat([data, physical], dim=-1))
            embedding = self.fused_norm(data + weights * physical)
            gate = weights.mean(-1)  # 供观察的平均权重，不代表物理贡献比例。
        return self.classifier(self.dropout(embedding)), gate
