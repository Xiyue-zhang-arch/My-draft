"""CWRU数据单分支对照。使用同一振动编码器和四分类MLP，移除整个物理融合模块。"""

import torch
from vibration_encoder import Network as BaseNetwork


class Network(BaseNetwork):
    def __init__(self, cfg):
        super().__init__(cfg, "data_only")
        del self.physics, self.gate, self.fused_norm
        self.variant = "data_only"

    def forward(self, x):
        """仅用振动波形计算四类logits；零占位返回值用于统一评估接口。"""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            seq = self.encoder(x)
        data = self.data_norm(seq.mean(1) + (seq.var(1, unbiased=False) + 1e-8).sqrt())
        return self.classifier(self.dropout(data)), torch.zeros(len(x), device=x.device)
