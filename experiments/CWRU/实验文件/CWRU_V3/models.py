"""复制V2当前方法，V3只保留数据单分支和原融合加共享头辅助CE。"""
import torch
from torch import nn
from context import R


# V3只允许这两组，不复制其他历史结构候选。
TRAINABLE_VARIANTS = ('baseline_data', 'data_aux_ce')


class Model(R.Classifier):
    def __init__(self,variant):
        if variant not in {'baseline_data','data_aux_ce'}:
            raise ValueError(variant)
        super().__init__('data_only' if variant=='baseline_data' else 'fusion',R.U.CFG)
        self.variant=variant
        self.aux_logits=None

    def forward(self,x,rpm):
        if self.variant.startswith('baseline_'):
            return super().forward(x,rpm)
        n=self.network
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=x.is_cuda):
            seq=n.encoder(x)
        d=n.data_norm(seq.mean(1)+(seq.var(1,unbiased=False)+1e-8).sqrt())
        features,_=self.evidence.extract(x,rpm/60)
        p=n.physics(self.evidence.standardize(features))
        gate=n.gate(torch.cat([d,p],dim=-1))
        embedding=n.fused_norm(d+gate*p)
        logits=n.classifier(n.dropout(embedding))
        self.aux_logits=None
        if self.variant=='data_aux_ce' and self.training:
            # 辅助前向不能改变主路径后续的dropout随机数序列。
            devices=[x.device.index] if x.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                self.aux_logits=n.classifier(n.dropout(d))
        return logits

