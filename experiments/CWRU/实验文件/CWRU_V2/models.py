"""V2单变量候选；骨干、物理描述和分类头结构沿用V1。"""
import torch
from torch import nn
from diagnose import R


# 2026-10-21后续实验仅训练这两组；Model仍支持加载全部历史变体。
TRAINABLE_VARIANTS = ('baseline_data', 'data_aux_ce')


class Model(R.Classifier):
    def __init__(self,variant):
        if variant not in {'baseline_fusion','baseline_data','residual_scale','data_aux_ce'}:
            raise ValueError(variant)
        super().__init__('data_only' if variant=='baseline_data' else 'fusion',R.U.CFG)
        self.variant=variant
        if variant=='residual_scale':
            self.alpha=nn.Parameter(torch.tensor(.1))
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
        alpha=self.alpha if self.variant=='residual_scale' else 1.
        embedding=n.fused_norm(d+alpha*gate*p)
        logits=n.classifier(n.dropout(embedding))
        self.aux_logits=None
        if self.variant=='data_aux_ce' and self.training:
            # 辅助前向不能改变主路径后续的dropout随机数序列。
            devices=[x.device.index] if x.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                self.aux_logits=n.classifier(n.dropout(d))
        return logits


def loss_for(model,logits,labels):
    main=nn.functional.cross_entropy(logits,labels,label_smoothing=0.)
    aux=main.new_zeros(())
    if model.variant=='data_aux_ce':
        if model.aux_logits is None:
            raise ValueError('辅助监督缺少训练前向')
        aux=nn.functional.cross_entropy(model.aux_logits,labels,label_smoothing=0.)
    return main+.1*aux,main,aux


@torch.no_grad()
def complexity(variant,device):
    R.U.seed_all(1706)
    model=Model(variant).to(device).eval()
    macs=0
    def hook(module,inputs,output):
        nonlocal macs
        if isinstance(module,nn.Conv1d):
            macs+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
        elif isinstance(module,nn.Linear):
            macs+=output.numel()*module.in_features
    handles=[m.register_forward_hook(hook) for m in model.modules() if isinstance(m,(nn.Conv1d,nn.Linear))]
    try:model(torch.zeros(1,1,24000,device=device),torch.full((1,),1750.,device=device))
    finally:
        for h in handles:h.remove()
    result=dict(variant=variant,parameters=sum(p.numel() for p in model.parameters()),
        conv_linear_flops=2*macs,flops_scope='batch1 Conv1d/Linear only; FFT, reductions, normalization, activation, elementwise excluded',
        auxiliary_training_head_flops_per_sample=2304 if variant=='data_aux_ce' else 0)
    # 32x32和32x4的两个Linear额外前向，共2304 FLOPs；不含其反向与逐元素算子。
    print('COMPLEXITY',result,flush=True)
    return result


def baseline_equivalence(device):
    """在基线已运行后检查共享结构及候选主前向，不训练，不保存假模型。"""
    wave=torch.linspace(-1,1,24000,device=device).reshape(1,1,-1)
    rpm=torch.tensor([1750.],device=device)
    for method,variant in [('fusion','baseline_fusion'),('data_only','baseline_data')]:
        R.U.seed_all(1706);original=R.Classifier(method,R.U.CFG).to(device)
        R.U.seed_all(1706);baseline=Model(variant).to(device)
        for key,value in original.state_dict().items():
            torch.testing.assert_close(value,baseline.state_dict()[key],rtol=0,atol=0)
        original.eval();baseline.eval()
        with torch.no_grad():
            torch.testing.assert_close(original(wave,rpm),baseline(wave,rpm),rtol=0,atol=0)
    R.U.seed_all(1706);original=R.Classifier('fusion',R.U.CFG).to(device).train()
    R.U.seed_all(1706);candidate=Model('data_aux_ce').to(device).train()
    R.U.seed_all(1234);expected=original(wave,rpm)
    cpu_after=torch.get_rng_state()
    cuda_after=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
    R.U.seed_all(1234);actual=candidate(wave,rpm)
    torch.testing.assert_close(expected,actual,atol=0,rtol=0)
    assert torch.equal(cpu_after,torch.get_rng_state())
    if cuda_after is not None:assert torch.equal(cuda_after,torch.cuda.get_rng_state(device))
    main=nn.functional.cross_entropy(actual,torch.tensor([1],device=device))
    total,reported,aux=loss_for(candidate,actual,torch.tensor([1],device=device))
    torch.testing.assert_close(reported,main)
    torch.testing.assert_close(total,main+.1*aux)
    print('BASELINE AND AUX MAIN-PATH EQUIVALENCE OK',flush=True)
