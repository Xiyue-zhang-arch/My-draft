"""ClassBD author implementation adapted to V3 parent-window evaluation.

Original files are preserved under official_classbd/upstream. This adapter fixes
the author code's detached physical losses and exposes logits to cross entropy.
"""
import sys
from pathlib import Path
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / 'official_classbd' / 'upstream'
sys.path.insert(0, str(UPSTREAM))

# fvcore is used only by the author's __main__ profiling example. Load the
# original definition with that single optional import removed in memory.
source = (UPSTREAM / 'Model' / 'BDCNN.py').read_text(encoding='utf-8')
source = source.replace('from fvcore.nn import FlopCountAnalysis, flop_count_str\n', '')
scope = {'__name__': 'classbd_upstream_model', '__file__': str(UPSTREAM / 'Model' / 'BDCNN.py')}
exec(compile(source, scope['__file__'], 'exec'), scope)
BDWDCNN = scope['BDWDCNN']
from Model.ClassBD_Module import get_envelope_frequency
import comparison_engine as base

VARIANT = 'ClassBD_WDCNN'
SEGMENT_LENGTH = 2048

def starts(length):
    if length < SEGMENT_LENGTH:
        raise ValueError('window shorter than 2048')
    indices = list(range(0, length - SEGMENT_LENGTH + 1, SEGMENT_LENGTH))
    if indices[-1] != length - SEGMENT_LENGTH:
        indices.append(length - SEGMENT_LENGTH)
    return indices

def differentiable_kurtosis(self, y, halfFilterlength=32):
    z = y.squeeze(1)[:, halfFilterlength:-halfFilterlength]
    z = z - z.mean()
    n = len(z)
    second = torch.sum(z.square(), dim=-1) / n
    return (torch.sum(z.pow(4), dim=-1) / n / second.square().clamp_min(1e-12)).mean()

def differentiable_lplq(self, y, p=2, q=4):
    spectrum, _ = get_envelope_frequency(y.clone(), fs=self.fs)
    return (torch.linalg.vector_norm(spectrum, ord=p, dim=-1) /
            torch.linalg.vector_norm(spectrum, ord=q, dim=-1).clamp_min(1e-12)).pow(p).mean()

class Model(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant != VARIANT:
            raise ValueError(variant)
        self.variant = variant
        self.net = BDWDCNN(4)
        import types
        self.net.classbd.funcKurtosis = types.MethodType(differentiable_kurtosis, self.net.classbd)
        self.net.classbd.g_lplq = types.MethodType(differentiable_lplq, self.net.classbd)
        self.extra = {}

    def forward(self, x, rpm):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        logits, ks, gs = [], [], []
        for offset in starts(x.shape[-1]):
            z = x[..., offset:offset+SEGMENT_LENGTH]
            a, k, g = self.net.classbd(z)
            y = self.net.cnn(a)
            y = self.net.fc1(y.reshape(y.shape[0], -1))
            y = self.net.fc2(self.net.dp(self.net.relu1(y)))
            logits.append(y.softmax(-1))
            ks.append(k)
            gs.append(g)
        self.extra = {'k': torch.stack(ks).mean(), 'g': torch.stack(gs).mean()}
        return torch.stack(logits).mean(0).clamp_min(1e-12).log()

def objective(model, logits, labels):
    ce = F.nll_loss(logits, labels)
    # The author's UW() creates log variances as -0.5 on each call and never
    # optimizes them. Keep the resulting fixed weights and constant exactly.
    losses = torch.stack([ce, model.extra['k'], model.extra['g']])
    log_variance = losses.new_full((3,), -.5)
    total = (losses / (3 * log_variance.exp()) + log_variance / 3).sum()
    return total, ce, total-ce

def load(folder, device):
    payload = torch.load(Path(folder)/'model.pt', map_location='cpu', weights_only=False)
    if payload.get('variant') != VARIANT:
        raise ValueError('not a ClassBD model')
    model = Model(VARIANT)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.to(device).eval()

@contextmanager
def _base_adapter():
    names = {'Model': Model, 'load': load, 'objective': objective, 'VARIANTS': (VARIANT,)}
    old = {key: getattr(base, key) for key in names}
    try:
        for key, value in names.items():
            setattr(base, key, value)
        yield
    finally:
        for key, value in old.items():
            setattr(base, key, value)

def train(x, w, protocol, variant, seed, subset, folder, device, spec=None):
    if protocol != 'clean_only':
        raise ValueError('ClassBD comparison is clean-only')
    with _base_adapter():
        return base.train(x, w, protocol, variant, seed, subset, folder, device, spec)

def predict_and_ce(model, x, rpm, labels, device):
    return base.predict_and_ce(model, x, rpm, labels, device)
