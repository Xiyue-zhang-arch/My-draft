"""Isolated V3 training engine with only the physical descriptor changed."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from experiments.CWRU.实验文件.CWRU_V3 import engine as original
from .physics import Evidence6

BASE_CONFIG = json.loads((original.R.SHARED/'model_config.json').read_text(encoding='utf-8'))
MODEL_CONFIG = deepcopy(BASE_CONFIG)
MODEL_CONFIG['physics_descriptor'] = dict(
    version='fullband_scheme1_six_v1', dimension=6, carrier_bands_hz=[],
    feature_order=['R_O','R_I','R_B','U_O','U_I','U_B'],
    R_orders=[3.5848,5.4152,2.35675], U_orders=[3.5848,5.4152,4.7135],
    harmonics=6, R_halfwidth='max(2Hz,0.015*h*f0)', U_lags=[1,2,3],
    U_halfwidth='max(1_sample,0.02*lag)',
    SES='abs(FFT(abs(Hilbert(x))**2-mean))**2/(2*N)',
    window='rectangular', resample=False, trim=False, prewhiten=False,
    added_CHNR_and_N=False)
# Top-level bands configure only the legacy temporary constructor. No evidence
# filters from it are instantiated or retained in this new model.


class SuiteModel(original.SuiteModel):
    def __init__(self, variant, spec=None):
        nn.Module.__init__(self)
        self.spec = engine.normalize_spec(spec)
        self.variant = variant
        self.aux_logits = None
        if variant == 'baseline_data':
            self.method = 'data_only'
            self.network = original.R.DataOnly(deepcopy(BASE_CONFIG))
        elif variant == 'data_aux_ce':
            self.method = 'fusion'
            self.network = original.R.Fusion(deepcopy(BASE_CONFIG))
            self.evidence = Evidence6(BASE_CONFIG)
            old = self.network.physics[0]
            with torch.random.fork_rng():
                self.network.physics[0] = nn.Linear(6, old.out_features, dtype=old.weight.dtype)
        else:
            raise ValueError(variant)


def load(folder, device):
    checkpoint = torch.load(Path(folder)/'model.pt', map_location=device, weights_only=True)
    if checkpoint.get('model_config') != MODEL_CONFIG:
        raise ValueError('Checkpoint does not match six-descriptor configuration')
    model = SuiteModel(checkpoint['variant'], checkpoint['suite_spec']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


spec = spec_from_file_location(f'{__package__}._engine', original.__file__)
engine = module_from_spec(spec)
spec.loader.exec_module(engine)
engine.R = SimpleNamespace(**{**vars(engine.R), 'U': SimpleNamespace(**{**vars(engine.R.U), 'CFG': deepcopy(MODEL_CONFIG)})})
engine.SuiteModel = SuiteModel
engine.load = load
train = engine.train
