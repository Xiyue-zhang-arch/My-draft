"""在独立的 V5 入口中复用 V3 模型和训练流程。"""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from experiments.CWRU.实验文件.CWRU_V3 import engine as v3_engine

from .physics_features import (
    BANDS_HZ,
    FAMILIES,
    MIN_TOLERANCE_HZ,
    ORDERS,
    RELATIVE_TOLERANCE,
    RMS_EPS,
    SPECTRUM_EPS,
    CompactEvidence,
)


_RUNTIME = v3_engine.R
_BASE_CONFIG = json.loads((_RUNTIME.SHARED / 'model_config.json').read_text(encoding='utf-8'))
MODEL_CONFIG = {
    **_BASE_CONFIG,
    'families': list(FAMILIES),
    'orders': list(ORDERS),
    'sham_multipliers': [],
    'physics_descriptor': {
        'version': 'cwru_v5_39',
        'dimension': CompactEvidence.dimension,
        'carrier_bands_hz': [list(band) for band in BANDS_HZ],
        'families': list(FAMILIES),
        'orders': list(ORDERS),
        'harmonics': list(range(1, 7)),
        'envelope_sampling_hz': 3000,
        'envelope_trim_each_end': 75,
        'relative_tolerance': RELATIVE_TOLERANCE,
        'minimum_tolerance_hz': MIN_TOLERANCE_HZ,
        'spectrum_epsilon': SPECTRUM_EPS,
        'rms_epsilon': RMS_EPS,
    },
}


class SuiteModel(v3_engine.SuiteModel):
    """保留 V3 的前向、训练统计拟合与辅助分类，只替换物理输入。"""

    def __init__(self, variant, spec=None):
        nn.Module.__init__(self)
        if variant not in v3_engine.VARIANTS:
            raise ValueError(f'不支持的模型类型: {variant}')
        self.spec = engine.normalize_spec(spec)
        self.variant = variant
        self.aux_logits = None
        cfg = deepcopy(_BASE_CONFIG)
        if variant == 'baseline_data':
            self.method = 'data_only'
            self.network = _RUNTIME.DataOnly(cfg)
            return

        self.method = 'fusion'
        self.network = _RUNTIME.Fusion(cfg)
        self.evidence = CompactEvidence(cfg)
        original = self.network.physics[0]
        device = original.weight.device
        devices = [device.index] if device.type == 'cuda' else []
        # 延续 V3 的初始化顺序，保留其他各层同种子初始权重及后续随机序列。
        with torch.random.fork_rng(devices=devices):
            self.network.physics[0] = nn.Linear(
                self.evidence.dimension, original.out_features, device=device, dtype=original.weight.dtype
            )


def load(folder, device):
    """严格加载 V5 检查点及其 39 维标准化缓冲区。"""
    from experiment_paths import resolve_result_path

    saved = torch.load(resolve_result_path(folder) / 'model.pt', map_location=device, weights_only=True)
    if saved.get('model_config') != MODEL_CONFIG or 'suite_spec' not in saved:
        raise ValueError('检查点缺少匹配的 V5 39 维物理特征和训练配置')
    model = SuiteModel(saved['variant'], saved['suite_spec']).to(device)
    model.load_state_dict(saved['state_dict'])
    return model.eval()


def _isolated_engine():
    """在私有模块内复用 V3 引擎，原引擎和公共配置继续服务历史模型。"""
    spec = spec_from_file_location(f'{__package__}._v3_engine', v3_engine.__file__)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    utils = SimpleNamespace(**{**vars(module.R.U), 'CFG': deepcopy(MODEL_CONFIG)})
    module.R = SimpleNamespace(**{**vars(module.R), 'U': utils})
    module.SuiteModel = SuiteModel
    module.load = load
    return module


engine = _isolated_engine()
train = engine.train
