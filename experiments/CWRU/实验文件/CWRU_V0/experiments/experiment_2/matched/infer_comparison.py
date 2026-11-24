"""CWRU四方法比较的推理入口。按训练协议、方法、训练比例和随机种子选择已训练模型。"""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
import json
import hashlib
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/experiment_2"))
from wdcnn_drsn_pg_models import FACTORIES as ORIGINAL
from attention_wavelet_hybrid import AttentionHybridAdapted

FACTORIES = dict(ORIGINAL, AW_Hybrid_adapted=AttentionHybridAdapted)


class Pipeline(torch.nn.Module):
    """将逐窗口标准化、所需物理计算和已训练分类器组合为推理流程。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, rpm):
        normalized = (x - x.mean(-1, keepdim=True)) / (
            x.std(-1, unbiased=False, keepdim=True) + 1e-8
        )
        return self.model(normalized, rpm)[0]


def load_pipeline(protocol, method, ratio=1.0, subset=0, seed=1706, device="cuda"):
    """按实验配置加载已有权重，返回eval模式的推理模块。"""
    assert protocol in ["clean_only", "awgn_mixed"] and method in FACTORIES
    dest = (
        ROOT.parents[3]
        / "experiments/CWRU/实验结果/CWRU/experiment_2/matched"
        / protocol
        / "models"
        / method
        / f"ratio_{int(ratio*100):03d}"
        / f"subset_{subset}"
        / f"seed_{seed}"
    )
    info = json.loads((dest / "frozen.json").read_text())
    assert (
        hashlib.sha256((dest / "model.pt").read_bytes()).hexdigest()
        == info["model_sha256"]
    )
    model = FACTORIES[method]().to(device)
    model.load_state_dict(
        torch.load(dest / "model.pt", weights_only=True, map_location=device)
    )
    return Pipeline(model).eval()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "shared"))
    from experiment_runner import inference_cli

    inference_cli()
