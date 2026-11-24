"""CWRU纯干净训练模型的推理入口。输入观测波形[B,1,24000]及转速rpm[B]，输出四类logits。"""

from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared"))

import training_utils as T
from model_loading import load_published_pipeline


class Pipeline(T.torch.nn.Module):
    """将逐窗口标准化、所需物理计算和已训练分类器组合为推理流程。"""

    def __init__(self, model, evidence, variant):
        super().__init__()
        self.model = model
        self.evidence = evidence
        self.variant = variant

    def forward(self, x, rpm):
        """每个观测独立归一化；转速由rpm换算为Hz后用于物理提取。"""
        x = (x - x.mean(-1, keepdim=True)) / (
            x.std(-1, unbiased=False, keepdim=True) + 1e-8
        )
        if self.variant == "fusion":
            features, _ = self.evidence.extract(x, rpm / 60)
            logits, _ = self.model(x, self.evidence.standardize(features))
        else:
            logits, _ = self.model(x)
        return logits


def load_pipeline(ratio=1.0, subset=0, seed=1706, variant="fusion"):
    """按实验配置加载已有权重，返回eval模式的推理模块。"""
    return load_published_pipeline(
        runtime=T,
        pipeline_type=Pipeline,
        root=ROOT,
        result_directory="experiment_1/clean_only",
        source_directory=ROOT / "shared",
        ratio=ratio,
        subset=subset,
        seed=seed,
        variant=variant,
        training_noise=False,
    )


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "shared"))
    from experiment_runner import inference_cli

    inference_cli()
