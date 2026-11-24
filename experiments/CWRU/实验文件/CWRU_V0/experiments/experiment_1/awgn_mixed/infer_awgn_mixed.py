"""CWRU混合AWGN训练模型的推理入口。复用统一推理流程，仅选择对应训练协议下的权重。"""

from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared"))

from model_loading import load_module, load_published_pipeline

S = load_module(
    "shared_published_clean_pipeline",
    ROOT / "experiments" / "experiment_1/clean_only" / "infer_clean_only.py",
)
T = S.T


def load_pipeline(ratio=1.0, subset=0, seed=1706, variant="fusion"):
    """按实验配置加载已有权重，返回eval模式的推理模块。"""
    return load_published_pipeline(
        runtime=T,
        pipeline_type=S.Pipeline,
        root=ROOT,
        result_directory="experiment_1/awgn_mixed",
        source_directory=ROOT / "shared",
        ratio=ratio,
        subset=subset,
        seed=seed,
        variant=variant,
        training_noise=True,
    )


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "shared"))
    from experiment_runner import inference_cli

    inference_cli()
