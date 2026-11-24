"""CWRU已训练模型的加载工具。读取权重和训练集标准化参数，供纯干净和AWGN推理入口共用。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def load_module(name: str, path: Path) -> ModuleType:
    """Load a Python module from a file without changing the package layout."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_frozen_info(destination: Path) -> dict[str, Any]:
    return json.loads((destination / "frozen.json").read_text(encoding="utf-8"))


def verify_artifacts(
    destination: Path,
    info: dict[str, Any],
    *,
    training_noise: bool,
) -> None:
    """Verify the immutable model artifacts before constructing a pipeline."""
    assert info["training_noise"] is training_noise
    assert info["paired_inputs"] is False
    for filename, digest_key in (
        ("model.pt", "model_sha256"),
        ("scaler.npz", "scaler_sha256"),
    ):
        assert sha256(destination / filename) == info[digest_key]


def load_scaler(runtime: ModuleType, evidence: Any, path: Path) -> None:
    """恢复当前训练子集拟合的物理描述中心和尺度。"""
    with runtime.np.load(path) as arrays:
        evidence.center.copy_(
            runtime.torch.from_numpy(arrays["center"]).to(runtime.DEVICE)
        )
        evidence.scale.copy_(
            runtime.torch.from_numpy(arrays["scale"]).to(runtime.DEVICE)
        )


def load_published_pipeline(
    *,
    runtime: ModuleType,
    pipeline_type: type,
    root: Path,
    result_directory: str,
    source_directory: Path,
    ratio: float,
    subset: int,
    seed: int,
    variant: str,
    training_noise: bool,
) -> Any:
    """Load one frozen clean-only or AWGN-trained model."""
    assert variant in {"fusion", "data_only"}
    if variant == "data_only":
        assert ratio == 1 and subset == 0

    destination = (
        root.parents[3]
        / "experiments/CWRU/实验结果/CWRU"
        / result_directory
        / "models"
        / variant
        / f"ratio_{int(ratio * 100):03d}"
        / f"subset_{subset}"
        / f"seed_{seed}"
    )
    info = read_frozen_info(destination)
    verify_artifacts(destination, info, training_noise=training_noise)

    module_id = result_directory.replace("/", "_").replace("\\", "_")
    network_name = (
        "physics_gated_fusion.py"
        if variant == "fusion"
        else "vibration_only_classifier.py"
    )
    network = load_module(
        f"published_{module_id}_{variant}_network",
        source_directory / network_name,
    )
    model = network.Network(runtime.CFG).to(runtime.DEVICE)
    model.load_state_dict(
        runtime.torch.load(
            destination / "model.pt",
            weights_only=True,
            map_location=runtime.DEVICE,
        )
    )

    evidence = None
    if variant == "fusion":
        evidence_module = load_module(
            f"published_{module_id}_evidence",
            source_directory / "bearing_physics_features.py",
        )
        evidence = evidence_module.Evidence(runtime.CFG).to(runtime.DEVICE)
        load_scaler(runtime, evidence, destination / "scaler.npz")

    return pipeline_type(model, evidence, variant).eval()
