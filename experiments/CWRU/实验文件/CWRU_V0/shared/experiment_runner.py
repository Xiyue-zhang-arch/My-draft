"""统一的从头训练、验证选模、测试预测和结果保存流程。"""

from pathlib import Path
from datetime import datetime
import argparse
import json
import platform
import sys
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
import training_utils as U
from bearing_physics_features import Evidence
from physics_gated_fusion import Network as Fusion
from vibration_only_classifier import Network as DataOnly
from cwru_dataset import load_dataset
from result_plots import scores, confusion_plot, learning_curve, summarize
from experiment_paths import output_directory

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[3]
sys.path.insert(0, str(ROOT / "experiments/experiment_2"))
sys.path.insert(0, str(ROOT / "experiments/experiment_1/awgn_mixed"))
from wdcnn_drsn_pg_models import FACTORIES, objective
from attention_wavelet_hybrid import AttentionHybridAdapted
from awgn_augmentation import observe, fixed

BASELINES = dict(FACTORIES, AW_Hybrid_adapted=AttentionHybridAdapted)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class Classifier(nn.Module):
    """统一模型接口。物理分支对当前观测在线提取，标准化仅拟合训练子集。"""

    def __init__(self, method, cfg):
        super().__init__()
        self.method = method
        if method == "fusion":
            self.network = Fusion(cfg)
            self.evidence = Evidence(cfg)
        elif method == "data_only":
            self.network = DataOnly(cfg)
        else:
            self.network = BASELINES[method]()

    def forward(self, x, rpm):
        if self.method == "fusion":
            features, _ = self.evidence.extract(x, rpm / 60)
            logits, _ = self.network(x, self.evidence.standardize(features))
            return logits, {}
        if self.method == "data_only":
            return self.network(x)[0], {}
        return self.network(x, rpm)

    @torch.no_grad()
    def fit(self, x, rpm):
        if self.method == "fusion":
            values = (
                torch.cat(
                    [
                        self.evidence.extract(x[i : i + 32], rpm[i : i + 32] / 60)[0]
                        for i in range(0, len(x), 32)
                    ]
                )
                .cpu()
                .numpy()
            )
            center = np.median(values, axis=0)
            scale = np.maximum(
                np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0),
                0.05,
            )
            self.evidence.center.copy_(torch.as_tensor(center, device=x.device))
            self.evidence.scale.copy_(torch.as_tensor(scale, device=x.device))
        elif hasattr(self.network, "front"):
            self.network.front.fit(x, rpm)


@torch.no_grad()
def predict(model, x, rpm, device):
    model.eval()
    return np.concatenate(
        [
            model(
                torch.as_tensor(x[i : i + 32], device=device),
                torch.as_tensor(rpm[i : i + 32], device=device),
            )[0]
            .softmax(-1)
            .cpu()
            .numpy()
            for i in range(0, len(x), 32)
        ]
    )


def conditions(x, protocol, testing):
    yield "clean", 0, x
    if protocol == "awgn_mixed":
        for snr in [-4, -2, 0, 2, 4]:
            for seed in ([7201, 7202, 7203, 7204, 7205] if testing else [6201]):
                yield f"white_{snr:+d}", seed, fixed(x, snr, seed, 2 if testing else 1)[
                    0
                ]


def train_case(x, windows, case, destination, updates, device):
    protocol, method, ratio, subset, seed = case
    U.seed_all(seed)
    selected = U.selected(windows, ratio, subset)
    validation = np.flatnonzero(windows.role.eq("validation"))
    y = windows.label.to_numpy()
    rpm = windows.rpm.to_numpy(np.float32)
    weights = U.weights_for(windows, validation)
    model = Classifier(method, U.CFG).to(device)
    U.sync()
    started = time.perf_counter()
    model.fit(
        torch.as_tensor(x[selected], device=device),
        torch.as_tensor(rpm[selected], device=device),
    )
    U.sync()
    fit_seconds = time.perf_counter() - started
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, updates, eta_min=0.00001
    )
    validation_views = list(conditions(x[validation], protocol, False))
    destination.mkdir(parents=True)
    windows.iloc[selected].to_csv(destination / "selected_windows.csv", index=False)
    best, best_update, step, epoch, next_validation = float("inf"), 0, 0, 0, 50
    history = []
    while step < updates:
        epoch += 1
        model.train()
        rng = np.random.default_rng(
            np.random.SeedSequence([subset, seed, epoch, 20261011])
        )
        order = U.sample_epoch(windows, selected, rng)
        U.sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss_sum, batches = 0.0, 0
        for offset in range(0, len(order), 32):
            indices = order[offset : offset + 32]
            observed = x[indices]
            if protocol == "awgn_mixed":
                rng = np.random.default_rng(
                    np.random.SeedSequence([20261012, subset, seed, epoch, offset])
                )
                observed = observe(observed, rng)[0]
            optimizer.zero_grad(set_to_none=True)
            logits, extra = model(
                torch.as_tensor(observed, device=device),
                torch.as_tensor(rpm[indices], device=device),
            )
            loss = objective(logits, extra, torch.as_tensor(y[indices], device=device))
            if not torch.isfinite(loss):
                raise ValueError("训练损失非有限值。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach())
            batches += 1
            step += 1
            if step >= updates:
                break
        U.sync()
        seconds = time.perf_counter() - started
        peak = (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else None
        )
        score = np.nan
        if epoch == 1 or step >= next_validation or step >= updates:
            values = [
                U.nll(
                    predict(model, wave, rpm[validation], device),
                    y[validation],
                    weights,
                )
                for _, _, wave in validation_views
            ]
            score = (
                values[0]
                if protocol == "clean_only"
                else 0.5 * values[0] + 0.1 * sum(values[1:])
            )
            if score < best - 1e-6:
                best, best_update = score, step
                torch.save(
                    dict(
                        method=method,
                        model_config=U.CFG,
                        state_dict={
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        },
                    ),
                    destination / "model.pt",
                )
            if step >= next_validation:
                next_validation += 50
            print(f"{case}: updates={step}, validation_NLL={score:.5f}", flush=True)
        history.append(
            dict(
                epoch=epoch,
                updates=step,
                train_loss=loss_sum / batches,
                validation_nll=score,
                train_seconds=seconds,
                peak_gpu_mib=peak,
            )
        )
        if step >= 200 and step - best_update >= 150:
            break
    history = pd.DataFrame(history)
    history.to_csv(destination / "training.csv", index=False)
    learning_curve(history, destination / "training.png")
    info = dict(
        protocol=protocol,
        method=method,
        ratio=ratio,
        subset=subset,
        seed=seed,
        best_update=best_update,
        updates_trained=step,
        validation_nll=best,
        training_windows=len(selected),
        validation_windows=len(validation),
        from_scratch=True,
        fit_seconds=fit_seconds,
        parameters=sum(p.numel() for p in model.parameters()),
        train_seconds_per_epoch=float(history.train_seconds.mean()),
        peak_gpu_mib=(
            float(history.peak_gpu_mib.max()) if device.type == "cuda" else None
        ),
    )
    write_json(destination / "parameters.json", info)
    del model
    return info


def load_model(model_path, device):
    checkpoint = torch.load(model_path, weights_only=True, map_location=device)
    model = Classifier(checkpoint["method"], checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


@torch.no_grad()
def inference_latency(model, raw, rpm, device):
    x = torch.as_tensor(raw[:1], device=device)
    speed = torch.as_tensor(rpm[:1], device=device)

    def once():
        observed = (x - x.mean(-1, keepdim=True)) / (
            x.std(-1, unbiased=False, keepdim=True) + 1e-8
        )
        model(observed, speed)

    for _ in range(10):
        once()
    U.sync()
    start = time.perf_counter()
    for _ in range(50):
        once()
    U.sync()
    return (time.perf_counter() - start) * 1000 / 50


def evaluate_case(x, raw, windows, info, folder, device):
    model = load_model(folder / "model.pt", device)
    test = np.flatnonzero(windows.role.eq("test"))
    labels = windows.iloc[test].label.to_numpy()
    rpm = windows.iloc[test].rpm.to_numpy(np.float32)
    metrics, arrays = [], {}
    for condition, noise_seed, observed in conditions(x[test], info["protocol"], True):
        probabilities = predict(model, observed, rpm, device)
        key = f"{condition}_n{noise_seed}"
        arrays[key] = probabilities
        metrics.append(
            {k: info[k] for k in ["protocol", "method", "ratio", "subset", "seed"]}
            | dict(
                condition=condition,
                noise_seed=noise_seed,
                **scores(labels, probabilities),
            )
        )
    for condition in sorted({row["condition"] for row in metrics}):
        probability = np.stack(
            [v for k, v in arrays.items() if k.startswith(condition + "_n")]
        )
        arrays[condition + "_confusion_counts"] = confusion_plot(
            labels, probability, folder / f"confusion_{condition}.png"
        )
    np.savez_compressed(
        folder / "predictions.npz",
        labels=labels,
        window_indices=windows.iloc[test].window_index.to_numpy(),
        array_indices=test,
        **arrays,
    )
    info["latency_ms"] = inference_latency(model, raw[test], rpm, device)
    return metrics, info


def main(protocol=None, comparison=False):
    parser = argparse.ArgumentParser(description="CWRU从原始MAT文件训练、评估并绘图")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT / "data/raw/CWRU/48k Drive End Bearking Fault Data",
    )
    parser.add_argument("--records", type=Path, help="可选的自定义记录表CSV")
    parser.add_argument(
        "--output",
        type=Path,
        help="必须是不存在的新目录；默认项目experiments/CWRU/实验结果/CWRU下的时间戳目录",
    )
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.1, 0.2, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1706, 1707, 1708])
    parser.add_argument(
        "--subset-seeds", type=int, nargs="+", default=[4101, 4102, 4103]
    )
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(BASELINES) if comparison else ["fusion", "data_only"],
        default=list(BASELINES) if comparison else ["fusion", "data_only"],
    )
    if comparison:
        parser.add_argument(
            "--protocols",
            nargs="+",
            choices=["clean_only", "awgn_mixed"],
            default=["clean_only", "awgn_mixed"],
        )
    args = parser.parse_args()
    if args.updates <= 0 or any(r <= 0 or r > 1 for r in args.ratios):
        parser.error("updates必须为正，ratios必须在(0,1]内。")
    args.ratios = list(dict.fromkeys(args.ratios))
    if len({round(r * 100) for r in args.ratios}) != len(args.ratios):
        parser.error("训练比例转换为整数百分比后不能重复。")
    device = torch.device(args.device)
    U.DEVICE = device
    U.configure()
    records, windows, raw = load_dataset(args.data_root, args.records)
    x = U.normalize(raw)
    name = "experiment_2" if comparison else f"experiment_1_{protocol}"
    out = (
        args.output
        or PROJECT / "experiments/CWRU/实验结果/CWRU" / f"{name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    out = output_directory(out, args.data_root)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(
        protocol=name,
        device=str(device),
        torch=str(torch.__version__),
        numpy=np.__version__,
        hardware=(
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        batch_size=32,
        learning_rate=0.0005,
        weight_decay=0.001,
        label_smoothing=0.05,
        minimum_updates=200,
        patience_updates=150,
        validation_interval=50,
        model_config=U.CFG,
        precision="BF16 proposed encoder on CUDA; remaining FP32",
        training_timing="per epoch; includes online features and augmentation; excludes validation",
        inference_timing="batch1; normalization and online features; excludes disk and transfer; 10 warmups, 50 repeats",
        confusion="sum counts across noise observations; row-normalized; repeated observations are not independent windows",
        noise_probability=0.5,
        training_snr=[-4, 4],
        evaluation_snrs=[-4, -2, 0, 2, 4],
        validation_noise_seed=6201,
        test_noise_seeds=[7201, 7202, 7203, 7204, 7205],
    )
    write_json(out / "config.json", settings)
    records.to_csv(out / "records.csv", index=False)
    windows.to_csv(out / "windows.csv", index=False)
    jobs = []
    for p in (args.protocols if comparison else [protocol]):
        for method in dict.fromkeys(args.methods):
            for ratio in args.ratios:
                if method == "data_only" and ratio != 1:
                    continue
                for subset in (dict.fromkeys(args.subset_seeds) if ratio < 1 else [0]):
                    for seed in dict.fromkeys(args.seeds):
                        case = (p, method, ratio, subset, seed)
                        folder = (
                            out
                            / p
                            / method
                            / f"ratio_{round(ratio*100):03d}_subset_{subset}_seed_{seed}"
                        )
                        jobs.append(
                            (
                                train_case(
                                    x, windows, case, folder, args.updates, device
                                ),
                                folder,
                            )
                        )
    if not jobs:
        raise ValueError("无实验组合；data_only只支持全样本。")
    # 全部训练和验证选模结束后才进行测试。
    metrics, complexity = [], []
    for info, folder in jobs:
        rows, timing = evaluate_case(x, raw, windows, info, folder, device)
        metrics.extend(rows)
        complexity.append(timing)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(complexity).to_csv(out / "complexity.csv", index=False)
    summarize(metrics, windows, out)
    print(f"完成。结果目录：{out}", flush=True)


def inference_cli():
    """对新训练权重推理；NPZ输入须包含waveforms和rpm，可选labels。"""
    parser = argparse.ArgumentParser(description="对新训练的CWRU模型推理")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    with np.load(args.input, allow_pickle=False) as data:
        x = data["waveforms"].astype(np.float32)
        rpm = data["rpm"].astype(np.float32)
        labels = data["labels"] if "labels" in data else None
    if x.ndim != 3 or x.shape[1:] != (1, 24000) or rpm.shape != (len(x),) or not len(x):
        raise ValueError("输入要求waveforms为[N,1,24000]，rpm为[N]。")
    if not np.isfinite(x).all() or not np.isfinite(rpm).all() or (rpm <= 0).any():
        raise ValueError("输入含无效值。")
    if labels is not None and (
        labels.shape != (len(x),) or not np.isin(labels, range(4)).all()
    ):
        raise ValueError("labels必须为[N]，取值0、1、2、3。")
    model = load_model(args.model, device)
    probabilities = predict(model, U.normalize(x), rpm, device)
    out = output_directory(args.output, args.input.parent)
    frame = pd.DataFrame(probabilities, columns=["p_Normal", "p_IR", "p_OR", "p_B"])
    frame["prediction"] = probabilities.argmax(1)
    if labels is not None:
        frame["label"] = labels
    frame.to_csv(out / "predictions.csv", index=False)
    if labels is not None:
        write_json(out / "metrics.json", scores(labels, probabilities))
        counts = confusion_plot(labels, probabilities, out / "confusion.png")
        np.save(out / "confusion_counts.npy", counts)
    print(out)
