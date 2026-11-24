"""List, check, plan and manually run a CWRU or PU experiment."""

import argparse
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"


@dataclass(frozen=True)
class Experiment:
    dataset: str
    version: str
    script: str
    prefix: str
    label: str
    args: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    style: str = "single"
    cpu_only: bool = False

    @property
    def code(self):
        return EXPERIMENTS / self.dataset / "实验文件" / self.version

    @property
    def result(self):
        version = "CWRU" if self.version == "CWRU_V0" else self.version
        return EXPERIMENTS / self.dataset / "实验结果" / version

    def python(self, script, *args):
        prefix = [sys.executable, "-X", "utf8", "-B", "-u"]
        path = self.code / script
        if self.version in {"CWRU_V5", "CWRU_V6"}:
            return [*prefix, "-m", ".".join(path.relative_to(ROOT).with_suffix("").parts), *args]
        return [*prefix, str(path), *args]


BASE = "CWRU_V1/baseline_20261020"
V3_SELECT = "CWRU_V3/run_20261022_2105"
V3_FULL = "CWRU_V3/load_full_20261024_0950"
V3_WINDOW = "CWRU_V3/window_random_20261023_0853"
V3_SMALL = "CWRU_V3/load_small_20261025_2017"
V4_SELECT = "CWRU_V4/clean_20261106_105008"
V5_SELECT = "CWRU_V5/selection_20261109_114155"
V5_CLEAN = "CWRU_V5/load_clean_20261109_142434"
def cwru(version, script, prefix, label, args=(), refs=(), style="single", cpu_only=False):
    return Experiment("CWRU", f"CWRU_V{version}", script, prefix, label, args, refs, style, cpu_only)


CATALOG = {
    "cwru-v0-clean": cwru(0, "experiments/experiment_1/clean_only/train_clean_only.py", "experiment_1_clean", "按负载，纯干净"),
    "cwru-v0-awgn": cwru(0, "experiments/experiment_1/awgn_mixed/train_awgn_mixed.py", "experiment_1_awgn", "按负载，混合噪声"),
    "cwru-v0-compare": cwru(0, "experiments/experiment_2/matched/train_matched_comparison.py", "experiment_2", "四方法匹配"),
    "cwru-v0-window": cwru(0, "experiments/experiment_3/run_window_split.py", "experiment_3", "随机窗口"),
    "cwru-v0-window-small": cwru(0, "experiments/experiment_3/supplement_data_small.py", "experiment_3_small", "随机窗口小样本补充", refs=("CWRU/experiment_3",)),
    "cwru-v0-exclusion": cwru(0, "experiments/experiment_4/run_exclusion.py", "experiment_4", "按负载，排除指定目标记录"),
    "cwru-v0-budget": cwru(0, "experiments/experiment_5/run_budget_matched.py", "experiment_5", "随机窗口训练预算匹配"),
    "cwru-v0-transfer": cwru(0, "experiments/experiment_6/run_transfer.py", "experiment_6", "单来源跨负载"),
    "cwru-v0-signal-outer": cwru(0, "experiments/signal_analysis/visualize_signal_quality.py", "signal_outer", "外圈信号分析", cpu_only=True),
    "cwru-v0-signal-others": cwru(0, "experiments/signal_analysis/visualize_other_classes.py", "signal_others", "其他类别信号分析", cpu_only=True),
    "cwru-v1": cwru(1, "run_baseline.py", "baseline", "全样本基线"),
    "cwru-v2-quick": cwru(2, "run_quick_exploration.py", "quick", "快速训练验证", refs=(BASE,)),
    "cwru-v2-optimization": cwru(2, "run_optimization.py", "optimization", "分阶段优化与评价", refs=(BASE,), style="optimization"),
    "cwru-v2-selected": cwru(2, "run_selected_experiment.py", "optimization_selected", "两阶段系数选择与正式训练", refs=(BASE,), style="selected"),
    "cwru-v3-selection": cwru(3, "run_experiment.py", "run", "V3 完整选参与后续训练", ("run",), (BASE,)),
    "cwru-v3-window": cwru(3, "run_window_experiment.py", "window_random", "随机窗口", ("run",), (BASE, V3_SELECT)),
    "cwru-v3-full": cwru(3, "run_load_experiment.py", "load_full", "全样本跨负载", ("run",), (BASE, V3_SELECT)),
    "cwru-v3-small": cwru(3, "run_load_small_experiment.py", "load_small", "小样本跨负载", ("run",), (BASE, V3_SELECT)),
    "cwru-v4-suite": cwru(4, "run_v4.py", "clean", "四频带完整实验", ("run",), (BASE, V3_SELECT)),
    "cwru-v4-full": cwru(4, "run_load_full.py", "load_full", "四频带全样本", ("run", "--replay"), (BASE, V3_SELECT, V4_SELECT)),
    "cwru-v4-small": cwru(4, "run_load_small.py", "load_small_10_20", "四频带 10% 与 20%", ("run", "--replay", "--budget", "10-20"), (BASE, V3_SELECT, V4_SELECT)),
    "cwru-v4-half": cwru(4, "run_load_small.py", "load_small_50", "四频带 50%", ("run", "--replay", "--budget", "50"), (BASE, V3_SELECT, V4_SELECT)),
    "cwru-v5-selection": cwru(5, "run_selection.py", "selection", "39 维描述符选参", ("run",), (BASE, V3_SELECT)),
    "cwru-v5-clean": cwru(5, "run_load.py", "load_clean", "39 维纯干净跨负载", ("run",), (BASE, V5_SELECT, V3_FULL, V3_SMALL)),
    "cwru-v5-awgn": cwru(5, "run_load_awgn.py", "load_awgn", "39 维 AWGN 完整跨负载", ("run",), (BASE, V5_SELECT, V5_CLEAN, V3_FULL, V3_SMALL)),
    "cwru-v5-window": cwru(5, "run_window.py", "window_random", "39 维随机窗口", ("run", "--replay"), (BASE, V5_SELECT, V3_WINDOW)),
    "cwru-v6": cwru(6, "run.py", "small10", "六维描述符 10% 小样本", ("run", "--replay"), (BASE, V3_WINDOW, V3_SMALL)),
    "pu-all": Experiment("PU", "PU_0_all_conditions_721", "run.py", "server", "四工况记录留出", style="pu"),
    "pu-small": Experiment("PU", "PU_0_small10_imbalance", "run.py", "server", "小样本与不平衡", style="pu"),
    "pu-concat": Experiment("PU", "PU_0_concat", "run.py", "server", "直接拼接与单模态", style="pu"),
    "pu-harmonic": Experiment("PU", "PU_harmonic_mean", "run_mean.py", "server", "四谐波跨工况", ("--portable",), style="harmonic"),
}

for label, method, author in (
    ("wdcnn", "WDCNN", False), ("drsn", "DRSN_CW", False),
    ("pg", "PG_RDGFN_adapted", False), ("aw-hybrid", "AW_Hybrid_adapted", False),
    ("dwcn", "DWCN_SDGFD", True), ("classbd", "ClassBD_WDCNN", True),
):
    for sample, reference in (("full", V3_FULL), ("small", V3_SMALL)):
        filename = "run_author_comparisons.py" if author else "run_load_comparisons.py"
        prefix = ("load_comparison_author_" if author else "load_comparison_") + f"{sample}_{label}"
        CATALOG[f"cwru-v3-{label}-{sample}"] = cwru(
            3, filename, prefix, f"{method} {sample}",
            ("run", "--method", method, "--sample", sample), (BASE, V3_SELECT, reference),
        )


def child_env():
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", MPLBACKEND="Agg")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return env


def check_environment(exp):
    if sys.version_info < (3, 10):
        raise RuntimeError("需要 Python 3.10 或更新版本")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print(f"Python {sys.version.split()[0]} | {sys.executable}")
    if exp is None or exp.dataset == "CWRU":
        packages = ("torch", "numpy", "scipy", "pandas", "sklearn", "matplotlib", "pywt", "threadpoolctl", "psutil")
    else:
        packages = ("torch", "numpy", "scipy", "matplotlib")
    modules = {}
    for name in packages:
        try:
            modules[name] = importlib.import_module(name)
        except (ImportError, OSError) as error:
            raise RuntimeError(f"当前解释器无法导入 {name}。检查已激活的环境和 requirements.txt。{error}") from error
    print(" | ".join(f"{name} {module.__version__}" for name, module in modules.items() if hasattr(module, "__version__")))
    if exp is not None and exp.cpu_only:
        return
    torch = modules["torch"]
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("当前 Python 中的 PyTorch 无法访问 NVIDIA CUDA。检查 nvidia-smi、GPU 分配和 CUDA 版 PyTorch。")
    try:
        with torch.no_grad():
            signal = torch.arange(2560, device="cuda", dtype=torch.float64)
            spectrum = torch.fft.rfft(signal)
            if not bool(torch.isfinite(spectrum).all()):
                raise RuntimeError("CUDA FFT returned nonfinite values")
            x = torch.ones((1, 1, 64), device="cuda")
            conv = torch.nn.Conv1d(1, 4, 3).cuda().eval()
            conv(x)
            x.reshape(8, 8) @ x.reshape(8, 8)
            if exp is None or (exp.dataset == "CWRU" and exp.version not in {"CWRU_V0", "CWRU_V1"}):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    conv(x)
            torch.cuda.synchronize()
    except Exception as error:
        raise RuntimeError(f"CUDA 小张量检查失败。核对显卡、驱动与 PyTorch 构建。{error}") from error
    free, total = torch.cuda.mem_get_info()
    print(f"CUDA {torch.version.cuda} | {torch.cuda.get_device_name()} | 可用显存 {free / 2**30:.1f}/{total / 2**30:.1f} GiB")


def require(paths):
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少运行输入：\n" + "\n".join(missing))


def check_inputs(exp):
    require([exp.code / exp.script])
    if not exp.result.is_dir():
        raise FileNotFoundError(f"实验结果父目录缺失：{exp.result}")
    if not os.access(exp.result, os.W_OK):
        raise PermissionError(f"实验结果父目录不可写：{exp.result}")
    print(f"结果磁盘剩余 {shutil.disk_usage(exp.result).free / 2**30:.1f} GiB")
    if exp.dataset == "CWRU":
        mats = ROOT / "data/raw/CWRU/48k Drive End Bearking Fault Data"
        if len(list(mats.glob("*.mat"))) != 40:
            raise FileNotFoundError(f"预期 40 个 48 kHz MAT 文件：{mats}")
        for ref in exp.references:
            path = exp.result.parent / ref
            if not path.is_dir() or not any(path.iterdir()):
                raise FileNotFoundError(f"缺少冻结参照：{path}")
        return

    mixed = EXPERIMENTS / "PU/实验文件/PU_0_all_conditions_721/data/prepared"
    shared = ROOT / "data/pu_shared/prepared_inputs"
    if exp.version == "PU_0_concat":
        inputs = shared
        require([exp.code / "data/source_20261115_a" / s for s in (
            "verification.json", "split_indices.npz", "normalization.json", "data_reference.json")])
    elif exp.style == "harmonic":
        inputs = mixed
        anchor = exp.result / "main_20261122_a"
        required = [anchor / s for s in ("RUN_COMPLETE.json", "config.json", "report/AUDIT_COMPLETE.json",
                                         "report/per_condition.csv")]
        for source in ("N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"):
            bank = anchor / "bank" / source
            required.extend(bank / name for name in ("descriptors.npy", "mean14/indices.npz",
                                                      "mean14/normalization.json"))
            required.extend(anchor / "models" / source / f"mean14_{seed}/training_trace.json"
                            for seed in range(1706, 1711))
        require(required)
    else:
        inputs = exp.code / "data/prepared"
        require([mixed / "verification.json"])
        if exp.version == "PU_0_all_conditions_721":
            require([shared / name for name in ("windows.json", "vibration_noise_templates.npy")])
            normalization = ("normalization.json",)
        else:
            require([mixed / name for name in ("windows.json", "split_indices.npz", "signals.npy",
                                               "vibration_noise_templates.npy", "physical_all.npy",
                                               "wrong_all.npy", "statistics_all.npy")])
            require([inputs / "parent_indices.npz"])
            normalization = tuple(f"normalization_{name}.json"
                                  for name in ("balanced10", "imbalanced", "matched_balanced"))
        require([inputs / name for name in (*normalization, "records.json", "wrong_all.npy", "statistics_all.npy")])
    fields = ["verification.json", "signals.npy", "windows.json", "vibration_noise_templates.npy"]
    if inputs != shared:
        fields.append("split_indices.npz")
    if exp.style == "pu":
        fields.append("archive_hashes.json")
    require([inputs / name for name in fields])
    if exp.style == "pu":
        archives = json.loads((inputs / "archive_hashes.json").read_text(encoding="utf-8-sig"))
        paths = []
        for row in archives.values():
            parts = PureWindowsPath(row["path"]).parts
            index = next(i for i in range(len(parts) - 3) if parts[i:i+3] == ("data", "raw", "PU"))
            paths.append(ROOT.joinpath(*parts[index:]))
        require(paths)
        if exp.version == "PU_0_all_conditions_721":
            tool = os.environ.get("PU_RAR_TOOL") or shutil.which("bsdtar") or shutil.which("tar")
            if not tool:
                raise RuntimeError("四工况预检需要 bsdtar 或可读取 RAR 的 tar。")
            trial = subprocess.run([tool, "-tf", str(paths[0])], capture_output=True, env=child_env())
            if trial.returncode or not trial.stdout:
                raise RuntimeError(f"{tool} 无法读取 PU RAR。请安装 bsdtar，或设置 PU_RAR_TOOL。")
            print(f"PU RAR 读取检查通过 | {tool}")


def commands(exp, tag):
    out = exp.result / f"{exp.prefix}_{tag}"
    if exp.style == "pu":
        cache = exp.code / "data/harmonic_calibration"
        probe = exp.code / "checks" / f"server_{tag}"
        concat = exp.version == "PU_0_concat"
        extra = ("--cache", str(cache)) if concat else ()
        prepare = [] if cache.exists() else [exp.python("prepare.py")]
        return out, [
            *prepare,
            exp.python("preflight.py" if concat else "check.py", "--output", str(probe), *extra),
            exp.python(exp.script, "--output", str(out), "--preflight", str(probe / "preflight.json"), *extra),
        ]
    if exp.style == "harmonic":
        return out, [
            exp.python(exp.script, *exp.args, "--preflight", "--output", str(out.with_name(f"check_{tag}"))),
            exp.python(exp.script, *exp.args, "--output", str(out)),
            exp.python("audit_mean.py", "--output", str(out)),
        ]
    if exp.style == "optimization":
        return out, [exp.python(exp.script, step, "--output", str(out)) for step in ("init", "run-autonomous")]
    if exp.style == "selected":
        return out, [exp.python(exp.script, "--selection-output", str(out.with_name(f"optimization_selection_{tag}")),
                                "--full-output", str(out))]
    return out, [exp.python(exp.script, *exp.args, "--output", str(out))]


def execute(exp, tag, *, dry_run):
    out, steps = commands(exp, tag)
    print(f"实验：{exp.label}\n输出：{out}", flush=True)
    if dry_run:
        for cmd in steps:
            print(shlex.join(cmd) if os.name != "nt" else subprocess.list2cmdline(cmd))
        return
    if out.exists():
        raise FileExistsError(f"结果目录已存在：{out}")
    check_environment(exp)
    check_inputs(exp)
    for cmd in steps:
        print(shlex.join(cmd) if os.name != "nt" else subprocess.list2cmdline(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, env=child_env(), check=True)
    print(f"入口已退出，结果和完成状态见 {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="列出可以独立手动运行的实验")
    check = sub.add_parser("check", help="检查环境，可按实验检查输入")
    check.add_argument("experiment", nargs="?", choices=CATALOG)
    for action in ("plan", "run"):
        selection = sub.add_parser(action)
        selection.add_argument("experiment", choices=CATALOG)
        selection.add_argument("--tag", help="输出名称中的批次标签")
    args = parser.parse_args()
    if args.action == "list":
        for name, exp in CATALOG.items():
            print(f"{name:<29} {exp.label}")
        return 0
    try:
        exp = CATALOG[args.experiment] if args.experiment else None
        if args.action == "check":
            check_environment(exp)
            if exp:
                check_inputs(exp)
                subprocess.run(exp.python(exp.script, "--help"), cwd=ROOT, env=child_env(),
                               check=True, stdout=subprocess.DEVNULL)
            print("检查通过")
        else:
            tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
            if len(tag) > 64 or not tag.isascii() or not tag.replace("_", "").replace("-", "").isalnum():
                parser.error("--tag 只接受 1 到 64 个英文字母、数字、连字符或下划线。")
            execute(exp, tag, dry_run=args.action == "plan")
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
