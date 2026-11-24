"""只读解析48 kHz CWRU MAT文件，按原始记录划分无重叠窗口。"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat

LABELS = ["Normal", "IR", "OR", "B"]
WINDOW = 24000


def load_dataset(data_root, records_csv=None):
    """记录表可覆盖文件名推断，列为path、variable、label、role、rpm、sample_rate。"""
    root = Path(data_root).resolve(strict=True)
    if records_csv:
        records = pd.read_csv(records_csv)
        required = {"path", "variable", "label", "role", "rpm", "sample_rate"}
        if not required.issubset(records.columns):
            raise ValueError(f"记录表缺少列：{required - set(records.columns)}")
    else:
        # 接受项目raw目录、CWRU目录或具体的48 kHz目录。
        for subdir in [
            "CWRU/48k Drive End Bearking Fault Data",
            "48k Drive End Bearking Fault Data",
        ]:
            if (root / subdir).is_dir():
                root = root / subdir
                break
        records = []
        for path in sorted(root.glob("*.mat")):
            match = re.fullmatch(r"(Normal|IR\d+|OR\d+|B\d+)_([0-3])", path.stem)
            if not match:
                raise ValueError(
                    f"无法确定标签和负载：{path.name}。请用--records提供记录表。"
                )
            kind, load = match.groups()
            label = next(i for i, name in enumerate(LABELS) if kind.startswith(name))
            records.append(
                dict(
                    path=path.name,
                    label=label,
                    load=int(load),
                    role=["train", "train", "validation", "test"][int(load)],
                )
            )
        records = pd.DataFrame(records)
    if records.empty:
        raise ValueError("目录中没有可用MAT文件。")

    signals, windows, metadata, seen = [], [], [], set()
    # 原始记录含多个DE变量时，当前公开数据的已知目标采集编号。
    aliases = {
        "Normal_2": "X099_DE_time",
        "IR014_1": "X175_DE_time",
        "IR021_3": "X217_DE_time",
    }
    for rid, row in enumerate(records.to_dict("records")):
        path = (root / row["path"]).resolve(strict=True)
        if path in seen:
            raise ValueError(f"同一原始文件重复分配：{path}")
        seen.add(path)
        if row["role"] not in {"train", "validation", "test"} or row[
            "label"
        ] not in range(4):
            raise ValueError(f"无效标签或划分：{row}")
        if records_csv and row["sample_rate"] != 48000:
            raise ValueError("当前物理频带和模型仅支持48 kHz输入。")
        mat = loadmat(path)
        variable = row.get("variable")
        if not variable:
            candidates = [key for key in mat if key.endswith("_DE_time")]
            variable = candidates[0] if len(candidates) == 1 else aliases.get(path.stem)
            if variable not in candidates:
                raise ValueError(f"{path.name}含多个DE变量，请提供--records。")
        signal = np.asarray(mat[variable]).ravel().astype(np.float32)
        rpm = row.get("rpm")
        if rpm is None:
            rpm_key = variable.split("_DE_time")[0] + "RPM"
            if rpm_key in mat:
                rpm = float(np.asarray(mat[rpm_key]).item())
                rpm_source = "MAT"
            elif path.stem in {"Normal_1", "Normal_2"}:
                rpm = {"Normal_1": 1772.0, "Normal_2": 1750.0}[path.stem]
                rpm_source = "nominal_load_table"
            else:
                raise ValueError(f"{path.name}缺少转速，请提供--records。")
        else:
            rpm_source = "records_csv"
        if not np.isfinite(signal).all() or not np.isfinite(rpm) or rpm <= 0:
            raise ValueError(f"信号或转速含无效值：{path}")
        if len(signal) < WINDOW:
            raise ValueError(f"{path.name}短于一个24000点窗口。")
        metadata.append(
            dict(
                record_index=rid,
                record_id=path.stem,
                path=str(path),
                variable=variable,
                label=int(row["label"]),
                role=row["role"],
                rpm=float(rpm),
                rpm_source=rpm_source,
                sample_rate=48000,
                samples=len(signal),
            )
        )
        for start in range(0, len(signal) - WINDOW + 1, WINDOW):
            signals.append(signal[start : start + WINDOW][None])
            windows.append(
                dict(
                    window_index=len(windows),
                    record_index=rid,
                    record_id=path.stem,
                    role=row["role"],
                    label=int(row["label"]),
                    rpm=float(rpm),
                    start_sample=start,
                    end_sample=start + WINDOW,
                )
            )
    windows = pd.DataFrame(windows)
    for role in ["train", "validation", "test"]:
        if set(windows.loc[windows.role == role, "label"]) != set(range(4)):
            raise ValueError(f"{role}必须包含四类窗口。")
    return pd.DataFrame(metadata), windows, np.stack(signals)
