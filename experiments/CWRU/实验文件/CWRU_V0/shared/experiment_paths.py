"""公共路径与输出保护，不导入模型或启动计算。"""

from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[3]
CWRU_RESULTS = ROOT.parents[1] / "实验结果"
RESULTS = CWRU_RESULTS / "CWRU"
CWRU_VERSIONS = ("CWRU", *(f"CWRU_V{version}" for version in range(1, 7)))


def resolve_result_path(path):
    """Resolve a stored CWRU result path after moving the repository or changing OS."""
    path = Path(path)
    local = path if path.is_absolute() else PROJECT / path
    if local.exists():
        return local
    parts = PureWindowsPath(str(path)).parts
    if '..' in parts:
        return path
    for index, part in enumerate(parts[:-1]):
        if parts[index + 1] not in CWRU_VERSIONS:
            continue
        if part == "results" or parts[max(0, index - 2):index + 1] == ("experiments", "CWRU", "实验结果"):
            return CWRU_RESULTS.joinpath(*parts[index + 1:])
    return path


def validate_output_path(path, data_root=None, *, existing=False):
    """拒绝写入数据、代码及已有实验内部；报告仅可首次生成。"""
    path = Path(path).resolve()
    protected = [PROJECT / "data"]
    if CWRU_RESULTS not in path.parents:
        protected.append(PROJECT / "experiments")
    if data_root is not None:
        protected.append(Path(data_root))
    for directory in protected:
        directory = directory.resolve()
        if path == directory or directory in path.parents:
            raise ValueError(f"输出位置受保护：{path}")
    # 新实验可在 CWRU 结果目录下创建，但不能嵌入已存在的实验结果中。
    for parent in path.parents:
        if parent.parent == RESULTS and parent.exists():
            raise ValueError(f"不能写入已有实验结果内部：{path}")
    if existing:
        if not path.is_dir():
            raise ValueError(f"报告输入目录不存在：{path}")
        if (path / "REPORT.md").exists():
            raise FileExistsError(f"已有正式报告，禁止覆盖：{path}")
    elif path.exists():
        raise FileExistsError(f"输出目录已存在：{path}")
    return path


def output_directory(path, data_root=None):
    """检查后新建独立输出目录。"""
    path = validate_output_path(path, data_root)
    path.mkdir(parents=True, exist_ok=False)
    return path
