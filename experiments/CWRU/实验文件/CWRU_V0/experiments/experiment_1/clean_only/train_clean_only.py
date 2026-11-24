"""CWRU纯干净训练入口。执行后从原始MAT数据训练、评估并保存图表。"""

from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "shared"))
from experiment_runner import main

if __name__ == "__main__":
    main(protocol="clean_only")
