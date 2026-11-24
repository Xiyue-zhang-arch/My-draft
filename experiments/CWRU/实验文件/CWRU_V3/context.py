"""V3只读复用底层网络和数据组件，不调用V1/V2训练入口。"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
sys.path.append(str(PROJECT / 'experiments/CWRU/实验文件/CWRU_V1'))
import runtime as R
BASE = PROJECT / 'experiments/CWRU/实验结果/CWRU_V1/baseline_20261020'
HISTORY = PROJECT / 'experiments/CWRU/实验结果/CWRU_V2/workflow_20261020'
