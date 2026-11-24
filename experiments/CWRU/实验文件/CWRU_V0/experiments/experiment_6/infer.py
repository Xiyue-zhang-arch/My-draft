"""单一来源模型推理，沿用公共固定212维接口。"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'shared'))
import experiment_runner as R
if __name__=='__main__':R.inference_cli()
