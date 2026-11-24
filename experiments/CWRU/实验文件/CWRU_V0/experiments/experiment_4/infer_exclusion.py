"""对排除实验从头训练的模型推理，使用包含结构配置的model.pt。"""
import sys
sys.dont_write_bytecode = True
from run_exclusion import R

if __name__=='__main__':
    R.inference_cli()
