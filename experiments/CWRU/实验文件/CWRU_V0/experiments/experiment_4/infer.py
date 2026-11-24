"""按设置选择本排除实验新训练的模型，不加载原200窗口实验的权重。"""
from pathlib import Path
import argparse, sys
sys.dont_write_bytecode = True
from run_exclusion import R, PROJECT


def load_pipeline(protocol='awgn_mixed',method='fusion',ratio=1.,subset=0,seed=1706,device='cuda'):
    path=PROJECT/'experiments/CWRU/实验结果/CWRU/experiment_4'/protocol/method/f'ratio_{round(ratio*100):03d}_subset_{subset}_seed_{seed}'/'model.pt'
    classifier=R.load_model(path,R.torch.device(device))
    class Pipeline(R.torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=classifier
        def forward(self,x,rpm):
            observed=(x-x.mean(-1,keepdim=True))/(x.std(-1,unbiased=False,keepdim=True)+1e-8)
            return self.model(observed,rpm)[0]
    return Pipeline().eval()


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol',choices=['clean_only','awgn_mixed'],default='awgn_mixed')
    parser.add_argument('--method',choices=['fusion','data_only'],default='fusion')
    parser.add_argument('--ratio',type=float,choices=[1.,.1,.2,.5],default=1.)
    parser.add_argument('--subset',type=int,default=0)
    parser.add_argument('--seed',type=int,default=1706)
    args,remaining=parser.parse_known_args()
    if args.method=='data_only' and args.ratio!=1:parser.error('单数据分支只做全样本')
    if (args.ratio==1 and args.subset!=0) or (args.ratio<1 and args.subset not in [4101,4102,4103]):parser.error('训练比例与子集编号不匹配')
    path=PROJECT/'experiments/CWRU/实验结果/CWRU/experiment_4'/args.protocol/args.method/f'ratio_{round(args.ratio*100):03d}_subset_{args.subset}_seed_{args.seed}'/'model.pt'
    sys.argv=[sys.argv[0],'--model',str(path)]+remaining
    R.inference_cli()
