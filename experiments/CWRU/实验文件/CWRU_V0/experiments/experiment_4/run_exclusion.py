"""从头训练固定212维排除实验，两协议共66模型，训练与验证保持原设置。"""
from pathlib import Path
from datetime import datetime
import argparse, sys, platform
sys.dont_write_bytecode = True
import torch
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PROJECT=ROOT.parents[3]
sys.path.insert(0,str(ROOT/'shared'))
import experiment_runner as R


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--updates',type=int,default=600)
    parser.add_argument('--seeds',nargs='+',type=int,default=[1706,1707,1708])
    parser.add_argument('--subset-seeds',nargs='+',type=int,default=[4101,4102,4103])
    parser.add_argument('--ratios',nargs='+',type=float,default=[1.,.1,.2,.5])
    parser.add_argument('--methods',nargs='+',choices=['fusion','data_only'],default=['fusion','data_only'])
    parser.add_argument('--protocols',nargs='+',choices=['clean_only','awgn_mixed'],default=['clean_only','awgn_mixed'])
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    if args.updates<=0 or not set(args.ratios).issubset({1.,.1,.2,.5}):parser.error('updates必须为正，比例仅支持1、0.1、0.2、0.5')
    device=torch.device(args.device);R.U.DEVICE=device;R.U.configure()
    data_root=PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data'
    records,full_windows,full_raw=R.load_dataset(data_root)
    historical=pd.read_csv(PROJECT/'experiments/CWRU/实验结果/CWRU/experiment_1/clean_only/windows.csv')
    columns=['record_id','role','label','start_sample','end_sample','rpm']
    pd.testing.assert_frame_equal(full_windows[columns],historical[columns],check_dtype=False)
    excluded=full_windows.record_id.eq('OR014_3').to_numpy()
    assert excluded.sum()==20 and full_windows.loc[excluded,'role'].eq('test').all()
    windows=full_windows.loc[~excluded].reset_index(drop=True)
    raw=full_raw[~excluded];x=R.U.normalize(raw)
    records=records[records.record_id!='OR014_3']
    assert windows.groupby('role').size().to_dict()=={'test':180,'train':287,'validation':200}
    # 按原200窗口顺序生成相同噪声方向，再仅保留180窗口；不读取旧模型或预测。
    original_test=full_windows.role.eq('test').to_numpy()
    full_test_x=R.U.normalize(full_raw[original_test]);test_keep=~excluded[original_test]
    original_conditions=R.conditions
    def aligned_conditions(observed,protocol,testing):
        if testing:
            assert len(observed)==180
            for condition,seed,wave in original_conditions(full_test_x,protocol,True):
                yield condition,seed,wave[test_keep]
        else:
            yield from original_conditions(observed,protocol,False)
    R.conditions=aligned_conditions
    out=R.output_directory(args.output or PROJECT/'experiments/CWRU/实验结果/CWRU'/f'experiment_4_{datetime.now():%Y%m%d_%H%M%S_%f}',data_root)
    records.to_csv(out/'records.csv',index=False);windows.to_csv(out/'windows.csv',index=False)
    (out/'PROTOCOL.md').write_text((HERE/'PROTOCOL.md').read_text(encoding='utf-8'),encoding='utf-8')
    settings=dict(experiment_id='experiment_4',execution='from_scratch',
        excluded_record='OR014_3',only_test_membership_changed=True,training_windows=287,validation_windows=200,
        test_windows=180,test_support=[20,60,40,60],updates=args.updates,seeds=args.seeds,
        subset_seeds=args.subset_seeds,ratios=args.ratios,methods=args.methods,protocols=args.protocols,
        model_config=R.U.CFG,hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
        torch=str(torch.__version__),numpy=np.__version__,device=str(device),cpu_threads=torch.get_num_threads(),
        precision='BF16 encoder on CUDA, remainder FP32',batch_size=32,
        optimizer='AdamW lr=0.0005 weight_decay=0.001; cosine eta_min=0.00001',label_smoothing=.05,
        early_stopping='epoch 1 and at least every 50 updates; min200/patience150, max600',
        selection='class-record balanced validation NLL; clean1 or clean0.5 and each noisy condition0.1',
        scaler='selected clean training subset median/IQR, floor0.05, clip[-8,8]',
        latency='batch1 normalization+live features+network, no disk/H2D; 10 warmups/50 repeats',
        training_timing='all actual epochs, including online features and backward; excludes validation',
        noise='per occurrence 50% clean, otherwise U[-4,4] dB AWGN; one CE, no paired inputs',
        test_noise='generated in original 200-window order before restriction; 7201..7205 for each fixed SNR',
        evaluation_snrs=[-4,-2,0,2,4],validation_noise_seed=6201,quality_input=False,
        frequency_correction=False,prior_test_exposed=True,model_reuse=False,prediction_reuse=False,
        stage='training')
    R.write_json(out/'config.json',settings)
    jobs=[]
    try:
        for protocol in dict.fromkeys(args.protocols):
            for method in dict.fromkeys(args.methods):
                for ratio in dict.fromkeys(args.ratios):
                    if method=='data_only' and ratio!=1:continue
                    for subset in (dict.fromkeys(args.subset_seeds) if ratio<1 else [0]):
                        for seed in dict.fromkeys(args.seeds):
                            folder=out/protocol/method/f'ratio_{round(ratio*100):03d}_subset_{subset}_seed_{seed}'
                            info=R.train_case(x,windows,(protocol,method,ratio,subset,seed),folder,args.updates,device)
                            jobs.append((info,folder))
                            print(f'Completed training {len(jobs)} models',flush=True)
        if not jobs:raise ValueError('无训练组合')
        settings.update(stage='all_models_selected',trained_models=len(jobs));R.write_json(out/'config.json',settings)
        print(f'All {len(jobs)} models selected. Starting restricted test.',flush=True)
        metrics=[];complexity=[]
        for info,folder in jobs:
            rows,timing=R.evaluate_case(x,raw,windows,info,folder,device)
            metrics.extend(rows);complexity.append(timing)
            print(f"Evaluated {info['protocol']} {info['method']} ratio={info['ratio']} subset={info['subset']} seed={info['seed']}",flush=True)
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(complexity).to_csv(out/'complexity.csv',index=False)
        from report_exclusion import report
        report(out)
        settings.update(stage='complete',trained_models=len(jobs),prediction_groups=len(metrics));R.write_json(out/'config.json',settings)
        print(f'Completed: {out}',flush=True)
    finally:
        R.conditions=original_conditions


if __name__=='__main__':main()
