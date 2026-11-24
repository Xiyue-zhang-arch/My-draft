"""每条记录先切窗再70/15/15随机划分，原固定212维模型从头训练。"""

from pathlib import Path
from datetime import datetime
import argparse
import sys
import time
sys.dont_write_bytecode=True
import torch
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CWRU=HERE.parents[1]
PROJECT=CWRU.parents[3]
sys.path.insert(0,str(CWRU/'shared'))
import experiment_runner as R
from window_split import split_windows,split_description,SPLIT_SEED


def train_case(*args):
    """沿用公共优化与选模，另记验证前向成本，不计入训练轮耗时。"""
    original=R.predict
    validation_seconds=0.
    validation_calls=0
    def timed(*values,**keywords):
        nonlocal validation_seconds,validation_calls
        R.U.sync();start=time.perf_counter()
        value=original(*values,**keywords)
        R.U.sync();validation_seconds+=time.perf_counter()-start;validation_calls+=1
        return value
    R.predict=timed
    start=time.perf_counter()
    try:
        info=R.train_case(*args)
    finally:
        R.predict=original
    folder=args[3]
    history=pd.read_csv(folder/'training.csv')
    info.update(validation_forward_seconds=validation_seconds,validation_forward_calls=validation_calls,
                total_train_seconds=float(history.train_seconds.sum()),
                fit_train_validation_save_seconds=time.perf_counter()-start)
    R.write_json(folder/'parameters.json',info)
    return info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data-root',type=Path,default=PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
    parser.add_argument('--split-seed',type=int,default=SPLIT_SEED)
    parser.add_argument('--updates',type=int,default=600)
    parser.add_argument('--seeds',type=int,nargs='+',default=[1706,1707,1708])
    parser.add_argument('--subset-seeds',type=int,nargs='+',default=[4101,4102,4103])
    parser.add_argument('--ratios',type=float,nargs='+',default=[1.,.1,.2,.5])
    parser.add_argument('--methods',nargs='+',choices=['fusion','data_only'],default=['fusion','data_only'])
    parser.add_argument('--protocols',nargs='+',choices=['clean_only','awgn_mixed'],default=['clean_only','awgn_mixed'])
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    if args.updates<=0 or not set(args.ratios).issubset({1.,.1,.2,.5}):
        parser.error('updates必须为正，比例支持1、0.1、0.2、0.5')
    device=torch.device(args.device);R.U.DEVICE=device;R.U.configure()
    records,original_windows,original_raw=R.load_dataset(args.data_root)
    # 原读取器的负载角色只用于保留历史来源，不用于本轮分配。
    windows,allocation=split_windows(original_windows,args.split_seed)
    keep=original_windows.record_id.ne('OR014_3').to_numpy()
    raw=original_raw[keep]
    np.testing.assert_array_equal(windows.window_index,original_windows.loc[keep,'window_index'])
    assert len(windows)==667 and windows.groupby('role').size().to_dict()=={'train':467,'validation':100,'test':100}
    x=R.U.normalize(raw)
    records=records.loc[records.record_id.ne('OR014_3')].rename(columns={'role':'historical_role'}).copy()
    roles=windows.groupby('record_id').role.apply(lambda g:','.join(sorted(set(g))))
    records['window_roles']=records.record_id.map(roles)
    support,loads,history,overlap=split_description(windows,allocation)
    cases=[(protocol,method,ratio,subset,seed)
           for protocol in dict.fromkeys(args.protocols)
           for method in dict.fromkeys(args.methods)
           for ratio in dict.fromkeys(args.ratios)
           if method!='data_only' or ratio==1
           for subset in (dict.fromkeys(args.subset_seeds) if ratio<1 else [0])
           for seed in dict.fromkeys(args.seeds)]
    if not cases:parser.error('无训练组合；数据单分支仅全样本')
    out=R.output_directory(args.output or PROJECT/'experiments/CWRU/实验结果/CWRU'/f'experiment_3_{datetime.now():%Y%m%d_%H%M%S_%f}',args.data_root)
    windows.to_csv(out/'windows.csv',index=False)
    records.to_csv(out/'records.csv',index=False)
    allocation.to_csv(out/'record_allocation.csv',index=False)
    support.to_csv(out/'class_support.csv');loads.to_csv(out/'load_support.csv')
    history.to_csv(out/'historical_role_distribution.csv');overlap.to_csv(out/'record_overlap.csv',index=False)
    (out/'PROTOCOL.md').write_text((HERE/'PROTOCOL.md').read_text(encoding='utf-8'),encoding='utf-8')
    budgets={str(ratio):len(R.U.selected(windows,ratio,args.subset_seeds[0] if ratio<1 else 0))
             for ratio in dict.fromkeys(args.ratios)}
    cfg=dict(experiment_id=HERE.name,stage='training',execution='from_scratch',planned_models=len(cases),
             trained_models=0,evaluated_models=0,split_seed=args.split_seed,split_ratio=[.7,.15,.15],
             split_unit='within each original recording after non-overlapping windows',
             training_windows=467,validation_windows=100,test_windows=100,training_budget_windows=budgets,
             test_support=support.loc['test'].to_list(),
             excluded_record='OR014_3',short_record_exception=allocation[allocation.short_record_exception].to_dict('records'),
             same_record_train_test_overlap=int(overlap.query("role_a=='train' and role_b=='test'").iloc[0].shared_records),
             cross_load_generalization=False,record_independent_test=False,prior_test_exposed=True,
             historical_test_windows_now_training=int(history.loc['train','test']),
             models_reused=False,predictions_reused=False,model_config=R.U.CFG,
             hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
             torch=str(torch.__version__),numpy=np.__version__,device=str(device),cpu_threads=torch.get_num_threads(),
             precision='BF16 encoder on CUDA, remainder FP32',batch_size=32,
             updates=args.updates,seeds=args.seeds,subset_seeds=args.subset_seeds,ratios=args.ratios,
             methods=args.methods,protocols=args.protocols,
             optimizer='AdamW lr0.0005 weight_decay0.001; cosine eta_min0.00001',
             early_stopping='min200 updates; patience150; max600; epoch1 and crossing50 updates',
             selection='class-record balanced NLL; clean1 or clean0.5+each noisy0.1',label_smoothing=.05,
             scaler='selected clean training subset median/IQR; floor0.05, clip[-8,8]',
             noise='per occurrence 50% clean, otherwise U[-4,4]dB AWGN; one CE, no pairs',
             evaluation_snrs=[-4,-2,0,2,4],validation_noise_seed=6201,test_noise_seeds=[7201,7202,7203,7204,7205],
             test_noise_order='new100 test windows in saved order; not old180/200 observations',
             training_timing='all actual epochs, online physical extraction and backward; excludes validation',
             latency='batch1 normalization+online physics+network; excludes disk/H2D; 10 warmups/50 repeats',
             started_at=datetime.now().astimezone().isoformat())
    R.write_json(out/'config.json',cfg)
    print(f'OUTPUT {out}; planned_models={len(cases)}; train467 val100 test100; budgets={budgets}',flush=True)
    jobs=[]
    try:
        for case in cases:
            protocol,method,ratio,subset,seed=case
            folder=out/protocol/method/f'ratio_{round(ratio*100):03d}_subset_{subset}_seed_{seed}'
            info=train_case(x,windows,case,folder,args.updates,device)
            jobs.append((info,folder))
            cfg.update(trained_models=len(jobs),last_completed_case=list(case))
            R.write_json(out/'config.json',cfg)
            print(f'TRAINED {len(jobs)}/{len(cases)}',flush=True)
        cfg['stage']='all_models_selected';R.write_json(out/'config.json',cfg)
        cfg['stage']='testing';R.write_json(out/'config.json',cfg)
        metrics=[];complexity=[]
        for info,folder in jobs:
            rows,timing=R.evaluate_case(x,raw,windows,info,folder,device)
            metrics.extend(rows);complexity.append(timing)
            R.write_json(folder/'parameters.json',timing)
            cfg.update(evaluated_models=len(complexity),prediction_groups=len(metrics))
            R.write_json(out/'config.json',cfg)
            print(f'EVALUATED {len(complexity)}/{len(cases)}',flush=True)
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(complexity).to_csv(out/'complexity.csv',index=False)
        cfg['stage']='reporting';R.write_json(out/'config.json',cfg)
        from window_report import report
        report(out)
        cfg.update(stage='complete',completed_at=datetime.now().astimezone().isoformat())
        R.write_json(out/'config.json',cfg)
        print(f'COMPLETE {out}',flush=True)
    except BaseException as error:
        cfg.update(stage='failed',failure_type=type(error).__name__,failure_message=str(error))
        R.write_json(out/'config.json',cfg)
        raise


if __name__=='__main__':main()
