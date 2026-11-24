"""补充实验3数据单分支小样本，不覆盖原66模型或重新测试旧融合模型。"""
from pathlib import Path
from datetime import datetime
import argparse
import sys
sys.dont_write_bytecode=True
import torch
import numpy as np
import pandas as pd
from run_window_split import R,train_case,PROJECT,HERE
from window_split import split_windows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--source',type=Path,default=PROJECT/'experiments/CWRU/实验结果/CWRU/experiment_3')
    p.add_argument('--data-root',type=Path,default=PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args();source=a.source.resolve()
    old=__import__('json').loads((source/'config.json').read_text(encoding='utf-8'))
    assert old['stage']=='complete'
    device=torch.device(a.device);R.U.DEVICE=device;R.U.configure()
    records,original,raw=R.load_dataset(a.data_root)
    windows,allocation=split_windows(original,old['split_seed'])
    pd.testing.assert_frame_equal(windows,pd.read_csv(source/'windows.csv'),check_dtype=False)
    raw=raw[original.record_id.ne('OR014_3').to_numpy()];x=R.U.normalize(raw)
    cases=[(protocol,'data_only',ratio,subset,seed) for protocol in ['clean_only','awgn_mixed']
        for ratio in [.1,.2,.5] for subset in [4101,4102,4103] for seed in [1706,1707,1708]]
    for protocol,method,ratio,subset,seed in cases:
        ref=source/protocol/'fusion'/f'ratio_{round(ratio*100):03d}_subset_{subset}_seed_{seed}'/'selected_windows.csv'
        pd.testing.assert_frame_equal(windows.iloc[R.U.selected(windows,ratio,subset)].reset_index(drop=True),pd.read_csv(ref),check_dtype=False)
    out=R.output_directory(a.output or PROJECT/'experiments/CWRU/实验结果/CWRU'/f'experiment_3_data_small_{datetime.now():%Y%m%d_%H%M%S_%f}',a.data_root)
    for name in ['windows.csv','records.csv','record_allocation.csv','class_support.csv','load_support.csv','historical_role_distribution.csv','record_overlap.csv']:
        (out/name).write_bytes((source/name).read_bytes())
    (out/'PROTOCOL.md').write_text((HERE/'SMALL_DATA_PROTOCOL.md').read_text(encoding='utf-8'),encoding='utf-8')
    cfg=dict(old,experiment_id='experiment_3_data_small_supplement',source_result=str(source),stage='training',
        planned_models=54,trained_models=0,evaluated_models=0,prediction_groups=0,planned_metric_rows=729,
        methods=['data_only'],ratios=[.1,.2,.5],seeds=[1706,1707,1708],subset_seeds=[4101,4102,4103],
        model_config=R.U.CFG,updates=600,new_models_from_scratch=True,old_fusion_predictions_reused_for_comparison=True,
        old_models_retrained=False,old_models_retested=False,hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        torch=str(torch.__version__),cpu_threads=torch.get_num_threads(),device=str(device),
        started_at=datetime.now().astimezone().isoformat())
    cfg.pop('completed_at',None);R.write_json(out/'config.json',cfg)
    print(f'OUTPUT {out}; planned_models54; budgets48/95/235; validation100; test100',flush=True)
    jobs=[]
    try:
        for case in cases:
            protocol,method,ratio,subset,seed=case
            folder=out/protocol/method/f'ratio_{round(ratio*100):03d}_subset_{subset}_seed_{seed}'
            info=train_case(x,windows,case,folder,600,device)
            jobs.append((info,folder));cfg.update(trained_models=len(jobs),last_completed_case=list(case))
            R.write_json(out/'config.json',cfg);print(f'TRAINED {len(jobs)}/54',flush=True)
        cfg['stage']='all_models_selected';R.write_json(out/'config.json',cfg)
        cfg['stage']='testing';R.write_json(out/'config.json',cfg)
        metrics=[];complexity=[]
        for info,folder in jobs:
            rows,timing=R.evaluate_case(x,raw,windows,info,folder,device)
            metrics.extend(rows);complexity.append(timing);R.write_json(folder/'parameters.json',timing)
            cfg.update(evaluated_models=len(complexity),prediction_groups=len(metrics));R.write_json(out/'config.json',cfg)
            print(f'EVALUATED {len(complexity)}/54',flush=True)
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(complexity).to_csv(out/'complexity.csv',index=False)
        cfg['stage']='reporting';R.write_json(out/'config.json',cfg)
        from small_data_report import report
        report(out,source)
        cfg.update(stage='complete',completed_at=datetime.now().astimezone().isoformat());R.write_json(out/'config.json',cfg)
        print('COMPLETE',out,flush=True)
    except BaseException as error:
        cfg.update(stage='failed',failure_type=type(error).__name__,failure_message=str(error));R.write_json(out/'config.json',cfg)
        raise


if __name__=='__main__':main()
