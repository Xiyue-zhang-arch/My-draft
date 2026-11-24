"""全部40记录的单一来源负载迁移，共同73训练窗，80模型、12方向。"""
from pathlib import Path
from datetime import datetime
import argparse
import sys
sys.dont_write_bytecode=True
import torch
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[5]
sys.path.insert(0,str(HERE.parents[1]/'shared'))
import experiment_runner as R
from transfer_split import split_source,for_target,SPLIT_SEED,SELECTION_SEED,QUOTAS
from transfer_training import train_case


def evaluate_target(x,raw,w,info,model_path,folder,device,latency):
    model=R.load_model(model_path,device)
    ix=np.flatnonzero(w.role.eq('test'))
    labels=w.iloc[ix].label.to_numpy();rpm=w.iloc[ix].rpm.to_numpy(np.float32)
    rows=[];arrays={}
    for condition,noise_seed,observed in R.conditions(x[ix],info['protocol'],True):
        probabilities=R.predict(model,observed,rpm,device)
        arrays[f'{condition}_n{noise_seed}']=probabilities
        rows.append({k:info[k] for k in ['protocol','method','ratio','subset','seed']}|dict(condition=condition,noise_seed=noise_seed,**R.scores(labels,probabilities)))
    for condition in sorted({row['condition'] for row in rows}):
        probability=np.stack([v for k,v in arrays.items() if k.startswith(condition+'_n')])
        arrays[condition+'_confusion_counts']=R.confusion_plot(labels,probability,folder/f'confusion_{condition}.png')
    np.savez_compressed(folder/'predictions.npz',labels=labels,window_indices=w.iloc[ix].window_index.to_numpy(),array_indices=ix,**arrays)
    measured=latency is None
    if measured:latency=R.inference_latency(model,raw[ix],rpm,device)
    return rows,dict(info,latency_ms=latency,latency_independently_measured_here=measured,
        latency_scope='one measurement on first target per source model, reused for other targets')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data-root',type=Path,default=PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=20)
    args=parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error('--epochs和--patience必须为正整数')
    device=torch.device(args.device);R.U.DEVICE=device;R.U.configure()
    records,original,raw=R.load_dataset(args.data_root)
    assert len(original)==687 and len(records)==40
    x=R.U.normalize(raw)
    out=R.output_directory(args.output or PROJECT/'experiments/CWRU/实验结果/CWRU'/f'experiment_6_{datetime.now():%Y%m%d_%H%M%S_%f}',args.data_root)
    records.rename(columns={'role':'historical_role'}).to_csv(out/'records.csv',index=False)
    (out/'PROTOCOL.md').write_text((HERE/'PROTOCOL.md').read_text(encoding='utf-8'),encoding='utf-8')
    cases=[(p,m,1.,0,s) for p in ['clean_only','awgn_mixed'] for m in ['fusion','data_only'] for s in [1706,1707,1708,1709,1710]]
    cfg=dict(experiment='single_source_load_transfer_all_records',stage='training',planned_models=80,trained_models=0,
        planned_evaluations=240,evaluated_models=0,evaluated_pairs=0,planned_metric_rows=3240,prediction_groups=0,
        from_scratch=True,old_models_reused=False,predictions_reused=False,source_model_reused_across_targets=True,
        source_loads=[0,1,2,3],directions=[[s,t] for s in range(4) for t in range(4) if s!=t],
        inner_split_seed=SPLIT_SEED,training_selection_seed=SELECTION_SEED,training_windows=73,
        training_class_quotas=list(QUOTAS),candidate_train_windows_by_source=[73,156,160,160],
        validation_windows_by_source=[19,39,40,40],validation_budget_matched_across_sources=False,
        test_windows_by_load=[92,195,200,200],excluded_records=[],original_records=40,original_windows=687,
        small_sample_experiment=False,protocols=['clean_only','awgn_mixed'],methods=['fusion','data_only'],
        seeds=[1706,1707,1708,1709,1710],max_epochs=args.epochs,batch_size=32,
        validation_interval_epochs=1,early_stopping=True,early_stopping_patience=args.patience,
        selection_min_delta=1e-6,loss='cross_entropy',label_smoothing=0.0,
        optimizer='AdamW lr0.0005 weight_decay0.001 cosine eta_min0.00001',scheduler_interval='epoch',
        sampling='class then record then window with replacement, exactly32 observations per update',
        selection='source-only class-record balanced NLL; clean1 or clean0.5+each noisy0.1',
        model_config=R.U.CFG,hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        torch=str(torch.__version__),device=str(device),cpu_threads=torch.get_num_threads(),
        precision='CUDA encoder BF16, remainder FP32',validation_noise_seed=6201,test_noise_seeds=[7201,7202,7203,7204,7205],
        latency='batch1 normalization+online physics+network; no disk/H2D; warmup10 repeat50; once per source model',
        prior_test_exposed=True,posthoc_exclusion=False,independent_bearing_generalization=False,
        started_at=datetime.now().astimezone().isoformat())
    R.write_json(out/'config.json',cfg)
    jobs=[];directions=[]
    print(f'OUTPUT {out}; planned_models=80; directions=12; train73; all40records',flush=True)
    try:
        for source in range(4):
            w=split_source(original,source)
            root=out/f'source_hp{source}';root.mkdir()
            w.to_csv(root/'windows.csv',index=False)
            pd.crosstab(w.role,w.label).to_csv(root/'class_support.csv')
            for target in range(4):
                if source==target:continue
                tw=for_target(w,target);d=root/f'test_hp{target}';d.mkdir()
                tw.to_csv(d/'windows.csv',index=False)
                directions.append(dict(source_load_hp=source,target_load_hp=target,training_windows=73,
                    validation_windows=int(w.role.eq('validation').sum()),test_windows=int(tw.role.eq('test').sum())))
            for case in cases:
                protocol,method,ratio,subset,seed=case
                folder=root/protocol/method/f'ratio_100_subset_0_seed_{seed}'
                info=train_case(R,x,w,case,folder,args.epochs,device,patience=args.patience)
                info['source_load_hp']=source;R.write_json(folder/'parameters.json',info)
                jobs.append((info,folder,w))
                cfg.update(trained_models=len(jobs),last_completed_case=[source,*case]);R.write_json(out/'config.json',cfg)
                print(f'TRAINED {len(jobs)}/80 source={source}',flush=True)
        pd.DataFrame(directions).to_csv(out/'directions.csv',index=False)
        cfg['stage']='all_source_models_selected';R.write_json(out/'config.json',cfg)
        cfg['stage']='testing';R.write_json(out/'config.json',cfg)
        metrics=[];complexity=[];evaluations=[]
        for info,folder,w in jobs:
            source=info['source_load_hp'];latency=None
            for target in range(4):
                if target==source:continue
                d=folder/f'test_hp{target}';d.mkdir()
                rows,timing=evaluate_target(x,raw,for_target(w,target),dict(info,target_load_hp=target),folder/'model.pt',d,device,latency)
                latency=timing['latency_ms']
                for row in rows:row.update(source_load_hp=source,target_load_hp=target)
                metrics.extend(rows);evaluations.append(timing);R.write_json(d/'parameters.json',timing)
                cfg.update(evaluated_pairs=len(evaluations),prediction_groups=len(metrics));R.write_json(out/'config.json',cfg)
                print(f'EVALUATED {len(evaluations)}/240 source={source} target={target}',flush=True)
            timing=dict(info,latency_ms=latency)
            complexity.append(timing);R.write_json(folder/'parameters.json',timing)
            cfg.update(evaluated_models=len(complexity));R.write_json(out/'config.json',cfg)
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(complexity).to_csv(out/'complexity.csv',index=False)
        pd.DataFrame(evaluations).to_csv(out/'evaluations.csv',index=False)
        cfg['stage']='reporting';R.write_json(out/'config.json',cfg)
        from transfer_report import report
        report(out)
        cfg.update(stage='complete',completed_at=datetime.now().astimezone().isoformat());R.write_json(out/'config.json',cfg)
        print('COMPLETE',out,flush=True)
    except BaseException as error:
        cfg.update(stage='failed',failure_type=type(error).__name__,failure_message=str(error));R.write_json(out/'config.json',cfg)
        raise


if __name__=='__main__':main()
