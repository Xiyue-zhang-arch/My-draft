"""独立V1全样本基线。默认执行100模型，仅写新输出目录。"""
from pathlib import Path
from datetime import datetime
import argparse
import sys
sys.dont_write_bytecode=True
import torch
import numpy as np
import pandas as pd
import runtime as R
from splits import save_splits,read_fixed
from reporting import scores,finish,markdown

HERE=Path(__file__).resolve().parent
SEEDS=[1706,1707,1708,1709,1710]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--prepared',type=Path,help='仅继续stage=prepared且无模型的本版新输出')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    if args.prepared and (args.output or args.prepare_only):
        parser.error('--prepared不能与--output或--prepare-only同时使用')
    device=torch.device(args.device)
    R.U.DEVICE=device;R.U.configure()
    data_root=R.PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data'
    records,original,raw=R.load_dataset(data_root)
    assert len(records)==40 and len(original)==687
    x=R.U.normalize(raw)
    if args.prepared:
        out=args.prepared.resolve(strict=True)
        if out.parent!=R.PROJECT/'experiments/CWRU/实验结果/CWRU_V1':
            raise ValueError('prepared仅接受experiments/CWRU/实验结果/CWRU_V1的直接子目录')
        cfg=__import__('json').loads((out/'config.json').read_text(encoding='utf-8'))
        if cfg.get('stage')!='prepared' or cfg.get('experiment')!='CWRU_V1_baseline' or (out/'models').exists():
            raise ValueError('仅允许启动未训练的新准备目录，不覆盖或续训已有模型')
        if cfg['device']!=str(device):
            raise ValueError('设备与准备阶段不一致')
        saved=pd.read_csv(out/'records.csv')
        pd.testing.assert_frame_equal(saved,records,check_dtype=False)
        for name in cfg['split_names']:
            w=read_fixed(out/'splits'/name)
            for c in ['window_index','record_id','label','rpm','start_sample','end_sample']:
                np.testing.assert_array_equal(w[c],original[c])
    else:
        out=(args.output or R.PROJECT/'experiments/CWRU/实验结果/CWRU_V1'/f'baseline_{datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
        if out.parent!=R.PROJECT/'experiments/CWRU/实验结果/CWRU_V1':
            raise ValueError('新输出必须直接位于experiments/CWRU/实验结果/CWRU_V1下')
        out.mkdir(parents=True,exist_ok=False)
        records.to_csv(out/'records.csv',index=False)
        tables=save_splits(original,records,out/'splits')
        cfg=dict(experiment='CWRU_V1_baseline',stage='prepared',started_at=datetime.now().astimezone().isoformat(),
            planned_models=100,trained_models=0,planned_evaluations=260,evaluated_pairs=0,planned_metric_rows=3510,
            split_names=list(tables),excluded_records=[],original_records=40,original_windows=687,
            source_training_windows=[73,156,160,160],source_validation_windows=[19,39,40,40],
            seeds=SEEDS,protocols=['clean_only','awgn_mixed'],methods=['fusion','data_only'],
            max_epochs=100,patience=20,validation_interval_epochs=1,selection_min_delta=1e-6,
            loss='cross_entropy',label_smoothing=0.,batch_size=32,from_scratch=True,
            optimizer='AdamW lr0.0005 weight_decay0.001',scheduler='cosine epoch T_max100 eta_min0.00001',
            sampling='class then record then window with replacement; ceil(N/32)*32 per epoch',
            selection='class-record balanced NLL; clean_only clean1; awgn clean0.5+each noise0.1',
            validation_noise_seed=6201,test_noise_seeds=[7201,7202,7203,7204,7205],
            small_sample=False,budget_matched=False,structure_exploration=False,prior_test_exposed=True,
            model_config=R.U.CFG,hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            device=str(device),torch=str(torch.__version__),precision='CUDA encoder BF16, remainder FP32',
            instability_threshold_percentage_points=.5)
        (out/'PROTOCOL.md').write_text((HERE/'PROTOCOL.md').read_text(encoding='utf-8'),encoding='utf-8')
        complexity=[R.complexity(m,device) for m in cfg['methods']]
        pd.DataFrame(complexity).to_csv(out/'complexity.csv',index=False)
        support=pd.read_csv(out/'dataset_support.csv')
        (out/'DATA_USAGE.md').write_text('# V1基线数据使用\n\n全部40个MAT记录，687完整无重叠窗，每窗24000点，48kHz。'
            'data只读；各记录尾部不足一窗部分不用，不选择性丢弃完整窗。\n\n'+markdown(support)+
            '\n\n逐文件原始路径、通道变量、原始采样点、尾部点和集合用量见[dataset_file_usage.csv](dataset_file_usage.csv)。'
            '固定窗口分配见splits下windows.csv及train/validation/test的txt。文本内为全局窗口编号，不是新增数据。'
            '窗口随机划分训练测试共享记录，IR014_0两窗不能同时分入三集合；按负载方案来源记录内8:2划分，'
            '未抽取共同预算，来源全部训练候选均保留。',encoding='utf-8')
        R.write_json(out/'config.json',cfg)
    print('OUTPUT',out,flush=True)
    if args.prepare_only:
        print('PREPARED ONLY; no model training or target inference',flush=True)
        return
    jobs=[]
    try:
        cfg['stage']='training';R.write_json(out/'config.json',cfg)
        for name in cfg['split_names']:
            # 每个配置使用同一份已保存划分，禁止训练函数内重新划分。
            for protocol in cfg['protocols']:
                for method in cfg['methods']:
                    for seed in cfg['seeds']:
                        w=read_fixed(out/'splits'/name)
                        folder=out/'models'/name/protocol/method/f'seed_{seed}'
                        info=R.train(x,w,protocol,method,seed,folder,device,epochs=cfg['max_epochs'],patience=cfg['patience'])
                        info.update(split='window_random' if name=='window_random' else 'load_transfer',
                            split_name=name,source_load_hp=-1 if name=='window_random' else int(name[-1]))
                        R.write_json(folder/'parameters.json',info)
                        jobs.append((folder,info))
                        cfg.update(trained_models=len(jobs),last_completed_model=str(folder.relative_to(out)))
                        R.write_json(out/'config.json',cfg)
                        print(f'TRAINED {len(jobs)}/100',flush=True)
        cfg['stage']='all_models_selected';R.write_json(out/'config.json',cfg)
        # 权重均由来源/训练域验证选出，此后不再训练。
        cfg['stage']='testing';R.write_json(out/'config.json',cfg)
        metric_rows=[];evaluation_rows=[];training_rows=[]
        for folder,info in jobs:
            w=read_fixed(out/'splits'/info['split_name'])
            model=R.load_model(folder,device)
            targets=[-1] if info['split']=='window_random' else [t for t in range(4) if t!=info['source_load_hp']]
            latency_ms=None
            for target in targets:
                mask=w.role.eq('test') if target==-1 else w.load_hp.eq(target)
                ix=np.flatnonzero(mask)
                labels=w.iloc[ix].label.to_numpy();rpm=w.iloc[ix].rpm.to_numpy(np.float32)
                destination=folder/('test_random' if target==-1 else f'test_hp{target}')
                destination.mkdir(exist_ok=False)
                metadata={k:info[k] for k in ['split','source_load_hp','protocol','method','seed']}
                metadata['target_load_hp']=target
                arrays=dict(labels=labels,window_indices=w.iloc[ix].window_index.to_numpy())
                for condition,noise_seed,observed in R.conditions(x[ix],info['protocol'],True):
                    p=R.predict(model,observed,rpm,R.U.DEVICE)
                    values,_=scores(labels,p)
                    arrays[f'{condition}_n{noise_seed}']=p
                    metric_rows.append(dict(**metadata,condition=condition,noise_seed=noise_seed,**values))
                np.savez_compressed(destination/'predictions.npz',**arrays)
                R.write_json(destination/'evaluation.json',metadata)
                if latency_ms is None:
                    latency_ms=R.latency(model,raw[ix],rpm,device)
                evaluation_rows.append(dict(**metadata,test_windows=len(ix),latency_ms=latency_ms))
                cfg.update(evaluated_pairs=len(evaluation_rows),metric_rows=len(metric_rows))
                R.write_json(out/'config.json',cfg)
                print(f'EVALUATED {len(evaluation_rows)}/260',flush=True)
            info['latency_ms']=latency_ms
            R.write_json(folder/'parameters.json',info)
            training_rows.append(info)
            del model
        pd.DataFrame(metric_rows).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(evaluation_rows).to_csv(out/'evaluations.csv',index=False)
        pd.DataFrame(training_rows).to_csv(out/'training_summary.csv',index=False)
        cfg['stage']='reporting';R.write_json(out/'config.json',cfg)
        finish(out)
        cfg.update(stage='complete',completed_at=datetime.now().astimezone().isoformat())
        R.write_json(out/'config.json',cfg)
        print('COMPLETE',out,flush=True)
    except BaseException as error:
        cfg.update(stage='failed',failure_type=type(error).__name__,failure_message=str(error))
        R.write_json(out/'config.json',cfg)
        raise


if __name__=='__main__':
    main()
