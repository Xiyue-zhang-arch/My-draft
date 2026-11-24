"""Scheme one, 10% only: window and cross-load clean/AWGN full evaluation."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
from unittest.mock import patch

import torch  # Required Windows DLL load order.
import numpy as np
import pandas as pd

from .model import MODEL_CONFIG, SuiteModel, engine, load, train, original
from experiments.CWRU_V4.run_v4 import read_dataset
from experiments.CWRU_V5.physics12_adapted_reference import extract as reference_extract
from metrics import scores, METRICS
from run_experiment import write_json, split_name, nested_subset
from splits import read_fixed

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = PROJECT/'results/CWRU_Physics6'
WINDOW_REF = PROJECT/'results/CWRU_V3/window_random_20261023_0853'
LOAD_REF = PROJECT/'results/CWRU_V3/load_small_20261025_2017'
SEEDS, SUBSETS = tuple(range(1706,1711)), (4101,4102,4103)
PROTOCOLS = ('clean_only','awgn_mixed')
CONDITIONS = (('clean',0), *((f'white_{snr:+d}', seed) for snr in (-4,-2,0,2,4) for seed in range(7201,7206)))
CASE_KEYS = ['scene','source_load_hp','subset','protocol','seed']
CASES = [dict(scene=scene, source_load_hp=source, subset=subset, protocol=protocol, seed=seed,
              domain='window_random' if scene=='window' else f'load_source_{source}', ratio=.1)
         for scene, sources in [('window',(-1,)),('load',range(4))]
         for source in sources for subset in SUBSETS for protocol in PROTOCOLS for seed in SEEDS]


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def spec_for(protocol):
    return engine.normalize_spec(dict(aux_weight=.3, physics_fit='clean' if protocol=='clean_only' else 'matched_noise'))


def casekey(case):
    return tuple(case[k] for k in CASE_KEYS)


def prepare(root):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this authorized experiment')
    engine.R.U.configure()
    records, original_windows, x = read_dataset()
    tables = {}
    for scene, reference in [('window',WINDOW_REF),('load',LOAD_REF)]:
        for source in ((-1,) if scene=='window' else range(4)):
            full = read_fixed(reference/'splits'/split_name('window_random' if source==-1 else f'load_source_{source}',1.,0)) if scene=='window' else None
            for subset in SUBSETS:
                name = split_name('window_random' if source==-1 else f'load_source_{source}',.1,subset)
                table = read_fixed(reference/'splits'/name)
                cols = ['window_index','record_index','record_id','rpm','label','start_sample','end_sample','load_hp']
                pd.testing.assert_frame_equal(table[cols], original_windows[cols], check_dtype=False)
                expected = 49 if source==-1 else (9,17,17,17)[source]
                if len(table[table.role.eq('train')]) != expected:
                    raise ValueError('Wrong 10% training count')
                if scene=='window':
                    pd.testing.assert_frame_equal(table, nested_subset(full,.1,subset), check_dtype=False)
                    assert table.role.eq('validation').sum()==103 and table.role.eq('test').sum()==103
                else:
                    assert table.role.eq('validation').sum()==(19,39,40,40)[source]
                    assert table[table.role.isin(['train','validation'])].load_hp.eq(source).all()
                dest = root/'splits'/name
                dest.mkdir(parents=True, exist_ok=False)
                for path in (reference/'splits'/name).iterdir():
                    if path.is_file():
                        shutil.copyfile(path,dest/path.name)
                        assert path.read_bytes()==(dest/path.name).read_bytes()
                tables[(scene,source,subset)] = table
    records.to_csv(root/'records.csv',index=False)
    return torch.device('cuda'), tables, x


def frozen_references():
    lookup = {}
    for scene, root in [('window',WINDOW_REF),('load',LOAD_REF)]:
        state = json.loads((root/'config.json').read_text(encoding='utf-8'))
        if state['stage']!='complete':
            raise ValueError('Frozen baseline not complete')
        jobs = pd.read_csv(root/'models_registry.csv')
        jobs = jobs[jobs.variant.eq('baseline_data') & jobs.ratio.eq(.1)]
        evaluations = pd.read_csv(root/'evaluation/evaluations.csv')
        for job in jobs.to_dict('records'):
            if scene=='window':
                job['source_load_hp']=-1
            job['scene']=scene
            selected = evaluations[evaluations.job_id.eq(job['job_id'])]
            expected_targets = {-1} if scene=='window' else set(range(4))-{int(job['source_load_hp'])}
            if set(selected.target_load_hp)!=expected_targets:
                raise ValueError('Frozen baseline target set mismatch')
            job['reference_evaluations'] = selected[['target_load_hp','prediction_path']].to_dict('records')
            job['source_root']=str(root)
            if casekey(job) in lookup:
                raise ValueError('Duplicate reference')
            if not (Path(job['folder'])/'model.pt').is_file():
                raise FileNotFoundError(job['folder'])
            lookup[casekey(job)] = job
    if set(lookup)!=set(map(casekey,CASES)):
        raise ValueError('150 matched baseline models required')
    return lookup


def verify(job, table, device='cpu', recompute=False, x=None):
    folder=Path(job['folder'])
    if not job['completed'] or job['aux_weight']!=.3 or job['variant']!='data_aux_ce':
        raise ValueError('Model record mismatch')
    pd.testing.assert_frame_equal(pd.read_csv(folder/'selected_windows.csv'),
        table[table.role.eq('train')].reset_index(drop=True),check_dtype=False)
    model=load(folder,device)
    assert model.network.physics[0].in_features==6
    assert model.spec==spec_for(job['protocol'])
    assert all(torch.isfinite(t).all() for t in model.state_dict().values())
    assert (model.evidence.scale>=.05).all()
    v=table[table.role.eq('validation')]
    with np.load(folder/'validation_predictions.npz',allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['window_indices'],v.window_index)
        np.testing.assert_array_equal(arrays['labels'],v.label)
    if recompute:
        usable=table.role.isin(['train','validation']).to_numpy()
        w=table[usable].reset_index(drop=True)
        rows, arrays=engine.validation(model,w,x[usable],job['protocol'],device,model.spec)
        with np.load(folder/'validation_predictions.npz',allow_pickle=False) as saved:
            for key in arrays:
                np.testing.assert_allclose(arrays[key],saved[key],rtol=1e-5,atol=1e-6)
    del model


def append_csv(path, rows):
    pd.DataFrame(rows).to_csv(path,index=False,mode='a',header=not path.exists())


class Run:
    def __init__(self, root):
        self.root=root.resolve()
        if self.root.parent!=RESULTS.resolve() or not self.root.name.startswith('small10_'):
            raise ValueError('Invalid new output location')
        self.root.mkdir(exist_ok=False)
        self.jobs=[]
        self.state=dict(version='physics6_scheme1_small10',stage='preparing',pid=os.getpid(),started_at=now(),
            planned_models=150,planned_new_models=150,planned_referenced_models=150,
            planned_pairs=780,planned_metric_rows=20280,trained_models=0,evaluated_pairs=0,metric_rows=0,
            selected_aux_weight=.3,lambda_locked=True,decisions_locked=False,weights_verified=False,
            target_test_accessed=False,automatic_resume=False,historical_test_exposure=True,
            training_specs={p:spec_for(p) for p in PROTOCOLS},model_config=MODEL_CONFIG,jobs=self.jobs)
        shutil.copyfile(HERE/'PROTOCOL.md',self.root/'PROTOCOL.md')
        hashes={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [HERE/'physics.py',HERE/'model.py',HERE/'run.py',HERE/'report.py',HERE/'PROTOCOL.md']}
        write_json(self.root/'code_snapshot_sha256.json',hashes)
        for p in HERE.glob('*.py'):
            dest=self.root/'code_snapshot'/p.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(p,dest)
        self.persist()

    def persist(self):
        self.state.update(updated_at=now(),trained_models=len(self.jobs))
        write_json(self.root/'config.json',self.state)
        pd.DataFrame(self.jobs).to_csv(self.root/'models_registry.csv',index=False)

    def run(self):
        self.references=frozen_references()
        self.device,self.tables,self.x=prepare(self.root)
        self.state.update(stage='training',hardware=torch.cuda.get_device_name(),split_count=len(self.tables))
        self.persist()
        for i,case in enumerate(CASES,1):
            jobid=f'm{i:04d}'
            table=self.tables[(case['scene'],case['source_load_hp'],case['subset'])]
            usable=table.role.isin(['train','validation']).to_numpy()
            self.state.update(active_model=dict(job_id=jobid,**case))
            self.persist()
            folder=self.root/'models'/jobid
            info=train(self.x[usable],table[usable].reset_index(drop=True),case['protocol'],
                       'data_aux_ce',case['seed'],case['subset'],folder,self.device,spec_for(case['protocol']))
            job=dict(info,**case,job_id=jobid,aux_weight=.3,folder=str(folder),method='fusion')
            verify(job,table)
            self.jobs.append(job)
            self.state.pop('active_model')
            self.persist()
            print('MODEL COMPLETE',jobid,case,flush=True)
        self.state.update(stage='verifying_weights')
        self.persist()
        for job in self.jobs:
            table=self.tables[(job['scene'],job['source_load_hp'],job['subset'])]
            verify(job,table)
        for ref in self.references.values():
            baseline=original.load(ref['folder'],'cpu')
            if baseline.variant!='baseline_data' or not all(torch.isfinite(t).all() for t in baseline.state_dict().values()):
                raise ValueError('Frozen baseline weights invalid')
            del baseline
        self.state.update(stage='decisions_locked',decisions_locked=True,weights_verified=True)
        self.persist()
        self.evaluate()

    def evaluate(self):
        if len(self.jobs)!=150 or not self.state['decisions_locked'] or self.state['target_test_accessed']:
            raise ValueError('All 150 weights must be locked before testing')
        dest=self.root/'evaluation'
        (dest/'predictions').mkdir(parents=True,exist_ok=False)
        self.state.update(stage='testing',target_test_accessed=True)
        self.persist()
        pairs=metric_count=0
        for scene,targets in [('window',(-1,)),('load',range(4))]:
            for target in targets:
                representative=self.tables[(scene,-1 if scene=='window' else (target+1)%4,SUBSETS[0])]
                indices=np.flatnonzero(representative.role.eq('test') if scene=='window' else representative.load_hp.eq(target))
                rows=representative.iloc[indices]
                labels,ids,rpm=rows.label.to_numpy(),rows.window_index.to_numpy(),rows.rpm.to_numpy(np.float32)
                views=list(engine.R.conditions(self.x[indices],'awgn_mixed',True))
                assert tuple((c,s) for c,s,_ in views)==CONDITIONS
                for job in self.jobs:
                    if job['scene']!=scene or (scene=='load' and job['source_load_hp']==target):
                        continue
                    table=self.tables[(scene,job['source_load_hp'],job['subset'])]
                    actual=table[table.role.eq('test') if scene=='window' else table.load_hp.eq(target)]
                    np.testing.assert_array_equal(actual.window_index,ids)
                    np.testing.assert_array_equal(actual.label,labels)
                    ref=self.references[casekey(job)]
                    frozen_path=next(Path(e['prediction_path']) for e in ref['reference_evaluations'] if e['target_load_hp']==target)
                    with np.load(frozen_path,allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['labels'],labels)
                        np.testing.assert_array_equal(saved['window_indices'],ids)
                        frozen_arrays={key:saved[key].copy() for key in saved.files}
                    for method in ('fusion','data'):
                        model=load(job['folder'],self.device) if method=='fusion' else None
                        arrays=dict(labels=labels,window_indices=ids)
                        metrics,confusion=[],[]
                        new_inferences=0
                        modelid=job['job_id'] if method=='fusion' else 'b'+job['job_id'][1:]
                        meta={k:job[k] for k in ['scene','source_load_hp','subset','protocol','seed','ratio','domain']}
                        meta.update(job_id=modelid,method=method,target_load_hp=target,aux_weight=.3 if method=='fusion' else 0.)
                        for condition,noise_seed,observed in views:
                            key=f'{condition}_n{noise_seed}'
                            if method=='data' and key in frozen_arrays:
                                probability=frozen_arrays[key]
                                origin='frozen_prediction'
                            else:
                                if model is None:
                                    model=original.load(ref['folder'],self.device)
                                probability=engine.R.predict(model,observed,rpm,self.device)
                                new_inferences+=1
                                origin='new_model' if method=='fusion' else 'new_noise_frozen_baseline'
                            arrays[key]=probability
                            values,counts=scores(labels,probability)
                            row=dict(meta,condition=condition,noise_seed=noise_seed,origin=origin)
                            metrics.append(dict(row,**values))
                            for a in range(4):
                                support=int(counts[a].sum())
                                for b in range(4):
                                    confusion.append(dict(row,true_label=a,predicted_label=b,count=int(counts[a,b]),
                                        support=support,row_fraction=float(counts[a,b]/support)))
                        del model
                        path=dest/'predictions'/f'{modelid}_target{target}.npz'
                        with path.open('xb') as f:
                            np.savez_compressed(f,**arrays)
                        append_csv(dest/'metrics.csv',metrics)
                        append_csv(dest/'confusion_matrices.csv',confusion)
                        append_csv(dest/'evaluations.csv',[dict(meta,prediction_path=str(path),test_windows=len(labels),
                            new_condition_inferences=new_inferences,reference_prediction=str(frozen_path) if method=='data' else '')])
                        pairs+=1
                        metric_count+=len(metrics)
                        self.state.update(evaluated_pairs=pairs,metric_rows=metric_count,active_target=dict(scene=scene,target=target))
                        self.persist()
                print('TARGET COMPLETE',scene,target,pairs,metric_count,flush=True)
        if (pairs,metric_count)!=(780,20280):
            raise ValueError('Evaluation count mismatch')
        self.state.update(stage='reporting')
        self.persist()
        from .report import finish
        finish(self.root)
        self.state.update(stage='complete',completed_at=now())
        self.persist()


def check(root):
    root.mkdir(parents=True,exist_ok=False)
    device,tables,x=prepare(root)
    frozen_references()
    rng=np.random.default_rng(1901)
    signal=rng.normal(size=(3,1,24000)).astype(np.float32)
    rpm=np.array([1720,1750,1797],np.float32)
    ev=SuiteModel('data_aux_ce',spec_for('clean_only')).evidence.to(device)
    gpu=ev.extract(torch.as_tensor(signal,device=device),torch.as_tensor(rpm/60,device=device))[0].cpu().numpy()
    cpu=np.stack([reference_extract(signal[i,0],rpm[i]/60)[0][:6] for i in range(3)])
    np.testing.assert_allclose(gpu,cpu,rtol=3e-4,atol=2e-5)
    a=ev.extract(torch.zeros(2,1,24000,device=device),torch.ones(2,device=device)*29)[0]
    assert torch.equal(a,torch.zeros_like(a))
    # New input dimension must not alter initialization of unrelated layers or RNG.
    original.R.U.seed_all(1706)
    old=original.SuiteModel('data_aux_ce',spec_for('clean_only'))
    rng_old=torch.random.get_rng_state()
    original.R.U.seed_all(1706)
    new=SuiteModel('data_aux_ce',spec_for('clean_only'))
    assert torch.equal(rng_old,torch.random.get_rng_state())
    for key,value in old.network.state_dict().items():
        if key not in ('physics.0.weight','physics.0.bias'):
            assert torch.equal(value,new.network.state_dict()[key]),key
    del old,new
    normalize=engine.normalize_spec
    def one_epoch(spec=None):
        result=normalize(dict(spec or {},max_epochs=200))
        result['max_epochs']=1
        return result
    with patch.object(engine,'normalize_spec',one_epoch):
        for scene in ('window','load'):
            for protocol in PROTOCOLS:
                source=-1 if scene=='window' else 0
                table=tables[(scene,source,4101)]
                ix=table.role.isin(['train','validation']).to_numpy()
                folder=root/f'{scene}_{protocol}'
                info=train(x[ix],table[ix].reset_index(drop=True),protocol,'data_aux_ce',1706,4101,folder,device,spec_for(protocol))
                job=dict(info,completed=True,aux_weight=.3,folder=str(folder))
                verify(job,table,device,recompute=True,x=x)
    write_json(root/'verification.json',dict(passed=True,gpu_cpu_max_error=float(np.abs(gpu-cpu).max()),
        fixed_split_count=len(tables),single_epoch_models=4,reference_preflight_models=150,target_test_accessed=False))
    print('CHECK PASSED',root,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','check','run'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.command=='plan':
        print('150 new fusion models; 150 frozen data models; 780 pairs; 20280 metric rows; lambda=.3')
        return
    if args.output is None:
        parser.error('--output required')
    RESULTS.mkdir(exist_ok=True)
    if args.command=='check':
        check(args.output)
        return
    lock=RESULTS/'.small10.lock'
    with lock.open('x',encoding='utf-8') as f:
        f.write(f'{os.getpid()} {args.output.resolve()}\n')
    try:
        existing=list(RESULTS.glob('small10_*'))
        if existing:
            raise FileExistsError(f'Existing formal run; no automatic duplicate/restart: {existing}')
        experiment=Run(args.output)
        with (experiment.root/'training_console.log').open('x',encoding='utf-8',buffering=1) as log:
            with redirect_stdout(log),redirect_stderr(log):
                try:
                    experiment.run()
                except BaseException as error:
                    experiment.state.update(stage='failed',error=str(error),error_type=type(error).__name__)
                    traceback.print_exc()
                    experiment.persist()
                    raise
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
