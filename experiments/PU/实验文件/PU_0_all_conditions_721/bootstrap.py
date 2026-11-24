"""Copy established numerical components and adapt only data scope and accounting."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'PU_0'
PREVIOUS=HERE.parent/'PU_0_within_condition'

def main():
    provenance={}
    for name in ('encoder.py','descriptors.py','model.py','reference.py','raw_prepare.py'):
        payload=(OLD/name).read_bytes(); (HERE/name).write_bytes(payload)
        provenance[name]=dict(original=str(OLD/name),sha256=hashlib.sha256(payload).hexdigest(),byte_identical=True)
    common=(OLD/'common.py').read_text(encoding='utf-8')
    common=common.replace('TARGETS=("N09_M07_F10","N15_M01_F10","N15_M07_F04")','TARGETS=("N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04")')
    common=common.replace('CONDITIONS=(SOURCE,*TARGETS)','CONDITIONS=TARGETS')
    common=common.replace("DATA=PROJECT/'data/prepared_inputs'","DATA=PROJECT/'data/prepared'")
    common=common.replace('EPOCHS=50','EPOCHS=50\nUPDATES_PER_EPOCH=175')
    common=common.replace('if len(result)!=63:','if len(result)!=84:').replace('Expected 3 times 21 cases','Expected 4 times 21 cases')
    (HERE/'common.py').write_text(common,encoding='utf-8')
    support=(PREVIOUS/'support.py').read_text(encoding='utf-8')
    support=support.replace('[:4000]','[:]').replace("'_source.npy'","'_all.npy'")
    support=support.replace("windows[:4000]","windows")
    (HERE/'support.py').write_text(support,encoding='utf-8')
    engine=(PREVIOUS/'engine.py').read_text(encoding='utf-8')
    engine=engine.replace('ROOT,PROJECT,DATA,TARGETS,METHODS,BATCH,EPOCHS,ETA','ROOT,PROJECT,DATA,TARGETS,METHODS,BATCH,EPOCHS,UPDATES_PER_EPOCH,ETA')
    engine=engine.replace('updates!=50 or not np.isfinite(ce) or not np.isfinite(grad).all() or min(grad[:2])<=0','updates!=UPDATES_PER_EPOCH or not np.isfinite(ce) or not np.isfinite(grad).all()')
    engine=engine.replace('updates=50*epoch','updates=UPDATES_PER_EPOCH*epoch').replace("row['updates']!=2500","row['updates']!=UPDATES_PER_EPOCH*EPOCHS")
    (HERE/'engine.py').write_text(engine,encoding='utf-8')
    provenance['engine.py']=dict(original=str(PREVIOUS/'engine.py'),changes=['expected updates 50 to 175','finite zero gradients recorded, NaN/Inf still rejected'],optimization_algorithm_unchanged=True)
    run=(PREVIOUS/'run.py').read_text(encoding='utf-8')
    run=run.replace('Thirty PU_0 replays, followed by within-condition testing and independent audit.','Thirty all-condition models followed by complete mixed-condition evaluation.')
    run=run.replace("ids=np.load(DATA/'split_indices.npz')['test']; windows=", "split=np.load(DATA/'split_indices.npz'); windows=")
    run=run.replace("    y=np.array([windows[int(i)]['label'] for i in ids],dtype='int64')\n",'')
    run=run.replace("    for case in cases():\n", "    for case in cases():\n        ids=split[case['target']]; y=np.array([windows[int(i)]['label'] for i in ids],dtype='int64')\n        noise_positions=np.searchsorted(split['test'],ids)\n")
    run=run.replace("noise[:,case['noise_instance']]","noise[noise_positions,case['noise_instance']]")
    run=run.replace('630','2520')
    run=run.replace("experiment_type='within_condition_record_holdout',previous_validation_exposure=True", "experiment_type='all_conditions_stratified_record_holdout_721',historical_dataset_exposure=True")
    start,end=run.index("        old=ROOT/'results/PU-3"),run.index('        if source_hashes()!=before:')
    run=run[:start]+run[end:]
    (HERE/'run.py').write_text(run,encoding='utf-8')
    audit=(PREVIOUS/'audit.py').read_text(encoding='utf-8')
    audit=audit.replace('CLASS_NAMES,METHODS,SEEDS,models','CLASS_NAMES,METHODS,SEEDS,CONDITIONS,models')
    audit=audit.replace('630','2520').replace('10080','40320').replace('(3200,400,400)','(11200,3200,1600)').replace("'_source.npy'","'_all.npy'").replace("row['updates']==2500","row['updates']==8750")
    audit=audit.replace("    y=np.array([windows[int(i)]['label'] for i in test]); matrices=[];", "    matrices=[];")
    audit=audit.replace("        for case in cases():\n", "        for case in cases():\n            condition_ids=split[case['target']]; y=np.array([windows[int(i)]['label'] for i in condition_ids])\n")
    audit=audit.replace("z['window_index'],test","z['window_index'],condition_ids")
    audit=audit.replace("dict(**spec,endpoint='clean'", "dict(**spec,condition=case['target'],endpoint='clean'")
    condition_audit='''    condition_report=csvread(out/'report/condition_endpoints_summary.csv')
    condition_seeds=csvread(out/'report/condition_per_seed_endpoints.csv')
    assert len(condition_report)==48 and len(condition_seeds)==240
    for condition in CONDITIONS:
        for role,expected in (('train',700),('val',200),('test',100)):
            chosen=[windows[int(i)] for i in split[role] if windows[int(i)]['condition']==condition]
            assert np.array_equal(np.bincount([w['label'] for w in chosen],minlength=4),np.full(4,expected))
        for method in METHODS:
            for endpoint in ('clean','noisy'):
                reported=next(r for r in condition_report if r['condition']==condition and r['method']==method and r['endpoint']==endpoint)
                for k in ('BA','Macro_F1'):
                    values=[]
                    for seed in SEEDS:
                        chosen=[r[k] for r in numerical if r['condition']==condition and r['method']==method and r['seed']==seed and r['endpoint']==endpoint]
                        assert len(chosen)==(1 if endpoint=='clean' else 20)
                        value=float(np.mean(chosen)); values.append(value)
                        rs=next(r for r in condition_seeds if r['condition']==condition and r['method']==method and int(r['seed'])==seed and r['endpoint']==endpoint)
                        error=max(error,abs(value-float(rs[k])))
                    error=max(error,abs(np.mean(values)-float(reported[k+'_mean'])),abs(np.std(values,ddof=1)-float(reported[k+'_SD'])))
'''
    audit=audit.replace("    protected=json.loads",condition_audit+"    protected=json.loads")
    (HERE/'audit.py').write_text(audit,encoding='utf-8')
    report=(PREVIOUS/'report.py').read_text(encoding='utf-8')
    report=report.replace('DATA,METHODS,SEEDS,SNRS,CLASS_NAMES','DATA,METHODS,SEEDS,SNRS,CLASS_NAMES,CONDITIONS')
    report=report.replace('630','2520').replace('10080','40320')
    report=report.replace('len(descriptions)!=21','len(descriptions)!=84')
    report=report.replace("ids=np.load(DATA/'split_indices.npz')['test']; windows=", "split=np.load(DATA/'split_indices.npz'); windows=")
    report=report.replace("    y=np.array([windows[int(i)]['label'] for i in ids],dtype='int64'); rows=[]; matrices=[]; hashes={}", "    rows=[]; matrices=[]; hashes={}")
    report=report.replace("        for case,desc in zip(cases(),descriptions):\n", "        for case,desc in zip(cases(),descriptions):\n            ids=split[case['target']]; y=np.array([windows[int(i)]['label'] for i in ids],dtype='int64')\n")
    report=report.replace("(1 if endpoint=='clean' else 20)","(4 if endpoint=='clean' else 80)")
    insert='''
def condition_summaries(rows):
    columns=('ACC','BA','Macro_F1','CE',*['recall_'+c for c in CLASS_NAMES])
    seeds=[]; summary=[]
    for condition in CONDITIONS:
        for method in METHODS:
            for endpoint in ('clean','noisy'):
                values=[]
                for seed in SEEDS:
                    chosen=[r for r in rows if r['condition']==condition and r['method']==method and r['seed']==seed and r['endpoint']==endpoint]
                    assert len(chosen)==(1 if endpoint=='clean' else 20)
                    v=dict(condition=condition,method=method,seed=seed,endpoint=endpoint,**{k:float(np.mean([r[k] for r in chosen])) for k in columns})
                    seeds.append(v); values.append(v)
                summary.append(dict(condition=condition,method=method,endpoint=endpoint,**{key:value for k in columns for key,value in
                    ((k+'_mean',float(np.mean([v[k] for v in values]))),(k+'_SD',float(np.std([v[k] for v in values],ddof=1))))}))
    return seeds,summary

'''
    report=report.replace('\ndef generate(out,figures=True):','\n'+insert+'\ndef generate(out,figures=True):')
    report=report.replace("    folder=out/'report'; folder.mkdir(exist_ok=False)","    condition_seeds,condition_summary=condition_summaries(rows)\n    folder=out/'report'; folder.mkdir(exist_ok=False)\n    write_csv(folder/'condition_per_seed_endpoints.csv',condition_seeds)\n    write_csv(folder/'condition_endpoints_summary.csv',condition_summary)")
    report=report.replace("endpoint+' within condition'","endpoint+' mixed conditions'")
    report=report.replace("folder/'within_condition_BA.png'","folder/'all_conditions_BA.png'").replace("folder/'within_condition_noise.png'","folder/'all_conditions_noise.png'")
    start=report.index("    replay=json.loads("); end=report.index("    for r in paired:",start)
    text='''    text=['# PU_0四工况混合7∶2∶1实验','',
        '四个工况均进入训练、验证和测试。按工况与轴承分层，完整重复记录组不跨集合，窗口数11200/3200/1600。每工况2800/800/400窗，每类每工况700/200/100窗。',
        '沿用六组模型、五种子、Adam0.001、batch64及50轮末轮。每轮175更新、每模型8750更新，不能声称与旧2500更新计算预算相同。30个模型全部核验后统一2520条件预测。','',*table,'',
        '主结果对四工况等权，受扰再对四SNR和五噪声实例等权。误差为五个训练种子的样本标准差。逐工况、900rpm、各类召回及配对种子差在CSV中完整保存。',
        '三个集合共享全部工况和同8个轴承实体，属于已知工况混合的记录留出评价，不能证明新工况或新轴承泛化。历史数据与目标工况已曝光，继续按探索性研究解释。',
        '原始640文件去重636条记录。为同时保持记录排他及严格7∶2∶1，只在必要KA04记录调整取窗配额。未以重复窗口补数。所有定标只拟合11200训练窗。',
        '正确物理调制相对等权的差仍须结合替代频率、统计量和自身条件对照解释。替代频率可能覆盖更高次故障成分，不能称物理信息完全移除。',
        '同一记录内窗口相关、共享噪声及五种子规模限制统计外推。若接近100%，按上限效应报告，不追加调参。有限零梯度或常量预测保留，不自动重训。','']
'''
    report=report[:start]+text+report[end:]
    (HERE/'report.py').write_text(report,encoding='utf-8')
    (HERE/'implementation_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
