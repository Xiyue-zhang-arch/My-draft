"""只读汇总完成后的作者代码与经典结构跨负载结果。"""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True

import torch  # Import torch before numpy in the configured Windows environment.
import numpy as np
import pandas as pd
from context import PROJECT
from metrics import METRICS
import author_comparison_report as verify

BASE=PROJECT/'experiments/CWRU/实验结果/CWRU_V3'
SMALL_REF=BASE/'load_small_20261025_2017'
FULL_REF=BASE/'load_full_20261024_0950'
METHODS=['WDCNN','DRSN_CW','DWCN_SDGFD','ClassBD_WDCNN']
ORDER=['baseline_data','data_aux_ce',*METHODS]
NAMES={'baseline_data':'数据单分支','data_aux_ce':'本文融合模型','WDCNN':'WDCNN（结构适配）',
       'DRSN_CW':'DRSN-CW（结构适配）','DWCN_SDGFD':'DWCN-SDGFD（作者代码适配）',
       'ClassBD_WDCNN':'ClassBD + WDCNN（作者代码修复适配）'}
OLD_SMALL={'WDCNN':BASE/'load_comparison_small_WDCNN_20261103_161215',
           'DRSN_CW':BASE/'load_comparison_small_DRSN_CW_20261103_164024'}
OLD_FULL={'WDCNN':BASE/'load_comparison_full_WDCNN_20261103_193929',
          'DRSN_CW':BASE/'load_comparison_full_DRSN_CW_20261103_194132'}
SPLITS=OLD_SMALL['WDCNN']/'splits'


def config(root):
    with (root/'config.json').open(encoding='utf-8') as handle:return json.load(handle)


def preflight(root,method,sample):
    d=config(root); scope=d.get('scope',{})
    n=180 if sample=='small' else 20
    if (d.get('stage')!='complete' or scope.get('method')!=method
        or scope.get('sample','small')!=sample or scope.get('protocol')!='clean_only'
        or d.get('trained_models')!=n or d.get('evaluated_pairs')!=n*3
        or d.get('metric_rows')!=n*3 or not d.get('decisions_locked')
        or not d.get('weights_verified') or not d.get('target_test_accessed')
        or not (root/'REPORT.md').is_file()):
        raise ValueError(f'实验未完成或范围不符，不读取其测试数据: {root}')
    if (scope.get('ratios')!=([.1,.2,.5] if sample=='small' else [1.])
        or scope.get('seeds')!=list(range(1706,1711))):
        raise ValueError(f'预算或种子不符: {root}')
    if method in ('DWCN_SDGFD','ClassBD_WDCNN') and (
        d.get('version')!='CWRU_V3_author_comparison_clean' or not (root/'METHOD_ADAPTATION.md').is_file()):
        raise ValueError(f'作者代码实验版本或方法说明缺失: {root}')
    return d


def aggregated(frame,keys):
    grouped=frame.groupby(keys+['seed'],as_index=False).agg(
        directions=('target_load_hp','size'),**{k:(k,'mean') for k in METRICS})
    if not grouped.directions.eq(12).all():raise ValueError('每配置每种子须有12个方向')
    ret=grouped.groupby(keys)[METRICS].agg(['mean','std','count']).reset_index()
    ret.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in ret.columns]
    if not ret.acc_count.eq(5).all():raise ValueError('每配置须有5个种子')
    return ret


def paired(frame):
    keys=['domain','ratio','subset','seed','source_load_hp','target_load_hp']
    pivot=frame.pivot(index=keys,columns='variant',values=METRICS)
    if pivot.isna().any().any():raise ValueError('配对数据不完整')
    out=[]
    for method in ['baseline_data',*METHODS]:
        d=pd.DataFrame({m:pivot[(m,'data_aux_ce')]-pivot[(m,method)] for m in METRICS}).reset_index()
        d['comparator']=method
        out.append(d)
    return pd.concat(out,ignore_index=True)


def detect_degenerate(root):
    e=pd.read_csv(root/'evaluation/evaluations.csv')
    result=[]
    for jid,g in e.groupby('job_id',sort=False):
        predictions=[]
        for r in g.itertuples(index=False):
            with np.load(root/'evaluation/predictions'/f'{jid}_target{r.target_load_hp}.npz',allow_pickle=False) as z:
                predictions.extend(np.argmax(z['clean_n0'],axis=1).tolist())
        classes=sorted(set(predictions))
        result.append(dict(job_id=jid,degenerate_single_class=len(classes)==1,
                           predicted_classes=','.join(map(str,classes))))
    return pd.DataFrame(result)


def table(frame,keys):
    rows=[]
    for ident,g in frame.groupby(keys,sort=True):
        ident=ident if isinstance(ident,tuple) else (ident,)
        lookup=g.set_index('variant')
        cells=[f"{100*lookup.loc[m,'balanced_acc_mean']:.2f}±{100*lookup.loc[m,'balanced_acc_std']:.2f}" for m in ORDER]
        rows.append('| '+' | '.join(str(i) for i in ident)+' | '+' | '.join(cells)+' |')
    return '\n'.join(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for opt in ['dwcn-small','dwcn-full','classbd-small','classbd-full']:
        ap.add_argument('--'+opt,type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    new_small={'DWCN_SDGFD':args.dwcn_small.resolve(),'ClassBD_WDCNN':args.classbd_small.resolve()}
    new_full={'DWCN_SDGFD':args.dwcn_full.resolve(),'ClassBD_WDCNN':args.classbd_full.resolve()}
    small={**OLD_SMALL,**new_small};full={**OLD_FULL,**new_full}
    out=args.output.resolve()
    if out.parent!=BASE.resolve() or not out.name.startswith('author_comparison_summary_') or out.exists():
        raise ValueError('输出须为新的 experiments/CWRU/实验结果/CWRU_V3/author_comparison_summary_* 目录')
    roots=list(small.values())+list(full.values())
    if len(set(roots))!=len(roots) or any(p.parent!=BASE.resolve() for p in roots):
        raise ValueError('输入目录重复或不在 V3 结果目录')
    # Check all eight statuses before accessing any test data.
    checked=[]
    for sample,runs in [('small',small),('full',full)]:
        for method,root in runs.items():
            d=preflight(root,method,sample)
            checked.append(dict(path=str(root),sample=sample,method=method,stage=d['stage'],
                                models=d['trained_models'],pairs=d['evaluated_pairs'],rows=d['metric_rows']))
    for root in (SMALL_REF,FULL_REF):
        if config(root).get('stage')!='complete':raise ValueError('V3 原参考未完成')

    for root in roots:
        verify._validate_splits(root)

    sf=[verify._verify(small[m],[.1,.2,.5],180,{m},SPLITS) for m in METHODS]
    ff=[verify._verify(full[m],[1.],20,{m},SPLITS) for m in METHODS]
    sf.append(verify._verify(SMALL_REF,[.1,.2,.5],360,set(verify.REFERENCE_METHODS),SPLITS))
    ff.append(verify._verify(FULL_REF,[1.],40,set(verify.REFERENCE_METHODS),SPLITS))
    sf=pd.concat(sf,ignore_index=True);ff=pd.concat(ff,ignore_index=True)
    for frame in (sf,ff):
        if frame.duplicated(verify.CASE+['condition','noise_seed']).any():raise ValueError('结果存在重复')
    ss=aggregated(sf,['ratio','subset','variant']); fs=aggregated(ff,['variant'])
    subsets=ss.groupby(['ratio','variant'])['balanced_acc_mean'].agg(['mean','std','count']).reset_index()
    if not subsets['count'].eq(3).all():raise ValueError('每比例须有3个子集')
    sp,fp=paired(sf),paired(ff)
    deg=[]
    for sample,runs in [('small',small),('full',full)]:
        for method,root in runs.items():
            d=detect_degenerate(root);d['method']=method;d['sample']=sample;deg.append(d)
    deg=pd.concat(deg,ignore_index=True)
    # Output is created only after every prediction and every source has passed verification.
    out.mkdir(exist_ok=False);report=out/'report';report.mkdir()
    ss.to_csv(report/'small_per_subset_seed_mean_sd.csv',index=False)
    fs.to_csv(report/'full_seed_mean_sd.csv',index=False)
    subsets.to_csv(report/'small_between_subset_mean_sd.csv',index=False)
    sp.to_csv(report/'small_paired_each_direction_seed.csv',index=False)
    fp.to_csv(report/'full_paired_each_direction_seed.csv',index=False)
    for label,pairs in [('small',sp),('full',fp)]:
        direction_keys=['ratio','subset','comparator','source_load_hp','target_load_hp']
        direction=pairs.groupby(direction_keys)[METRICS].agg(['mean','std','count'])
        direction.columns=['_'.join(c) for c in direction.columns]
        if not direction['balanced_acc_count'].eq(5).all():raise ValueError('每个配对方向须有5个种子')
        direction.to_csv(report/f'{label}_paired_direction_seed_mean_sd.csv')
        equal=(pairs.groupby(['ratio','subset','comparator','seed'],as_index=False)
               .agg(directions=('target_load_hp','size'),**{k:(k,'mean') for k in METRICS}))
        if not equal.directions.eq(12).all():raise ValueError('每个配对须有12个方向')
        equal=equal.groupby(['ratio','subset','comparator'])[METRICS].agg(['mean','std','count'])
        equal.columns=['_'.join(c) for c in equal.columns]
        if not equal['balanced_acc_count'].eq(5).all():raise ValueError('每个配对须有5个种子')
        equal.to_csv(report/f'{label}_paired_equal_directions_seed_mean_sd.csv')
    for label,data in [('small',sf),('full',ff)]:
        data.groupby(['variant','source_load_hp','target_load_hp'])[METRICS].mean().to_csv(report/f'{label}_direction_mean.csv')
    deg.to_csv(report/'degenerate_model_predictions.csv',index=False)
    (out/'verification.json').write_text(json.dumps(checked,ensure_ascii=False,indent=2),encoding='utf-8')
    head=' | '.join(NAMES[x] for x in ORDER)
    smalltable=table(ss,['ratio','subset'])
    fulltable=table(fs.assign(ratio=1),['ratio'])
    degcount=deg.groupby(['sample','method']).degenerate_single_class.agg(['sum','count']).reset_index()
    degtext='\n'.join(f"| {r['sample']} | {NAMES[r['method']]} | {r['sum']} / {r['count']} |" for _,r in degcount.iterrows())
    txt=('# V3 跨负载对比汇总\n\n四种对照与本文方法均为纯干净跨负载结果；逐条预测与窗口标签及指标复算通过。'
         '数据单分支和本文方法来自已冻结的 V3 结果。模型设置、数据处理及训练预算见各实验的独立协议。\n\n'
         '## 小样本 BA（12方向等权、五种子均值±样本标准差，%）\n\n'
         f'| 比例 | 子集 | {head} |\n|:--|:--|'+'--:|'*len(ORDER)+'\n'+smalltable+'\n\n'
         '## 全样本 BA（12方向等权、五种子均值±样本标准差，%）\n\n'
         f'| 比例 | {head} |\n|:--|'+'--:|'*len(ORDER)+'\n'+fulltable+'\n\n'
         '子集均值与子集之间的标准差单独见 `report/small_between_subset_mean_sd.csv`；不得称为种子标准差。'
         '每个来源目标方向和种子的完整配对差见 `report/*_paired_each_direction_seed.csv`，其五种子均值及标准差另见 `report/*_paired_direction_seed_mean_sd.csv`。配对差定义为本文减对照，正负值均保留。'
         '各方向均值见 `report/*_direction_mean.csv`。\n\n'
         '## 单类别预测\n\n按同一模型全部三个目标负载的预测类别检查，不删除退化模型。\n\n'
         '| 样本 | 方法 | 模型数 |\n|:--|:--|--:|\n'+degtext+'\n\n'
         '## 解释边界\n\nWDCNN、DRSN 是既有结构适配版本，后者尚须与作者源码核对。DWCN 和 ClassBD 依据作者代码接入，'
         'ClassBD 修复物理损失断梯度、分类损失与优化器重置，并对父窗口切段汇总；DWCN 改用来源验证而非目标标签选模。'
         '两者输入及优化器与本文不同，不称为原文结果的精确复现，具体版本与改动见各结果目录 `METHOD_ADAPTATION.md`。'
         '不同方法统一数据划分不代表训练与额外信息完全相同，也不能将差异归因于特定模块。'
         '训练与验证共享来源记录、不同来源预算不等、历史测试曝光；结果仅适用于此数据设置，不构成独立轴承确认性证据。\n')
    (out/'REPORT.md').write_text(txt,encoding='utf-8')
    print('SUMMARY COMPLETE',out,'small rows',len(sf),'full rows',len(ff))


if __name__=='__main__':main()
