"""合并 8 个对比方法目录与两个 V3 参考，生成跨方法、跨预算总报告；只读，不训练。"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import torch  # Windows 环境须先于 numpy/pandas 导入。
import numpy as np
import pandas as pd
from context import PROJECT
from metrics import METRICS
import comparison_report as cr

BASE = PROJECT/'experiments/CWRU/实验结果/CWRU_V3'
SMALL_REF = BASE/'load_small_20261025_2017'
FULL_REF = BASE/'load_full_20261024_0950'
SMALL_RUNS = {
    'WDCNN': BASE/'load_comparison_small_WDCNN_20261103_161215',
    'DRSN_CW': BASE/'load_comparison_small_DRSN_CW_20261103_164024',
    'PG_RDGFN_adapted': BASE/'load_comparison_small_PG_RDGFN_adapted_20261103_174118',
    'AW_Hybrid_adapted': BASE/'load_comparison_small_AW_Hybrid_adapted_20261103_180621',
}
FULL_RUNS = {
    'WDCNN': BASE/'load_comparison_full_WDCNN_20261103_193929',
    'DRSN_CW': BASE/'load_comparison_full_DRSN_CW_20261103_194132',
    'PG_RDGFN_adapted': BASE/'load_comparison_full_PG_RDGFN_adapted_20261103_195544',
    'AW_Hybrid_adapted': BASE/'load_comparison_full_AW_Hybrid_adapted_20261103_200104',
}
METHODS = ['WDCNN', 'DRSN_CW', 'PG_RDGFN_adapted', 'AW_Hybrid_adapted']
ORDER = ['baseline_data', 'data_aux_ce', *METHODS]
CASE = cr.CASE
PAIR = cr.PAIR
LABEL = cr.LABELS
SPLITS = SMALL_RUNS['WDCNN']/'splits'


def _flatten(frame):
    frame.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in frame.columns]
    return frame.reset_index(drop=True)


def _verify(run, sample, ratios, models, variants):
    cfg = json.loads((Path(run)/'config.json').read_text(encoding='utf-8'))
    if cfg.get('stage') != 'complete':
        raise ValueError(f'{run} 未完成: {cfg.get("stage")}')
    rows = cr._verify(run, ratios, models, set(variants), SPLITS)
    return rows


def _cell(row, metric='balanced_acc'):
    return f"{100*float(row[f'{metric}_mean']):.2f}±{100*float(row[f'{metric}_std']):.2f}"


def _md_table(header, rows):
    lines=[header, '|'+':--|'*(header.count('|')-1)]
    lines += rows
    return '\n'.join(lines)


def _summary(frame, keys):
    per_seed=(frame.groupby(keys+['seed'],as_index=False)
              .agg(directions=('target_load_hp','size'),**{k:(k,'mean') for k in METRICS}))
    if not per_seed.directions.eq(12).all():
        raise ValueError(f'每个配置必须有 12 个方向: {keys}')
    out=per_seed.groupby(keys)[METRICS].agg(['mean','std','count']).reset_index()
    out.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in out.columns]
    if not out.acc_count.eq(5).all():
        raise ValueError(f'每个配置必须有五种子: {keys}')
    return per_seed,out


def _paired(frame, level_keys):
    piv=frame.pivot_table(index=['domain','ratio','subset','seed','source_load_hp','target_load_hp'],
                          columns='variant',values=METRICS,aggfunc='mean')
    if piv.isna().any().any():
        raise ValueError('方法间配对不完整')
    out={}
    for method in METHODS+['baseline_data']:
        d=pd.DataFrame({k:piv[(k,'data_aux_ce')]-piv[(k,method)] for k in METRICS}).reset_index()
        d=d.groupby(level_keys+['seed'],as_index=False)[METRICS].mean()
        g=d.groupby(level_keys)[['balanced_acc','acc']].agg(['mean','std']).reset_index()
        g.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in g.columns]
        g['comparator']=method
        out[method]=g
    return pd.concat(out.values(),ignore_index=True), piv


def _direction(frame, value='balanced_acc'):
    return (frame.groupby(['variant','source_load_hp','target_load_hp'])[value]
            .mean().unstack('variant').reindex(columns=ORDER))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args=parser.parse_args()
    stamp=pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y%m%d_%H%M%S')
    out=Path(args.output) if args.output else BASE/f'comparison_summary_{stamp}'
    out=out.resolve()
    if out.parent != BASE.resolve() or not out.name.startswith('comparison_summary_'):
        raise ValueError('输出必须是 experiments/CWRU/实验结果/CWRU_V3 下全新的 comparison_summary_* 目录')
    if out.exists():
        raise FileExistsError('输出目录已存在，禁止覆盖')

    verified=[]
    small=[_verify(SMALL_RUNS[m],'small',[.1,.2,.5],180,[m]) for m in METHODS]
    full=[_verify(FULL_RUNS[m],'full',[1.0],20,[m]) for m in METHODS]
    small.append(cr._verify(SMALL_REF,[.1,.2,.5],360,set(cr.REFERENCE_METHODS),SPLITS))
    full.append(cr._verify(FULL_REF,[1.0],40,set(cr.REFERENCE_METHODS),SPLITS))
    small_frame=pd.concat(small,ignore_index=True)
    full_frame=pd.concat(full,ignore_index=True)
    small_frame['sample']='small'; full_frame['sample']='full'
    for name,root in list(SMALL_RUNS.items())+list(FULL_RUNS.items()):
        cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
        verified.append(dict(name=root.name,sample=cfg['scope'].get('sample','small'),method=cfg['scope']['method'],
                             stage=cfg['stage'],models=cfg['trained_models'],pairs=cfg['evaluated_pairs'],rows=cfg['metric_rows']))
    verified.append(dict(name=SMALL_REF.name,sample='small',method='reference',stage='complete',models=360,pairs=1080,rows=1080))
    verified.append(dict(name=FULL_REF.name,sample='full',method='reference',stage='complete',models=40,pairs=120,rows=120))

    _,small_sd=_summary(small_frame,['ratio','subset','variant'])
    _,full_sd=_summary(full_frame,['variant'])
    small_subset=(small_sd.groupby(['ratio','variant'])[['balanced_acc_mean']]
                  .agg(['mean','std']).reset_index())
    small_subset.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in small_subset.columns]
    small_pair,small_piv=_paired(small_frame,['ratio','subset'])
    full_pair,full_piv=_paired(full_frame,['ratio'])
    small_dir=_direction(small_frame)
    full_dir=_direction(full_frame)
    small_dir_diff={}
    for method in METHODS+['baseline_data']:
        d=(small_piv[('balanced_acc','data_aux_ce')]-small_piv[('balanced_acc',method)]).rename('diff').reset_index()
        small_dir_diff[method]=d.groupby(['source_load_hp','target_load_hp'])['diff'].mean()
    full_dir_diff={}
    for method in METHODS+['baseline_data']:
        d=(full_piv[('balanced_acc','data_aux_ce')]-full_piv[('balanced_acc',method)]).rename('diff').reset_index()
        full_dir_diff[method]=d.groupby(['source_load_hp','target_load_hp'])['diff'].mean()

    trend=[]
    for v in ORDER:
        sm=small_sd[small_sd.variant.eq(v)].balanced_acc_mean.mean()
        fu=full_sd[full_sd.variant.eq(v)].balanced_acc_mean.iloc[0]
        trend.append(dict(variant=v,small_mean_9configs=float(sm),full=float(fu),
                          sample='small' if v in cr.REFERENCE_METHODS else 'comparison'))

    costs=[]
    for name,root in list(SMALL_RUNS.items())+list(FULL_RUNS.items()):
        cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
        frame=pd.DataFrame(cfg['jobs']).assign(cost_source=root.name,sample=cfg['scope'].get('sample','small'))
        costs.append(frame)
    for root,sample,ratios in [(SMALL_REF,'small',[.1,.2,.5]),(FULL_REF,'full',[1.0])]:
        cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
        frame=pd.DataFrame(cfg['jobs'])
        frame=frame[frame.protocol.eq('clean_only')&frame.ratio.isin(ratios)]
        costs.append(frame.assign(cost_source=root.name,sample=sample))
    costs=pd.concat(costs,ignore_index=True)

    out.mkdir(parents=True)
    rep=out/'report'; rep.mkdir()
    small_sd.to_csv(rep/'small_equal_12_directions_seed_mean_sd.csv',index=False)
    full_sd.to_csv(rep/'full_equal_12_directions_seed_mean_sd.csv',index=False)
    small_subset.to_csv(rep/'small_between_subset_mean_sd.csv',index=False)
    small_pair.to_csv(rep/'small_paired_equal_12_directions.csv',index=False)
    full_pair.to_csv(rep/'full_paired_equal_12_directions.csv',index=False)
    small_dir.to_csv(rep/'small_direction_balanced_acc.csv')
    full_dir.to_csv(rep/'full_direction_balanced_acc.csv')
    pd.DataFrame({k:v for k,v in small_dir_diff.items()}).to_csv(rep/'small_direction_paired_ba.csv')
    pd.DataFrame({k:v for k,v in full_dir_diff.items()}).to_csv(rep/'full_direction_paired_ba.csv')
    pd.DataFrame(trend).to_csv(rep/'small_to_full_trend.csv',index=False)
    costs.to_csv(rep/'comparison_cost.csv',index=False)
    (out/'verification.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2),encoding='utf-8')

    names=[LABEL[v] for v in ORDER]
    small_rows=[]
    for (ratio,subset),group in small_sd.groupby(['ratio','subset'],sort=True):
        cells=[]
        for v in ORDER:
            row=group[group.variant.eq(v)]
            cells.append(_cell(row.iloc[0]))
        small_rows.append(f'| {int(round(ratio*100))}% | {int(subset)} | '+' | '.join(cells)+' |')
    full_rows=[]
    for v in ORDER:
        row=full_sd[full_sd.variant.eq(v)].iloc[0]
        full_rows.append(f'| {LABEL[v]} | {_cell(row)} | {_cell(row,"acc")} | {_cell(row,"f1_macro")} |')
    psmall=[]
    for (ratio,subset),group in small_pair.groupby(['ratio','subset'],sort=True):
        cells=[]
        for m in METHODS+['baseline_data']:
            row=group[group.comparator.eq(m)]
            cells.append(_cell(row.iloc[0]))
        psmall.append(f'| {int(round(ratio*100))}% | {int(subset)} | '+' | '.join(cells)+' |')
    pfull=[]
    for m in METHODS+['baseline_data']:
        row=full_pair[full_pair.comparator.eq(m)].iloc[0]
        pfull.append(f'| {LABEL[m]} | {_cell(row)} | {_cell(row,"acc")} |')
    dir_rows=[]
    for (src,tgt),row in small_dir.iterrows():
        cells=[f'{100*float(row[v]):.2f}' for v in ORDER]
        dir_rows.append(f'| {src}→{tgt} | '+' | '.join(cells)+' |')
    full_dir_rows=[]
    for (src,tgt),row in full_dir.iterrows():
        cells=[f'{100*float(row[v]):.2f}' for v in ORDER]
        full_dir_rows.append(f'| {src}→{tgt} | '+' | '.join(cells)+' |')
    trend_rows=[f'| {LABEL[r["variant"]]} | {100*r["small_mean_9configs"]:.2f} | {100*r["full"]:.2f} | {100*(r["full"]-r["small_mean_9configs"]):+.2f} |' for _,r in pd.DataFrame(trend).iterrows()]
    cost_rows=[]
    for v in ORDER:
        for sample in ('small','full'):
            sel=costs[costs.variant.eq(v)&costs['sample'].eq(sample)]
            if sel.empty: continue
            cost_rows.append(f'| {LABEL[v]} | {sample} | {int(sel.parameters.iloc[0])} | '
                             f'{sel.latency_ms.mean():.2f} | {sel.peak_gpu_mib.max():.0f} | '
                             f'{sel.total_train_seconds.sum():.0f} | {len(sel)} |')
    header_small='| 训练比例 | 子集 | '+' | '.join(names)+' |'
    header_full='| 方法 | BA | ACC | F1 |'
    header_pair='| 训练比例 | 子集 | '+' | '.join(LABEL[m] for m in METHODS+['baseline_data'])+' |'
    header_dir='| 方向 | '+' | '.join(names)+' |'
    text=('# V3 跨负载四方法对比总报告\n\n'
      '本报告只读合并 8 个已完成对比目录与两个 V3 参考目录，并逐条复算全部预测；'
      '不训练、不修改任何已有结果。四方法为 WDCNN、DRSN-CW、PG-RDGFN-adapted、AW-Hybrid-adapted，'
      '均在小样本（10%/20%/50%×3 子集）与全样本上按来源训练、来源验证选模，再评价另外三个负载，纯干净。\n\n'
      '## 核验\n\n'
      '每个目录的预测文件均按目标窗口、标签和保存指标逐条复算，全部一致；'
      '每配置恰有 12 个方向、五种子。逐目录模型数与组合数见 `verification.json`。\n\n'
      '| 目录 | 样本 | 方法 | 阶段 | 模型 | 组合 | 指标 |\n|:--|:--|:--|:--|--:|--:|--:|\n'
      + '\n'.join(f'| {v["name"]} | {v["sample"]} | {v["method"]} | {v["stage"]} | {v["models"]} | {v["pairs"]} | {v["rows"]} |' for v in verified) + '\n\n'
      '## 小样本结果（12 方向等权 BA，五种子均值±样本标准差，%）\n\n'
      + _md_table(header_small,small_rows) + '\n\n'
      'ACC、F1 与三子集间波动见 `report/small_equal_12_directions_seed_mean_sd.csv` 和 '
      '`report/small_between_subset_mean_sd.csv`。\n\n'
      '## 全样本结果（12 方向等权，%）\n\n'
      + _md_table(header_full,full_rows) + '\n\n'
      '## 配对差值（本文融合加辅助分类 减 对照，BA）\n\n### 小样本\n\n'
      + _md_table(header_pair,psmall) + '\n\n### 全样本\n\n'
      '| 对照 | BA 差 | ACC 差 |\n|:--|--:|--:|\n' + '\n'.join(pfull) + '\n\n'
      '正值表示本文方法较高；全部逐方向结果见 `report/*_direction_paired_ba.csv`，负值不省略。\n\n'
      '## 小样本到全样本\n\n'
      '| 方法 | 小样本（9 配置均值） | 全样本 | 变化 |\n|:--|--:|--:|--:|\n' + '\n'.join(trend_rows) + '\n\n'
      '小样本列是 9 个配置（3 预算×3 子集）均值的平均，只用于描述趋势，不是正式统计量。\n\n'
      '## 方向明细（BA，%）\n\n### 小样本（五种子与三子集平均）\n\n'
      + _md_table(header_dir,dir_rows) + '\n\n### 全样本（五种子平均）\n\n'
      + _md_table(header_dir,full_dir_rows) + '\n\n'
      '## 计算成本\n\n'
      '| 方法 | 样本 | 参数量 | 延迟 ms | 峰值 MiB | 训练合计 s | 记录数 |\n|:--|:--|--:|--:|--:|--:|--:|\n'
      + '\n'.join(cost_rows) + '\n\n'
      '峰值显存与延迟的测量口径在 V3 参考与新模型间不同（V3 含首批梯度诊断），不能直接当作效率排序。\n\n'
      '## 解释边界\n\n'
      '四种方法均为结构或方法适配，不是官方实现；PG 额外使用转速与轴承几何信息，AW 使用小波包与统计特征。'
      '统一训练规则不保证各方法达到最优。训练与验证共享来源记录，λ 与划分来自既有全负载开发，历史测试已曝光，'
      '因此这是固定公开数据上的探索性跨负载比较，不支持严格未见域或独立轴承的确认性结论。'
      '小样本中 DRSN-CW 12/180、WDCNN 3/180 个模型退化到单一类别（BA=0.25），按协议保留；全样本无退化模型。\n')
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    print('SUMMARY OK', out)
    print('rows small/full:',len(small_frame),len(full_frame))
    print('variants:',sorted(small_frame.variant.unique()))


if __name__=='__main__':
    main()
