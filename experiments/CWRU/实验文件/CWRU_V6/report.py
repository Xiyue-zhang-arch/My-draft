"""Recalculate saved predictions before any experiment summary is accepted."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CWRU_V0/shared'))
from experiment_paths import resolve_result_path



def finish(root, expected_counts=(780,20280)):
    from .run import CONDITIONS, CASE_KEYS, METRICS, scores, write_json
    root=Path(root)
    evaluation=root/'evaluation'
    entries=pd.read_csv(evaluation/'evaluations.csv')
    metrics=pd.read_csv(evaluation/'metrics.csv')
    confusion=pd.read_csv(evaluation/'confusion_matrices.csv')
    keys=['job_id','target_load_hp','condition','noise_seed']
    expected_pairs, expected_metrics=expected_counts
    assert len(entries)==expected_pairs and len(metrics)==expected_metrics and len(confusion)==16*expected_metrics
    assert not metrics.duplicated(keys).any()
    assert not confusion.duplicated(keys+['true_label','predicted_label']).any()
    max_error=0.
    support_checks=0
    for entry in entries.itertuples(index=False):
        path=resolve_result_path(entry.prediction_path)
        assert path.resolve().parent==(evaluation/'predictions').resolve()
        rows=metrics[metrics.job_id.eq(entry.job_id)&metrics.target_load_hp.eq(entry.target_load_hp)]
        cells=confusion[confusion.job_id.eq(entry.job_id)&confusion.target_load_hp.eq(entry.target_load_hp)]
        assert len(rows)==26 and set(zip(rows.condition,rows.noise_seed))==set(CONDITIONS)
        with np.load(path,allow_pickle=False) as arrays:
            assert set(arrays.files)=={'labels','window_indices',*(f'{c}_n{s}' for c,s in CONDITIONS)}
            assert len(arrays['labels'])==entry.test_windows
            assert len(np.unique(arrays['window_indices']))==entry.test_windows
            # Verify target labels and ordering against the copied immutable split.
            name=f'{entry.domain}_r010_s{entry.subset}'
            table=pd.read_csv(root/'splits'/name/'windows.csv')
            expected=table[table.role.eq('test') if entry.scene=='window' else table.load_hp.eq(entry.target_load_hp)]
            np.testing.assert_array_equal(arrays['labels'],expected.label)
            np.testing.assert_array_equal(arrays['window_indices'],expected.window_index)
            for condition,seed in CONDITIONS:
                p=arrays[f'{condition}_n{seed}']
                assert p.shape==(entry.test_windows,4) and np.isfinite(p).all()
                np.testing.assert_allclose(p.sum(-1),1.,atol=2e-5,rtol=0)
                assert (p>=0).all() and (p<=1).all()
                actual,counts=scores(arrays['labels'],p)
                saved=rows[rows.condition.eq(condition)&rows.noise_seed.eq(seed)].iloc[0]
                err=max(abs(actual[k]-saved[k]) for k in METRICS)
                max_error=max(max_error,err)
                assert err<1e-12
                matrix=cells[cells.condition.eq(condition)&cells.noise_seed.eq(seed)].sort_values(['true_label','predicted_label'])
                np.testing.assert_array_equal(matrix['count'].to_numpy().reshape(4,4),counts)
                np.testing.assert_array_equal(matrix.support.to_numpy().reshape(4,4),np.repeat(counts.sum(1)[:,None],4,axis=1))
                np.testing.assert_allclose(matrix.row_fraction.to_numpy().reshape(4,4),counts/counts.sum(1,keepdims=True),atol=1e-12)
                support_checks+=1
    out=root/'report'
    out.mkdir(exist_ok=False)
    group=['scene','source_load_hp','target_load_hp','subset','protocol','method','seed','condition']
    # Average noise instances first. Each direction/subset subsequently has equal weight.
    per_case=metrics.groupby(group,as_index=False)[METRICS].mean()
    per_case.to_csv(out/'model_snr_metrics.csv',index=False)
    per_seed=per_case.groupby(['scene','protocol','method','condition','seed'],as_index=False)[METRICS].mean()
    per_seed.to_csv(out/'seed_equal_direction_subset.csv',index=False)
    summary=per_seed.groupby(['scene','protocol','method','condition'])[METRICS].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in summary.columns]
    summary.to_csv(out/'summary.csv',index=False)
    subset=per_case.groupby(['scene','protocol','method','condition','subset'],as_index=False)[METRICS].mean()
    subset.to_csv(out/'subset_means.csv',index=False)
    direction=per_case[per_case.scene.eq('load')].groupby(
        ['source_load_hp','target_load_hp','protocol','method','condition'],as_index=False)[METRICS].mean()
    direction.to_csv(out/'directions.csv',index=False)
    pair_keys=[k for k in group if k!='method']
    paired=per_case[per_case.method.eq('fusion')].merge(per_case[per_case.method.eq('data')],
        on=pair_keys,validate='one_to_one',suffixes=('_fusion','_data'))
    for metric in METRICS:
        paired[metric+'_difference']=paired[metric+'_fusion']-paired[metric+'_data']
    paired.to_csv(out/'paired_differences.csv',index=False)
    cm=confusion.groupby(['scene','protocol','method','condition','true_label','predicted_label'],as_index=False).row_fraction.mean()
    cm.to_csv(out/'mean_normalized_confusion.csv',index=False)
    verification=dict(passed=True,prediction_files=expected_pairs,metric_rows=expected_metrics,confusion_cells=16*expected_metrics,
        recomputed_conditions=support_checks,max_metric_error=max_error,noise_instances_averaged_first=True,
        source_target_directions_equally_weighted=True,subsets_equally_weighted=True)
    write_json(out/'verification.json',verification)
    lines=['# 方案一六维物理描述符10%实验报告','',
        '固定λ=0.3。窗口与跨负载均包含纯干净及AWGN混合训练，测试干净及五档AWGN，每档五个固定实例。',
        '新增150个融合模型，引用150个冻结数据单分支模型。窗口纯干净数据单分支缺少的375个加噪条件首次补测；其他旧概率只读复用。',
        '已从保存预测复算全部20280行指标和324480个混淆单元，目标标签和窗口顺序均已核验。','',
        '|划分|训练|测试|方法|ACC %|BA %|Macro-F1 %|滚动体召回 %|',
        '|---|---|---|---|---|---|---|---|']
    for row in summary.to_dict('records'):
        scene='窗口' if row['scene']=='window' else '跨负载'
        protocol='纯干净' if row['protocol']=='clean_only' else 'AWGN混合'
        method='融合' if row['method']=='fusion' else '数据单分支'
        values=[f"{100*row[m+'_mean']:.2f} ± {100*row[m+'_std']:.2f}" for m in ['acc','balanced_acc','f1_macro','recall_B']]
        lines.append('|'+ '|'.join([scene,protocol,row['condition'],method,*values])+'|')
    lines += ['',
        '先在同一模型和SNR内平均五次加噪，再对方向及三个子集等权，最后计算五训练种子的均值与样本标准差。标准差不等于独立重复试验的总体不确定性。',
        '窗口实验来自同一批记录，不能证明跨记录泛化。10%只约束训练窗，窗口仍使用103个验证标签，跨负载使用来源完整验证标签。',
        '数据分支、门控、辅助损失及网络设置沿用既有协议，仅整体替换物理描述符；不能据此分离各物理指标的因果贡献。',
        '历史测试曝光继续存在。跨负载只在本数据集四种负载和已知转速下评估，不外推至任意设备、自然噪声或明显变转速。',
        '纯干净与混合训练同时涉及训练观测、标准化拟合及验证选模变化。个别方向或类别退化须据directions.csv和混淆矩阵逐项披露。','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('REPORT VERIFIED',verification,flush=True)
