"""合成测试：不加载或训练任何正式模型，不修改正式结果。"""
import torch  # Windows 下必须先于 numpy 导入。
import json
from contextlib import contextmanager
import shutil
import tempfile
import uuid
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import author_comparison_report as report

RATIOS=[.1,.2,.5]
SUBSETS=[4101,4102,4103]
SEEDS=list(range(1706,1711))


@contextmanager
def test_directory():
    # 临时检查只写系统临时目录，结束后删除，不留在项目目录。
    base=Path(tempfile.gettempdir()).resolve()
    path=base/('comparison_report_test_'+uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != base or not path.name.startswith('comparison_report_test_'):
            raise ValueError('临时测试目录路径不符')
        shutil.rmtree(path)


def synthetic_frames():
    """纯干净小样本合成结果：只含本轮方法与两个参考方法。"""
    shifts={'ClassBD_WDCNN':0.,'baseline_data':.06,'data_aux_ce':.10}
    records=[]
    for source in range(4):
        for ratio in RATIOS:
            for subset in SUBSETS:
                for method in shifts:
                    for seed in SEEDS:
                        for target in range(4):
                            if target==source:continue
                            sv=(subset-4101)*.0001
                            value=.35+shifts[method]+source*.003+target*.002+(seed-1706)*.001+sv
                            item=dict(domain=f'load_source_{source}',ratio=ratio,subset=subset,
                                protocol='clean_only',variant=method,seed=seed,
                                source_load_hp=source,target_load_hp=target,
                                condition='clean',noise_seed=0,
                                job_id=f'{method}_{source}_{ratio}_{subset}_{seed}')
                            item.update({k:value for k in report.METRICS})
                            records.append(item)
    all_=pd.DataFrame(records)
    new=all_[all_.variant.eq('ClassBD_WDCNN')].copy()
    old=all_[~all_.variant.eq('ClassBD_WDCNN')].copy()
    return new,old


def synthetic_full():
    """全样本纯干净合成结果：四来源 × 五种子 × 三目标。"""
    shifts={'ClassBD_WDCNN':0.,'baseline_data':.06,'data_aux_ce':.10}
    records=[]
    for source in range(4):
        for method in shifts:
            for seed in SEEDS:
                for target in range(4):
                    if target==source:continue
                    value=.35+shifts[method]+source*.003+target*.002+(seed-1706)*.001
                    item=dict(domain=f'load_source_{source}',ratio=1.,subset=0,protocol='clean_only',
                        variant=method,seed=seed,source_load_hp=source,target_load_hp=target,
                        condition='clean',noise_seed=0,job_id=f'{method}_{source}_{seed}')
                    item.update({k:value for k in report.METRICS})
                    records.append(item)
    all_=pd.DataFrame(records)
    return all_[all_.variant.eq('ClassBD_WDCNN')].copy(), all_[~all_.variant.eq('ClassBD_WDCNN')].copy()


def _write_sources(root):
    source=root/'old_small'
    source.mkdir()
    (source/'config.json').write_text(json.dumps({'jobs':[]}),encoding='utf8')
    return source


class ReportTests(unittest.TestCase):
    def test_summary_seed_subset_pair_and_output_guard(self):
        new,old=synthetic_frames()
        with test_directory() as directory:
            root=Path(directory)
            source=_write_sources(root)
            (root/'config.json').write_text(json.dumps(dict(
                jobs=[{}]*180,decisions_locked=True,target_test_accessed=True,
                scope=dict(method='ClassBD_WDCNN',ratios=RATIOS,protocol='clean_only',subsets=SUBSETS))),encoding='utf8')
            def fake_verify(path,*unused):
                return new if path==root else old
            with patch.object(report,'SMALL',source),patch.object(report,'_validate_splits',return_value=40), \
                 patch.object(report,'_verify',side_effect=fake_verify):
                report.finish(root,scope=dict(method='ClassBD_WDCNN',ratios=RATIOS,protocol='clean_only',subsets=SUBSETS))
                mean=pd.read_csv(root/'report/equal_12_directions_seed_mean_sd.csv')
                sel=mean[(mean.ratio==.1)&(mean.subset==4101)&(mean.variant=='ClassBD_WDCNN')&(mean.condition=='clean')].iloc[0]
                # 12 方向中各来源、各目标分别出现三次。
                self.assertAlmostEqual(sel.balanced_acc_mean,.35+.0045+.003+.002,places=10)
                self.assertAlmostEqual(sel.balanced_acc_std,np.std(np.arange(5)*.001,ddof=1),places=10)
                subset=pd.read_csv(root/'report/between_subset_mean_sd.csv')
                row=subset[(subset.ratio==.1)&(subset.variant=='ClassBD_WDCNN')&(subset.condition=='clean')].iloc[0]
                self.assertAlmostEqual(row.balanced_acc_mean_std,.0001,places=10)
                pair=pd.read_csv(root/'report/paired_equal_12_directions_seed_mean_sd.csv')
                row=pair[(pair.ratio==.1)&(pair.subset==4101)&(pair.comparator=='ClassBD_WDCNN')&(pair.condition=='clean')].iloc[0]
                self.assertAlmostEqual(row.balanced_acc_mean,.10,places=10)
                self.assertAlmostEqual(row.balanced_acc_std,0.,places=10)
                text=(root/'REPORT.md').read_text(encoding='utf8')
                self.assertIn('本文融合加辅助分类',text)
                self.assertIn('10%',text)
                with self.assertRaises(FileExistsError): report.finish(root)

    def test_full_sample_report_has_no_subset_block(self):
        new,old=synthetic_full()
        scope=dict(method='ClassBD_WDCNN',sample='full',ratios=[1.],subsets=[0],protocol='clean_only')
        with test_directory() as directory:
            root=Path(directory)
            source=_write_sources(root)
            (root/'config.json').write_text(json.dumps(dict(
                jobs=[{}]*20,decisions_locked=True,target_test_accessed=True,scope=scope)),encoding='utf8')
            seen=[]
            def fake_verify(path,ratios,models,variants,split_root):
                seen.append((path,list(ratios),models,set(variants)))
                return new if path==root else old
            with patch.object(report,'FULL',source),patch.object(report,'_validate_splits',return_value=40), \
                 patch.object(report,'_verify',side_effect=fake_verify):
                report.finish(root,scope=scope)
            self.assertEqual(seen[0][1],[1.0])
            self.assertEqual(seen[0][2],20)
            self.assertEqual(seen[1][2],40)
            sd=pd.read_csv(root/'report/equal_12_directions_seed_mean_sd.csv')
            self.assertEqual(len(sd),3)
            self.assertFalse((root/'report/between_subset_mean_sd.csv').exists())
            text=(root/'REPORT.md').read_text(encoding='utf8')
            self.assertIn('全样本',text)
            self.assertNotIn('10%/20%/50%',text)
            self.assertIn('| 训练比例 | ',text)

    def test_ratio_scope_restriction(self):
        new,old=synthetic_frames()
        scope=dict(method='ClassBD_WDCNN',ratios=[.1],protocol='clean_only',subsets=SUBSETS)
        with test_directory() as directory:
            root=Path(directory)
            source=_write_sources(root)
            (root/'config.json').write_text(json.dumps(dict(
                jobs=[{}]*60,decisions_locked=True,target_test_accessed=True,scope=scope)),encoding='utf8')
            seen=[]
            def fake_verify(path,ratios,models,variants,split_root):
                seen.append((path,list(ratios),models,set(variants)))
                data=new if path==root else old
                return data[data.ratio.isin(list(ratios))]
            with patch.object(report,'SMALL',source),patch.object(report,'_validate_splits',return_value=40), \
                 patch.object(report,'_verify',side_effect=fake_verify):
                report.finish(root,scope=scope)
            self.assertEqual(seen[0][1],[.1])
            self.assertEqual(seen[0][2],60)
            self.assertEqual(seen[1][2],120)
            mean=pd.read_csv(root/'report/equal_12_directions_seed_mean_sd.csv')
            self.assertEqual(set(mean.ratio),{.1})

    def test_incomplete_direction_seed_or_subset_rejected(self):
        new,old=synthetic_frames()
        scope=dict(method='ClassBD_WDCNN',ratios=RATIOS,protocol='clean_only',subsets=SUBSETS)
        masks=[
            (new.variant=='ClassBD_WDCNN')&(new.source_load_hp==0)&(new.target_load_hp==1)&(new.ratio==.1)&(new.subset==4101)&(new.seed==1706),
            (new.variant=='ClassBD_WDCNN')&(new.ratio==.1)&(new.subset==4101)&(new.seed==1706),
            (new.variant=='ClassBD_WDCNN')&(new.ratio==.1)&(new.subset==4101),
        ]
        for mask in masks:
            with self.subTest(removed=int(mask.sum())),test_directory() as root:
                source=_write_sources(root)
                (root/'config.json').write_text(json.dumps(dict(
                    jobs=[{}]*180,decisions_locked=True,target_test_accessed=True,scope=scope)),encoding='utf8')
                bad=new[~mask].copy()
                def fake_verify(path,*unused):
                    return bad if path==root else old
                with patch.object(report,'SMALL',source),patch.object(report,'_validate_splits',return_value=40), \
                     patch.object(report,'_verify',side_effect=fake_verify):
                    with self.assertRaises(ValueError):report.finish(root,scope=scope)

    def test_wrong_model_count_or_unlocked_rejected(self):
        new,old=synthetic_frames()
        scope=dict(method='ClassBD_WDCNN',ratios=RATIOS,protocol='clean_only',subsets=SUBSETS)
        for state in [dict(jobs=[{}]*179),dict(jobs=[{}]*180,decisions_locked=False),
                      dict(jobs=[{}]*180,decisions_locked=True,target_test_accessed=False)]:
            with self.subTest(state=state.keys()),test_directory() as root:
                source=_write_sources(root)
                body=dict(target_test_accessed=True,decisions_locked=True)
                body.update(state)
                body['scope']=scope
                (root/'config.json').write_text(json.dumps(body),encoding='utf8')
                with patch.object(report,'SMALL',source),patch.object(report,'_validate_splits',return_value=40), \
                     patch.object(report,'_verify',return_value=new):
                    with self.assertRaises(ValueError):report.finish(root,scope=scope)

    def test_real_prediction_audit_and_reject_tampering(self):
        with test_directory() as directory:
            root=Path(directory);ev=root/'evaluation';pred=ev/'predictions';pred.mkdir(parents=True)
            split=root/'splits'/'load_source_0_r010_s4101';split.mkdir(parents=True)
            frame=pd.DataFrame(dict(window_index=range(16),load_hp=np.repeat(range(4),4),
                label=list(range(4))*4,rpm=[1750.]*16))
            frame['role']=['train']*4+['test']*12
            frame.to_csv(split/'windows.csv',index=False)
            labels=np.arange(4)
            probability=np.eye(4)*.9+.025
            vals=report._counts(labels,probability)
            base=dict(job_id='m0001',domain='load_source_0',ratio=.1,subset=4101,protocol='clean_only',
                      variant='ClassBD_WDCNN',seed=1706,source_load_hp=0,target_load_hp=1)
            rows=[];evaluations=[]
            for target in [1,2,3]:
                row=dict(base,target_load_hp=target)
                evaluations.append(row)
                item=dict(row,condition='clean',noise_seed=0);item.update(dict(zip(report.METRICS,vals)))
                rows.append(item)
                ix=np.flatnonzero(frame.load_hp.eq(target).to_numpy())
                np.savez_compressed(pred/f'm0001_target{target}.npz',
                    labels=frame.label.to_numpy()[ix],window_indices=frame.window_index.to_numpy()[ix],
                    clean_n0=probability)
            # 非纯干净的记录必须先被过滤，不参与本轮核对。
            noise=dict(base,job_id='m0002',protocol='awgn_mixed',condition='white_-4',noise_seed=7201)
            evaluations.append(dict(noise));item=dict(noise);item.update(dict(zip(report.METRICS,vals)));rows.append(item)
            pd.DataFrame(evaluations).to_csv(ev/'evaluations.csv',index=False)
            df=pd.DataFrame(rows);df.to_csv(ev/'metrics.csv',index=False)
            report._verify(root,[.1],1,['ClassBD_WDCNN'],root/'splits')
            altered=df.copy();altered.loc[0,'balanced_acc']=.5;altered.to_csv(ev/'metrics.csv',index=False)
            with self.assertRaises(ValueError):report._verify(root,[.1],1,['ClassBD_WDCNN'],root/'splits')
            df.to_csv(ev/'metrics.csv',index=False)
            path=pred/'m0001_target1.npz'
            arrays=dict(labels=frame.label.to_numpy()[4:8],window_indices=frame.window_index.to_numpy()[4:8],clean_n0=probability)
            arrays['labels']=arrays['labels'][::-1];np.savez_compressed(path,**arrays)
            with self.assertRaises(AssertionError):report._verify(root,[.1],1,['ClassBD_WDCNN'],root/'splits')
            arrays['labels']=array_labels=frame.label.to_numpy()[4:8]
            np.savez_compressed(path,labels=array_labels,window_indices=frame.window_index.to_numpy()[4:8],
                white__4_n7201=probability)
            with self.assertRaises(ValueError):report._verify(root,[.1],1,['ClassBD_WDCNN'],root/'splits')


if __name__=='__main__':
    unittest.main()
