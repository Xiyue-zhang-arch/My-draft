"""模拟：测试阶段因文件占用中断后，从已保存权重续测缺失模型；不训练、不碰正式结果。"""
from pathlib import Path
import json
import sys
import tempfile
import shutil
import types
import uuid
from unittest.mock import patch

import torch
import numpy as np
import pandas as pd
import run_load_comparisons as runner

METRIC_COLS = dict(acc=.9, balanced_acc=.9, f1_macro=.9, recall_N=.9, recall_IR=.9, recall_OR=.9, recall_B=.9)


def main():
    base = Path(tempfile.gettempdir()).resolve()/('cmp_recovery_sim_'+uuid.uuid4().hex)
    base.mkdir()
    try:
        project = base
        (project/'experiments/CWRU/实验结果'/'CWRU_V3').mkdir(parents=True)
        root = project/'experiments/CWRU/实验结果'/'CWRU_V3'/'load_comparison_small_AW_Hybrid_adapted_sim'
        (root/'models'/'m0001').mkdir(parents=True)
        (root/'models'/'m0002').mkdir(parents=True)
        (root/'splits'/'load_source_0_r010_s4101').mkdir(parents=True)
        (root/'evaluation'/'predictions').mkdir(parents=True)
        records = pd.DataFrame(dict(window_index=range(16), load_hp=np.repeat(range(4), 4),
                                    role=['train']*4+['test']*12, label=list(range(4))*4, rpm=[1750.]*16))
        records.to_csv(root/'splits'/'load_source_0_r010_s4101'/'windows.csv', index=False)
        jobs = []
        for i, seed in enumerate((1706, 1707), 1):
            jid = f'm{i:04d}'
            (root/'models'/jid/'model.pt').write_bytes(b'simulated')
            (root/'models'/jid/'completed.json').write_text('{}', encoding='utf-8')
            jobs.append(dict(job_id=jid, domain='load_source_0', ratio=.1, subset=4101, protocol='clean_only',
                             variant='AW_Hybrid_adapted', seed=seed, folder=str(root/'models'/jid),
                             source_load_hp=0, completed=True, best_epoch=1))
        # 已完成 m0001 的三个目标负载
        evals, metrics = [], []
        for target in (1, 2, 3):
            meta = dict(jobs[0], target_load_hp=target)
            np.savez_compressed(root/'evaluation'/'predictions'/f'm0001_target{target}.npz',
                                labels=records.label.to_numpy()[4*target:4*target+4],
                                window_indices=records.window_index.to_numpy()[4*target:4*target+4],
                                clean_n0=np.eye(4)*.9+.025)
            metrics.append(dict(meta, condition='clean', noise_seed=0, **METRIC_COLS))
            evals.append(dict(meta, prediction_path='x', test_windows=4))
        pd.DataFrame(metrics).to_csv(root/'evaluation'/'metrics.csv', index=False)
        pd.DataFrame(evals).to_csv(root/'evaluation'/'evaluations.csv', index=False)
        state = dict(stage='failed', error_type='PermissionError', error='[WinError 5] simulated',
                     decisions_locked=True, weights_verified=True, verified_models=2,
                     target_test_accessed=True, planned_models=2, planned_test_pairs=6, planned_metric_rows=6,
                     trained_models=2, evaluated_pairs=3, metric_rows=3, jobs=jobs,
                     scope=dict(method='AW_Hybrid_adapted', ratios=[.1], protocol='clean_only',
                                seeds=[1706], subsets=[4101], sources=['load_source_0']))
        (root/'config.json').write_text(json.dumps(state), encoding='utf-8')
        cases = [{k: j[k] for k in runner.CASE_KEYS} for j in jobs]

        class Engine:
            @staticmethod
            def load(folder, device):
                return object()

            @staticmethod
            def predict_and_ce(model, x, rpm, labels, device):
                return np.full((len(labels), 4), .25, dtype=np.float32), 0.

        def conditions(x, protocol, test):
            assert protocol == 'clean_only'
            yield 'clean', 0, x

        reports = []

        def finish(path, scope=None):
            reports.append(str(path))
            (path/'REPORT.md').write_text('synthetic recovery only', encoding='utf-8')

        class Sim(runner.ComparisonRun):
            def setup(self, resume=False):
                self.device = torch.device('cpu')
                self.tables = {'load_source_0_r010_s4101': records}
                self.x = np.zeros((16, 1, 8), dtype=np.float32)

            def verify_checkpoint(self, job):
                return None

        with patch.object(runner, 'PROJECT', project), \
             patch.object(runner, 'scope_cases', lambda method, sample, ratios: iter(cases)), \
             patch.object(runner.R, 'conditions', conditions), \
             patch.dict(sys.modules, {'comparison_engine': Engine,
                                      'comparison_report': types.SimpleNamespace(finish=finish)}):
            # 1) 非文件占用类失败必须拒绝
            bad = dict(state, error_type='ValueError')
            (root/'config.json').write_text(json.dumps(bad), encoding='utf-8')
            try:
                Sim.open_for_recovery(root)
                raise AssertionError('非文件占用失败未被拒绝')
            except ValueError:
                pass
            # 2) 已测试模型不是前缀必须拒绝
            (root/'config.json').write_text(json.dumps(state), encoding='utf-8')
            shuffled = pd.read_csv(root/'evaluation'/'evaluations.csv')
            shuffled['job_id'] = ['m0002', 'm0002', 'm0002']
            shuffled.to_csv(root/'evaluation'/'evaluations.csv', index=False)
            try:
                Sim.open_for_recovery(root)
                raise AssertionError('非前缀已完成模型未被拒绝')
            except ValueError:
                pass
            pd.DataFrame(evals).to_csv(root/'evaluation'/'evaluations.csv', index=False)
            # 3) 正常续测
            with runner.exclusive_run_lock(root):
                experiment = Sim.open_for_recovery(root)
                experiment.recover_testing()
            final = json.loads((root/'config.json').read_text(encoding='utf-8'))
            assert final['stage'] == 'complete', final['stage']
            assert final['evaluated_pairs'] == 6 and final['metric_rows'] == 6
            assert final['test_recovery'][0]['completed_before'] == 1
            assert final['test_recovery'][0]['missing_models'] == 1
            assert len(pd.read_csv(root/'evaluation'/'evaluations.csv')) == 6
            assert len(pd.read_csv(root/'evaluation'/'metrics.csv')) == 6
            assert len(list((root/'evaluation'/'predictions').glob('*.npz'))) == 6
            assert reports and (root/'REPORT.md').is_file()
            # 4) 完成后再续测必须拒绝
            try:
                Sim.open_for_recovery(root)
                raise AssertionError('已完成后仍允许续测')
            except ValueError:
                pass
        print('模拟通过：拒绝非文件占用失败与非前缀现场，仅补测 m0002 后 6 组合 6 指标并生成报告')
    finally:
        if base.parent == Path(tempfile.gettempdir()).resolve() and base.name.startswith('cmp_recovery_sim_'):
            shutil.rmtree(base)


if __name__ == '__main__':
    main()
