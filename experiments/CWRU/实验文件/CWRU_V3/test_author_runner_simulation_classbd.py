"""模拟主控的训练锁定、目标测试、报告及失败保护，不使用正式数据。"""
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import json
import sys
import tempfile
import types
import uuid
import shutil
import torch
import numpy as np
import pandas as pd
import run_author_comparisons as runner


@contextmanager
def workspace_temp():
    # 模拟只写系统临时目录，结束后删除，不留在项目目录。
    path = Path(tempfile.gettempdir()).resolve()/('comparison_runner_sim_' + uuid.uuid4().hex)
    path.mkdir(parents=True, exist_ok=False)
    print('模拟目录:', path, flush=True)
    try:
        yield path
    finally:
        if path.parent != Path(tempfile.gettempdir()).resolve() or not path.name.startswith('comparison_runner_sim_'):
            raise ValueError('模拟目录路径不符')
        shutil.rmtree(path)


def main():
    assert runner.parse_ratio('all') == [.1, .2, .5]
    assert runner.parse_ratio('10') == [.1]
    assert runner.parse_ratio('20%') == [.2]
    assert runner.plan_counts('small', [.1, .2, .5]) == (180, 540, 540)
    assert runner.plan_counts('small', [.1]) == (60, 180, 180)
    assert runner.plan_counts('full', [.1, .2, .5]) == (20, 60, 60)
    assert len(list(runner.scope_cases('ClassBD_WDCNN', 'small', [.1, .2, .5]))) == 180
    assert len(list(runner.scope_cases('ClassBD_WDCNN', 'full', [.1, .2, .5]))) == 20
    assert {c['ratio'] for c in runner.scope_cases('ClassBD_WDCNN', 'full', [.1])} == {1.0}
    assert {c['protocol'] for c in runner.scope_cases('ClassBD_WDCNN', 'small', [.1])} == {'clean_only'}
    # 在系统临时目录下测试，不访问正式结果目录。
    with workspace_temp() as temporary:
        project = Path(temporary)
        (project/'experiments/CWRU/实验结果'/'CWRU_V3').mkdir(parents=True)
        root = project/'experiments/CWRU/实验结果'/'CWRU_V3'/'load_comparison_author_sim'
        records = pd.DataFrame(dict(window_index=range(16), load_hp=np.repeat(range(4),4),
            role=['train']*4+['test']*12, label=list(range(4))*4, rpm=[1750.]*16))
        examples = [dict(domain='load_source_0', ratio=.1, subset=4101, protocol='clean_only',
                         variant='ClassBD_WDCNN', seed=seed) for seed in (1706,1707)]
        pause_requested=[]
        class Engine:
            @staticmethod
            def train(x,w,protocol,variant,seed,subset,folder,device,spec):
                assert set(w.role).issubset({'train','validation'})
                assert protocol == 'clean_only'
                folder.mkdir(); (folder/'model.pt').write_bytes(b'simulated')
                (folder/'completed.json').write_text('{}',encoding='utf-8')
                if not pause_requested:
                    runner.request_pause(root)
                    pause_requested.append(True)
                return dict(best_epoch=1)
            @staticmethod
            def load(folder,device): return object()
            @staticmethod
            def predict_and_ce(model,x,rpm,labels,device):
                return np.eye(4,dtype=np.float32)[labels], 0.
        def conditions(x,protocol,test):
            assert protocol == 'clean_only'
            yield 'clean',0,x
        report_called=[]
        audited=[]
        def finish(path,scope=None):
            state=json.loads((path/'config.json').read_text(encoding='utf-8'))
            assert state['decisions_locked'] and state['target_test_accessed'] and state['stage']=='reporting'
            assert scope and scope['method']=='ClassBD_WDCNN'
            (path/'REPORT.md').write_text('synthetic only',encoding='utf-8');report_called.append(True)
        class Simulated(runner.ComparisonRun):
            def setup(self, resume=False):
                self.device=torch.device('cpu')
                self.tables={'load_source_0_r010_s4101':records}
                self.x=np.zeros((16,1,8),dtype=np.float32)
            def verify_checkpoint(self, job):
                audited.append(job['job_id'])
        with patch.object(runner,'PROJECT',project), patch.object(runner,'scope_cases',lambda method,sample,ratios: iter(examples)), \
             patch.object(runner,'plan_counts',lambda sample,ratios:(2,6,6)),patch.object(runner.R,'conditions',conditions), \
             patch.object(runner,'get_engine',lambda method:Engine), patch.dict(sys.modules,{'author_comparison_report':types.SimpleNamespace(finish=finish)}):
            with runner.exclusive_run_lock():
                try:
                    with runner.exclusive_run_lock():
                        raise AssertionError('并发运行锁未生效')
                except RuntimeError:
                    pass
            # 正常退出后允许再次取得同一锁，残留锁文件不会阻止后续运行。
            with runner.exclusive_run_lock():
                pass
            run=Simulated(root,'ClassBD_WDCNN','small',[.1])
            assert run.state['scope']==dict(method='ClassBD_WDCNN',sample='small',ratios=[.1],protocol='clean_only',
                seeds=list(range(1706,1711)),subsets=[4101,4102,4103],
                sources=[f'load_source_{i}' for i in range(4)])
            assert run.state['planned_models']==2 and run.state['planned_test_pairs']==6
            try: run.evaluate(); raise AssertionError('测试锁未生效')
            except ValueError: pass
            with runner.exclusive_run_lock(root):
                run.run()
            state=json.loads((root/'config.json').read_text(encoding='utf-8'))
            assert state['stage']=='paused' and state['pause_boundary_models']==1 and not state['target_test_accessed']
            assert len(list((root/'models').iterdir())) == 1
            for corruption in [dict(stage='failed'),dict(active_model={'job_id':'m0002'}),
                               dict(target_test_accessed=True),dict(pause_boundary_verified=False),
                               dict(pause_boundary_models=0),dict(planned_models=3)]:
                bad=dict(state,**corruption)
                (root/'config.json').write_text(json.dumps(bad),encoding='utf-8')
                try:
                    Simulated.open_paused(root)
                    raise AssertionError('不安全恢复未被拒绝')
                except ValueError:
                    pass
            (root/'config.json').write_text(json.dumps(state),encoding='utf-8')
            extra=root/'models'/'m0002'
            extra.mkdir()
            try:
                Simulated.open_paused(root)
                raise AssertionError('未登记模型目录未被拒绝')
            except ValueError:
                pass
            extra.rmdir()
            with runner.exclusive_run_lock(root):
                resumed=Simulated.open_paused(root)
                resumed.activate_resume()
                assert resumed.train_all()
                resumed.evaluate()
            state=json.loads((root/'config.json').read_text(encoding='utf-8'))
            assert state['stage']=='complete' and state['trained_models']==2
            assert state['resume_history'][0]['completed_models']==1
            assert state['weights_verified'] and audited==['m0001','m0001','m0002']
            assert state['evaluated_pairs']==6 and state['metric_rows']==6 and report_called
            assert len(pd.read_csv(root/'evaluation'/'confusion_matrices.csv'))==6*16
            try: resumed.evaluate(); raise AssertionError('重复测试保护未生效')
            except ValueError: pass
            try: Simulated.open_paused(root); raise AssertionError('测试后恢复保护未生效')
            except ValueError: pass
            try: Simulated(root,'ClassBD_WDCNN','small',[.1]); raise AssertionError('重复启动保护未生效')
            except FileExistsError: pass
            try: Simulated(root.parent/'other_run','ClassBD_WDCNN','small',[.1]); raise AssertionError('目录命名保护未生效')
            except ValueError: pass
        print('模拟通过：范围解析、训练锁定、6组合、6指标、混淆计数、报告、暂停/恢复与重复启动保护')


if __name__=='__main__':main()
