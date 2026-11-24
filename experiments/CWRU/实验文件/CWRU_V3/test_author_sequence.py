"""Safety tests with temporary synthetic state; never invokes real training."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

import author_sequence as seq


@contextmanager
def workspace():
    base=Path(tempfile.gettempdir()).resolve()
    root=base/('author_seq_sim_'+uuid.uuid4().hex)
    root.mkdir()
    try:
        yield root
    finally:
        if root.parent != base or not root.name.startswith('author_seq_sim_'):
            raise RuntimeError('temporary root invalid')
        shutil.rmtree(root)


def completed(path,method,sample):
    path.mkdir()
    n=180 if sample=='small' else 20
    state=dict(version='CWRU_V3_author_comparison_clean',stage='complete',trained_models=n,
       planned_models=n,planned_test_pairs=3*n,planned_metric_rows=3*n,evaluated_pairs=3*n,metric_rows=3*n,
       decisions_locked=True,weights_verified=True,target_test_accessed=True,
       scope=dict(method=method,sample=sample,protocol='clean_only',
                  ratios=[.1,.2,.5] if sample=='small' else [1.],seeds=list(range(1706,1711))))
    (path/'config.json').write_text(json.dumps(state),encoding='utf8')
    (path/'REPORT.md').write_text('synthetic',encoding='utf8')


@contextmanager
def no_lock(*args,**kwargs):
    yield


class SequenceTests(unittest.TestCase):
    def test_wait_rejects_stopped_incomplete_run(self):
        with workspace() as root:
            path=root/'run'
            completed(path,'DWCN_SDGFD','small')
            state=seq.read_state(path);state['stage']='training'
            (path/'config.json').write_text(json.dumps(state),encoding='utf8')
            with patch.object(seq.runner,'exclusive_run_lock',no_lock):
                with self.assertRaisesRegex(ValueError,'已退出'):
                    seq.wait_for_current(path,{},lambda:None,1)

    def test_wait_rejects_failed_or_paused_run(self):
        for stage in ('failed','paused'):
            with self.subTest(stage=stage),workspace() as root:
                path=root/'run'
                completed(path,'DWCN_SDGFD','small')
                state=seq.read_state(path);state['stage']=stage
                (path/'config.json').write_text(json.dumps(state),encoding='utf8')
                with self.assertRaises(RuntimeError):
                    seq.wait_for_current(path,{},lambda:None,1)

    def test_complete_incomplete_and_scope(self):
        with workspace() as root:
            path=root/'run'
            completed(path,'DWCN_SDGFD','small')
            seq.validate_complete(path,'DWCN_SDGFD','small')
            state=seq.read_state(path);state['trained_models']=179
            (path/'config.json').write_text(json.dumps(state),encoding='utf8')
            with self.assertRaises(RuntimeError):seq.validate_complete(path,'DWCN_SDGFD','small')
            with self.assertRaises(ValueError):seq.validate_scope(state,'ClassBD_WDCNN','small')

    def test_sequential_order_and_duplicate_guard(self):
        with workspace() as root:
            watched=root/'load_comparison_author_small_DWCN_SDGFD_original'
            completed(watched,'DWCN_SDGFD','small')
            launched=[]
            class Child:
                pid=123
                def __init__(self,cmd,**kwargs):
                    method=cmd[cmd.index('--method')+1];sample=cmd[cmd.index('--sample')+1]
                    out=Path(cmd[cmd.index('--output')+1]);launched.append((method,sample))
                    completed(out,method,sample)
                def wait(self):return 0
            args=['author_sequence.py','--watch',str(watched),'--tag','sim']
            with patch.object(seq,'PARENT',root),patch.object(seq.runner,'PROJECT',root), \
                 patch.object(seq.runner,'file_lock',no_lock),patch.object(seq.subprocess,'Popen',Child), \
                 patch.object(sys,'argv',args):
                seq.main()
                with self.assertRaises(FileExistsError):seq.main()
            self.assertEqual(launched,[('DWCN_SDGFD','full'),('ClassBD_WDCNN','small'),('ClassBD_WDCNN','full')])
            progress=json.loads((root/'author_sequence_sim/progress.json').read_text(encoding='utf8'))
            self.assertEqual(progress['stage'],'complete')

    def test_failure_stops_later_steps(self):
        with workspace() as root:
            watched=root/'load_comparison_author_small_DWCN_SDGFD_original'
            completed(watched,'DWCN_SDGFD','small')
            launched=[]
            class Child:
                pid=123
                def __init__(self,cmd,**kwargs):launched.append(cmd[cmd.index('--method')+1])
                def wait(self):return 1
            with patch.object(seq,'PARENT',root),patch.object(seq.runner,'PROJECT',root), \
                 patch.object(seq.runner,'file_lock',no_lock),patch.object(seq.subprocess,'Popen',Child), \
                 patch.object(sys,'argv',['author_sequence.py','--watch',str(watched),'--tag','fail']):
                with self.assertRaises(RuntimeError):seq.main()
            self.assertEqual(launched,['DWCN_SDGFD'])
            progress=json.loads((root/'author_sequence_fail/progress.json').read_text(encoding='utf8'))
            self.assertEqual(progress['stage'],'failed')
            self.assertEqual([s['status'] for s in progress['steps']],['failed','pending','pending'])


if __name__=='__main__':unittest.main()
