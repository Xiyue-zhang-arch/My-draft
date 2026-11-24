"""完全使用临时数据测试等待器，不启动真实训练或汇总。"""
from contextlib import contextmanager,ExitStack
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
import torch
import author_summary_waiter as w

@contextmanager
def temp_root():
    base=Path(tempfile.gettempdir()).resolve();p=base/('summary_wait_sim_'+uuid.uuid4().hex);p.mkdir()
    try:yield p
    finally:
        if p.parent!=base or not p.name.startswith('summary_wait_sim_'):raise ValueError('临时目录无效')
        shutil.rmtree(p)

@contextmanager
def no_lock(*args,**kwargs):yield

def sequence(stage,complete=False):
    steps=[]
    for method,sample,path in [('DWCN_SDGFD','full',w.DWCN_FULL),('ClassBD_WDCNN','small',w.CLASSBD_SMALL),('ClassBD_WDCNN','full',w.CLASSBD_FULL)]:
        steps.append(dict(method=method,sample=sample,output=str(path),status='complete' if complete else 'pending',exit_code=0 if complete else None))
    return dict(stage=stage,pid=123,watched=str(w.DWCN_SMALL),steps=steps)

class Tests(unittest.TestCase):
    def test_wait_complete_and_reject_partial(self):
        with patch.object(w,'read_sequence',return_value=sequence('complete',True)):
            w.wait(1,{},lambda:None)
        with patch.object(w,'read_sequence',return_value=sequence('complete',False)):
            with self.assertRaises(ValueError):w.wait(1,{},lambda:None)

    def test_failure_and_dead_process(self):
        with patch.object(w,'read_sequence',return_value=sequence('failed')):
            with self.assertRaises(RuntimeError):w.wait(1,{},lambda:None)
        with patch.object(w,'read_sequence',return_value=sequence('running')), \
             patch.object(w,'sequence_lock_held',return_value=True),patch.object(w,'process_identity',return_value=(42,0)):
            with self.assertRaises(RuntimeError):w.wait(1,{},lambda:None)

    def test_wait_then_summarize_and_no_repeat(self):
        with temp_root() as root:
            output=root/'output';state=root/'waiter'
            class FakeChild:
                pid=88
                def __init__(self,*args,**kwargs):
                    output.mkdir();(output/'REPORT.md').write_text('synthetic');(output/'verification.json').write_text('[]')
                def wait(self):return 0
            with patch.object(w,'STATE',state),patch.object(w,'OUTPUT',output),patch.object(w,'BASE',root), \
                 patch.object(w.runner,'file_lock',no_lock),patch.object(w,'read_sequence',return_value=sequence('complete',True)), \
                 patch.object(w.subprocess,'Popen',FakeChild),patch.object(sys,'argv',['waiter']):
                w.main()
                self.assertEqual(json.loads((state/'progress.json').read_text(encoding='utf8'))['stage'],'complete')
                with self.assertRaises(FileExistsError):w.main()

    def test_existing_output_stops_without_launch(self):
        with temp_root() as root:
            output=root/'output';output.mkdir();state=root/'waiter'
            with patch.object(w,'STATE',state),patch.object(w,'OUTPUT',output),patch.object(w,'BASE',root), \
                 patch.object(w.runner,'file_lock',no_lock),patch.object(w.subprocess,'Popen') as pop,patch.object(sys,'argv',['waiter']):
                with self.assertRaises(FileExistsError):w.main()
                pop.assert_not_called()
            self.assertEqual(json.loads((state/'progress.json').read_text(encoding='utf8'))['stage'],'failed')

if __name__=='__main__':unittest.main()
