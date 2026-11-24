"""Wait for the authorized comparison sequence, then summarize exactly once."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import time

import run_author_comparisons as runner
from run_experiment import now, write_json

BASE=(runner.PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve()
SEQUENCE=BASE/'author_sequence_20261104_chain'
STATE=BASE/'author_summary_waiter_20261104_chain'
OUTPUT=BASE/'author_comparison_summary_20261104_chain'
SUMMARY=Path(__file__).resolve().with_name('author_comparison_summary.py')
DWCN_SMALL=BASE/'load_comparison_author_small_DWCN_SDGFD_20261104_1940'
DWCN_FULL=BASE/'load_comparison_author_full_DWCN_SDGFD_20261104_chain'
CLASSBD_SMALL=BASE/'load_comparison_author_small_ClassBD_WDCNN_20261104_chain'
CLASSBD_FULL=BASE/'load_comparison_author_full_ClassBD_WDCNN_20261104_chain'


def read_sequence():
    with (SEQUENCE/'progress.json').open(encoding='utf-8') as f:return json.load(f)


def process_identity(pid):
    """Return (creation time, exit status) from Windows, without sending a signal."""
    if sys.platform!='win32':raise RuntimeError('仅支持 Windows 进程核验')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype=wintypes.BOOL
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
    kernel.GetProcessTimes.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:raise RuntimeError(f'无法核验进程 {pid}: Windows error {ctypes.get_last_error()}')
    try:
        code=wintypes.DWORD();creation=wintypes.FILETIME();exit_time=wintypes.FILETIME()
        kernel_time=wintypes.FILETIME();user_time=wintypes.FILETIME()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
            raise RuntimeError('无法读取顺序控制器进程状态')
        if not kernel.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit_time),
                                      ctypes.byref(kernel_time),ctypes.byref(user_time)):
            raise RuntimeError('无法读取顺序控制器启动时间')
        return (creation.dwHighDateTime<<32)|creation.dwLowDateTime,code.value
    finally:kernel.CloseHandle(handle)


def sequence_lock_held():
    try:
        with runner.file_lock(BASE/'.author_sequence.lock'):
            return False
    except RuntimeError:
        return True


def validate_sequence(seq):
    if Path(seq.get('watched','')).resolve()!=DWCN_SMALL:
        raise ValueError('监控对象与指定 DWCN 小样本目录不一致')
    expected=[('DWCN_SDGFD','full',DWCN_FULL),('ClassBD_WDCNN','small',CLASSBD_SMALL),
              ('ClassBD_WDCNN','full',CLASSBD_FULL)]
    steps=seq.get('steps') or []
    if len(steps)!=3:raise ValueError('顺序控制器计划不完整')
    for step,(m,s,p) in zip(steps,expected):
        if (step.get('method'),step.get('sample'),Path(step.get('output','')).resolve())!=(m,s,p):
            raise ValueError('顺序控制器计划与预定方法不一致')
    if not isinstance(seq.get('pid'),int) or seq['pid']<=0:raise ValueError('进程号无效')
    return steps


def wait(poll, state, persist):
    identity=None
    while True:
        seq=read_sequence();steps=validate_sequence(seq);stage=seq.get('stage')
        if stage=='complete':
            if not all(s.get('status')=='complete' and s.get('exit_code')==0 for s in steps):
                raise ValueError('顺序控制器显示完成但步骤状态不完整')
            return
        if stage not in {'initializing','waiting','running'}:
            raise RuntimeError(f'顺序控制器状态为 {stage!r}，停止汇总')
        try:
            if not sequence_lock_held():raise RuntimeError('顺序控制器运行锁已释放')
            created,code=process_identity(seq['pid'])
            if code!=259:raise RuntimeError('顺序控制器进程已退出')
            if identity is None:identity=created
            elif identity!=created:raise RuntimeError('顺序控制器进程身份已改变')
        except RuntimeError:
            # A just-completed coordinator may have exited after the first read.
            latest=read_sequence()
            if latest.get('stage')=='complete':
                validate_sequence(latest)
                if all(s.get('status')=='complete' and s.get('exit_code')==0 for s in latest['steps']):return
            raise
        state.update(stage='waiting',sequence_stage=stage,sequence_pid=seq['pid'],updated_at=now())
        persist();time.sleep(poll)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--poll-seconds',type=int,default=20)
    args=ap.parse_args()
    if not 1<=args.poll_seconds<=60:ap.error('轮询间隔必须为1至60秒')
    with runner.file_lock(BASE/'.author_summary_waiter.lock'):
        if STATE.exists():raise FileExistsError('等待器状态目录已存在；禁止重试')
        STATE.mkdir(exist_ok=False)
        state=dict(stage='initializing',started_at=now(),output=str(OUTPUT),sequence=str(SEQUENCE))
        def persist():write_json(STATE/'progress.json',state)
        persist()
        try:
            if OUTPUT.exists():raise FileExistsError('汇总目录已存在，禁止覆盖')
            wait(args.poll_seconds,state,persist)
            if OUTPUT.exists():raise FileExistsError('汇总目录已存在，禁止覆盖')
            cmd=[sys.executable,'-X','utf8','-B',str(SUMMARY),
                 '--dwcn-small',str(DWCN_SMALL),'--dwcn-full',str(DWCN_FULL),
                 '--classbd-small',str(CLASSBD_SMALL),'--classbd-full',str(CLASSBD_FULL),
                 '--output',str(OUTPUT)]
            with (STATE/'summary.log').open('x',encoding='utf-8',buffering=1) as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=str(runner.PROJECT))
                state.update(stage='summarizing',summary_pid=child.pid,updated_at=now());persist()
                code=child.wait()
            if code!=0:raise RuntimeError(f'汇总进程退出码 {code}；保留现场，不重试')
            if not (OUTPUT/'REPORT.md').is_file() or not (OUTPUT/'verification.json').is_file():
                raise RuntimeError('汇总进程退出但结果不完整；保留现场')
            state.update(stage='complete',exit_code=code,completed_at=now(),updated_at=now());persist()
        except BaseException as error:
            state.update(stage='failed',error_type=type(error).__name__,error=str(error),updated_at=now());persist()
            raise


if __name__=='__main__':main()
