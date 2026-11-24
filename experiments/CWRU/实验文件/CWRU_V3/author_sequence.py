"""Continue an authorized four-cohort run sequentially, without retrying failures."""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import run_author_comparisons as runner
from run_experiment import now, write_json

PARENT = (runner.PROJECT / 'experiments/CWRU/实验结果/CWRU_V3').resolve()
RUNNER = Path(__file__).resolve().with_name('run_author_comparisons.py')
CURRENT_STAGES = {'preparing', 'training', 'verifying_weights', 'decisions_locked', 'testing', 'reporting'}


def read_state(path):
    with (path/'config.json').open(encoding='utf-8') as handle:
        return json.load(handle)


def validate_scope(state, method, sample):
    scope = state.get('scope') or {}
    if (state.get('version') != 'CWRU_V3_author_comparison_clean'
        or scope.get('method') != method or scope.get('sample') != sample
        or scope.get('protocol') != 'clean_only'):
        raise ValueError('结果目录的方法、样本范围或协议不匹配')
    expected_ratios = [0.1, 0.2, 0.5] if sample == 'small' else [1.0]
    if scope.get('ratios') != expected_ratios or scope.get('seeds') != list(range(1706, 1711)):
        raise ValueError('训练比例或种子与固定计划不符')
    n = 180 if sample == 'small' else 20
    if (state.get('planned_models'), state.get('planned_test_pairs'), state.get('planned_metric_rows')) != (n, n*3, n*3):
        raise ValueError('预定模型或测试数量不符')
    return n


def validate_complete(path, method, sample):
    state = read_state(path)
    n = validate_scope(state, method, sample)
    if (state.get('stage') != 'complete' or state.get('trained_models') != n
        or state.get('evaluated_pairs') != n*3 or state.get('metric_rows') != n*3
        or not state.get('decisions_locked') or not state.get('weights_verified')
        or not state.get('target_test_accessed') or not (path/'REPORT.md').is_file()):
        raise RuntimeError('前一组未完整完成或报告缺失，禁止继续')
    return state


def owner_root():
    lock = PARENT/'.load_comparison_process.lock'
    with lock.open('rb') as handle:
        handle.seek(1)
        owner = json.loads(handle.read().decode('utf-8'))
    return Path(owner['root']).resolve(), owner['pid']


def wait_for_current(path, progress, persist, poll_seconds):
    while True:
        state = read_state(path)
        validate_scope(state, 'DWCN_SDGFD', 'small')
        stage = state.get('stage')
        if stage == 'complete':
            validate_complete(path, 'DWCN_SDGFD', 'small')
            return
        if stage not in CURRENT_STAGES:
            raise RuntimeError(f'当前实验进入不可继续状态 {stage!r}；保留现场')
        try:
            with runner.exclusive_run_lock():
                # The process might have completed between reading state and acquiring the lock.
                refreshed = read_state(path)
                if refreshed.get('stage') == 'complete':
                    validate_complete(path, 'DWCN_SDGFD', 'small')
                    return
                raise ValueError('当前训练进程已退出，但结果未完成；禁止自动恢复')
        except RuntimeError:
            owner, pid = owner_root()
            if owner != path or pid != state.get('pid'):
                raise RuntimeError('运行锁所有者与监控实验不一致，禁止继续')
        progress.update(stage='waiting', current_stage=stage, trained_models=state.get('trained_models'),
                        active_model=state.get('active_model'), updated_at=now())
        persist()
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', type=Path, required=True)
    parser.add_argument('--tag', required=True, help='unique ASCII tag for new result names')
    parser.add_argument('--poll-seconds', type=int, default=20)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]{1,40}', args.tag) or not 1 <= args.poll_seconds <= 60:
        parser.error('tag 只能包含英文字母数字下划线；轮询间隔必须为1至60秒')
    watched = args.watch.resolve()
    if watched.parent != PARENT or not watched.name.startswith('load_comparison_author_small_DWCN_SDGFD_'):
        raise ValueError('只能监控指定 V3 DWCN 小样本独立结果目录')
    entries = [
        ('DWCN_SDGFD', 'full'),
        ('ClassBD_WDCNN', 'small'),
        ('ClassBD_WDCNN', 'full'),
    ]
    plan = [(m,s,PARENT/f'load_comparison_author_{s}_{m}_{args.tag}') for m,s in entries]
    sequence_dir = PARENT/f'author_sequence_{args.tag}'
    with runner.file_lock(PARENT/'.author_sequence.lock'):
        if sequence_dir.exists() or any(p.exists() for _,_,p in plan):
            raise FileExistsError('计划目录或进度目录已存在；禁止重复启动和自动续跑')
        validate_scope(read_state(watched), 'DWCN_SDGFD', 'small')
        sequence_dir.mkdir(exist_ok=False)
        progress = dict(stage='initializing', watched=str(watched), started_at=now(), pid=os.getpid(),
                        steps=[dict(method=m,sample=s,output=str(p),status='pending') for m,s,p in plan])
        def persist():
            write_json(sequence_dir/'progress.json',progress)
        persist()
        try:
            wait_for_current(watched, progress, persist, args.poll_seconds)
            for idx, (method, sample, out) in enumerate(plan):
                # Revalidate immediately before each launch and reject all existing output.
                preceding = watched if idx == 0 else plan[idx-1][2]
                prev_m, prev_s = ('DWCN_SDGFD','small') if idx == 0 else plan[idx-1][:2]
                validate_complete(preceding, prev_m, prev_s)
                if out.exists():
                    raise FileExistsError(f'新结果目录已存在，禁止覆盖: {out}')
                step=progress['steps'][idx]
                logfile=sequence_dir/f'{idx+1:02d}_{method}_{sample}.log'
                cmd=[sys.executable,'-X','utf8','-B',str(RUNNER),'run','--output',str(out),'--method',method,'--sample',sample]
                with logfile.open('x',encoding='utf-8',buffering=1) as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=str(runner.PROJECT))
                    step.update(status='running',pid=child.pid,started_at=now(),log=str(logfile))
                    progress.update(stage='running',current_step=idx+1,updated_at=now())
                    persist()
                    exit_code=child.wait()
                step.update(exit_code=exit_code,ended_at=now())
                if exit_code != 0:
                    step['status']='failed'
                    raise RuntimeError(f'{method}/{sample} 进程退出码 {exit_code}，后续组不启动')
                validate_complete(out,method,sample)
                step['status']='complete'
                progress['updated_at']=now()
                persist()
            progress.update(stage='complete',completed_at=now(),updated_at=now())
            persist()
        except BaseException as error:
            progress.update(stage='failed',error=str(error),error_type=type(error).__name__,updated_at=now())
            persist()
            raise


if __name__ == '__main__':
    main()
