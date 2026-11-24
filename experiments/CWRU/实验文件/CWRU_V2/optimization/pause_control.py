"""在已登记的系数模型全部完成后停止旧版运行进程，保留可恢复状态。"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import json
import time
import psutil

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'CWRU_V0/shared'))
from experiment_paths import resolve_result_path


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def expected_cases():
    return {(r,s,p,t) for r in (.1,.2) for s in (4101,4102,4103)
            for p in ('clean_only','awgn_mixed') for t in (1706,1707,1708)}


def boundary_complete(state, weight):
    jobs = [j for j in state['jobs'] if j['variant']=='data_aux_ce'
            and j['domain']=='window_random' and j['spec']['aux_weight']==weight]
    actual = {(j['ratio'],j['subset'],j['protocol'],j['seed']) for j in jobs}
    return actual == expected_cases()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--pid',required=True,type=int)
    parser.add_argument('--after-lambda',required=True,type=float,choices=[.03])
    args=parser.parse_args()
    root=args.output.resolve()
    parent=Path(__file__).resolve().parents[5]/'experiments/CWRU/实验结果/CWRU_V2'
    if root.parent!=parent.resolve() or not root.name.startswith('optimization_'):
        raise ValueError('只允许指定新优化实验目录')
    process=psutil.Process(args.pid)
    command=process.cmdline()
    if str(root) not in command or not any(Path(c).name=='run_optimization.py' for c in command):
        raise ValueError('PID与本实验训练入口不符')
    created=process.create_time()
    request_path=root/'pause_request.json'
    if request_path.exists():
        raise FileExistsError('已有暂停请求，禁止启动重复监控')
    request=dict(status='watching',requested_at=now(),after_lambda=args.after_lambda,
                 pid=args.pid,process_created=created,automatic_resume=False)
    atomic_json(request_path,request)
    print('PAUSE REQUEST ACTIVE: complete lambda=0.03, then stop before remaining coefficients',flush=True)
    while True:
        if not process.is_running() or process.create_time()!=created:
            request.update(status='process_exited_before_boundary',updated_at=now())
            atomic_json(request_path,request)
            return
        try:
            state=json.loads((root/'config.json').read_text(encoding='utf-8'))
        except (json.JSONDecodeError,PermissionError):
            time.sleep(.1)
            continue
        if state['stage']!='lambda':
            raise RuntimeError('系数阶段已发生异常或越过边界，停止自动控制并保留现场')
        if not boundary_complete(state,args.after_lambda):
            time.sleep(.1)
            continue
        # 已保存完成记录后才暂停进程，避免切断任何目标模型训练或保存。
        process.suspend()
        try:
            try:
                state=json.loads((root/'config.json').read_text(encoding='utf-8'))
            except (json.JSONDecodeError,PermissionError):
                process.resume()
                time.sleep(.1)
                continue
            if not boundary_complete(state,args.after_lambda):
                raise RuntimeError('暂停边界复核失败')
            registered={resolve_result_path(j['folder']).resolve() for j in state['jobs'] if j['origin']=='suite'}
            actual={p.resolve() for p in (root/'models').iterdir() if p.is_dir()}
            if actual!=registered:
                raise RuntimeError('出现未完成新模型目录，不能标记为可恢复暂停')
            for folder in registered:
                info=json.loads((folder/'parameters.json').read_text(encoding='utf-8'))
                if not info.get('completed'):
                    raise RuntimeError('存在未完成模型，不能标记为可恢复暂停')
            process.terminate()
            process.wait(timeout=30)
        except BaseException:
            if process.is_running():
                process.resume()
            raise
        state['stage']='paused'
        state['stages']['lambda']['status']='paused'
        state['paused_at']=now()
        state['pause_reason']='user_requested_after_lambda_0.03'
        state['updated_at']=now()
        state['pause_boundary_verified']=True
        atomic_json(root/'config.json',state)
        request.update(status='paused',paused_at=now(),trained_models=state['trained_models'],
                       completed_lambda_0_03_models=36)
        atomic_json(request_path,request)
        print('PAUSED: lambda=0.01 and 0.03 completed; no automatic resume',flush=True)
        return


if __name__=='__main__':
    main()
