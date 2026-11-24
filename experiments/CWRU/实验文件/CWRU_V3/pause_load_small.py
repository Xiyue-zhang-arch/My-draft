"""完成来源0和1的360模型后暂停。用文件屏障阻止2HP训练，等待原进程自然退出。"""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import json
import os
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone, timedelta
from pathlib import Path

import psutil


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CWRU_V0/shared'))
from experiment_paths import resolve_result_path

PROJECT = Path(__file__).resolve().parents[4]
CASE_KEYS = ['domain', 'ratio', 'subset', 'protocol', 'variant', 'seed']
BARRIER_TEXT = 'V3 user-requested pause after source 1 HP; 360 complete models. Control file, not a model.\n'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix+'.pause_controller_pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(pending, path)


def expected_cases():
    return [dict(domain=f'load_source_{source}', ratio=ratio, subset=subset,
                 protocol=protocol, variant=variant, seed=seed)
            for source in (0, 1) for ratio in (.1, .2, .5) for subset in (4101, 4102, 4103)
            for protocol in ('clean_only', 'awgn_mixed')
            for variant in ('baseline_data', 'data_aux_ce') for seed in range(1706, 1711)]


def check_cohort(root, state):
    """只读检查完整360模型，拒绝任意未完成模型或已经测试的输出。"""
    root = Path(root).resolve()
    if (state.get('version') != 'CWRU_V3_load_small' or state.get('target_test_accessed')
            or state.get('decisions_locked') or state.get('selected_aux_weight') != .3):
        raise ValueError('仅允许未测试、固定lambda0.3的V3跨负载小样本边界')
    jobs = state['jobs']
    expected = expected_cases()
    if len(jobs) != 360 or state.get('trained_models') != 360:
        raise ValueError('暂停边界必须累计完成360模型')
    registered = set()
    for number, (job, case) in enumerate(zip(jobs, expected), 1):
        if any(job[k] != case[k] for k in CASE_KEYS) or job['job_id'] != f'm{number:04d}':
            raise ValueError('模型登记必须是原计划的完整来源0和1前缀')
        weight = .3 if job['variant'] == 'data_aux_ce' else 0.
        if (not job.get('completed') or not job.get('from_scratch') or job.get('max_epochs') != 200
                or job['aux_weight'] != weight or job['source_load_hp'] != int(job['domain'][-1])):
            raise ValueError('完成模型的设置或完成标记不符')
        folder = resolve_result_path(job['folder']).resolve()
        if folder.parent != (root/'models').resolve() or folder.name != job['job_id']:
            raise ValueError('模型路径必须位于当前输出且与编号一致')
        registered.add(folder)
        for name in ['model.pt', 'parameters.json', 'training.csv', 'training_spec.json',
                     'selected_windows.csv', 'validation_metrics.csv', 'validation_curve.csv',
                     'validation_predictions.npz']:
            path = folder/name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'模型文件未完整保存: {path}')
        info = read_json(folder/'parameters.json')
        if (not info.get('completed') or info['seed'] != job['seed']
                or info['validation_nll'] != job['validation_nll']
                or info['epochs_trained'] != job['epochs_trained']):
            raise ValueError('模型参数记录与登记不一致')
        with (folder/'training.csv').open(encoding='utf-8', newline='') as handle:
            history = list(csv.DictReader(handle))
        if len(history) != job['epochs_trained'] or [int(r['epoch']) for r in history] != list(range(1, len(history)+1)):
            raise ValueError('逐轮训练记录未完整落盘')
    actual = {p.resolve() for p in (root/'models').iterdir() if p.is_dir()}
    if actual != registered:
        raise ValueError('存在额外或未完成模型目录，不能标记边界暂停')
    barrier = root/'models/m0361'
    if not barrier.is_file() or barrier.read_text(encoding='utf-8') != BARRIER_TEXT:
        raise ValueError('第361模型阻挡文件缺失或内容不符')
    with (root/'models_registry.csv').open(encoding='utf-8', newline='') as handle:
        registry = list(csv.DictReader(handle))
    if len(registry) != 360 or [r['job_id'] for r in registry] != [j['job_id'] for j in jobs]:
        raise ValueError('models_registry.csv未完整落盘')
    if (root/'evaluation').exists():
        raise ValueError('暂停前不得创建测试目录')
    return jobs


def barrier_stop(state, barrier):
    return (state.get('stage') == 'failed' and state.get('error_type') == 'FileExistsError'
            and state.get('error') == str(barrier)
            and state.get('active_model', {}).get('job_id') == 'm0361'
            and state.get('active_model', {}).get('domain') == 'load_source_2')


def watch(root, pid):
    root = Path(root).resolve()
    if root.parent != (PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve() or not root.name.startswith('load_small_'):
        raise ValueError('只允许当前V3跨负载小样本直接结果目录')
    request_path = root/'pause_request.json'
    if request_path.exists():
        raise FileExistsError('已有暂停请求，禁止重复安装控制器')
    process = psutil.Process(pid)
    command = process.cmdline()
    if (not any(Path(c).name == 'run_load_small_experiment.py' for c in command)
            or not any(Path(c).resolve() == root for c in command if c and not c.startswith('-'))):
        raise ValueError('PID的命令行与指定实验入口或结果路径不一致')
    created = process.create_time()
    state = read_json(root/'config.json')
    if (state.get('pid') != pid or state.get('stage') != 'load_transfer_small'
            or state.get('version') != 'CWRU_V3_load_small' or state.get('target_test_accessed')
            or state.get('decisions_locked') or len(state['jobs']) >= 360):
        raise ValueError('必须在来源1完成之前、训练正常运行时安装暂停控制')
    barrier = root/'models/m0361'
    with barrier.open('x', encoding='utf-8') as handle:
        handle.write(BARRIER_TEXT)
    request = dict(status='watching', requested_at=now(), controller_pid=os.getpid(), training_pid=pid,
                   process_created=created, after_source_load_hp=1, expected_completed_models=360,
                   automatic_resume=False, test_before_pause=False, training_barrier=str(barrier),
                   stop_method='existing-path guard before model initialization; wait for natural process exit')
    atomic_json(request_path, request)
    print('PAUSE REQUEST ACTIVE: finish sources 0 and 1, stop before source 2; no testing.', flush=True)
    try:
        while True:
            try:
                state = read_json(root/'config.json')
            except (json.JSONDecodeError, PermissionError):
                time.sleep(.2)
                continue
            alive = process.is_running()
            if alive and process.create_time() != created:
                raise RuntimeError('训练PID已复用，禁止操作其他进程')
            expected_stop = barrier_stop(state, barrier)
            if state.get('stage') == 'failed' and not expected_stop:
                raise RuntimeError('训练出现其他失败，保留现场，不转为正常暂停')
            if len(state['jobs']) > 360 or state.get('target_test_accessed'):
                raise RuntimeError('已越过预定暂停边界，停止控制并保留现场')
            if not alive:
                if not expected_stop:
                    raise RuntimeError('训练未通过已知屏障正常停止，不能标记可恢复暂停')
                break
            # 不杀死可能正在写登记CSV的进程。屏障在任何第361模型训练动作之前生效。
            time.sleep(.2 if len(state['jobs']) >= 359 else 1.)
        state = read_json(root/'config.json')
        check_cohort(root, state)
        state.setdefault('pause_control_history', []).append(dict(
            stopped_at=now(), reason='user_requested_after_source_1_hp',
            stop_stage=state['stage'], stop_error_type=state['error_type'], stop_error=state['error'],
            blocked_model=state.get('active_model'), training_pid=pid, process_created=created))
        for key in ('active_model', 'error', 'error_type'):
            state.pop(key, None)
        state.update(stage='paused', paused_at=now(), updated_at=now(), automatic_resume=False,
                     pause_reason='user_requested_after_source_1_hp', pause_after_source_load_hp=1,
                     pause_boundary_verified=True, decisions_locked=False)
        atomic_json(root/'config.json', state)
        request.update(status='paused', paused_at=state['paused_at'], boundary_verified=True,
                       trained_models=360, completed_source_0_models=180, completed_source_1_models=180)
        atomic_json(request_path, request)
        print('PAUSED: 360 models complete; no source 2/3 training or testing; explicit resume required.', flush=True)
        return request
    except BaseException as error:
        request.update(status='failed', error=str(error), error_type=type(error).__name__, updated_at=now())
        atomic_json(request_path, request)
        # 仍在运行时不与训练器竞争写config；屏障保留以阻止越过来源1。
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.parent != (PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve() or not root.name.startswith('load_small_'):
        parser.error('只允许experiments/CWRU/实验结果/CWRU_V3/load_small_*目录')
    with (root/'pause_control.log').open('x', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                watch(root, args.pid)
            except BaseException:
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
