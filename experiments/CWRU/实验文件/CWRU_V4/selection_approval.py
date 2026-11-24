"""核验已运行 V4 的选参停止边界，登记待审批状态。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True

from run_v4 import (BANDS, CASE_KEYS, COUNTS, RESULTS, STAGES, now, np, pd,
                    select_lambda, selection_cases, split_name, torch, transfer_cases, write_json)
import psutil

from experiment_paths import resolve_result_path

BARRIER_TEXT = 'CWRU_V4 pause after 30 selection models. Control file. Approval required for transfer training.\n'


def read_json(path):
    for attempt in range(8):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (PermissionError, json.JSONDecodeError):
            if attempt == 7:
                raise
            time.sleep(.25 * (attempt + 1))


def verify_boundary(root, state, barrier, pid):
    first_transfer = dict(job_id='m0031', aux_weight=0., **next(transfer_cases(True)))
    if (state['stage'] != 'failed' or state['pid'] != pid
            or state.get('error_type') != 'FileExistsError'
            or resolve_result_path(state.get('error') or '').resolve() != barrier.resolve()
            or state.get('active_model') != first_transfer):
        raise ValueError('原进程未停在预定的首次迁移训练入口')
    if (barrier.read_text(encoding='utf-8') != BARRIER_TEXT or (root / 'evaluation').exists()
            or state['decisions_locked'] or state['selection_test_accessed'] or state['target_test_accessed']):
        raise ValueError('训练屏障或测试状态与选参审批协议不符')
    jobs = state['jobs']
    registry = pd.read_csv(root / 'models_registry.csv')
    ids = [f'm{i:04d}' for i in range(1, COUNTS[0] + 1)]
    if (state['trained_models'] != COUNTS[0] or [j['job_id'] for j in jobs] != ids
            or list(registry.job_id) != ids
            or {p.name for p in (root / 'models').iterdir()} != {*ids, barrier.name}):
        raise ValueError('选参模型登记或目录数量异常')
    split = pd.read_csv(root / 'splits' / split_name('window_random', 1., 0) / 'windows.csv')
    validation = split[split.role.eq('validation')]
    for job, (case, weight) in zip(jobs, selection_cases(), strict=True):
        folder = root / 'models' / job['job_id']
        if (not folder.is_dir() or resolve_result_path(job['folder']).resolve() != folder
                or job['stage'] != STAGES[0] or job['aux_weight'] != weight
                or any(job[k] != case[k] for k in CASE_KEYS) or not job['completed']):
            raise ValueError(f'选参配置异常 {job["job_id"]}')
        if not read_json(folder / 'parameters.json')['completed']:
            raise ValueError(f'模型未完成 {job["job_id"]}')
        saved = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
        if (saved['variant'] != case['variant'] or saved['model_config']['bands_hz'] != BANDS
                or (case['variant'] == 'data_aux_ce' and saved['suite_spec']['aux_weight'] != weight)):
            raise ValueError(f'权重配置异常 {job["job_id"]}')
        with np.load(folder / 'validation_predictions.npz', allow_pickle=False) as result:
            np.testing.assert_array_equal(result['labels'], validation.label)
            np.testing.assert_array_equal(result['window_indices'], validation.window_index)
        val = pd.read_csv(folder / 'validation_metrics.csv')
        if (len(val) != 1 or val.iloc[0].condition != 'clean'
                or abs(val.iloc[0].validation_nll - job['validation_nll']) > 2e-6):
            raise ValueError(f'保存验证CE与选模登记不一致 {job["job_id"]}')
    selection = read_json(root / 'selection.json')
    winner, rows = select_lambda(jobs)
    if (not state['lambda_locked'] or winner != state['selected_aux_weight']
            or winner != selection['selected_aux_weight'] or rows != selection['candidates']
            or selection['test_accessed'] is not False
            or not (root / 'selection' / 'REPORT.md').is_file()):
        raise ValueError('选参结果与复算不一致')
    return selection


def watch(root, pid):
    root = root.resolve()
    if root.parent != RESULTS.resolve() or not root.name.startswith('clean_'):
        raise ValueError('仅允许本项目V4正式结果直接子目录')
    barrier = root / 'models' / 'm0031'
    cfg_path = root / 'config.json'
    request_path = root / 'selection_approval.json'
    initial = read_json(cfg_path)
    process = psutil.Process(pid)
    cmd = process.cmdline()
    if (initial['pid'] != pid or initial['stage'] != STAGES[0] or initial['trained_models'] >= COUNTS[0]
            or not any(Path(arg).name == 'run_v4.py' for arg in cmd)
            or '--output' not in cmd or Path(cmd[cmd.index('--output') + 1]).resolve() != root
            or barrier.read_text(encoding='utf-8') != BARRIER_TEXT):
        raise ValueError('训练进程、阶段或屏障无法核验')
    request = dict(status='waiting_for_selection', approval_granted=False, requested_at=now(),
                   controller_pid=os.getpid(), training_pid=pid, expected_selection_models=COUNTS[0],
                   automatic_resume=False, barrier=str(barrier))
    with request_path.open('x', encoding='utf-8') as stream:
        json.dump(request, stream, ensure_ascii=False, indent=2)
    print(f'WAITING FOR SELECTION: {root}', flush=True)
    try:
        process.wait()
        if (RESULTS / '.clean_run.lock').exists():
            raise ValueError('原进程退出后运行锁仍存在')
        state = read_json(cfg_path)
        selection = verify_boundary(root, state, barrier, pid)
        state['approval_stop_event'] = dict(
            stage=state['stage'], error=state.pop('error'), error_type=state.pop('error_type'),
            blocked_model=state.pop('active_model'), original_training_pid=pid)
        state.update(stage='awaiting_selection_approval', approval_required=True, approval_granted=False,
                     pause_boundary_verified=True, automatic_resume=False, paused_at=now(), updated_at=now())
        write_json(cfg_path, state)
        request.update(status='awaiting_approval', verified_at=now(), selection_models=COUNTS[0],
                       transfer_models=0, selected_aux_weight=selection['selected_aux_weight'],
                       candidates=selection['candidates'], report=str(root / 'selection' / 'REPORT.md'))
        write_json(request_path, request)
        print('SELECTION VERIFIED; AWAITING USER APPROVAL', flush=True)
    except BaseException as error:
        request.update(status='failed', failed_at=now(), error=str(error), error_type=type(error).__name__)
        write_json(request_path, request)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pid', required=True, type=int)
    args = parser.parse_args()
    watch(args.output, args.pid)
