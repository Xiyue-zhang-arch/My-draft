"""用户明确授权的中断恢复：保留m0312现场，复用311模型后按指定范围继续。"""
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import json
import os
import shutil
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import psutil

from context import R, PROJECT, BASE, HERE
from splits import read_fixed
from run_experiment import split_name, nested_subset, now, write_json
from run_window_experiment import SOURCE, source_state
from run_load_small_experiment import LoadSmallExperiment, small_cases, CASE_KEYS
import engine

from experiment_paths import resolve_result_path

REQUIRED_FILES = ['model.pt', 'parameters.json', 'training_spec.json', 'training.csv',
                  'selected_windows.csv', 'validation_metrics.csv', 'validation_curve.csv',
                  'validation_predictions.npz']
_CHECK_MODELS = {}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def assert_no_training_process(root, previous_pid):
    if psutil.pid_exists(previous_pid):
        raise ValueError('原PID仍存在，必须先人工核对身份，禁止重复启动')
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.info['pid'] == os.getpid() or 'python' not in (process.info['name'] or '').lower():
            continue
        command = process.info['cmdline']
        if command is None:
            raise ValueError('无法核对其他Python进程命令行，禁止重复启动')
        if any(str(root).replace('\\', '/').lower() == str(c).replace('\\', '/').lower() for c in command):
            raise ValueError('已有进程使用本轮结果目录')


def check_checkpoint(folder, job):
    """只读加载并检查权重结构，不前向、不重新评价已完成模型。"""
    saved = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
    if (saved.get('variant') != job['variant'] or saved.get('suite_spec') != job['spec']
            or saved.get('model_config') != R.U.CFG):
        raise ValueError('权重中的方法或训练设置与登记不一致')
    if not isinstance(saved.get('state_dict'), dict) or not all(
            isinstance(t, torch.Tensor) and torch.isfinite(t).all().item() for t in saved['state_dict'].values()):
        raise ValueError('模型权重缺失或包含非有限值')
    if job['variant'] not in _CHECK_MODELS:
        _CHECK_MODELS[job['variant']] = engine.SuiteModel(job['variant'], job['spec']).cpu()
    _CHECK_MODELS[job['variant']].load_state_dict(saved['state_dict'], strict=True)


def check_completed(root, state, count, abandoned):
    expected = list(small_cases())[:count]
    jobs = state['jobs']
    if len(jobs) != count or state['trained_models'] != count:
        raise ValueError('完整模型登记数不符')
    folders = set()
    for number, (job, case) in enumerate(zip(jobs, expected), 1):
        if job['job_id'] != f'm{number:04d}' or any(job[k] != case[k] for k in CASE_KEYS):
            raise ValueError('已完成模型不等于原计划的有序前缀')
        folder = resolve_result_path(job['folder']).resolve()
        if folder.parent != (root/'models').resolve():
            raise ValueError('已完成模型路径越界')
        folders.add(folder)
        if not job.get('completed') or not job.get('from_scratch') or job.get('max_epochs') != 200:
            raise ValueError('不能复用未完成或设置不符的模型')
        if job['aux_weight'] != (.3 if job['variant'] == 'data_aux_ce' else 0.):
            raise ValueError('模型系数与原锁定设置不符')
        expected_spec = engine.normalize_spec(dict(aux_weight=.3,
            physics_fit='matched_noise' if job['protocol'] == 'awgn_mixed' else 'clean'))
        if job['spec'] != expected_spec:
            raise ValueError('模型训练规则与V3固定设置不符')
        for name in REQUIRED_FILES:
            path = folder/name
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f'已登记模型文件不完整: {path}')
        info = read_json(folder/'parameters.json')
        for key in ['completed', 'variant', 'protocol', 'seed', 'subset', 'spec',
                    'validation_nll', 'epochs_trained', 'updates_trained']:
            if info[key] != job[key]:
                raise ValueError(f'模型配置与登记不一致: {job["job_id"]}/{key}')
        check_checkpoint(folder, job)
        with (folder/'training.csv').open(encoding='utf-8', newline='') as handle:
            history = list(csv.DictReader(handle))
        if len(history) != job['epochs_trained'] or [int(r['epoch']) for r in history] != list(range(1, len(history)+1)):
            raise ValueError('已完成模型训练记录不完整')
        fixed = read_fixed(root/'splits'/split_name(job['domain'], job['ratio'], job['subset']))
        pd.testing.assert_frame_equal(pd.read_csv(folder/'selected_windows.csv'),
            fixed[fixed.role.eq('train')].reset_index(drop=True), check_dtype=False)
    actual = {p.resolve() for p in (root/'models').iterdir() if p.is_dir()}
    if len(folders) != count or actual != folders | {abandoned.resolve()}:
        raise ValueError('存在其他未登记目录，禁止自动恢复')
    with (root/'models_registry.csv').open(encoding='utf-8', newline='') as handle:
        registry = list(csv.DictReader(handle))
    if len(registry) != count or [r['job_id'] for r in registry] != [j['job_id'] for j in jobs]:
        raise ValueError('模型登记CSV与配置不一致')
    if state.get('target_test_accessed') or state.get('decisions_locked') or (root/'evaluation').exists():
        raise ValueError('恢复及完整性核验前不能已访问测试或锁定全部模型')


class ContinueToSourceOne(LoadSmallExperiment):
    def __init__(self, root, stop_after_source=1):
        if stop_after_source not in (1, 3):
            raise ValueError('只允许补齐来源1或完成全部来源')
        self.stop_after_source = stop_after_source
        self.stop_models = 180*(stop_after_source+1)
        self.root = Path(root).resolve()
        if self.root.parent != (PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve() or not self.root.name.startswith('load_small_'):
            raise ValueError('仅允许原V3小样本结果目录')
        self.state = read_json(self.root/'config.json')
        self.jobs = self.state['jobs']
        if (self.state.get('version') != 'CWRU_V3_load_small' or self.state.get('stage') != 'load_transfer_small'
                or self.state.get('planned_models') != 720 or self.state.get('selected_aux_weight') != .3
                or self.state.get('trained_models') != 311):
            raise ValueError('本次授权只处理311完整模型之后的已知中断，不适用于其他失败或暂停')
        assert_no_training_process(self.root, self.state['pid'])
        _, weight = source_state(SOURCE)
        if weight != .3:
            raise ValueError('原锁定系数不一致')
        self.abandoned = self.root/'models/m0312'
        self.retry = self.root/'models/m0312_retry1'
        active = self.state.get('active_model', {})
        expected = list(small_cases())[311]
        if active.get('job_id') != 'm0312' or any(active.get(k) != v for k, v in expected.items()):
            raise ValueError('中断对象不等于预定第312个模型')
        check_completed(self.root, self.state, 311, self.abandoned)
        if not self.abandoned.is_dir() or (self.abandoned/'parameters.json').exists() or self.retry.exists():
            raise ValueError('中断现场或补训目录不符合条件，禁止覆盖')
        interrupted_spec = read_json(self.abandoned/'training_spec.json')
        if (interrupted_spec['protocol'] != expected['protocol'] or interrupted_spec['variant'] != expected['variant']
                or interrupted_spec['seed'] != expected['seed'] or interrupted_spec['subset'] != expected['subset']
                or interrupted_spec['spec'] != engine.normalize_spec(dict(aux_weight=.3, physics_fit='matched_noise'))):
            raise ValueError('中断模型设置与原计划不一致')
        self.interrupted_history = pd.read_csv(self.abandoned/'training.csv')
        if len(self.interrupted_history) != 53 or int(self.interrupted_history.epoch.iloc[-1]) != 53:
            raise ValueError('中断现场与已核对的53轮不符')
        if (self.root/'pause_request.json').exists() or (self.root/'continuation_request.json').exists():
            raise ValueError('已有暂停控制请求，须先核对，禁止重复恢复')

    def setup_existing(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        R.U.DEVICE = self.device
        R.U.configure()
        self.tables = {}
        for case in small_cases():
            name = split_name(case['domain'], case['ratio'], case['subset'])
            if name in self.tables:
                continue
            w = read_fixed(self.root/'splits'/name)
            pd.testing.assert_frame_equal(w, read_fixed(SOURCE/'splits'/name))
            full = read_fixed(BASE/'splits'/case['domain'])
            pd.testing.assert_frame_equal(w, nested_subset(full, case['ratio'], case['subset']))
            self.tables[name] = w
        _, original, raw = R.load_dataset(PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
        original['load_hp'] = original.record_id.str.rsplit('_', n=1).str[-1].astype(int)
        for w in self.tables.values():
            for key in ['window_index', 'record_id', 'label', 'load_hp', 'rpm', 'start_sample', 'end_sample']:
                np.testing.assert_array_equal(w[key], original[key])
        self.x = R.U.normalize(raw)

    def train_retry(self, case):
        spec = engine.normalize_spec(dict(aux_weight=.3, physics_fit='matched_noise'))
        w = self.tables[split_name(case['domain'], case['ratio'], case['subset'])]
        mask = w.role.isin(['train', 'validation']).to_numpy()
        self.state.update(stage='load_transfer_small', active_model=dict(job_id='m0312', **case,
            aux_weight=0., attempt=2, folder=str(self.retry)))
        self.persist()
        info = engine.train(self.x[mask], w.loc[mask].reset_index(drop=True), case['protocol'],
            case['variant'], case['seed'], case['subset'], self.retry, self.device, spec)
        job = dict(info, **{k:v for k,v in case.items() if k not in info}, job_id='m0312',
            stage='load_transfer_small', aux_weight=0., folder=str(self.retry), source_load_hp=1,
            attempt=2, interrupted_attempt_folder=str(self.abandoned))
        self.jobs.append(job)
        self.state.pop('active_model', None)
        self.persist()
        print('MODEL COMPLETE m0312; explicit from-scratch retry; original attempt preserved', flush=True)

    def run(self):
        request = dict(status='preparing', requested_at=now(), controller='built_in_training_boundary',
            training_pid=os.getpid(), after_source_load_hp=self.stop_after_source,
            expected_completed_models=self.stop_models, automatic_resume=False,
            test_after_all_models=self.stop_models == 720, user_authorized_interrupted_retry='m0312',
            preserved_attempt_folder=str(self.abandoned), retry_folder=str(self.retry))
        request_path = self.root/('pause_request.json' if self.stop_models == 360 else 'continuation_request.json')
        with request_path.open('x', encoding='utf-8') as handle:
            json.dump(request, handle, ensure_ascii=False, indent=2)
        continuation_protocol = self.root/'CONTINUATION_PROTOCOL.md'
        with continuation_protocol.open('xb') as target, (HERE/'LOAD_SMALL_RESUME_PROTOCOL.md').open('rb') as source:
            shutil.copyfileobj(source, target)
        self.state.setdefault('interruption_history', []).append(dict(observed_at=now(),
            previous_pid=self.state['pid'], previous_stage=self.state['stage'],
            previous_active_model=self.state.get('active_model'), completed_models=311,
            interrupted_epoch=53, interrupted_updates=int(self.interrupted_history.updates.iloc[-1]),
            reason='process_exited_without_completion; cause_unknown',
            optimizer_state_saved=False, preserved_folder=str(self.abandoned),
            user_authorized_action=('retrain_m0312_from_scratch_in_new_folder; stop_after_source_1'
                if self.stop_models == 360 else 'retrain_m0312_from_scratch_in_new_folder; finish_all_720_then_test')))
        self.state.update(pid=os.getpid(), stage='resuming_after_interruption', resume_started_at=now(),
            pause_after_source_load_hp=1 if self.stop_models == 360 else None,
            automatic_resume=False, resume_stop_models=self.stop_models,
            preserved_incomplete_folders=[str(self.abandoned)], interrupted_retrain_authorized=True)
        self.persist()
        self.setup_existing()
        request.update(status='watching', updated_at=now())
        write_json(request_path, request)
        cases = list(small_cases())
        self.train_retry(cases[311])
        for case in cases[312:self.stop_models]:
            self.ensure(case, .3, 'load_transfer_small')
        check_completed(self.root, self.state, self.stop_models, self.abandoned)
        self.state.pop('active_model', None)
        if self.stop_models == 360:
            self.state.update(stage='paused', paused_at=now(), pause_boundary_verified=True,
                pause_reason='user_requested_after_source_1_hp', decisions_locked=False,
                target_test_accessed=False, automatic_resume=False)
            self.persist()
            request.update(status='paused', paused_at=self.state['paused_at'], boundary_verified=True,
                trained_models=360, completed_source_0_models=180, completed_source_1_models=180)
            write_json(request_path, request)
            print('PAUSED: 360 complete models; source 2/3 and testing not started.', flush=True)
        else:
            self.verify_cohort()
            self.state.update(stage='decisions_locked', decisions_locked=True, decisions_locked_at=now())
            self.persist()
            request.update(status='testing', trained_models=720, decisions_locked_at=self.state['decisions_locked_at'])
            write_json(request_path, request)
            self.evaluate()
            request.update(status='complete', completed_at=self.state['completed_at'],
                evaluated_pairs=self.state['evaluated_pairs'], metric_rows=self.state['metric_rows'])
            write_json(request_path, request)

    def evaluate(self):
        if self.stop_models != 720:
            raise RuntimeError('仅补齐来源1的请求禁止测试')
        return super().evaluate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'resume-to-source1', 'resume-all'])
    parser.add_argument('--stop-after-source', choices=[1, 3], type=int, default=1,
                        help='仅plan使用；正式执行范围由resume-to-source1或resume-all确定')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    stop = args.stop_after_source if args.command == 'plan' else (3 if args.command == 'resume-all' else 1)
    experiment = ContinueToSourceOne(args.output, stop_after_source=stop)
    if args.command == 'plan':
        print(f'Validated 311 completed models; preserve incomplete m0312; {experiment.stop_models-311} new training calls; '
              + ('stop at 360; no testing.' if stop == 1 else 'finish 720 models, then unified testing and report.'))
        return
    log_path = experiment.root/'training_resume.log'
    with log_path.open('x', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                experiment.run()
            except BaseException as error:
                experiment.state.update(stage='failed', error=str(error), error_type=type(error).__name__, automatic_resume=False)
                experiment.persist()
                request_path = experiment.root/('pause_request.json' if stop == 1 else 'continuation_request.json')
                if request_path.exists():
                    request = read_json(request_path)
                    request.update(status='failed', error=str(error), updated_at=now())
                    write_json(request_path, request)
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
