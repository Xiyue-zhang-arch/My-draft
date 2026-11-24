"""V3跨负载四方法对比：仅小样本、纯干净。逐方法手动运行；默认只显示计划。"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import os
import shutil
import traceback
import uuid
from pathlib import Path
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# 与 V3 一致，先导入 torch，避免 Windows DLL 加载顺序问题。
import torch
import numpy as np
import pandas as pd
from context import R, HERE, PROJECT, BASE
from splits import read_fixed
from run_experiment import split_name, nested_subset, now, write_json, SEEDS, SUBSETS, RATIOS
from run_window_experiment import SOURCE
from metrics import scores

from experiment_paths import resolve_result_path

METHODS = ['WDCNN', 'DRSN_CW', 'PG_RDGFN_adapted', 'AW_Hybrid_adapted']
CASE_KEYS = ['domain', 'ratio', 'subset', 'protocol', 'variant', 'seed']
PROTOCOL = 'clean_only'
SAMPLE_CHOICES = ('small', 'full')
TRAIN_SMALL = {0:[9,16,37], 1:[17,33,78], 2:[17,34,80], 3:[17,34,80]}
TRAIN_FULL = [73,156,160,160]
VALIDATION = [19,39,40,40]
TARGET = [92,195,200,200]


def scope_cases(method, sample, ratios):
    """单个方法的跨负载纯干净实验：small 用三嵌套子集，full 用该来源完整训练池。"""
    if method not in METHODS:
        raise ValueError(f'未知对比方法: {method}')
    if sample not in SAMPLE_CHOICES:
        raise ValueError(f'未知样本范围: {sample}')
    ratios = [1.0] if sample == 'full' else list(ratios)
    for source in range(4):
        for ratio in ratios:
            if ratio not in RATIOS and abs(ratio-1.0) > 1e-9:
                raise ValueError(f'训练比例不合法: {ratio}')
            for subset in ([0] if ratio == 1. else SUBSETS):
                for seed in SEEDS:
                    yield dict(domain=f'load_source_{source}', ratio=ratio, subset=subset,
                               protocol=PROTOCOL, variant=method, seed=seed)


def plan_counts(sample, ratios):
    """模型数、模型—目标组合数、逐条件指标行数；纯干净每个组合仅 1 行。"""
    ratios = [1.0] if sample == 'full' else list(ratios)
    subsets = 1 if sample == 'full' else len(SUBSETS)
    models = 4*len(ratios)*subsets*len(SEEDS)
    pairs = models*3
    return models, pairs, pairs


def parse_ratio(text):
    """接受 all、0.1、10 或 10%。"""
    if text is None or str(text).strip().lower() in ('', 'all'):
        return list(RATIOS)
    value = str(text).strip().rstrip('%')
    try:
        number = float(value)
    except ValueError as error:
        raise SystemExit(f'比例必须为 all、0.1、0.2 或 0.5（也可写 10、20、50）: {text}') from error
    if number > 1:
        number /= 100
    match = [r for r in RATIOS if abs(r-number) < 1e-9]
    if not match:
        raise SystemExit(f'比例必须为 all、0.1、0.2 或 0.5（也可写 10、20、50）: {text}')
    return match


def case_id(case):
    return tuple(case[k] for k in CASE_KEYS)


def prepare_splits(root):
    """逐行读取和核对原 V3 40 个负载划分，目标只能用作测试。"""
    tables = {}
    for source in range(4):
        domain = f'load_source_{source}'
        full = read_fixed(SOURCE/'splits'/split_name(domain, 1., 0))
        pd.testing.assert_frame_equal(full, read_fixed(BASE/'splits'/domain))
        for ratio in [1., *RATIOS]:
            for subset in ([0] if ratio == 1. else SUBSETS):
                name = split_name(domain, ratio, subset)
                w = read_fixed(SOURCE/'splits'/name)
                pd.testing.assert_frame_equal(w, full if ratio == 1. else nested_subset(full, ratio, subset))
                expected = TRAIN_FULL[source] if ratio == 1. else TRAIN_SMALL[source][RATIOS.index(ratio)]
                if len(w) != 687 or w.record_id.nunique() != 40 or not w.window_index.is_unique:
                    raise ValueError(f'固定划分记录或窗口不符: {name}')
                if int(w.role.eq('train').sum()) != expected or int(w.role.eq('validation').sum()) != VALIDATION[source]:
                    raise ValueError(f'训练或验证窗口数不符: {name}')
                if set(w.loc[w.role.isin(['train','validation']), 'load_hp']) != {source}:
                    raise ValueError(f'目标负载进入训练或验证: {name}')
                if any(int(w.load_hp.eq(t).sum()) != TARGET[t] for t in range(4)):
                    raise ValueError(f'目标窗口数不符: {name}')
                shutil.copytree(SOURCE/'splits'/name, root/'splits'/name)
                tables[name] = w
    if len(tables) != 40:
        raise ValueError('固定来源划分必须是 40 个')
    for filename in ['dataset_support.csv', 'dataset_file_usage.csv']:
        data = pd.read_csv(SOURCE/filename)
        data = data[data.domain.str.startswith('load_source_')].copy()
        if data.empty:
            raise ValueError(f'原用量表为空: {filename}')
        data.to_csv(root/filename, index=False, mode='x')
    return tables


class ComparisonRun:
    def __init__(self, root, method, sample, ratios):
        self.root = Path(root).resolve()
        parent = (PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve()
        if self.root.parent != parent or not self.root.name.startswith('load_comparison_'):
            raise ValueError('只能使用 experiments/CWRU/实验结果/CWRU_V3 下全新 load_comparison_* 直接子目录')
        if method not in METHODS:
            raise ValueError(f'未知对比方法: {method}')
        if sample not in SAMPLE_CHOICES:
            raise ValueError(f'未知样本范围: {sample}')
        ratios = [1.0] if sample == 'full' else list(ratios)
        if not ratios or any(r not in RATIOS and abs(r-1.0) > 1e-9 for r in ratios):
            raise ValueError('训练比例不合法')
        if self.root.exists():
            raise FileExistsError('结果目录已存在；禁止覆盖、重训、恢复或自动重试')
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root/'models').mkdir()
        shutil.copyfile(HERE/'COMPARISON_LOAD_PROTOCOL.md', self.root/'PROTOCOL.md')
        self.jobs = []
        self._cohort_audited = False
        models, pairs, rows = plan_counts(sample, ratios)
        self.state = dict(version='CWRU_V3_load_comparison_clean', stage='preparing', started_at=now(), pid=os.getpid(),
            scope=dict(method=method, sample=sample, ratios=ratios, protocol=PROTOCOL, seeds=SEEDS,
                       subsets=([0] if sample == 'full' else SUBSETS),
                       sources=[f'load_source_{i}' for i in range(4)]),
            planned_models=models, planned_test_pairs=pairs, planned_metric_rows=rows,
            trained_models=0, target_test_accessed=False, historical_test_exposure=True,
            decisions_locked=False, automatic_resume=False, fixed_split_source=str(SOURCE/'splits'),
            seeds=SEEDS, ratios=ratios, subsets=([0] if sample == 'full' else SUBSETS), methods=[method], jobs=self.jobs)
        self.persist()

    def persist(self):
        self.state.update(updated_at=now(), trained_models=len(self.jobs))
        write_json(self.root/'config.json', self.state)
        if self.jobs:
            pd.DataFrame(self.jobs).to_csv(self.root/'models_registry.csv', index=False)

    def setup(self, resume=False):
        if not torch.cuda.is_available():
            raise RuntimeError('未检测到可用 CUDA，已停止；请检查 pytorch_GPU 环境及显卡驱动')
        self.device = torch.device('cuda')
        R.U.DEVICE = self.device
        R.U.configure()
        scope = self.state['scope']
        names = {split_name(c['domain'],c['ratio'],c['subset'])
                 for c in scope_cases(scope['method'], scope.get('sample','small'), scope['ratios'])}
        if resume:
            self.tables = {}
            for name in sorted(names):
                saved = read_fixed(self.root/'splits'/name)
                pd.testing.assert_frame_equal(saved, read_fixed(SOURCE/'splits'/name))
                self.tables[name] = saved
            if set(self.tables) != names:
                raise ValueError('保存的固定负载划分数量不正确')
        else:
            self.tables = prepare_splits(self.root)
        records, original, raw = R.load_dataset(PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
        if len(original) != 687 or original.record_id.nunique() != 40:
            raise ValueError('数据必须包含 40 条记录和 687 个完整窗口')
        original['load_hp'] = original.record_id.str.rsplit('_', n=1).str[-1].astype(int)
        for w in self.tables.values():
            for key in ['window_index','record_id','label','load_hp','rpm','start_sample','end_sample']:
                np.testing.assert_array_equal(w[key], original[key])
        if not resume:
            records.to_csv(self.root/'records.csv', index=False, mode='x')
        self.x = R.U.normalize(raw)
        self.state.update(hardware=torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU',
                          precision='comparison models FP32; V3 reference encoder BF16 on CUDA')
        if not resume:
            self.persist()

    @classmethod
    def open_paused(cls, root):
        root = Path(root).resolve()
        if root.parent != (PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve() or not root.name.startswith('load_comparison_'):
            raise ValueError('恢复目录不合法')
        state = json.loads((root/'config.json').read_text(encoding='utf-8'))
        if not state.get('scope'):
            raise ValueError('该目录不是本入口生成的小样本纯干净对比实验，不能恢复')
        if (state.get('stage') != 'paused' or not state.get('pause_boundary_verified')
            or state.get('active_model') or state.get('target_test_accessed') or state.get('decisions_locked')
            or (root/'evaluation').exists()):
            raise ValueError('仅允许恢复已核验模型边界暂停且尚未测试的实验；异常中断不可恢复')
        jobs = state.get('jobs', [])
        if state.get('pause_boundary_models') != len(jobs) or state.get('trained_models') != len(jobs):
            raise ValueError('暂停登记模型数不一致')
        plan = list(scope_cases(state['scope']['method'], state['scope'].get('sample','small'), state['scope']['ratios']))
        if len(jobs) > len(plan) or len(plan) != state.get('planned_models'):
            raise ValueError('恢复计划长度不一致')
        expected_dirs = {f'm{i:04d}' for i in range(1,len(jobs)+1)}
        actual_dirs = {p.name for p in (root/'models').iterdir()}
        if expected_dirs != actual_dirs:
            raise ValueError('存在未登记模型目录或缺失已完成模型')
        for i, job in enumerate(jobs):
            if (case_id(job) != case_id(plan[i]) or job.get('job_id') != f'm{i+1:04d}'
                or resolve_result_path(job['folder']).resolve() != (root/'models'/job['job_id']).resolve()
                or not job.get('completed')):
                raise ValueError('已完成模型不是原计划的完整前缀')
        if pending_pause_requests(root, state):
            raise ValueError('存在未处理的暂停请求')
        self = object.__new__(cls)
        self.root, self.jobs, self.state, self._cohort_audited = root, jobs, state, False
        self.setup(resume=True)
        for index, job in enumerate(jobs, 1):
            self.verify_checkpoint(job)
            if index == 1 or index % 25 == 0 or index == len(jobs):
                print(f'RESUME VERIFIED {index}/{len(jobs)}', flush=True)
        return self

    @classmethod
    def open_for_recovery(cls, root):
        """仅续测因文件占用中断的测试：权重已核验、已完成模型是原计划前缀。"""
        root = Path(root).resolve()
        if root.parent != (PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve() or not root.name.startswith('load_comparison_'):
            raise ValueError('续测目录不合法')
        state = json.loads((root/'config.json').read_text(encoding='utf-8'))
        scope = state.get('scope')
        if not scope:
            raise ValueError('该目录不是本入口生成的小样本纯干净对比实验，不能续测')
        if state.get('stage') != 'failed' or state.get('error_type') != 'PermissionError':
            raise ValueError('仅允许续测因文件占用中断的测试阶段；其他失败必须保留现场')
        if not state.get('decisions_locked') or not state.get('weights_verified'):
            raise ValueError('权重尚未全部核验，不能续测')
        if not state.get('target_test_accessed'):
            raise ValueError('该目录尚未进入目标测试，不能用续测入口')
        if (root/'report').exists() or (root/'REPORT.md').exists():
            raise ValueError('报告已存在，不能续测')
        ev = root/'evaluation'
        if not ev.is_dir() or not (ev/'evaluations.csv').is_file() or not (ev/'metrics.csv').is_file():
            raise ValueError('缺少测试记录，不能续测')
        jobs = state.get('jobs', [])
        total = state.get('planned_models')
        if len(jobs) != total or state.get('trained_models') != total:
            raise ValueError('训练登记不完整，不能续测')
        for i, job in enumerate(jobs, 1):
            folder = resolve_result_path(job['folder']).resolve()
            if (folder.parent != (root/'models').resolve() or job.get('job_id') != f'm{i:04d}'
                or not job.get('completed') or not (folder/'model.pt').is_file()
                or not (folder/'completed.json').is_file()):
                raise ValueError(f'模型登记或权重不完整: {job.get("job_id")}')
        e = pd.read_csv(ev/'evaluations.csv')
        m = pd.read_csv(ev/'metrics.csv')
        done = list(dict.fromkeys(e.job_id))
        if len(e) != 3*len(done) or len(m) != len(e):
            raise ValueError('既有测试记录行数不一致')
        if done != [j['job_id'] for j in jobs[:len(done)]]:
            raise ValueError('已测试模型不是原计划的完整前缀')
        if not done or len(done) >= total:
            raise ValueError('没有可续测的模型或模型已全部测试')
        if (state.get('evaluated_pairs'), state.get('metric_rows')) != (len(e), len(m)):
            raise ValueError('配置与既有测试记录不一致')
        for job in jobs[:len(done)]:
            rows = e[e.job_id.eq(job['job_id'])]
            if sorted(rows.target_load_hp) != [t for t in range(4) if t != job['source_load_hp']]:
                raise ValueError(f'已完成模型的目标不完整: {job["job_id"]}')
            for target in range(4):
                if target != job['source_load_hp'] and not (
                        ev/'predictions'/f'{job["job_id"]}_target{target}.npz').is_file():
                    raise ValueError(f'已完成模型的预测缺失: {job["job_id"]} target{target}')
        self = object.__new__(cls)
        self.root, self.jobs, self.state, self._cohort_audited = root, jobs, state, False
        self.setup(resume=True)
        print(f'RECOVERY READY completed={len(done)}/{total}', flush=True)
        return self

    def recover_testing(self):
        """补齐缺失模型的测试后生成报告；不重训、不重测已完成模型。"""
        self.verify_cohort(deep=True)
        done = list(dict.fromkeys(pd.read_csv(self.root/'evaluation'/'evaluations.csv').job_id))
        self.state.setdefault('test_recovery', []).append(dict(
            recovered_at=now(), action='complete_missing_predictions',
            completed_before=len(done), missing_models=self.state['planned_models']-len(done),
            reason='permission_error_during_progress_write', pid=os.getpid()))
        self.persist()
        self.evaluate(recover=True)

    def activate_resume(self):
        self.state.setdefault('resume_history', []).append(dict(resumed_at=now(), completed_models=len(self.jobs), pid=os.getpid()))
        self.state.update(stage='training', pid=os.getpid(), pause_boundary_verified=False)
        self.persist()

    def train_all(self):
        import comparison_engine as engine
        scope = self.state['scope']
        plan = list(scope_cases(scope['method'], scope.get('sample','small'), scope['ratios']))
        for case in plan[len(self.jobs):]:
            jid = f'm{len(self.jobs)+1:04d}'
            folder = self.root/'models'/jid
            w = self.tables[split_name(case['domain'],case['ratio'],case['subset'])]
            # 此处只传训练和验证，排除所有目标观测。
            mask = w.role.isin(['train','validation']).to_numpy()
            self.state.update(stage='training', active_model=dict(job_id=jid, **case))
            self.persist()
            info = engine.train(self.x[mask], w.loc[mask].reset_index(drop=True), case['protocol'],
                                case['variant'], case['seed'], case['subset'], folder, self.device, {})
            job = dict(info)
            job.update(case, job_id=jid, folder=str(folder), source_load_hp=int(case['domain'][-1]), completed=True)
            if not (folder/'model.pt').is_file():
                raise ValueError(f'模型权重缺失: {jid}')
            self.jobs.append(job)
            with control_lock(self.root):
                self.state.pop('active_model', None)
                requests = pending_pause_requests(self.root, self.state)
                if requests:
                    self.state.update(stage='paused', pause_boundary_verified=True,
                                      pause_boundary_models=len(self.jobs), paused_at=now(),
                                      consumed_pause_requests=self.state.get('consumed_pause_requests', []) + requests)
                    self.persist()
                    print('PAUSED AT MODEL BOUNDARY',len(self.jobs),flush=True)
                    return False
                if len(self.jobs) == self.state['planned_models']:
                    self.state['stage'] = 'verifying_weights'
                self.persist()
            print('MODEL COMPLETE', jid, case, flush=True)
        self.state.update(stage='verifying_weights')
        self.persist()
        self.verify_cohort(deep=True)
        self.state.update(stage='decisions_locked', decisions_locked=True, decisions_locked_at=now(),
                          weights_verified=True, weights_verified_at=now())
        self.persist()
        return True

    def verify_checkpoint(self, job):
        """CPU 严格加载和有限值核验，并将完成记录关联回原固定划分。"""
        import comparison_engine as engine
        path = resolve_result_path(job['folder'])
        info = json.loads((path/'completed.json').read_text(encoding='utf-8'))
        for key in ('variant','protocol','seed','subset'):
            if info.get(key) != job[key]:
                raise ValueError(f'{job["job_id"]} 完成记录 {key} 不匹配')
        table = self.tables[split_name(job['domain'],job['ratio'],job['subset'])]
        expected_train = table.loc[table.role.eq('train'), 'window_index'].to_numpy()
        selected = pd.read_csv(path/'selected_windows.csv')
        if not np.array_equal(selected.window_index.to_numpy(), expected_train):
            raise ValueError(f'{job["job_id"]} 训练窗口与固定划分不一致')
        if info.get('training_windows') != len(expected_train) or info.get('validation_windows') != int(table.role.eq('validation').sum()):
            raise ValueError(f'{job["job_id"]} 训练或验证数量不一致')
        model = engine.load(path, torch.device('cpu'))  # 内含严格键与形状检查
        if model.variant != job['variant']:
            raise ValueError(f'{job["job_id"]} 权重方法名称不一致')
        for name, tensor in model.state_dict().items():
            if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'{job["job_id"]} 权重或缓冲 {name} 含非有限值')
        del model

    def verify_cohort(self, deep=False):
        scope = self.state['scope']
        plan = list(scope_cases(scope['method'], scope.get('sample','small'), scope['ratios']))
        expected = {case_id(c) for c in plan}
        total = self.state['planned_models']
        if len(self.jobs) != total or len(expected) != total or {case_id(j) for j in self.jobs} != expected:
            raise ValueError(f'预定的 {total} 个模型尚未完整训练登记')
        for index, j in enumerate(self.jobs, 1):
            path = resolve_result_path(j['folder']).resolve()
            if path.parent != (self.root/'models').resolve() or not (path/'model.pt').is_file() or not (path/'completed.json').is_file() or not j.get('completed'):
                raise ValueError(f'权重或完成记录不完整: {j["job_id"]}')
            if deep:
                self.verify_checkpoint(j)
                if index == 1 or index % 25 == 0 or index == total:
                    self.state['verified_models'] = index
                    self.persist()
                    print(f'VERIFIED {index}/{total}', flush=True)
        if deep:
            self._cohort_audited = True

    def evaluate(self, recover=False):
        import comparison_engine as engine
        if recover:
            # 仅续测缺失模型；已完成预测不重算、不覆盖。
            dest = self.root/'evaluation'
            existing = pd.read_csv(dest/'evaluations.csv')
            done = list(dict.fromkeys(existing.job_id))
            pairs, metrics = len(existing), len(pd.read_csv(dest/'metrics.csv'))
            if pairs != 3*len(done) or metrics != pairs or (pairs, metrics) != (self.state.get('evaluated_pairs'), self.state.get('metric_rows')):
                raise ValueError('既有测试记录不完整，不能续测')
            missing = [j for j in self.jobs if j['job_id'] not in set(done)]
            if not missing:
                raise ValueError('没有待续测的模型')
            if (pairs, metrics) != (3*(len(self.jobs)-len(missing)), 3*(len(self.jobs)-len(missing))):
                raise ValueError('已完成模型不是原计划的前缀')
            todo = missing
            self.state.update(stage='testing', target_test_accessed=True)
            self.persist()
        else:
            if not self.state['decisions_locked'] or not self._cohort_audited or self.state['target_test_accessed']:
                raise ValueError('须全部训练完成且只进行一次正式测试')
            self.verify_cohort()
            dest = self.root/'evaluation'
            dest.mkdir(exist_ok=False)
            (dest/'predictions').mkdir()
            self.state.update(stage='testing', target_test_accessed=True)
            self.persist()
            pairs = metrics = 0
            todo = self.jobs
        for j in todo:
            w = self.tables[split_name(j['domain'],j['ratio'],j['subset'])]
            model = engine.load(j['folder'], self.device)
            metric_rows, confusion_rows, eval_rows = [], [], []
            for target in [t for t in range(4) if t != j['source_load_hp']]:
                ix = np.flatnonzero(w.load_hp.eq(target))
                labels = w.iloc[ix].label.to_numpy()
                rpm = w.iloc[ix].rpm.to_numpy(np.float32)
                meta = {k:j[k] for k in ['job_id',*CASE_KEYS,'source_load_hp']}
                meta['target_load_hp'] = target
                arrays = dict(labels=labels, window_indices=w.iloc[ix].window_index.to_numpy())
                for condition, noise_seed, observed in R.conditions(self.x[ix], j['protocol'], True):
                    prediction, _ = engine.predict_and_ce(model, observed, rpm, labels, self.device)
                    arrays[f'{condition}_n{noise_seed}'] = prediction
                    values, counts = scores(labels, prediction)
                    m = dict(meta, condition=condition, noise_seed=noise_seed)
                    metric_rows.append(dict(m,**values))
                    for a in range(4):
                        for b in range(4):
                            confusion_rows.append(dict(m, true_label=a, predicted_label=b, count=int(counts[a,b]),
                                support=int(counts[a].sum()), row_percent=float(100*counts[a,b]/counts[a].sum())))
                path = dest/'predictions'/f'{j["job_id"]}_target{target}.npz'
                with path.open('xb') as handle:
                    np.savez_compressed(handle,**arrays)
                eval_rows.append(dict(meta, prediction_path=str(path), test_windows=len(ix)))
            del model
            for filename, rows in [('metrics.csv',metric_rows), ('confusion_matrices.csv',confusion_rows),
                                   ('evaluations.csv',eval_rows)]:
                path = dest/filename
                pd.DataFrame(rows).to_csv(path,index=False,mode='a',header=not path.exists())
            pairs += len(eval_rows)
            metrics += len(metric_rows)
            self.state.update(evaluated_pairs=pairs, metric_rows=metrics)
            self.persist()
            print('TEST COMPLETE', j['job_id'], flush=True)
        if (pairs,metrics) != (self.state['planned_test_pairs'],self.state['planned_metric_rows']):
            raise ValueError('测试组合或指标行数错误')
        self.state.update(stage='reporting', testing_completed_at=now())
        self.persist()
        import comparison_report
        comparison_report.finish(self.root, self.state['scope'])
        self.state.update(stage='complete', completed_at=now())
        self.persist()

    def run(self):
        self.setup()
        if self.train_all():
            self.evaluate()


class Tee:
    """前台显示训练进度，同时保留完整日志。"""
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, s):
        self.console.write(s)
        self.log.write(s)
        self.flush()
        return len(s)
    def flush(self):
        self.console.flush()
        self.log.flush()


@contextmanager
def file_lock(lock_path, blocking=False, owner_root=None):
    """进程级锁；文件残留无妨，操作系统在进程退出时释放锁。"""
    with lock_path.open('a+b') as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError('另一个跨负载对比进程正在运行，请勿重复启动') from error
            try:
                if owner_root is not None:
                    lock.seek(1); lock.truncate(); lock.write(json.dumps({'root':str(Path(owner_root).resolve()),'pid':os.getpid()}).encode('utf-8')); lock.flush()
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except OSError as error:
                raise RuntimeError('另一个跨负载对比进程正在运行，请勿重复启动') from error
            try:
                if owner_root is not None:
                    lock.seek(1); lock.truncate(); lock.write(json.dumps({'root':str(Path(owner_root).resolve()),'pid':os.getpid()}).encode('utf-8')); lock.flush()
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def exclusive_run_lock(owner_root=None):
    return file_lock((PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve()/'.load_comparison_process.lock',owner_root=owner_root)


def control_lock(root):
    return file_lock(Path(root)/'.control.lock', blocking=True)


def pending_pause_requests(root, state):
    folder = Path(root)/'pause_requests'
    if not folder.exists():
        return []
    consumed = set(state.get('consumed_pause_requests', []))
    return sorted(p.name for p in folder.glob('*.json') if p.name not in consumed)


def request_pause(root):
    root = Path(root).resolve()
    if root.parent != (PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve() or not root.name.startswith('load_comparison_'):
        raise ValueError('暂停目录不合法')
    try:
        with exclusive_run_lock():
            raise ValueError('没有正在运行的跨负载对比进程；不能对中断现场请求暂停')
    except RuntimeError:
        pass  # 运行锁被占用，仍需在控制锁内核验本目录阶段。
    with ((PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve()/'.load_comparison_process.lock').open('rb') as handle:
        handle.seek(1)
        try:
            owner = json.loads(handle.read().decode('utf-8'))
        except (ValueError, UnicodeError) as error:
            raise ValueError('无法核验正在运行的结果目录') from error
    if Path(owner.get('root','')).resolve() != root:
        raise ValueError('该结果目录不是当前正在运行的实验')
    with control_lock(root):
        state = json.loads((root/'config.json').read_text(encoding='utf-8'))
        if (state.get('stage') != 'training' or state.get('target_test_accessed') or state.get('decisions_locked')
            or state.get('pid') != owner.get('pid')):
            raise ValueError('只有训练阶段且未开始测试时可以请求暂停')
        folder = root/'pause_requests'
        folder.mkdir(exist_ok=True)
        path = folder/(uuid.uuid4().hex+'.json')
        with path.open('x',encoding='utf-8') as f:
            json.dump(dict(requested_at=now(), request='pause_after_completed_model'),f,ensure_ascii=False)
        print('暂停请求已记录；当前模型完成后才会暂停。请用 status 查询是否已暂停。',flush=True)


def status(root):
    import json
    root = Path(root).resolve()
    parent = (PROJECT/'experiments/CWRU/实验结果/CWRU_V3').resolve()
    if root.parent != parent or not root.name.startswith('load_comparison_'):
        raise ValueError('状态只能查看 experiments/CWRU/实验结果/CWRU_V3 下的 load_comparison_* 目录')
    state = json.loads((root/'config.json').read_text(encoding='utf-8'))
    scope = state.get('scope') or {}
    if scope:
        print(f"method: {scope.get('method')}; ratios: {scope.get('ratios')}; protocol: {scope.get('protocol')}")
    for key in ('stage','trained_models','planned_models','verified_models','pause_boundary_verified','pause_boundary_models','evaluated_pairs','planned_test_pairs',
                'metric_rows','planned_metric_rows','active_model','updated_at','error'):
        if key in state:
            print(f'{key}: {state[key]}')
    print(f'pending_pause_requests: {len(pending_pause_requests(root,state))}')
    active = state.get('active_model') or {}
    if active.get('job_id'):
        progress = root/'models'/active['job_id']/'training.csv'
        if progress.exists():
            try:
                frame = pd.read_csv(progress)
                if not frame.empty:
                    row = frame.iloc[-1]
                    print(f'active_epoch: {int(row.epoch)}; updates: {int(row.updates)}; validation_nll: {row.validation_nll:.6f}')
            except (pd.errors.EmptyDataError, OSError, ValueError, KeyError):
                print('active_epoch: 日志正在写入，请稍后查询')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','run','status','pause','resume','recover-testing'], nargs='?', default='plan')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--method', choices=METHODS, help='本次运行的对比方法；一次只跑一个')
    parser.add_argument('--sample', choices=list(SAMPLE_CHOICES), default='small',
                        help='small（10/20/50%%，默认）或 full（完整训练池）')
    parser.add_argument('--ratio', default='all', help='all（默认）或 0.1/0.2/0.5，也可写 10/20/50；full 忽略此项')
    args = parser.parse_args()
    if args.command == 'plan':
        ratios = parse_ratio(args.ratio)
        targets = [args.method] if args.method else METHODS
        for method in targets:
            models, pairs, rows = plan_counts(args.sample, ratios)
            if args.sample == 'full':
                scope_desc = '4 sources x 1 full training pool x 5 seeds'
            else:
                scope_desc = f'4 sources x {len(ratios)} budgets x 3 subsets x 5 seeds'
            print(f'{method}: {models} new models; sample={args.sample}; {scope_desc}; '
                  f'{pairs} target pairs; {rows} metric rows; clean-only cross-load.')
        print('固定划分与种子与 V3 对应实验一致；本文方法结果只读复用，不重训。')
        return
    if args.output is None:
        parser.error('run/status/pause/resume/recover-testing 需要 --output 目录')
    if args.command == 'status':
        status(args.output)
        return
    if args.command == 'pause':
        request_pause(args.output)
        return
    if args.command == 'recover-testing':
        with exclusive_run_lock(args.output):
            experiment = ComparisonRun.open_for_recovery(args.output)
            logname = f'testing_recover_{uuid.uuid4().hex}.log'
            with (experiment.root/logname).open('x', encoding='utf-8', buffering=1) as log:
                with redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
                    try:
                        experiment.recover_testing()
                    except BaseException as error:
                        experiment.state.update(stage='failed',error=str(error),error_type=type(error).__name__)
                        experiment.persist()
                        traceback.print_exc()
                        raise
        return
    if args.command == 'run' and not args.method:
        parser.error('run 必须指定 --method：' + '/'.join(METHODS))
    ratios = [1.0] if args.sample == 'full' else parse_ratio(args.ratio)
    with exclusive_run_lock(args.output):
        if args.command == 'resume':
            experiment = ComparisonRun.open_paused(args.output)
            if args.method and args.method != experiment.state['scope']['method']:
                raise ValueError('恢复方法与原目录登记的方法不一致')
        else:
            experiment = ComparisonRun(args.output, args.method, args.sample, ratios)
        logname = f'training_resume_{uuid.uuid4().hex}.log' if args.command == 'resume' else 'training_console.log'
        with (experiment.root/logname).open('x', encoding='utf-8', buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
                try:
                    if args.command == 'resume':
                        experiment.activate_resume()
                        if experiment.train_all():
                            experiment.evaluate()
                    else:
                        experiment.run()
                except BaseException as error:
                    experiment.state.update(stage='failed',error=str(error),error_type=type(error).__name__)
                    experiment.persist()
                    traceback.print_exc()
                    raise


if __name__ == '__main__':
    main()
