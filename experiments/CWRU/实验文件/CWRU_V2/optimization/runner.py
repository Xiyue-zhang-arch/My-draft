"""阶段控制、历史引用、验证决策及决策锁定后的独立测试。"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import copy
import json
import sys
sys.dont_write_bytecode = True
import torch
import numpy as np
import pandas as pd
from diagnose import R, BASE, OUT as HISTORY, HERE
from splits import read_fixed
from reporting import scores
from models import baseline_equivalence
from . import engine, design
from . import reporting as suite_reporting

from experiment_paths import resolve_result_path

PROJECT = HERE.parents[3]
RESULT_PARENT = PROJECT / 'experiments/CWRU/实验结果/CWRU_V2'
PROTOCOL_FILE = HERE / 'OPTIMIZATION_SUITE.md'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path, value):
    R.write_json(path, value)


def checked_root(path, new=False):
    root = Path(path).resolve()
    if root.parent != RESULT_PARENT.resolve() or not root.name.startswith('optimization_'):
        raise ValueError('结果只能是experiments/CWRU/实验结果/CWRU_V2下新的optimization_*直接子目录')
    if new and root.exists():
        raise FileExistsError(root)
    if not new and (not root.is_dir() or not (root/'config.json').is_file()):
        raise ValueError('请先用init创建新的独立实验目录')
    return root


def initialize(path, include_convergence=False):
    root = checked_root(path, new=True)
    for reference in [BASE, HISTORY]:
        state = json.loads((reference/'config.json').read_text(encoding='utf-8'))
        if state['stage'] != 'complete':
            raise ValueError(f'参照未完成: {reference}')
    stages = list(design.STAGES)
    if include_convergence:
        stages.insert(stages.index('expansion'), 'convergence')
    root.mkdir(parents=True, exist_ok=False)
    state = dict(schema_version=1, created_at=now(), stage='prepared',
                 historical_test_exposure=True, target_test_accessed=False,
                 decisions_locked=False, current_spec=copy.deepcopy(design.DEFAULT_SPEC),
                 stage_order=stages, stages={s:dict(status='pending') for s in stages},
                 trained_models=0, referenced_models=0, include_convergence=include_convergence,
                 lambda_grid=design.LAMBDA_GRID, seeds=design.SEEDS, full_seeds=design.FULL_SEEDS,
                 subsets=design.SUBSETS, screening_ratios=design.SCREEN_RATIOS,
                 decision_rule='validation_gain_and_preservation_v2_sd_descriptive',
                 baseline=str(BASE), history=str(HISTORY), jobs=[],
                 transfer_scope='exploratory_locked_random_window_configuration')
    write_json(root/'config.json', state)
    (root/'PROTOCOL.md').write_text(PROTOCOL_FILE.read_text(encoding='utf-8'), encoding='utf-8')
    (root/'DATA_USAGE.md').write_text(
        '# 数据使用\n\n全部40条记录，687完整窗口，data只读。复用V1已固定的481/103/103窗口划分，'
        '10%/20%/50%复用V2的49/98/242训练窗及3个嵌套子集。所有小样本仍使用103验证标签。'
        '\n\n原始MAT路径、通道、各集合逐文件数量见'
        '[V1数据用量](../../CWRU_V1/baseline_20261020/dataset_file_usage.csv)。'
        '小样本逐文件用量见[固定V2子集](../workflow_20261020/training_subset_file_usage.csv)。'
        '\n\n本轮每模型选中的窗口和来源在models_registry.csv、dataset_file_usage.csv及'
        'training_subset_file_usage.csv中记录。复用模型保留原路径；没有复制或改写原始数据。'
        '\n\n若进入迁移，来源训练73/156/160/160、验证19/39/40/40，目标92/195/200/200窗。'
        '随机窗口开发已经使用全部负载，不称目标负载未参与开发或独立确认性迁移。', encoding='utf-8')
    print(f'PREPARED {root}; 未训练或读取测试预测')
    return state


def cases(kind='screen'):
    if kind == 'transfer':
        return [dict(domain=f'load_source_{s}',ratio=1.,subset=0,protocol=p,seed=seed)
                for s in range(4) for p in design.PROTOCOLS for seed in design.FULL_SEEDS]
    budgets = design.SCREEN_RATIOS if kind == 'screen' else [.5, 1.]
    return [dict(domain='window_random',ratio=r,subset=subset,protocol=p,seed=seed)
            for r in budgets for subset in (design.SUBSETS if r < 1 else [0])
            for p in design.PROTOCOLS for seed in (design.SEEDS if r < 1 else design.FULL_SEEDS)]


def split_path(case):
    if case['domain'] != 'window_random':
        return BASE/'splits'/case['domain']
    if case['ratio'] == 1:
        return BASE/'splits/window_random'
    return HISTORY/'subsets'/f"ratio_{int(case['ratio']*100):03d}_subset_{case['subset']}"


def effective_spec(variant, protocol, spec):
    value = copy.deepcopy(spec)
    if protocol == 'clean_only':
        value['val_noise_seeds'] = [6201]
        value['physics_fit'] = 'clean'
    if variant == 'baseline_data':
        value['aux_weight'] = .1
        value['aux_head'] = 'shared'
        value['physics_fit'] = 'clean'
    if value['physics_fit'] == 'clean':
        value['fit_noise_seed'] = 6301
        value['fit_noise_repeats'] = 3
    return value


def case_key(case):
    return tuple(case[k] for k in ['domain','ratio','subset','protocol','seed'])


class Suite:
    def __init__(self, root):
        self.root = checked_root(root)
        self.state = json.loads((self.root/'config.json').read_text(encoding='utf-8'))
        if self.state.get('schema_version') != 1:
            raise ValueError('不支持的状态格式')
        design.validate_spec(self.state['current_spec'])
        # 运行配置随init锁定，不能通过改网格或种子隐式改变既定实验。
        for key, expected in [('lambda_grid',design.LAMBDA_GRID),('seeds',design.SEEDS),
                              ('full_seeds',design.FULL_SEEDS),('subsets',design.SUBSETS),
                              ('screening_ratios',design.SCREEN_RATIOS)]:
            if self.state[key] != expected:
                raise ValueError(f'配置与已声明方案不一致: {key}')
        if self.state.get('decision_rule') != 'validation_gain_and_preservation_v2_sd_descriptive':
            raise ValueError('判定规则版本不匹配，不能修改已有实验的判据')
        self.jobs = self.state['jobs']
        self.device = None
        self.raw = self.x = self.original = self.records = None
        self.reference_costs = {}

    def persist(self):
        new = {j['folder'] for j in self.jobs if j['origin'] == 'suite'}
        old = {j['folder'] for j in self.jobs if j['origin'] != 'suite'}
        self.state.update(updated_at=now(), jobs=self.jobs,
                          trained_models=len(new), referenced_models=len(old))
        write_json(self.root/'config.json', self.state)
        if self.jobs:
            pd.DataFrame(self.jobs).to_csv(self.root/'models_registry.csv', index=False)

    def setup(self):
        if self.device is not None:
            return
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        R.U.DEVICE = self.device
        R.U.configure()
        baseline_equivalence(self.device)
        self.records, self.original, self.raw = R.load_dataset(
            PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
        self.x = R.U.normalize(self.raw)
        if len(self.original) != 687 or self.original.record_id.nunique() != 40:
            raise ValueError('原始数据范围不符合固定687窗/40记录协议')
        self.state['hardware'] = torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU'
        self.state['precision'] = 'CUDA encoder BF16; remaining FP32'
        if not (self.root/'records.csv').exists():
            self.records.to_csv(self.root/'records.csv', index=False)

    def windows(self, case):
        w = read_fixed(split_path(case))
        for column in ['window_index','record_id','label','start_sample','end_sample']:
            np.testing.assert_array_equal(w[column], self.original[column])
        if case['domain'] == 'window_random':
            fixed = read_fixed(BASE/'splits/window_random')
            for role in ['validation','test']:
                np.testing.assert_array_equal(w.loc[w.role.eq(role),'window_index'],
                                              fixed.loc[fixed.role.eq(role),'window_index'])
        return w

    def old_folder(self, variant, spec, case):
        if effective_spec(variant,case['protocol'],spec) != effective_spec(
                variant,case['protocol'],design.DEFAULT_SPEC):
            return None
        if case['domain'] == 'window_random' and case['ratio'] < 1:
            path = HISTORY/'models/window_random'/variant/f"ratio_{int(case['ratio']*100):03d}_subset_{case['subset']}"/case['protocol']/f"seed_{case['seed']}"
        elif variant == 'baseline_data':
            path = BASE/'models'/case['domain']/case['protocol']/'data_only'/f"seed_{case['seed']}"
        else:
            return None
        return path if (path/'model.pt').is_file() and (path/'parameters.json').is_file() else None

    def historical_cost(self, folder, variant, case):
        """只读引用已测计算成本，不能重跑冻结模型的基准测试。"""
        is_v1 = BASE in folder.parents
        root = BASE if is_v1 else HISTORY
        if root not in self.reference_costs:
            self.reference_costs[root] = (
                pd.read_csv(root/'evaluations.csv'), pd.read_csv(root/'complexity.csv'))
        evaluations, complexities = self.reference_costs[root]
        if is_v1:
            method = 'data_only' if variant == 'baseline_data' else 'fusion'
            match = evaluations[evaluations['split'].eq(case['domain']) &
                evaluations.protocol.eq(case['protocol']) & evaluations.method.eq(method) &
                evaluations.seed.eq(case['seed'])]
            cost = complexities[complexities.method.eq(method)]
        else:
            match = evaluations[evaluations.domain.eq(case['domain']) &
                evaluations.variant.eq(variant) & evaluations.ratio.eq(case['ratio']) &
                evaluations.subset.eq(case['subset']) & evaluations.protocol.eq(case['protocol']) &
                evaluations.seed.eq(case['seed'])]
            cost = complexities[complexities.variant.eq(variant)]
        # 同一来源的不同目标若有不同延迟，报告已保存测量的平均值并注明来源。
        return dict(latency_ms=float(match.latency_ms.mean()) if len(match) else None,
                    conv_linear_flops=int(cost.conv_linear_flops.iloc[0]) if len(cost) else None,
                    historical_cost_source=str(root/'evaluations.csv'))

    def ensure(self, variant, spec, case, stage):
        design.validate_spec(spec)
        cid = design.config_id(variant,spec)
        found = next((j for j in self.jobs if j['config_id'] == cid and case_key(j) == case_key(case)), None)
        if found is not None:
            return found
        # 简短登记号避免Windows路径长度限制；科学配置完整保存在registry中。
        jid = f'm{len(self.jobs)+1:04d}'
        self.setup()
        w = self.windows(case)
        reuse = next((j for j in self.jobs if j['variant'] == variant and case_key(j) == case_key(case)
                      and effective_spec(variant,case['protocol'],j['spec']) == effective_spec(variant,case['protocol'],spec)), None)
        folder = resolve_result_path(reuse['folder']) if reuse else self.old_folder(variant,spec,case)
        origin = reuse['origin'] if reuse else ('historical' if folder else 'suite')
        if folder is None:
            folder = self.root/'models'/jid
            # 所有结构与配置先打印开销，再允许训练。
            complexity = engine.complexity(variant,spec,self.device)
            info = engine.train(self.x,w,case['protocol'],variant,case['seed'],case['subset'],
                                folder,self.device,spec)
            write_json(folder/'complexity.json',complexity)
        else:
            info = json.loads((folder/'parameters.json').read_text(encoding='utf-8'))
            selected = pd.read_csv(folder/'selected_windows.csv')
            np.testing.assert_array_equal(selected.window_index,w.loc[w.role.eq('train'),'window_index'])
            if int(info['seed']) != case['seed'] or int(info['validation_windows']) != int(w.role.eq('validation').sum()):
                raise ValueError('历史模型的种子或验证集合不匹配')
        model = engine.load(folder,self.device)
        rows, arrays = engine.validation(model,w,self.x,case['protocol'],self.device,spec)
        del model
        trained_spec = info.get('spec', design.DEFAULT_SPEC)
        if case['protocol'] == 'clean_only' or trained_spec['val_noise_seeds'] == spec['val_noise_seeds']:
            if not np.isclose(engine.validation_score(rows,case['protocol']),info['validation_nll'],rtol=0,atol=2e-6):
                raise ValueError('已选模型的验证NLL与保存记录不一致，不能作为有效参照')
        if origin == 'historical':
            info.update(self.historical_cost(folder,variant,case))
            info['inference_parameters'] = info['parameters']
            info['auxiliary_only_parameters'] = 0
            info['hardware'] = self.state['hardware']
            info['precision'] = 'CUDA encoder BF16; remaining FP32'
            info['batch_size'] = 32
            info['latency_scope'] = 'historical measurement; batch1 normalization+online physics+network; 10 warmups/50 repeats'
        validation_dir = self.root/'validation'
        validation_dir.mkdir(exist_ok=True)
        np.savez_compressed(validation_dir/f'{jid}.npz',**arrays)
        source = -1 if case['domain'] == 'window_random' else int(case['domain'][-1])
        meta = dict(job_id=jid,config_id=cid,variant=variant,stage=stage,**case,
                    source_load_hp=source,target_load_hp=-1,
                    aux_weight=spec['aux_weight'] if variant=='data_aux_ce' else 0.,
                    physics_fit=spec['physics_fit'],aux_head=spec['aux_head'],
                    val_noise_count=len(spec['val_noise_seeds']),epochs_limit=spec['max_epochs'],
                    control_config_id=design.config_id('baseline_data',spec))
        frame = pd.DataFrame([{**r,**meta} for r in rows])
        frame.to_csv(validation_dir/f'{jid}.csv',index=False)
        job = dict(**meta,folder=str(folder),origin=origin,spec=copy.deepcopy(spec),
                   split_path=str(split_path(case)),validation_metrics=str(validation_dir/f'{jid}.csv'),
                   validation_predictions=str(validation_dir/f'{jid}.npz'),
                   training_windows=int(w.role.eq('train').sum()),validation_windows=int(w.role.eq('validation').sum()),
                   parameters=info.get('parameters'),epochs_trained=info.get('epochs_trained'),
                   updates_trained=info.get('updates_trained'),best_epoch=info.get('best_epoch'),
                   fit_seconds=info.get('fit_seconds'),total_train_seconds=info.get('total_train_seconds'),
                   train_seconds_per_epoch=info.get('train_seconds_per_epoch'),validation_seconds=info.get('validation_seconds'),
                   peak_gpu_mib=info.get('peak_gpu_mib'),latency_ms=info.get('latency_ms'))
        for field in ['inference_parameters','conv_linear_flops','auxiliary_only_parameters',
                      'gradient_diagnostic_seconds','total_training_with_diagnostics_seconds',
                      'hardware','precision','batch_size','latency_scope','peak_memory_scope']:
            job[field] = info.get(field)
        self.jobs.append(job)
        self.persist()
        print(f"JOB COMPLETE {jid}; origin={origin}",flush=True)
        return job

    def ensure_config(self, variant, spec, kind, stage):
        for case in cases(kind):
            self.ensure(variant,spec,case,stage)
        return design.config_id(variant,spec)

    def metrics(self, ids, kind='screen'):
        required = {case_key(c) for c in cases(kind)}
        frames = [pd.read_csv(j['validation_metrics']) for j in self.jobs
                  if j['config_id'] in ids and case_key(j) in required]
        if not frames:
            raise ValueError('缺少验证数据')
        return pd.concat(frames,ignore_index=True)

    def data_control(self, spec, kind, stage):
        return self.ensure_config('baseline_data',spec,kind,stage)

    def screen_gate(self):
        spec = self.state['current_spec']
        cid = design.config_id('data_aux_ce',spec)
        data = design.config_id('baseline_data',spec)
        return design.assess(self.metrics([cid,data]),cid,data,data)

    def stage_lambda(self, out):
        base = copy.deepcopy(design.DEFAULT_SPEC)
        data = self.data_control(base,'screen','lambda')
        ids = []
        for value in design.LAMBDA_GRID:
            spec = {**base,'aux_weight':value}
            ids.append(self.ensure_config('data_aux_ce',spec,'screen','lambda'))
        reference = design.config_id('data_aux_ce',base)
        metrics = self.metrics([data,*ids])
        decision = design.select_lambda(metrics,ids,reference,data)
        selected = next(j['spec'] for j in self.jobs if j['config_id'] == decision['selected_config_id'])
        self.state['current_spec'] = copy.deepcopy(selected)
        decision['purpose'] = '通过才更新系数；否则保留0.1，失败结果保留'
        return metrics,decision

    def stage_physics_fit(self, out):
        before = copy.deepcopy(self.state['current_spec'])
        after = {**before,'physics_fit':'matched_noise'}
        data = self.data_control(before,'screen','physics_fit')
        reference = self.ensure_config('data_aux_ce',before,'screen','physics_fit')
        candidate = self.ensure_config('data_aux_ce',after,'screen','physics_fit')
        diagnostic_frames = []
        for job in self.jobs:
            if (job['config_id'] not in [reference,candidate] or job['protocol'] != 'awgn_mixed'
                    or job['seed'] != design.SEEDS[0] or case_key(job) not in {case_key(c) for c in cases()}):
                continue
            model = engine.load(resolve_result_path(job['folder']),self.device)
            frame = engine.physical_standardization_diagnostic(model,self.x,self.windows(job),self.device,job['spec'])
            del model
            for key in ['job_id','config_id','ratio','subset','seed']:
                frame[key] = job[key]
            diagnostic_frames.append(frame)
        pd.concat(diagnostic_frames,ignore_index=True).to_csv(out/'physical_standardization_diagnosis.csv',index=False)
        metrics = self.metrics([reference,candidate,data])
        decision = design.assess(metrics,candidate,reference,data)
        if decision['passed']:
            self.state['current_spec'] = after
        decision.update(candidate_spec=after,adopted=bool(decision['passed']),
                        physical_diagnosis='physical_standardization_diagnosis.csv',
                        diagnosis_scope='training/validation only; fixed seed per subset; scale shrinkage is not performance evidence')
        return metrics,decision

    def stage_aux_head(self, out):
        before = copy.deepcopy(self.state['current_spec'])
        reference = self.ensure_config('data_aux_ce',before,'screen','aux_head')
        data = self.data_control(before,'screen','aux_head')
        probes = []
        for job in self.jobs:
            if job['config_id'] != reference or job['protocol'] != 'awgn_mixed' or job['ratio'] not in design.SCREEN_RATIOS:
                continue
            model = engine.load(resolve_result_path(job['folder']),self.device)
            probe = engine.gradient_probe(model,self.x,self.windows(job),job['protocol'],self.device,before)
            probes.append(dict(job_id=job['job_id'],**probe))
            del model
        write_json(out/'gradient_diagnosis.json',dict(
            phase='best_checkpoint_fixed_training_observations_no_optimizer',measurements=probes))
        fractions = [p.get('head_negative_fraction') for p in probes]
        finite = [float(v) for v in fractions if v is not None and np.isfinite(float(v))]
        if not finite:
            raise ValueError('共享分类头梯度诊断缺失，不能自动决定进入独立头实验')
        fraction = float(np.mean(finite))
        if fraction < .2:
            return self.metrics([reference,data]), dict(skipped=True,adopted=False,
                reason='固定训练观测上的共享头负余弦比例未达到预设20%，不新增独立头实验',
                head_negative_fraction=fraction,conflict_threshold=.2)
        after = {**before,'aux_head':'independent'}
        candidate = self.ensure_config('data_aux_ce',after,'screen','aux_head')
        metrics = self.metrics([reference,candidate,data])
        decision = design.assess(metrics,candidate,reference,data)
        if decision['passed']:
            self.state['current_spec'] = after
        decision.update(candidate_spec=after,adopted=bool(decision['passed']),head_negative_fraction=fraction,
                        conflict_threshold=.2,gradient_conflict_is_not_proof_of_generalization_failure=True)
        return metrics,decision

    def stage_validation_noise(self, out):
        before = copy.deepcopy(self.state['current_spec'])
        after = {**before,'val_noise_seeds':[6201,6202,6203]}
        reference = self.ensure_config('data_aux_ce',before,'screen','validation_noise')
        old_data = self.data_control(before,'screen','validation_noise')
        data = self.data_control(after,'screen','validation_noise')
        candidate = self.ensure_config('data_aux_ce',after,'screen','validation_noise')
        # 对前后模型均使用相同3实例验证面板评价，训练选模差别仍仅1与3实例。
        panel_rows = []
        ids = [reference,old_data,data,candidate]
        for job in self.jobs:
            if job['config_id'] not in ids or case_key(job) not in {case_key(c) for c in cases()}:
                continue
            model = engine.load(resolve_result_path(job['folder']),self.device)
            rows,_ = engine.validation(model,self.windows(job),self.x,job['protocol'],self.device,after)
            meta = {k:job[k] for k in ['job_id','config_id','variant','domain','ratio','subset','protocol','seed',
                    'source_load_hp','target_load_hp','control_config_id','aux_weight','aux_head','physics_fit']}
            panel_rows.extend([{**r,**meta} for r in rows])
            del model
        metrics = pd.DataFrame(panel_rows)
        decision = design.assess(metrics,candidate,reference,data,require_gain=False)
        data_check = design.assess(metrics,data,old_data,old_data,require_gain=False)
        decision['data_protocol_check'] = data_check
        # 两方法共同更改选模，数据对照也不得在验证均值上出现明显退化。
        data_non_regression = data_check['noisy_gain_pp'] >= -.5 and all(
            v >= -.5 for v in data_check['clean_gain_pp'].values())
        adopt = bool(decision['passed'] and data_non_regression)
        if adopt:
            self.state['current_spec'] = after
        decision.update(candidate_spec=after,adopted=adopt,data_non_regression=data_non_regression,
                        evaluation_panel=[6201,6202,6203])
        return metrics,decision

    def stage_convergence(self, out):
        before = copy.deepcopy(self.state['current_spec'])
        after = {**before,'max_epochs':200}
        ids = []
        for variant in ['baseline_data','data_aux_ce']:
            ids.append(self.ensure_config(variant,before,'screen','convergence'))
            ids.append(self.ensure_config(variant,after,'screen','convergence'))
        return self.metrics(ids),dict(adopted=False,diagnostic_only=True,
            reason='额外100epoch只检查训练充分性，不更改最多100epoch的主实验，不据此选择主方案',
            extended_max_epochs=200,schedule_epochs=100,lr_after_schedule=1e-5)

    def stage_expansion(self, out):
        gate = self.screen_gate()
        spec = copy.deepcopy(self.state['current_spec'])
        if not gate['passed']:
            ids = [design.config_id(v,spec) for v in ['baseline_data','data_aux_ce']]
            return self.metrics(ids),dict(skipped=True,passed=False,reason='基础两组比较未通过预设收益及退化保护规则',screen_gate=gate)
        data = self.data_control(spec,'expansion','expansion')
        candidate = self.ensure_config('data_aux_ce',spec,'expansion','expansion')
        metrics = self.metrics([data,candidate],kind='expansion')
        half = design.assess(metrics[metrics.ratio.eq(.5)],candidate,data,data)
        full = design.assess(metrics[metrics.ratio.eq(1.)],candidate,data,data,full=True)
        return metrics,dict(passed=bool(half['passed'] and full['passed']),half=half,full=full,screen_gate=gate)

    def stage_transfer(self, out):
        expansion = self.state['stages']['expansion'].get('decision',{})
        if not expansion.get('passed',False):
            spec = self.state['current_spec']
            ids = [design.config_id(v,spec) for v in ['baseline_data','data_aux_ce']]
            return self.metrics(ids),dict(skipped=True,passed=False,reason='50%及全样本扩展未通过，停止跨负载')
        spec = copy.deepcopy(self.state['current_spec'])
        data = self.data_control(spec,'transfer','transfer')
        candidate = self.ensure_config('data_aux_ce',spec,'transfer','transfer')
        return self.metrics([data,candidate],kind='transfer'),dict(completed=True,
            selected_spec=spec,source_only_checkpoint_selection=True,
            scope='参数在含全部负载的随机窗口开发后锁定；探索性12方向迁移，不能声称目标负载未参与开发')

    def write_usage(self):
        rows = []
        for job in self.jobs:
            w = read_fixed(resolve_result_path(job['split_path']))
            for rid,g in w.groupby('record_id'):
                for role in ['train','validation','test']:
                    if job['domain'] != 'window_random' and role == 'test':
                        continue
                    rows.append(dict(job_id=job['job_id'],config_id=job['config_id'],record_id=rid,
                                     role=role,used_windows=int(g.role.eq(role).sum()),split_path=job['split_path']))
                if job['domain'] != 'window_random':
                    source=int(job['domain'][-1])
                    for target in range(4):
                        if target != source:
                            rows.append(dict(job_id=job['job_id'],config_id=job['config_id'],record_id=rid,
                                role=f'test_hp{target}',used_windows=int(g.load_hp.eq(target).sum()),split_path=job['split_path']))
        frame = pd.DataFrame(rows)
        if len(frame):
            records = pd.read_csv(self.root/'records.csv')
            columns = [c for c in ['record_id','path','variable','samples','label'] if c in records]
            frame = frame.merge(records[columns],on='record_id',validate='many_to_one')
            frame.to_csv(self.root/'dataset_file_usage.csv',index=False)
            frame[frame.role.eq('train')].to_csv(self.root/'training_subset_file_usage.csv',index=False)

    def run_next(self):
        if self.state['stage']=='paused' and not self.state.get('resume_authorized',False):
            raise ValueError('实验处于用户指定暂停状态，须明确恢复后继续')
        if (self.state['target_test_accessed'] and not self.state.get('continuation_after_partial_test_authorized')) or self.state['decisions_locked']:
            raise ValueError('决策已锁定或测试已读取，不允许新增训练/重新筛选')
        paused = [s for s in self.state['stage_order'] if self.state['stages'][s]['status']=='paused']
        if paused and not self.state.get('resume_authorized',False):
            raise ValueError('用户请求的暂停尚未恢复，请使用resume-autonomous明确继续')
        if any(v['status'] in ['running','failed'] for v in self.state['stages'].values()):
            raise ValueError('有中断/失败阶段，保留现场；不自动覆盖或重启。请用新的独立实验目录处理')
        name = paused[0] if paused else next((s for s in self.state['stage_order'] if self.state['stages'][s]['status']=='pending'),None)
        if name is None:
            self.lock()
            return False
        out = self.root/'stages'/name
        resuming = bool(paused)
        if resuming:
            if name!='lambda' or not self.state.get('pause_boundary_verified'):
                raise ValueError('仅支持已核验完整模型边界的系数阶段暂停恢复')
            if any((out/p).exists() for p in ['decision.json','VALIDATION_REPORT.md','validation_metrics.csv']):
                raise ValueError('暂停阶段已有决策产物，禁止恢复覆盖')
            registered={resolve_result_path(j['folder']).resolve() for j in self.jobs if j['origin']=='suite'}
            actual={p.resolve() for p in (self.root/'models').iterdir() if p.is_dir()}
            if actual!=registered:
                raise ValueError('存在未登记模型，不能自动重训或恢复')
            for folder in registered:
                if not json.loads((folder/'parameters.json').read_text(encoding='utf-8')).get('completed'):
                    raise ValueError('存在未完成模型，禁止自动恢复')
            self.state.update(resumed_at=now(),resume_authorized=False)
            request=json.loads((self.root/'pause_request.json').read_text(encoding='utf-8'))
            request.update(status='resumed',resumed_at=self.state['resumed_at'])
            write_json(self.root/'pause_request.json',request)
        out.mkdir(parents=True,exist_ok=resuming)
        entry = self.state['stages'][name]
        entry.update(status='running',started_at=now(),input_spec=copy.deepcopy(self.state['current_spec']))
        self.state['stage'] = name
        self.persist()
        try:
            metrics,decision = getattr(self,f'stage_{name}')(out)
            metrics.to_csv(out/'validation_metrics.csv',index=False)
            write_json(out/'decision.json',decision)
            suite_reporting.write_validation_report(out,metrics,decision)
            entry.update(status='skipped' if decision.get('skipped') else 'complete',completed_at=now(),
                         decision=decision,output_spec=copy.deepcopy(self.state['current_spec']))
            self.write_usage()
            self.persist()
            if self.state.get('pause_after_stage')==name:
                self.state.update(stage='paused',paused_at=now(),pause_boundary_verified=True,
                                  pause_kind='completed_stage',pause_completed_stage=name,
                                  pause_reason='user_requested_stage_review',resume_authorized=False)
                self.persist()
                print(f'STAGE {name} {entry["status"]}; PAUSED FOR RESULT REVIEW',flush=True)
                return True
            if all(v['status'] in ['complete','skipped'] for v in self.state['stages'].values()):
                self.lock()
            print(f'STAGE {name} {entry["status"]}',flush=True)
            return True
        except BaseException as error:
            entry.update(status='failed',error_type=type(error).__name__,error=str(error))
            self.state['stage'] = 'failed'
            self.persist()
            raise

    def lock(self):
        self.state.update(decisions_locked=True,decisions_locked_at=now(),stage='decisions_locked')
        selection = dict(selected_spec=self.state['current_spec'],
                         decisions_locked=True,decisions_locked_at=self.state['decisions_locked_at'],
                         earlier_partial_test_exposure=self.state.get('partial_test_status')=='complete',
                         decisions={k:v.get('decision',{}) for k,v in self.state['stages'].items()},
                         methodology='全部候选包括失败者统一报告；历史测试曝光；固定验证重复使用；仅两组方法',
                         selected_is_validated=bool(self.state['stages']['expansion'].get('decision',{}).get('passed',False)))
        write_json(self.root/'selection.json',selection)
        self.persist()

    def evaluate(self, partial=False):
        if partial:
            if self.state['stage']!='paused' or not self.state.get('pause_boundary_verified'):
                raise ValueError('分阶段测试仅允许在用户指定的完整系数边界暂停后执行')
            jobs=[j for j in self.jobs if j['domain']=='window_random' and
                  (j['variant']=='baseline_data' or j['spec']['aux_weight'] in (.01,.03))]
            expected={case_key(c) for c in cases()}
            for variant,weight in [('baseline_data',None),('data_aux_ce',.01),('data_aux_ce',.03)]:
                group=[j for j in jobs if j['variant']==variant and (weight is None or j['spec']['aux_weight']==weight)]
                if len(group)!=36 or {case_key(j) for j in group}!=expected:
                    raise ValueError('两系数和数据对照必须各完整36模型后才能测试')
            selection=dict(partial_test=True,cohort_locked=True,decisions_locked=False,
                locked_at=now(),coefficients=[.01,.03],model_jobs=[j['job_id'] for j in jobs],
                scope='用户明确要求的两系数提前测试；全网格选择未完成，不根据本次测试选参',
                decision_rule=self.state['decision_rule'],historical_test_exposure=True)
        elif not self.state['decisions_locked']:
            raise ValueError('所有阶段结束并锁定决策前禁止测试')
        else:
            jobs=self.jobs
            selection=json.loads((self.root/'selection.json').read_text(encoding='utf-8'))
        out = self.root/('partial_evaluation_lambda_001_003' if partial else 'evaluation')
        if out.exists() or (self.state['target_test_accessed'] and (partial or self.state.get('partial_test_status')!='complete')):
            raise FileExistsError('测试仅执行一次；不覆盖或自动重测')
        self.setup()
        out.mkdir()
        write_json(out/'selection.json',selection)
        self.state.update(target_test_accessed=True,stage='partial_testing' if partial else 'testing')
        if partial:
            self.state.update(partial_test_status='running',partial_test_started_at=now(),
                              partial_test_user_authorized=True,partial_test_coefficients=[.01,.03])
        self.persist()
        metrics=[];evaluations=[]; cache={}
        if not partial and self.state.get('partial_test_status')=='complete':
            previous=pd.read_csv(self.root/'partial_evaluation_lambda_001_003/evaluations.csv')
            cache={(row['folder'],int(row['target_load_hp'])):(resolve_result_path(row['prediction_path']),row['latency_ms'])
                   for row in previous.to_dict('records')}
        try:
            for job in jobs:
                w = self.windows(job)
                source = job['source_load_hp']
                targets = [-1] if source==-1 else [t for t in range(4) if t!=source]
                for target in targets:
                    ix=np.flatnonzero(w.role.eq('test') if target==-1 else w.load_hp.eq(target))
                    tag='test_random' if target==-1 else f'test_hp{target}'
                    key=(job['folder'],target)
                    path=resolve_result_path(job['folder'])/tag/'predictions.npz'
                    if key in cache:
                        path,latency=cache[key]
                    elif path.exists():
                        latency=job['latency_ms']
                        cache[key]=(path,latency)
                    else:
                        model=engine.load(resolve_result_path(job['folder']),self.device)
                        arrays={'labels':w.iloc[ix].label.to_numpy(),'window_indices':w.iloc[ix].window_index.to_numpy()}
                        rpm=w.iloc[ix].rpm.to_numpy(np.float32)
                        for condition,seed,observed in R.conditions(self.x[ix],job['protocol'],True):
                            arrays[f'{condition}_n{seed}']=R.predict(model,observed,rpm,self.device)
                        dest=out/'predictions'/job['job_id']
                        dest.mkdir(parents=True,exist_ok=True)
                        path=dest/f'{tag}.npz'
                        np.savez_compressed(path,**arrays)
                        latency=job['latency_ms']
                        if latency is None:
                            latency=R.latency(model,self.raw[ix],rpm,self.device)
                        del model
                        cache[key]=(path,latency)
                    meta={k:v for k,v in job.items() if k not in ['spec','validation_metrics','validation_predictions']}
                    meta['target_load_hp']=target
                    with np.load(path) as saved:
                        np.testing.assert_array_equal(saved['window_indices'],w.iloc[ix].window_index)
                        np.testing.assert_array_equal(saved['labels'],w.iloc[ix].label)
                        expected={'clean_n0'}
                        if job['protocol']=='awgn_mixed':
                            expected|={f'white_{snr:+d}_n{seed}' for snr in [-4,-2,0,2,4] for seed in range(7201,7206)}
                        if set(saved.files)-{'labels','window_indices'} != expected:
                            raise ValueError('测试条件与固定协议不一致')
                        for name in sorted(expected):
                            condition,noise_seed=name.rsplit('_n',1)
                            values,_=scores(saved['labels'],saved[name])
                            metrics.append(dict(**meta,condition=condition,noise_seed=int(noise_seed),**values))
                    evaluations.append({**meta,'test_windows':len(ix),'prediction_path':str(path),'latency_ms':latency})
            pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
            pd.DataFrame(evaluations).to_csv(out/'evaluations.csv',index=False)
            suite_reporting.write_report(out)
            if partial:
                self.state.update(stage='paused',partial_test_status='complete',partial_test_completed_at=now(),
                                  partial_test_pairs=len(evaluations),partial_test_metric_rows=len(metrics))
            else:
                self.state.update(stage='complete',completed_at=now(),evaluated_pairs=len(evaluations),metric_rows=len(metrics))
            self.persist()
        except BaseException as error:
            self.state.update(stage='partial_test_failed' if partial else 'test_failed',error_type=type(error).__name__,error=str(error))
            if partial:
                self.state['partial_test_status']='failed'
            self.persist()
            raise


def print_plan():
    rows=[('1 lambda','0.01/0.03/0.1/0.3/1，10%和20%，两协议3子集3种子','144新模型；0.1与数据对照只读复用'),
          ('2 physics_fit','锁定系数，只改AWGN物理标准化拟合分布','18新模型；干净条件引用'),
          ('3 aux_head','固定训练观测共享头负梯度余弦比例>=20%才测试独立辅助头','触发时36新模型'),
          ('4 validation_noise','仅将选模噪声从1实例改为3固定实例，两方法共同更改','36新模型；干净条件引用'),
          ('可选 convergence','两组最多200epoch诊断；100后lr维持下限，不改变主方案','72新模型；仅init显式开启'),
          ('5 expansion','基础两组收益及退化保护通过后补50%及全样本','最多56；可复用完全匹配的数据基线'),
          ('6 transfer','扩展通过后，锁定方案、四来源五种子、12方向','最多80；探索性迁移'),
          ('evaluate','所有决策锁定后单独执行，报告全部成功与失败候选','固定测试噪声；历史预测只读复用')]
    for stage,task,count in rows:
        print(f'{stage}: {task}\n  {count}')
    print('默认只打印计划。未通过的优化不采用；后续诊断可以继续，扩展门槛不放宽。')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='plan',choices=['plan','init','status','run-next','run-all','run-autonomous','resume-autonomous','evaluate'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--include-convergence-diagnostic',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='plan':
        print_plan();return
    if args.output is None:
        parser.error('该命令需要--output指定独立optimization_*结果目录')
    if args.include_convergence_diagnostic and args.command!='init':
        parser.error('收敛诊断只能在init时预先声明')
    if args.command=='init':
        initialize(args.output,args.include_convergence_diagnostic);return
    suite=Suite(args.output)
    if args.command=='status':
        print(json.dumps({k:v for k,v in suite.state.items() if k!='jobs'},ensure_ascii=False,indent=2));return
    if args.command=='evaluate':
        suite.evaluate();return
    if args.command=='run-next':
        suite.run_next();return
    if args.command=='resume-autonomous':
        if suite.state['stage']!='paused' or not suite.state.get('pause_boundary_verified'):
            raise ValueError('只允许恢复用户请求且完整模型边界已核验的暂停')
        suite.state['resume_authorized']=True
        if suite.state['target_test_accessed']:
            if suite.state.get('partial_test_status')!='complete':
                raise ValueError('提前测试未完成，不能自动恢复')
            suite.state['continuation_after_partial_test_authorized']=True
    while not suite.state['decisions_locked']:
        suite.run_next()
        if suite.state['stage']=='paused':
            print('已完成指定阶段并暂停，保留结果，等待用户继续。',flush=True)
            return
    if args.command in ('run-autonomous','resume-autonomous'):
        suite.evaluate()
    else:
        print('所有阶段决策已锁定；测试尚未运行。使用evaluate执行一次统一测试。')
