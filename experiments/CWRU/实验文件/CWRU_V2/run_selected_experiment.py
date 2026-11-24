"""固定混合拟合，补齐30轮系数验证，再执行锁定配置的完整实验。"""
import sys
sys.dont_write_bytecode = True
import argparse
import copy
import json
import traceback
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from optimization.runner import Suite, initialize, effective_spec, now, write_json, PROJECT
from optimization import design, reporting

from experiment_paths import resolve_result_path

QUICK = PROJECT/'experiments/CWRU/实验结果/CWRU_V2/quick_20261022'
PROTOCOL = Path(__file__).with_name('SELECTED_PROTOCOL.md')


class SelectionSuite(Suite):
    def old_folder(self, variant, spec, case):
        if variant == 'data_aux_ce' and spec['max_epochs'] == 30 and case['subset'] == 4101:
            state = json.loads((QUICK/'config.json').read_text(encoding='utf-8'))
            if state['stage'] != 'complete':
                raise ValueError('短程参照未完成')
            for job in state['jobs']:
                if all(job[k] == case[k] for k in ['ratio','subset','protocol','seed']) and effective_spec(variant,case['protocol'],job['spec']) == effective_spec(variant,case['protocol'],spec):
                    return resolve_result_path(job['folder'])
        return super().old_folder(variant, spec, case)

    def historical_cost(self, folder, variant, case):
        if QUICK in folder.parents:
            info = json.loads((folder/'parameters.json').read_text(encoding='utf-8'))
            return {k: info.get(k) for k in ['latency_ms','conv_linear_flops']}
        return super().historical_cost(folder, variant, case)


def prepare(path, stage, spec, planned):
    initialize(path)
    suite = SelectionSuite(path)
    suite.state.update(stage=stage,current_spec=spec,stage_order=[stage],
                       stages={stage:dict(status='running')},planned_model_cases=planned,
                       user_authorized_complete_experiment=True)
    (suite.root/'PROTOCOL.md').write_text(PROTOCOL.read_text(encoding='utf-8'),encoding='utf-8')
    suite.persist()
    return suite


def select(suite):
    spec = suite.state['current_spec']
    data = suite.data_control(spec,'screen','lambda_selection')
    ids = []
    for weight in design.LAMBDA_GRID:
        ids.append(suite.ensure_config('data_aux_ce',{**spec,'aux_weight':weight},'screen','lambda_selection'))
    metrics = suite.metrics([data,*ids])
    decision = design.select_lambda(metrics,ids,design.config_id('data_aux_ce',spec),data)
    reporting.write_validation_report(suite.root/'stages/lambda_selection',metrics,decision)
    suite.write_usage()
    suite.state.update(stage='complete',completed_at=now(),decisions_locked=True,
                       selected_aux_weight=decision['selected_aux_weight'],selection=decision)
    suite.state['stages']['lambda_selection'] = dict(status='complete',decision=decision)
    suite.persist()
    write_json(suite.root/'selection.json',decision)
    return decision['selected_aux_weight']


def complete(suite):
    spec = suite.state['current_spec']
    for kind in ['screen','expansion','transfer']:
        suite.state['stage'] = 'full_'+kind
        suite.persist()
        for variant in ['baseline_data','data_aux_ce']:
            suite.ensure_config(variant,spec,kind,'full_'+kind)
        ids = [design.config_id(v,spec) for v in ['baseline_data','data_aux_ce']]
        reporting.write_validation_report(suite.root/'stages'/kind,suite.metrics(ids,kind),
            dict(selected_aux_weight=spec['aux_weight'],configuration_fixed=True,
                 scope='完整评价，无论对照优劣均保留；验证结果不再改变系数'))
    suite.write_usage()
    suite.state.update(decisions_locked=True,decisions_locked_at=now(),stage='decisions_locked')
    write_json(suite.root/'selection.json',dict(selected_spec=spec,decisions_locked=True,
        historical_test_exposure=True,selection_source=suite.state['selection_source'],
        methodology='30轮固定验证选参；100轮独立从头完整训练；不根据测试修改配置'))
    suite.persist()
    suite.evaluate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-output',type=Path,required=True)
    parser.add_argument('--full-output',type=Path,required=True)
    args = parser.parse_args()
    if args.selection_output.resolve() == args.full_output.resolve() or args.full_output.exists():
        raise FileExistsError('两个阶段必须使用不同的全新结果目录')
    short = {**design.DEFAULT_SPEC,'physics_fit':'matched_noise','max_epochs':30}
    suite = prepare(args.selection_output,'lambda_selection',short,216)
    with (suite.root/'training_console.log').open('x',encoding='utf-8',buffering=1) as log, (suite.root/'training_errors.log').open('x',encoding='utf-8',buffering=1) as err:
        with redirect_stdout(log), redirect_stderr(err):
            try:
                weight = select(suite)
                full = prepare(args.full_output,'full_training',{**short,'aux_weight':weight,'max_epochs':100},208)
                full.state['selection_source'] = str(suite.root/'selection.json')
                full.persist()
                suite = full
                complete(full)
            except BaseException as error:
                suite.state.update(stage='failed',error=str(error),error_type=type(error).__name__)
                suite.persist()
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
