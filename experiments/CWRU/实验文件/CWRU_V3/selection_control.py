"""等待已启动进程完成30模型，停止后续训练，锁定lambda、测试本批并暂停。"""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import os
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
import psutil


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CWRU_V0/shared'))
from experiment_paths import resolve_result_path

BARRIER_TEXT = 'V3 user-requested stop after 30 selection models; control barrier, not a model.\n'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix+'.controller_pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(pending, path)


def check_cohort(root, state):
    """核验完整30模型及落盘文件，不读取测试或模型参数内容。"""
    if state.get('version') != 'CWRU_V3' or state.get('target_test_accessed'):
        raise ValueError('仅允许尚未测试的V3选参阶段')
    jobs = state['jobs']
    expected = {('baseline_data', 0., seed) for seed in range(1706,1711)}
    expected |= {('data_aux_ce', weight, seed) for weight in (.01,.03,.1,.3,1.) for seed in range(1706,1711)}
    actual = {(j['variant'],j['aux_weight'],j['seed']) for j in jobs}
    if len(jobs)!=30 or actual!=expected:
        raise ValueError('必须完整登记数据对照5模型及五档lambda各5模型')
    folders = set()
    for j in jobs:
        if not (j['stage']=='lambda_selection' and j['domain']=='window_random' and j['ratio']==1.
                and j['subset']==0 and j['protocol']=='clean_only' and j['max_epochs']==200):
            raise ValueError('模型不属于本批全样本纯干净200轮选参')
        folder = resolve_result_path(j['folder']).resolve()
        if folder.parent != (root/'models').resolve():
            raise ValueError('本批只能使用当前V3输出内的模型')
        folders.add(folder)
        for name in ['model.pt','parameters.json','training.csv','training_spec.json',
                     'selected_windows.csv','validation_metrics.csv','validation_predictions.npz']:
            if not (folder/name).is_file() or (folder/name).stat().st_size == 0:
                raise ValueError(f'模型尚未完整保存: {folder/name}')
        info = read_json(folder/'parameters.json')
        if not info.get('completed') or info['seed']!=j['seed'] or info['validation_nll']!=j['validation_nll']:
            raise ValueError('模型完成状态与登记不一致')
    actual_folders = {p.resolve() for p in (root/'models').iterdir() if p.is_dir()}
    if actual_folders != folders:
        raise ValueError('存在额外或未完成模型目录，保留现场，不声称精确30模型边界')
    return jobs


def summarize(out, jobs, candidates, weight):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from metrics import METRICS
    import report
    m = pd.read_csv(out/'metrics.csv')
    if len(m)!=30 or m.job_id.nunique()!=30 or not m.condition.eq('clean').all():
        raise ValueError('本批须恰为30组干净测试指标')
    summary = report.aggregate(m, ['variant','aux_weight'], out/'mean_std.csv')
    data = m[m.variant.eq('baseline_data')].set_index('seed')
    paired = []
    for row in m[m.variant.eq('data_aux_ce')].to_dict('records'):
        baseline = data.loc[row['seed']]
        paired.append(dict(aux_weight=row['aux_weight'], seed=row['seed'],
                           **{metric:row[metric]-float(baseline[metric]) for metric in METRICS}))
    diff = pd.DataFrame(paired)
    diff.to_csv(out/'paired_seed_differences.csv', index=False)
    report.aggregate(diff, ['aux_weight'], out/'paired_mean_std.csv')
    val = report.validation_table(jobs)
    val.to_csv(out/'validation_metrics.csv',index=False)
    vs = val.groupby(['variant','aux_weight'])[METRICS+['validation_nll']].agg(['mean','std','count'])
    vs.columns = ['_'.join(c) for c in vs.columns]
    vs.reset_index().to_csv(out/'validation_mean_std.csv',index=False)
    sensitivity = pd.DataFrame(candidates)
    sensitivity.to_csv(out/'lambda_validation_sensitivity.csv',index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.errorbar(sensitivity.aux_weight, sensitivity.mean_best_validation_ce,
                yerr=sensitivity.sample_sd, marker='o', capsize=3)
    ax.set(xscale='log',xlabel='Auxiliary CE weight',ylabel='Validation CE (mean +/- sample SD)')
    fig.tight_layout();fig.savefig(out/'lambda_validation_sensitivity.png',dpi=180);plt.close(fig)
    cm = pd.read_csv(out/'confusion_matrices.csv')
    pooled = cm.groupby(['variant','aux_weight','true_label','predicted_label'],as_index=False)['count'].sum()
    pooled['prediction_support'] = pooled.groupby(['variant','aux_weight','true_label'])['count'].transform('sum')
    pooled['row_percent'] = 100*pooled['count']/pooled.prediction_support
    pooled.to_csv(out/'pooled_confusion_matrices.csv',index=False)
    costs = pd.DataFrame(jobs)
    keep = ['job_id','variant','aux_weight','seed','training_windows','validation_windows','epochs_trained',
            'updates_trained','best_epoch','parameters','conv_linear_flops','fit_seconds','train_seconds_per_epoch',
            'total_train_seconds','validation_seconds','peak_gpu_mib','latency_ms','hardware','precision','batch_size','latency_scope']
    costs[keep].to_csv(out/'computational_cost.csv',index=False)
    display = summary[['variant','aux_weight','acc_mean','acc_std','balanced_acc_mean','balanced_acc_std',
                       'f1_macro_mean','f1_macro_std','recall_N_mean','recall_IR_mean','recall_OR_mean','recall_B_mean']]
    report.exclusive(out/'REPORT.md', '# V3基础选参及本批测试报告\n\n'
        f'完成30个全样本纯干净模型。每模型最多200epoch、patience20、每配置5训练种子。'
        f'统一λ为{weight}，已在测试前根据五种子最佳验证CE的均值锁定。选参规则和网格均未根据测试调整。\n\n'
        '用户在训练期间明确授权这30模型完成后先测试并结束当日实验。本报告覆盖数据单分支5模型及五档λ各5模型。'
        '使用固定103个干净测试窗口，共30组预测，不增加AWGN测试。今天到此暂停，后续实验未运行。\n\n'
        '## 验证选参\n\n'+sensitivity.to_markdown(index=False)+'\n\n'
        '## 固定干净测试\n\n'+display.round(6).to_markdown(index=False)+'\n\n'
        '所有指标使用0至1比例，std为五种子的样本标准差。各类召回的完整均值和标准差见mean_std.csv。'
        '原始混淆计数、每类支持数与逐行百分比见confusion_matrices.csv。五种子累加的预测支持数不能称独立测试样本数。'
        '每档λ均与同一组数据对照按种子配对，不重复计算五套数据对照，不按测试排序重新选λ。\n\n'
        '逐轮记录、最优权重、配置及所选窗口保留在原模型目录。成本见computational_cost.csv；延迟包含标准化、'
        '在线物理特征和网络，不含磁盘/H2D；10次预热、50次重复。Conv1d/Linear FLOPs不包含FFT。\n\n'
        '本阶段已提前访问V3测试，后续即使恢复也不能称完整实验前从未访问测试。统一λ保持固定。未来完整报告只能只读复用'
        '本批选中λ及数据对照的10份测试预测，不重新测试或覆盖本批输出。其余20份作为已曝光的敏感性结果单独保留。\n\n'
        +report.LIMITS)


def finish_selection(root, state):
    # 在原训练进程已停止后，使用新的进程完成验证锁定与一次性测试。
    import torch
    import numpy as np
    import pandas as pd
    from context import R, PROJECT, BASE
    from run_experiment import select_lambda, split_name
    from splits import read_fixed
    from metrics import scores
    import engine
    root = Path(root).resolve()
    jobs = check_cohort(root, state)
    out = root/'selection_evaluation'
    if out.exists():
        raise FileExistsError('本批测试目录已存在，不自动重测或覆盖')
    weight, candidates = select_lambda(jobs)
    if state.get('lambda_locked') and state['selected_aux_weight']!=weight:
        raise ValueError('已锁定lambda与验证规则复算不一致')
    if (root/'selection.json').exists():
        previous=read_json(root/'selection.json')
        if previous['selected_aux_weight']!=weight or previous['candidates']!=candidates:
            raise ValueError('已有选参结果与原验证规则不一致，禁止覆盖')
    out.mkdir()
    decision = dict(selected_aux_weight=weight,candidates=candidates,
        selection_rule=state['selection_rule'],locked_at=now(),lambda_locked=True,
        cohort_locked=True,decisions_locked=False,test_accessed=False,
        cohort=[j['job_id'] for j in jobs],scope='30 full-sample clean models; fixed103 clean test only')
    atomic_json(out/'selection.json',decision)
    state.update(stage='selection_test_preparing', selected_aux_weight=weight, lambda_locked=True,
        lambda_locked_at=state.get('lambda_locked_at',now()), selection_test_status='preparing',
        selection_test_user_authorized=True, pause_after_stage='selection_test',automatic_resume=False)
    atomic_json(root/'config.json',state)
    fixed = read_fixed(root/'splits'/split_name('window_random',1.,0))
    historical = read_fixed(BASE/'splits/window_random')
    pd.testing.assert_frame_equal(fixed,historical)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    R.U.DEVICE=device;R.U.configure()
    _, original, raw = R.load_dataset(PROJECT/'data/raw/CWRU/48k Drive End Bearking Fault Data')
    for key in ['window_index','record_id','label','start_sample','end_sample']:
        np.testing.assert_array_equal(fixed[key],original[key])
    ix = np.flatnonzero(fixed.role.eq('test'))
    if len(ix)!=103:
        raise ValueError('固定测试窗口必须为103')
    x = R.U.normalize(raw[ix]);labels=fixed.iloc[ix].label.to_numpy();rpm=fixed.iloc[ix].rpm.to_numpy(np.float32)
    (out/'predictions').mkdir()
    state.update(stage='selection_testing',target_test_accessed=True,selection_test_status='running',
                 selection_test_started_at=now(),updated_at=now())
    atomic_json(root/'config.json',state)
    for index,j in enumerate(jobs,1):
        model = engine.load(j['folder'],device)
        p = R.predict(model,x,rpm,device)
        del model
        meta = {k:j[k] for k in ['job_id','variant','aux_weight','seed']}
        meta.update(condition='clean',noise_seed=0)
        np.savez_compressed(out/'predictions'/f'{j["job_id"]}.npz',labels=labels,
                            window_indices=fixed.iloc[ix].window_index.to_numpy(),clean_n0=p)
        values,cm=scores(labels,p)
        rows=[]
        for a in range(4):
            for b in range(4):
                rows.append(dict(meta,true_label=a,predicted_label=b,count=int(cm[a,b]),
                                 support=int(cm[a].sum()),row_percent=float(100*cm[a,b]/cm[a].sum())))
        for filename,frame in [('metrics.csv',pd.DataFrame([dict(meta,**values)])),
                               ('confusion_matrices.csv',pd.DataFrame(rows))]:
            path=out/filename
            frame.to_csv(path,index=False,mode='a',header=not path.exists())
        state.update(selection_test_models=index,updated_at=now())
        atomic_json(root/'config.json',state)
        print('SELECTION TEST COMPLETE',j['job_id'],flush=True)
    summarize(out,jobs,candidates,weight)
    state.pop('active_model',None)
    state.update(stage='paused',paused_at=now(),updated_at=now(),selection_test_status='complete',
        selection_test_completed_at=now(),selection_test_models=30,pause_boundary_verified=True,
        pause_reason='user_requested_end_after_selection_and_30_model_test',
        selection_test_report=str(out/'REPORT.md'),decisions_locked=False,automatic_resume=False)
    atomic_json(root/'config.json',state)


def watch(root, pid):
    root = Path(root).resolve()
    parent = Path(__file__).resolve().parents[4]/'experiments/CWRU/实验结果/CWRU_V3'
    if root.parent != parent.resolve():
        raise ValueError('只允许V3结果直接子目录')
    process = psutil.Process(pid)
    command = process.cmdline()
    if ('--output' not in command or Path(command[command.index('--output')+1]).resolve()!=root
            or not any(Path(c).name=='run_experiment.py' for c in command)):
        raise ValueError('PID与本轮V3入口和输出不符')
    created = process.create_time()
    request_path = root/'selection_test_request.json'
    if request_path.exists():
        raise FileExistsError('已存在控制请求，禁止重复监控')
    initial=read_json(root/'config.json')
    if initial['stage']!='lambda_selection' or len(initial['jobs'])>=30:
        raise ValueError('必须在选参30模型完成前安装停止控制')
    # 原进程已经载入旧代码。即使监控延迟或退出，训练器也会因该路径已存在而在第31模型训练前拒绝。
    barrier=root/'models/m0031'
    with barrier.open('x',encoding='utf-8') as f:
        f.write(BARRIER_TEXT)
    request = dict(status='watching',requested_at=now(),controller_pid=os.getpid(),training_pid=pid,
        process_created=created,expected_models=30,test_scope='fixed 103 clean windows for all 30 models',
        stop_after_testing=True,automatic_resume=False,training_barrier=str(barrier))
    atomic_json(request_path,request)
    print('WAITING FOR 30 COMPLETE SELECTION MODELS',flush=True)
    try:
        while True:
            try:
                state=read_json(root/'config.json')
            except (json.JSONDecodeError,PermissionError):
                time.sleep(.02);continue
            alive=process.is_running()
            if alive and process.create_time()!=created:
                raise RuntimeError('训练PID发生复用，禁止控制其他进程')
            expected_barrier_stop=(state.get('stage')=='failed'
                and state.get('error_type')=='FileExistsError' and state.get('error')==str(barrier)
                and state.get('active_model',{}).get('job_id')=='m0031'
                and barrier.read_text(encoding='utf-8')==BARRIER_TEXT)
            if state.get('stage')=='failed' and not expected_barrier_stop:
                raise RuntimeError('原训练失败，不自动重启或进行不完整选参测试')
            if len(state['jobs'])<30:
                if not alive:
                    raise RuntimeError('原训练进程在完整30模型边界前退出，不自动恢复或测试')
                time.sleep(.02 if len(state['jobs'])>=29 else 1.)
                continue
            if not alive and not expected_barrier_stop:
                raise RuntimeError('训练进程异常退出，不自动将其转换为正常暂停')
            if alive:
                try:
                    process.suspend()
                except psutil.NoSuchProcess:
                    state=read_json(root/'config.json')
                    if not (state.get('error_type')=='FileExistsError' and state.get('error')==str(barrier)):
                        raise
            # 到达边界后先结束旧全流程，禁止后续训练。所有正式模型必须已登记完整。
            state=read_json(root/'config.json')
            if process.is_running():
                process.terminate();process.wait(timeout=30)
            check_cohort(root,state)
            state.pop('active_model',None)
            if state.get('error_type')=='FileExistsError' and state.get('error')==str(barrier):
                state['training_stop_via_barrier']=True
                state.pop('error',None);state.pop('error_type',None)
            state.update(stage='selection_training_complete',pause_boundary_verified=True,
                         training_stopped_at=now(),automatic_resume=False)
            atomic_json(root/'config.json',state)
            request.update(status='testing',training_stopped_at=now(),boundary_verified=True)
            atomic_json(request_path,request)
            finish_selection(root,state)
            request.update(status='complete_paused',completed_at=now(),tested_models=30)
            atomic_json(request_path,request)
            print('30 MODEL TEST COMPLETE; PAUSED; NO AUTOMATIC RESUME',flush=True)
            return
    except BaseException as error:
        request.update(status='failed',error=str(error),updated_at=now())
        atomic_json(request_path,request)
        # 已停止训练后的失败写真实状态；仍运行时不与训练进程争写config。
        if not process.is_running():
            state=read_json(root/'config.json')
            state.update(stage='selection_control_failed',error=str(error),updated_at=now(),automatic_resume=False)
            atomic_json(root/'config.json',state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pid',type=int,required=True)
    args=parser.parse_args()
    from contextlib import redirect_stdout,redirect_stderr
    with (args.output/'selection_control.log').open('x',encoding='utf-8',buffering=1) as log:
        with redirect_stdout(log),redirect_stderr(log):
            try:watch(args.output,args.pid)
            except BaseException:
                traceback.print_exc()
                raise


if __name__=='__main__':
    main()
