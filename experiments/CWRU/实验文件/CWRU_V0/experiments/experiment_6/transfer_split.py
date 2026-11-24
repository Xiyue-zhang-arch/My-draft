"""全部40记录，来源内80/20划分后抽取共同73窗训练预算。"""
import numpy as np
import pandas as pd

SPLIT_SEED = 20261019
SELECTION_SEED = 5101
QUOTAS = (8, 17, 24, 24)


def split_source(original, source_load):
    if source_load not in range(4):
        raise ValueError('来源负载必须是0至3 HP')
    w = original.copy().reset_index(drop=True)
    w['historical_role'] = w.role
    w['load_hp'] = w.record_id.str.rsplit('_', n=1).str[-1].astype(int)
    w['role'] = 'unused'
    w['source_load_hp'] = source_load
    w['split_seed'] = SPLIT_SEED
    for rid, group in w.groupby('record_index', sort=True):
        if int(group.load_hp.iloc[0]) != source_load:
            continue
        ix = group.index.to_numpy()
        if len(ix)<2:
            raise ValueError('来源记录不足2完整窗')
        n = max(1,min(len(ix)-1,(8*len(ix)+5)//10))
        order = np.random.default_rng(np.random.SeedSequence([SPLIT_SEED,int(rid),8020])).permutation(ix)
        w.loc[order[:n],'role'] = 'train'
        w.loc[order[n:],'role'] = 'validation'
    w['candidate_role'] = w.role
    chosen=[]
    for label,count in enumerate(QUOTAS):
        rng=np.random.default_rng(np.random.SeedSequence([SELECTION_SEED,label,20261019]))
        cp=np.flatnonzero(w.role.eq('train') & w.label.eq(label))
        if len(cp)<count:
            raise ValueError('来源训练候选不足共同类别配额')
        records=rng.permutation(w.iloc[cp].record_index.unique())
        queues={rid:list(rng.permutation(cp[w.iloc[cp].record_index.to_numpy()==rid])) for rid in records}
        order=[]
        while any(queues.values()):
            for rid in records:
                if queues[rid]:order.append(queues[rid].pop())
        chosen.extend(order[:count])
    w.loc[w.role.eq('train'),'role']='unused_source_train'
    w.loc[sorted(chosen),'role']='train'
    assert len(w)==687 and w.window_index.is_unique and w.record_id.nunique()==40
    assert tuple(w.loc[w.role.eq('train')].groupby('label').size())==QUOTAS
    assert set(w.loc[w.role.isin(['train','validation']),'load_hp'])=={source_load}
    assert w.role.eq('validation').sum()==[19,39,40,40][source_load]
    for _,g in w.groupby('record_id'):
        g=g.sort_values('start_sample')
        assert (g.end_sample-g.start_sample).eq(24000).all()
        assert (g.start_sample.to_numpy()[1:]>=g.end_sample.to_numpy()[:-1]).all()
    return w


def for_target(windows,target):
    source=int(windows.source_load_hp.iloc[0])
    if target not in range(4) or target==source:
        raise ValueError('目标必须与来源不同')
    w=windows.copy()
    w['target_load_hp']=target
    w.loc[w.load_hp.eq(target),'role']='test'
    assert w.role.eq('test').sum()==[92,195,200,200][target]
    assert not set(w.loc[w.role.eq('test'),'record_id']) & set(w.loc[w.role.isin(['train','validation']),'record_id'])
    return w


def selected(windows,ratio,subset):
    if ratio!=1 or subset!=0:
        raise ValueError('仅共同73窗预算，本轮无小样本实验')
    ix=np.flatnonzero(windows.role.eq('train'))
    assert tuple(np.bincount(windows.iloc[ix].label,minlength=4))==QUOTAS
    return ix
