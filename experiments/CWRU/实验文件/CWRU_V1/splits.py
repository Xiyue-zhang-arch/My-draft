"""V1固定划分，全部40记录、全样本，不匹配训练预算。"""
import numpy as np
import pandas as pd

WINDOW_SEED = 20261018
LOAD_SEED = 20261019


def check(w):
    assert len(w)==687 and w.record_id.nunique()==40 and w.window_index.is_unique
    for role in ['train','validation','test']:
        if w.role.eq(role).any():
            assert set(w.loc[w.role.eq(role),'label'])==set(range(4))
    for _, group in w.groupby('record_index'):
        g=group.sort_values('start_sample')
        assert (g.end_sample-g.start_sample).eq(24000).all()
        assert np.all(g.start_sample.to_numpy()[1:]>=g.end_sample.to_numpy()[:-1])


def base(original):
    w=original.copy().reset_index(drop=True)
    w['historical_role']=w.role
    w['load_hp']=w.record_id.str.rsplit('_',n=1).str[-1].astype(int)
    w['role']='unused'
    return w


def window_split(original):
    w=base(original)
    w['split_seed']=WINDOW_SEED
    rng=np.random.default_rng(np.random.SeedSequence([WINDOW_SEED,701515]))
    record_ids=rng.permutation(np.sort(w.record_index.unique()))
    extra_val=bool(rng.integers(0,2))
    for rid in record_ids:
        ix=w.index[w.record_index.eq(rid)].to_numpy()
        n=len(ix)
        if n<2:
            raise ValueError('每记录至少2窗')
        nt=max(1,min(n-1,(7*n+5)//10))
        nv=(n-nt)//2
        if (n-nt)%2:
            nv+=int(extra_val)
            extra_val=not extra_val
        shuffled=np.random.default_rng(np.random.SeedSequence([WINDOW_SEED,int(rid),24000])).permutation(ix)
        w.loc[shuffled[:nt],'role']='train'
        w.loc[shuffled[nt:nt+nv],'role']='validation'
        w.loc[shuffled[nt+nv:],'role']='test'
    check(w)
    return w


def source_split(original,source):
    w=base(original)
    w['split_seed']=LOAD_SEED
    for rid,g in w.groupby('record_index',sort=True):
        if int(g.load_hp.iloc[0])!=source:
            continue
        ix=g.index.to_numpy()
        nt=max(1,min(len(ix)-1,(8*len(ix)+5)//10))
        order=np.random.default_rng(np.random.SeedSequence([LOAD_SEED,int(rid),8020])).permutation(ix)
        w.loc[order[:nt],'role']='train'
        w.loc[order[nt:],'role']='validation'
    check(w)
    assert w.role.eq('train').sum()==[73,156,160,160][source]
    assert w.role.eq('validation').sum()==[19,39,40,40][source]
    assert set(w.loc[w.role.isin(['train','validation']),'load_hp'])=={source}
    return w


def save_splits(original,records,destination):
    destination.mkdir(parents=True,exist_ok=False)
    tables={'window_random':window_split(original)}
    tables.update({f'load_source_{s}':source_split(original,s) for s in range(4)})
    usage=[]; supports=[]
    for name,w in tables.items():
        folder=destination/name
        folder.mkdir()
        w.to_csv(folder/'windows.csv',index=False)
        role_indices={role:w.loc[w.role.eq(role),'window_index'].to_list() for role in ['train','validation']}
        if name=='window_random':
            role_indices['test']=w.loc[w.role.eq('test'),'window_index'].to_list()
        else:
            source=int(name[-1])
            for target in range(4):
                if target!=source:
                    role_indices[f'test_hp{target}']=w.loc[w.load_hp.eq(target),'window_index'].to_list()
        for role,ids in role_indices.items():
            (folder/f'{role}.txt').write_text(''.join(f'{i}\n' for i in ids),encoding='utf-8')
            a=w.loc[w.window_index.isin(ids)]
            counts=np.bincount(a.label,minlength=4)
            supports.append(dict(split=name,role=role,windows=len(a),**dict(zip(['N','IR','OR','B'],map(int,counts)))))
            for _,record in records.iterrows():
                used=a.loc[a.record_id.eq(record.record_id)]
                usage.append(dict(split=name,role=role,record_id=record.record_id,raw_file=record.path,
                    DE_variable=record.variable,label=int(record.label),load_hp=int(record.record_id.rsplit('_',1)[1]),
                    raw_samples=int(record.samples),complete_windows=int(original.record_id.eq(record.record_id).sum()),
                    unused_tail_samples=int(record.samples%24000),used_windows=len(used)))
    pd.DataFrame(usage).to_csv(destination.parent/'dataset_file_usage.csv',index=False)
    pd.DataFrame(supports).to_csv(destination.parent/'dataset_support.csv',index=False)
    return tables


def read_fixed(folder):
    w=pd.read_csv(folder/'windows.csv')
    check(w)
    for role in ['train','validation']:
        saved=np.loadtxt(folder/f'{role}.txt',dtype=int,ndmin=1)
        np.testing.assert_array_equal(saved,w.loc[w.role.eq(role),'window_index'].to_numpy())
    for p in folder.glob('test*.txt'):
        saved=np.loadtxt(p,dtype=int,ndmin=1)
        expected=w.loc[w.role.eq('test') if p.stem=='test' else w.load_hp.eq(int(p.stem[-1])),'window_index']
        np.testing.assert_array_equal(saved,expected.to_numpy())
    return w
