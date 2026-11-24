"""Synthetic end-to-end report/label/probability consistency check."""
from pathlib import Path
import json
from uuid import uuid4

import torch  # Windows: Torch must precede NumPy/Pandas.
import numpy as np
import pandas as pd

from .run import CONDITIONS, RESULTS, append_csv, scores
from .report import finish


def main():
    RESULTS.mkdir(exist_ok=True)
    root=RESULTS/f'report_check_{uuid4().hex}'
    root.mkdir()
    dest=root/'evaluation'
    (dest/'predictions').mkdir(parents=True)
    entries,metrics,cells=[],[],[]
    for scene,source,target in [('window',-1,-1),('load',0,1)]:
        domain='window_random' if scene=='window' else 'load_source_0'
        table=pd.DataFrame(dict(window_index=np.arange(8),label=np.tile(np.arange(4),2),
            role=['test']*8,load_hp=[target]*8))
        split=root/'splits'/f'{domain}_r010_s4101'
        split.mkdir(parents=True)
        table.to_csv(split/'windows.csv',index=False)
        for method in ('fusion','data'):
            jobid=f'{scene}_{method}'
            labels=table.label.to_numpy()
            arrays=dict(labels=labels,window_indices=np.arange(8))
            meta=dict(job_id=jobid,scene=scene,source_load_hp=source,target_load_hp=target,
                subset=4101,protocol='clean_only',seed=1706,method=method,domain=domain,ratio=.1)
            for c,s in CONDITIONS:
                p=np.full((8,4),.05)
                p[np.arange(8),labels]=.85
                if method=='data':
                    p[0]=[.05,.85,.05,.05]
                arrays[f'{c}_n{s}']=p
                values,counts=scores(labels,p)
                info=dict(meta,condition=c,noise_seed=s)
                metrics.append(dict(info,**values))
                for a in range(4):
                    for b in range(4):
                        cells.append(dict(info,true_label=a,predicted_label=b,count=int(counts[a,b]),
                            support=int(counts[a].sum()),row_fraction=float(counts[a,b]/counts[a].sum())))
            path=dest/'predictions'/f'{jobid}.npz'
            np.savez_compressed(path,**arrays)
            entries.append(dict(meta,prediction_path=str(path),test_windows=8))
    for name,rows in [('evaluations.csv',entries),('metrics.csv',metrics),('confusion_matrices.csv',cells)]:
        append_csv(dest/name,rows)
    finish(root,expected_counts=(4,104))
    paired=pd.read_csv(root/'report/paired_differences.csv')
    assert np.allclose(paired.acc_difference,.125)
    summary=pd.read_csv(root/'report/summary.csv')
    assert (summary[summary.method.eq('fusion')].balanced_acc_mean==1).all()
    cm=pd.read_csv(root/'report/mean_normalized_confusion.csv')
    assert np.allclose(cm.groupby(['scene','protocol','method','condition','true_label']).row_fraction.sum(),1)
    # The verifier must reject a manipulated metric without trusting the report.
    frame=pd.read_csv(dest/'metrics.csv')
    frame.loc[0,'acc']=.123
    frame.to_csv(dest/'metrics.csv',index=False)
    try:
        finish(root,expected_counts=(4,104))
    except AssertionError:
        pass
    else:
        raise AssertionError('Tampered metrics accepted')
    print(json.dumps(dict(passed=True,scope='synthetic_report_only',path=str(root),
        verifies_paired_difference=True,rejects_tampered_metric=True)))


if __name__=='__main__':
    main()
