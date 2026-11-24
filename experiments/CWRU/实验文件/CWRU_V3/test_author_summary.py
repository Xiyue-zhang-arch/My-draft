"""仅用合成数据测试统计、负差保留及不完整实验的读取保护。"""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
import torch  # Windows DLL initialization must precede numpy/pandas.
import numpy as np
import pandas as pd
import author_comparison_summary as s

@contextmanager
def workspace():
    base=Path(tempfile.gettempdir()).resolve();root=base/('author_summary_sim_'+uuid.uuid4().hex);root.mkdir()
    try:yield root
    finally:
        if root.parent!=base or not root.name.startswith('author_summary_sim_'):raise ValueError('临时目录不安全')
        shutil.rmtree(root)

def rows(method,sample):
    out=[]
    for ratio in ([.1,.2,.5] if sample=='small' else [1.]):
        for subset in ([4101,4102,4103] if sample=='small' else [0]):
            for seed in range(1706,1711):
                for src in range(4):
                    for tgt in range(4):
                        if src==tgt:continue
                        val=.7+(seed-1706)*.001+(.1 if method=='DWCN_SDGFD' else 0)
                        row=dict(domain=f'load_source_{src}',ratio=ratio,subset=subset,seed=seed,
                           source_load_hp=src,target_load_hp=tgt,protocol='clean_only',variant=method,
                           condition='clean',noise_seed=0)
                        row.update({k:val for k in s.METRICS});out.append(row)
    return pd.DataFrame(out)

class SummaryTests(unittest.TestCase):
    def test_seed_sd_and_negative_pair(self):
        f=pd.concat([rows(m,'small') for m in s.ORDER],ignore_index=True)
        result=s.aggregated(f,['ratio','subset','variant'])
        row=result[(result.ratio==.1)&(result.subset==4101)&(result.variant=='DWCN_SDGFD')].iloc[0]
        self.assertAlmostEqual(row.balanced_acc_std,np.std(np.arange(5)*.001,ddof=1))
        diff=s.paired(f)
        self.assertTrue((diff[diff.comparator=='DWCN_SDGFD'].balanced_acc<0).all())

    def test_missing_direction_rejected(self):
        f=rows('DWCN_SDGFD','small').iloc[1:]
        with self.assertRaises(ValueError):s.aggregated(f,['ratio','subset','variant'])

    def test_incomplete_preflight_before_any_prediction_read(self):
        with workspace() as root:
            oldsmall={m:root/('small_'+m) for m in ['WDCNN','DRSN_CW']}
            oldfull={m:root/('full_'+m) for m in ['WDCNN','DRSN_CW']}
            argv=['summary','--dwcn-small',str(root/'ds'),'--dwcn-full',str(root/'df'),
                  '--classbd-small',str(root/'cs'),'--classbd-full',str(root/'cf'),
                  '--output',str(root/'author_comparison_summary_test')]
            def check(path,method,sample):
                if method=='DWCN_SDGFD':raise ValueError('incomplete')
                return dict(stage='complete',trained_models=180,evaluated_pairs=540,metric_rows=540)
            with patch.object(s,'BASE',root),patch.object(s,'OLD_SMALL',oldsmall),patch.object(s,'OLD_FULL',oldfull), \
                 patch.object(s,'preflight',side_effect=check),patch.object(s.verify,'_verify') as reader,patch.object(sys,'argv',argv):
                with self.assertRaises(ValueError):s.main()
                reader.assert_not_called()
            self.assertFalse((root/'author_comparison_summary_test').exists())

    def test_full_synthetic_report(self):
        with workspace() as root:
            oldsmall={m:root/('small_'+m) for m in ['WDCNN','DRSN_CW']}
            oldfull={m:root/('full_'+m) for m in ['WDCNN','DRSN_CW']}
            ds,df,cs,cf=[root/i for i in ['ds','df','cs','cf']]
            mapping={**{p:(m,'small') for m,p in oldsmall.items()},**{p:(m,'full') for m,p in oldfull.items()},
                     ds:('DWCN_SDGFD','small'),df:('DWCN_SDGFD','full'),cs:('ClassBD_WDCNN','small'),cf:('ClassBD_WDCNN','full')}
            sr=root/'reference_small';fr=root/'reference_full'
            def fake_verify(path,ratios,models,variants,split_root):
                if path in (sr,fr):return pd.concat([rows(m,'small' if path==sr else 'full') for m in s.verify.REFERENCE_METHODS])
                return rows(*mapping[path])
            def check(path,method,sample):
                n=180 if sample=='small' else 20
                return dict(stage='complete',trained_models=n,evaluated_pairs=3*n,metric_rows=3*n)
            argv=['summary','--dwcn-small',str(ds),'--dwcn-full',str(df),'--classbd-small',str(cs),
                  '--classbd-full',str(cf),'--output',str(root/'author_comparison_summary_test')]
            with patch.object(s,'BASE',root),patch.object(s,'OLD_SMALL',oldsmall),patch.object(s,'OLD_FULL',oldfull), \
                 patch.object(s,'SMALL_REF',sr),patch.object(s,'FULL_REF',fr),patch.object(s,'preflight',side_effect=check), \
                 patch.object(s,'config',return_value=dict(stage='complete')),patch.object(s.verify,'_verify',side_effect=fake_verify), \
                 patch.object(s.verify,'_validate_splits',return_value=40), \
                 patch.object(s,'detect_degenerate',return_value=pd.DataFrame([dict(job_id='m1',degenerate_single_class=False,predicted_classes='0,1')])), \
                 patch.object(sys,'argv',argv):s.main()
            out=root/'author_comparison_summary_test'
            self.assertTrue((out/'REPORT.md').is_file())
            p=pd.read_csv(out/'report/full_paired_direction_seed_mean_sd.csv')
            self.assertTrue((p[p.comparator=='DWCN_SDGFD'].balanced_acc_mean<0).all())
            self.assertEqual(len(json.loads((out/'verification.json').read_text(encoding='utf8'))),8)

if __name__=='__main__':unittest.main()
