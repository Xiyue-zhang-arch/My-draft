"""Reuse frozen PU_0 numerics with protocol-specific counts and normalization."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'PU_0_all_conditions_721'

def main():
    provenance={}
    for name in ('encoder.py','model.py','descriptors.py','reference.py','raw_prepare.py'):
        payload=(PARENT/name).read_bytes(); (HERE/name).write_bytes(payload)
        provenance[name]=dict(original=str(PARENT/name),sha256=hashlib.sha256(payload).hexdigest(),byte_identical=True)
    common=(PARENT/'common.py').read_text(encoding='utf-8')
    common=common.replace('UPDATES_PER_EPOCH=175',"PROTOCOLS=('balanced10','imbalanced','matched_balanced')\nTRAIN_COUNTS={'balanced10':1120,'imbalanced':364,'matched_balanced':364}\nUPDATES_PER_EPOCH={'balanced10':18,'imbalanced':6,'matched_balanced':6}")
    start,end=common.index('def models():'),common.index('\n\ndef cases():')
    common=common[:start]+'''def models():
    result=[]
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            for method in METHODS:
                result.append(dict(model_id=f"m{len(result)+1:04d}",protocol=protocol,seed=seed,method=method))
    return result
'''+common[end:]
    (HERE/'common.py').write_text(common,encoding='utf-8')
    support=(PARENT/'support.py').read_text(encoding='utf-8')
    support=support.replace("'implementation_provenance.json')","'implementation_provenance.json','report_helpers.py')")
    support=support.replace('def load_bank(cache):','def load_bank(cache,protocol):')
    support=support.replace("cache/'normalization.json'","cache/('normalization_'+protocol+'.json')")
    support=support.replace("norm=norm,cache=cache)","norm=norm,cache=cache,normalization_path=cache/('normalization_'+protocol+'.json'))")
    support=support.replace("ids=split[role].copy()","ids=split['train_'+protocol if role=='train' else role].copy()")
    (HERE/'support.py').write_text(support,encoding='utf-8')
    engine=(PARENT/'engine.py').read_text(encoding='utf-8')
    engine=engine.replace('updates!=UPDATES_PER_EPOCH','updates!=UPDATES_PER_EPOCH[spec[\'protocol\']]')
    engine=engine.replace('updates=UPDATES_PER_EPOCH*epoch',"updates=UPDATES_PER_EPOCH[spec['protocol']]*epoch")
    engine=engine.replace("sha(bank['cache']/'normalization.json')","sha(bank['normalization_path'])")
    engine=engine.replace('def verify_all(out,completed,bank):','def verify_all(out,completed,banks):')
    engine=engine.replace('len(completed)!=30','len(completed)!=90').replace('All exact 30 models','All exact 90 models').replace('verified_count=30','verified_count=90')
    engine=engine.replace("('model_id','seed','method')","('model_id','protocol','seed','method')")
    engine=engine.replace("    for row in completed:\n","    for row in completed:\n        bank=banks[row['protocol']]\n")
    engine=engine.replace("row['updates']!=UPDATES_PER_EPOCH*EPOCHS","row['updates']!=UPDATES_PER_EPOCH[row['protocol']]*EPOCHS")
    engine=engine.replace("seed=row['seed']","seed=(row['protocol'],row['seed'])")
    engine=engine.replace("global_window_permutation_sha256=array_hash(ids)","global_window_permutation_sha256=array_hash(ids),ordinal_permutation_sha256=array_hash(permutation)")
    (HERE/'engine.py').write_text(engine,encoding='utf-8')
    report=(PARENT/'report.py').read_text(encoding='utf-8')
    helpers='import numpy as np\nfrom common import METHODS,SEEDS,SNRS,CLASS_NAMES,CONDITIONS\n\n'+report[report.index('def summarize(rows):'):report.index('def generate(out,figures=True):')]
    (HERE/'report_helpers.py').write_text(helpers,encoding='utf-8')
    provenance['engine.py']=dict(original=str(PARENT/'engine.py'),optimization_unchanged=True,
        changes=['90-model barrier','protocol-specific training sizes and updates','protocol-specific train-only normalization','trace grouping by protocol and seed'])
    (HERE/'implementation_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
