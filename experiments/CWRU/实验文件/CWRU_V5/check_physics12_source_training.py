"""Read-only support check on source-0 TRAIN windows, no labels used for decisions."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from physics12_adapted_reference import extract


def run():
    project = Path(__file__).resolve().parents[4]
    frozen = project/'experiments/CWRU/实验结果/CWRU_V5/load_clean_20261109_142434'
    with (frozen/'splits/load_source_0_r100_s0/windows.csv').open(encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['role'] == 'train']
    assert len(rows) == 73 and all(r['load_hp'] == '0' for r in rows)
    with (frozen/'records.csv').open(encoding='utf-8-sig') as f:
        records = {r['record_id']: r for r in csv.DictReader(f)}
    cache, values, orders, failures, caps, hits = {}, [], [], [], [], []
    for row in rows:
        rid = row['record_id']
        if rid not in cache:
            rec = records[rid]
            cache[rid] = loadmat(rec['path'], variable_names=[rec['variable']])[rec['variable']].reshape(-1)
        start, end = int(row['start_sample']), int(row['end_sample'])
        try:
            v, meta = extract(cache[rid][start:end], float(row['rpm'])/60)
            values.append(v)
            orders.append(meta['ar_order'])
            caps.append(meta['ar_order_cap'])
            hits.append(meta['ar_cap_hit'])
        except Exception as e:
            failures.append({'window_index': row['window_index'], 'error': repr(e)})
    matrix = np.asarray(values)
    report = {'scope': 'source_0_training_support_only', 'windows': len(rows),
              'labels_used_for_selection': False, 'validation_windows_processed': 0,
              'target_windows_processed': 0, 'models_trained': 0,
              'failures': failures, 'valid_windows': len(values),
              'BIC_orders': dict(Counter(orders)), 'AR_cap_hits': sum(hits),
              'AR_cap_range': [min(caps), max(caps)],
              'median_features': np.median(matrix, axis=0).tolist(),
              'zero_detected_harmonic_fraction': (matrix[:, 9:] == 0).mean(axis=0).tolist()}
    assert not failures and np.all(np.isfinite(matrix))
    with Path(__file__).with_name('physics12_source_training_check_v2.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
