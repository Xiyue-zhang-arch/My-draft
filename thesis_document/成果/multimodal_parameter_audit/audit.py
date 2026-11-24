"""Read-only sampling audit. No training, feature fitting, or raw-data writes."""
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import openpyxl
from scipy.io import loadmat
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'data/raw'
OUT = Path(__file__).parent
result = {'scope': 'sampled metadata and signal acquisition audit; not full-file certification',
          'PU': [], 'Ottawa': [], 'MAFAULDA': []}

for code in ['K001', 'KA01', 'KI01']:
    archive = next((RAW/'PU').rglob(code+'.rar'))
    names = subprocess.run(['tar', '-tf', str(archive)], capture_output=True,
                           text=True, check=True).stdout.splitlines()
    for condition in ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10']:
        name = next(n for n in names if condition in n and n.endswith('.mat'))
        raw = subprocess.run(['tar', '-xOf', str(archive), name],
                             capture_output=True, check=True).stdout
        mat = loadmat(io.BytesIO(raw), simplify_cells=True)
        v = next(v for k,v in mat.items() if not k.startswith('__'))
        item = {'archive': str(archive.relative_to(ROOT)), 'member': name, 'channels': []}
        for y in v['Y']:
            t = np.asarray(v['X'][int(y['XIndex'])-1]['Data'])
            a = np.asarray(y['Data'])
            c = {'name': y['Name'], 'raster': y['Raster'], 'x_index': int(y['XIndex']),
                 'n': len(a), 'time_n': len(t), 'start': float(t[0]), 'end': float(t[-1]),
                 'effective_fs': float((len(t)-1)/(t[-1]-t[0])),
                 'time_monotone': bool(np.all(np.diff(t)>0)),
                 'finite': bool(np.isfinite(a).all())}
            if y['Name']=='speed':
                c.update(rpm_min=float(a.min()), rpm_median=float(np.median(a)), rpm_max=float(a.max()))
            item['channels'].append(c)
        result['PU'].append(item)
        print('PU', name, flush=True)

for name in ['H_1_0.xlsx', 'I_1_1.xlsx', 'O_6_2.xlsx', 'B_11_2.xlsx', 'C_16_1.xlsx']:
    path = RAW/'Ottawa'/name
    w = openpyxl.load_workbook(path, read_only=True, data_only=True)
    s = w.active
    rows = s.iter_rows(values_only=True)
    header = list(next(rows))
    a = np.asarray(list(rows), dtype=float)
    item = {'file': name, 'header': header, 'shape': list(a.shape), 'finite': bool(np.isfinite(a).all())}
    for j in [2,3]:
        vals = a[:,j]
        nz = np.flatnonzero(vals!=0)
        item[header[j]] = {'nonzero_count': len(nz), 'first_nonzero_indices': nz[:12].tolist(),
                           'nonzero_min': float(vals[nz].min()) if len(nz) else None,
                           'nonzero_median': float(np.median(vals[nz])) if len(nz) else None,
                           'nonzero_max': float(vals[nz].max()) if len(nz) else None}
    result['Ottawa'].append(item)
    w.close()
    print('Ottawa', name, flush=True)

for category in ['normal', 'ball_fault', 'cage_fault', 'outer_race']:
    files = sorted((RAW/'MAFAULDA'/category).rglob('*.csv'), key=lambda p: float(p.stem))
    for path in [files[0], files[-1]]:
        a = np.loadtxt(path, delimiter=',')
        tach = a[:,0]
        # Empirical peak frequency only. Pulses/revolution is not assumed established.
        peaks, _ = find_peaks(tach, height=float((np.quantile(tach,.05)+np.quantile(tach,.95))/2),
                              prominence=float(np.ptp(tach)*.4), distance=250)
        periods = np.diff(peaks)/50000.
        item = {'file': str(path.relative_to(ROOT)), 'shape': list(a.shape),
                'finite': bool(np.isfinite(a).all()), 'filename_number': float(path.stem),
                'tach_peak_count': len(peaks), 'pulse_hz_median_at_50k': float(1/np.median(periods)),
                'pulse_hz_p05_p95': np.quantile(1/periods,[.05,.95]).tolist()}
        result['MAFAULDA'].append(item)
        print('MAFAULDA', str(path.relative_to(RAW)), flush=True)

result['geometry_orders_zero_contact_angle'] = {}
for name,z,d,D in [('PU',8,6.75,28.55),('Ottawa',8,6.77,28.50),('MAFAULDA',8,7.145,28.519)]:
    r=d/D
    result['geometry_orders_zero_contact_angle'][name] = {
        'BPFO': z/2*(1-r), 'BPFI': z/2*(1+r), 'BSF': (1-r*r)/(2*r), 'FTF': (1-r)/2}
target = OUT/'evidence.json'
with target.open('x',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
print('Saved', target, flush=True)
