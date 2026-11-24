"""Fit and verify four-harmonic descriptor caches for existing PU splits."""
import hashlib
import json
from pathlib import Path, PureWindowsPath
from uuid import uuid4

import numpy as np
import torch

from shared_descriptor import describe_four_harmonic


VERSION = 'PU_harmonic_mean_20261122_a'
HERE = Path(__file__).resolve().parent
CURRENT_SOURCE = HERE / 'current_descriptor.py'
if not CURRENT_SOURCE.is_file():
    CURRENT_SOURCE = HERE / 'runtime/current_descriptor.py'
SHARED_SOURCES = {
    'descriptor_cache.py': Path(__file__).resolve(),
    'shared_descriptor.py': HERE / 'shared_descriptor.py',
    'current_descriptor.py': CURRENT_SOURCE,
}


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def archive_path(project, recorded):
    """Resolve a PU archive from the portable suffix in a frozen manifest."""
    parts = PureWindowsPath(recorded).parts
    for index in range(len(parts) - 3):
        if parts[index:index + 3] == ('data', 'raw', 'PU'):
            tail = parts[index + 3:]
            if '..' not in tail:
                return project.joinpath('data', 'raw', 'PU', *tail)
    raise ValueError(f'Unsupported PU archive path: {recorded}')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def check_inputs(base, filenames):
    proof = read_json(base / 'verification.json')
    for name in filenames:
        if file_sha(base / name) != proof['files_sha256'][name]:
            raise ValueError(f'Frozen input changed: {base / name}')


def fit_physical(features, train_ids):
    q1, center, q3 = np.quantile(features[train_ids].astype(np.float64), [.25, .5, .75], axis=0)
    return dict(center=center.tolist(), scale=np.maximum(q3 - q1, 1e-6).tolist())


def apply_in_batches(raw, legacy_describe):
    return torch.cat([describe_four_harmonic(chunk, legacy_describe) for chunk in raw.split(128)])


def build_cache(out, signals, row_count, split_dir, normalization_roles, source_hashes,
                legacy_describe, input_proofs, device):
    """Write only the new descriptors, train-only calibrations and provenance."""
    if out.exists():
        raise FileExistsError(out)
    initial_sources = source_hashes()
    initial_proofs = [file_sha(path) for path in input_proofs]
    stage = out.parent / f'.harmonic-{uuid4().hex}'
    stage.mkdir()
    try:
        raw = np.load(signals, mmap_mode='r')
        if raw.ndim != 3 or raw.shape[1:] != (2, 2560) or not 0 < row_count <= len(raw):
            raise ValueError('Unexpected angle-domain waveform cache')
        features = np.empty((row_count, 2, 4), dtype=raw.dtype)
        for start in range(0, row_count, 128):
            stop = min(start + 128, row_count)
            chunk = torch.from_numpy(np.array(raw[start:stop])).to(device)
            features[start:stop] = describe_four_harmonic(chunk, legacy_describe).cpu().numpy()
        np.save(stage / 'physical.npy', features, allow_pickle=False)
        with np.load(split_dir / 'split_indices.npz', allow_pickle=False) as split:
            for name, role in normalization_roles.items():
                train = split[role]
                if len(train) == 0 or train.min() < 0 or train.max() >= row_count:
                    raise ValueError(f'Invalid training rows for {name}')
                norm = read_json(split_dir / name)
                norm['physical'] = fit_physical(features, train)
                write_json(stage / name, norm)
        if initial_sources != source_hashes() or initial_proofs != [file_sha(path) for path in input_proofs]:
            raise ValueError('Input proof or descriptor code changed during preparation')
        proof = dict(descriptor_version=VERSION, stage='descriptor_cache_prepared', windows=row_count,
                     input_proof_sha256=initial_proofs, descriptor_source_sha256=initial_sources,
                     files_sha256={path.name: file_sha(path) for path in sorted(stage.iterdir())})
        write_json(stage / 'verification.json', proof)
        stage.rename(out)
    finally:
        if stage.is_dir():
            for path in stage.iterdir():
                path.unlink()
            stage.rmdir()
    return proof


def verify_cache(cache, input_proofs, source_hashes, output_names):
    proof = read_json(cache / 'verification.json')
    if (proof.get('descriptor_version') != VERSION or proof.get('stage') != 'descriptor_cache_prepared'
            or proof.get('descriptor_source_sha256') != source_hashes()
            or proof.get('input_proof_sha256') != [file_sha(path) for path in input_proofs]
            or set(proof.get('files_sha256', ())) != {'physical.npy', *output_names}):
        raise ValueError('The four-harmonic descriptor cache does not match the current inputs or code')
    for name, digest in proof['files_sha256'].items():
        if file_sha(cache / name) != digest:
            raise ValueError(f'Descriptor cache changed: {cache / name}')
    return proof
