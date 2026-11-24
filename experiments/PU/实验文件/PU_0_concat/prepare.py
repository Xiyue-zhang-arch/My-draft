"""Prepare new descriptors for the existing 3200/800 source split."""
import sys
from pathlib import Path

import argparse

import numpy as np

from common import DATA, FROZEN_SPLIT, PROJECT, read, setup, sha
from descriptors import describe
if not (PROJECT / 'descriptor_cache.py').is_file():
    sys.path.insert(0, str(PROJECT.parent / 'PU_harmonic_mean'))
from descriptor_cache import (
    SHARED_SOURCES, VERSION, apply_in_batches, build_cache, check_inputs, file_sha, verify_cache,
)

def source_path(name):
    return SHARED_SOURCES[name] if name in SHARED_SOURCES else PROJECT / name


def descriptor_hashes():
    names = ('descriptors.py', *SHARED_SOURCES)
    return {name: file_sha(source_path(name)) for name in names}


def describe_physical(raw):
    return apply_in_batches(raw, describe)

CACHE = PROJECT / 'data/harmonic_calibration'
INPUT_PROOFS = (DATA / 'verification.json', FROZEN_SPLIT / 'verification.json')
NORMALIZATION_ROLES = {'normalization.json': 'train'}


def data_verify(cache):
    return verify_cache(cache, INPUT_PROOFS, descriptor_hashes, NORMALIZATION_ROLES)


def verify_frozen_inputs():
    check_inputs(DATA, ('signals.npy', 'windows.json', 'vibration_noise_templates.npy', 'archive_hashes.json'))
    check_inputs(FROZEN_SPLIT, ('data_reference.json', 'normalization.json', 'split_indices.npz'))
    if sha(DATA / 'verification.json') != read(FROZEN_SPLIT / 'data_reference.json')['data_verification_sha256']:
        raise ValueError('Frozen split and waveform cache do not match')
    windows = read(DATA / 'windows.json')
    with np.load(FROZEN_SPLIT / 'split_indices.npz') as split:
        train, val = split['train'], split['val']
        if len(train) != 3200 or len(val) != 800 or not np.array_equal(np.sort(np.r_[train, val]), np.arange(4000)):
            raise ValueError('Unexpected source windows')
        if {windows[i]['record_index'] for i in train} & {windows[i]['record_index'] for i in val}:
            raise ValueError('Source record overlap')


def prepare(out):
    setup()
    verify_frozen_inputs()
    return build_cache(out, DATA / 'signals.npy', 4000, FROZEN_SPLIT, NORMALIZATION_ROLES,
                       descriptor_hashes, describe, INPUT_PROOFS, 'cuda')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=CACHE)
    cache = parser.parse_args().output.resolve()
    proof = prepare(cache)
    print('PREPARED', cache, proof['descriptor_version'], proof['windows'], flush=True)
