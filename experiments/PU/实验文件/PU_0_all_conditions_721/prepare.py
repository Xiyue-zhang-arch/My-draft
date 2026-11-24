"""Prepare four-harmonic descriptors on the frozen record split."""
import hashlib
import sys

import numpy as np
import torch

from common import DATA, PROJECT, WRONG_ORDERS, setup
from descriptors import describe, statistics
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
INPUT_PROOFS = (DATA / 'verification.json',)
OLD_DATA = PROJECT.parents[3] / 'data/pu_shared/prepared_inputs'
NORMALIZATION_ROLES = {'normalization.json': 'train'}

def descriptor_banks(raw):
    return {
        'physical': describe_physical(raw),
        'wrong': statistics_bank(raw, lambda chunk: describe(chunk, WRONG_ORDERS)),
        'statistics': statistics_bank(raw, statistics),
    }


def statistics_bank(raw, fn):
    return torch.cat([fn(chunk) for chunk in raw.split(128)])


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def descriptor_path(profile, cache=CACHE):
    return cache / 'physical.npy' if profile == 'physical' else DATA / f'{profile}_all.npy'


def data_verify(cache=CACHE):
    return verify_cache(cache, INPUT_PROOFS, descriptor_hashes, NORMALIZATION_ROLES)


def verify_frozen_inputs():
    check_inputs(DATA, (
        'signals.npy', 'windows.json', 'split_indices.npz', 'vibration_noise_templates.npy',
        'records.json', 'archive_hashes.json', 'wrong_all.npy', 'statistics_all.npy',
        *NORMALIZATION_ROLES,
    ))


def prepare():
    setup()
    verify_frozen_inputs()
    return build_cache(CACHE, DATA / 'signals.npy', 16000, DATA, NORMALIZATION_ROLES,
                       descriptor_hashes, describe, INPUT_PROOFS, 'cuda')


if __name__ == '__main__':
    proof = prepare()
    print('PREPARED', CACHE, proof['descriptor_version'], proof['windows'], flush=True)
