"""Rebuild paired confusion figures from the frozen evaluation counts."""

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "experiments/CWRU/实验结果"
V3 = RESULTS / "CWRU_V3"
V5 = RESULTS / "CWRU_V5"
CROSS_CLEAN = V5 / "load_clean_20261109_142434"
CROSS_NOISE = (
    V5 / "load_awgn_full_10_20261109_231117",
    V5 / "load_awgn_20_50_20261110_100114",
)
WINDOW = V5 / "window_random_20261110_151114"
BUDGETS = (0.1, 0.2, 0.5, 1.0)
CONDITIONS = (
    ("clean_only", "clean"),
    ("awgn_mixed", "clean"),
    *(("awgn_mixed", f"white_{snr:+d}") for snr in (-4, -2, 0, 2, 4)),
)
KEYS = [
    "ratio", "protocol", "condition", "source_load_hp", "target_load_hp",
    "subset", "seed", "noise_seed",
]
METRICS = ["acc", "balanced_acc", "recall_N", "recall_IR", "recall_OR", "recall_B"]
CLASSES = pd.MultiIndex.from_product((range(4), range(4)))


def read_counts(runs, variant, protocols):
    all_counts = []
    all_metrics = []
    for run in runs:
        folder = run / "evaluation"
        counts = pd.read_csv(folder / "confusion_matrices.csv")
        metrics = pd.read_csv(folder / "metrics.csv")
        all_counts.append(counts.loc[
            (counts.variant == variant) & counts.protocol.isin(protocols)
        ])
        all_metrics.append(metrics.loc[
            (metrics.variant == variant) & metrics.protocol.isin(protocols)
        ])

    counts = pd.concat(all_counts, ignore_index=True)
    labels = ["true_label", "predicted_label"]
    wide = counts.pivot(index=KEYS, columns=labels, values="count")
    np.testing.assert_array_equal(wide.columns.to_list(), CLASSES.to_list())
    assert wide.notna().all().all() and (wide >= 0).all().all()
    cubes = wide.to_numpy().reshape(-1, 4, 4)
    support = cubes.sum(axis=2)
    assert (support > 0).all()
    archived_support = counts.pivot(index=KEYS, columns=labels, values="support")
    np.testing.assert_array_equal(
        archived_support.reindex(wide.index).to_numpy().reshape(-1, 4, 4),
        np.repeat(support[:, :, None], 4, axis=2),
    )
    normalized = cubes / support[:, :, None]
    recalls = np.diagonal(normalized, axis1=1, axis2=2)
    verified = np.column_stack((
        np.trace(cubes, axis1=1, axis2=2) / support.sum(axis=1),
        recalls.mean(axis=1),
        recalls,
    ))
    archived = pd.concat(all_metrics, ignore_index=True).set_index(KEYS)
    assert archived.index.is_unique and set(wide.index) == set(archived.index)
    np.testing.assert_allclose(
        verified, archived.loc[wide.index, METRICS], rtol=0, atol=1e-12
    )
    return wide.index, support, normalized, verified


def read_reference(run, protocol=None):
    table = pd.read_csv(run / "evaluation" / "comparison_budget_mean_std.csv")
    if "protocol" in table:
        table = table[table.protocol == protocol]
    if "condition" not in table:
        table["condition"] = "clean"
    return table.set_index(["ratio", "condition", "method"])


def expected_keys(directions):
    return {
        (budget, protocol, condition, source, target, subset, seed, noise)
        for budget, (protocol, condition), (source, target) in product(
            BUDGETS, CONDITIONS, directions
        )
        for subset, seed, noise in product(
            (0,) if budget == 1 else (4101, 4102, 4103),
            range(1706, 1711),
            (0,) if condition == "clean" else range(7201, 7206),
        )
    }


def main():
    protocols = ("clean_only", "awgn_mixed")
    directions = [
        [(-1, -1)],
        [(source, target) for source in range(4) for target in range(4) if source != target],
    ]
    sources = [
        [
            read_counts((V3 / "window_random_20261023_0853",), "baseline_data", protocols),
            read_counts((WINDOW,), "data_aux_ce", protocols),
        ],
        [
            read_counts(
                (V3 / "load_full_20261024_0950", V3 / "load_small_20261025_2017"),
                "baseline_data", protocols,
            ),
            # Use matched training protocols. The noise runs also have a
            # clean-trained fusion arm, which has no matched noisy baseline.
            read_counts((CROSS_CLEAN,), "data_aux_ce", ("clean_only",)),
        ],
    ]
    cross_noise = read_counts(CROSS_NOISE, "data_aux_ce", ("awgn_mixed",))
    index, support, normalized, metrics = sources[1][1]
    sources[1][1] = (
        index.append(cross_noise[0]),
        np.concatenate((support, cross_noise[1])),
        np.concatenate((normalized, cross_noise[2])),
        np.concatenate((metrics, cross_noise[3])),
    )
    references = [
        [read_reference(WINDOW, protocol) for protocol in protocols],
        [
            read_reference(CROSS_CLEAN),
            pd.concat([read_reference(run) for run in CROSS_NOISE], verify_integrity=True),
        ],
    ]
    matrices = np.empty((4, 4, 2, len(CONDITIONS), len(BUDGETS), 2))
    scores = np.empty((len(METRICS), 2, len(CONDITIONS), len(BUDGETS), 2))

    for scene, pair in enumerate(sources):
        expected = expected_keys(directions[scene])
        baseline_keys, baseline_support = pair[0][:2]
        for method, (keys, support, normalized, verified) in enumerate(pair):
            assert keys.is_unique and set(keys) == expected
            np.testing.assert_array_equal(support, baseline_support[baseline_keys.get_indexer(keys)])
            for condition_idx, (protocol, condition) in enumerate(CONDITIONS):
                ref = references[scene][protocol == "awgn_mixed"]
                method_key = ("v3_data" if method == 0 else
                              "v5_awgn" if scene == 1 and protocol == "awgn_mixed" else "v5_39")
                for budget_idx, budget in enumerate(BUDGETS):
                    keep = ((keys.get_level_values("ratio") == budget)
                            & (keys.get_level_values("protocol") == protocol)
                            & (keys.get_level_values("condition") == condition))
                    # Every axis is balanced by the complete key check, so
                    # averaging normalized rows gives equal weight to each axis.
                    matrices[:, :, method, condition_idx, budget_idx, scene] = normalized[keep].mean(axis=0)
                    average = verified[keep].mean(axis=0)
                    scores[:, method, condition_idx, budget_idx, scene] = average
                    archived = ref.loc[(budget, condition, method_key), [f"{name}_mean" for name in METRICS]]
                    np.testing.assert_allclose(average, archived, rtol=0, atol=1e-12)
            print(f"scene {scene}, method {method}: {len(keys)} count matrices verified")

    np.testing.assert_allclose(matrices.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    old = loadmat(HERE / "data_confusion.mat")
    np.testing.assert_allclose(matrices[:, :, :, 0, 3, 0], old["matrices"][:, :, :, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(matrices[:, :, :, 0, 3, 1], old["matrices"][:, :, :, 1:].mean(axis=3), rtol=0, atol=1e-12)
    savemat(HERE / "data_confusion_extension.mat", {
        "matrices": matrices, "acc": 100 * scores[0], "ba": 100 * scores[1],
        "cmap": old["cmap"],
    }, do_compression=True)
    print("Verified 56 paired groups against frozen accuracy and recall summaries.")


if __name__ == "__main__":
    main()
