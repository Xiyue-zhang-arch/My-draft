"""Recompute harmonic predictions and pair them with the retained comparison metrics."""
import argparse
import csv
from pathlib import Path

import numpy as np

from run_mean import CASES, HERE, METHOD, REFERENCE, SEEDS, CONDITIONS, read, parent, verify_frozen_trace, verify_reference
from common import sha, write_csv, write_json

METHODS = ("equal", "v2", "high34", METHOD)
FIELDS = ("BA", "ACC", "Macro_F1", "recall_healthy", "recall_outer", "recall_inner", "recall_compound",
          "KA01", "KA04", "KI05", "KI14")


def audit_models(out, state, splits, labels):
    assert [(r["source"], r["seed"]) for r in state["models"]] == [(c, s) for c in CONDITIONS for s in SEEDS]
    original = {(r["source"], r["seed"]): r for r in read(REFERENCE / "config.json")["models"]}
    for source in CONDITIONS:
        bank = out / "bank" / source
        features = np.load(bank / "descriptors.npy")
        assert features.shape == (3600, 2, 4) and np.isfinite(features).all()
        q1, mid, q3 = np.quantile(features[:2800].astype(np.float64), [.25, .5, .75], axis=0)
        norm = read(bank / METHOD / "normalization.json")
        np.testing.assert_array_equal(mid, norm["physical"]["center"])
        np.testing.assert_array_equal(np.maximum(q3-q1, 1e-6), norm["physical"]["scale"])
        with np.load(bank / METHOD / "indices.npz") as ids:
            for role in ("train", "val"):
                np.testing.assert_array_equal(np.sort(ids[role]), np.sort(splits[source][role]))
    for row in state["models"]:
        source, seed = row["source"], row["seed"]
        folder = parent.paths(out, source, METHOD, seed)
        assert row["updates"] == 2200 and row["epochs"] == 50 and not row["functional"]
        anchor = original[source, seed]
        matches_initialization = row["role_initialization_sha256"] == anchor["role_initialization_sha256"]
        portable = state.get("portable", False)
        if not portable:
            assert matches_initialization
        ref = parent.paths(REFERENCE, source, METHOD, seed)
        assert sha(ref / "training_trace.json") == anchor["artifacts_sha256"]["training_trace.json"]
        matches_trace = verify_frozen_trace(read(folder / "training_trace.json"), read(ref / "training_trace.json"), portable)
        if portable:
            assert row["frozen_initialization_matched"] == matches_initialization
            assert row["frozen_cuda_trace_matched"] == matches_trace
        for filename, value in row["artifacts_sha256"].items():
            assert sha(folder / filename) == value
        with np.load(out / "bank" / source / METHOD / "indices.npz") as bank, np.load(folder / "validation.npz") as val:
            np.testing.assert_array_equal(val["y"], labels[bank["val"]])
            assert np.isfinite(val["logits"]).all()


def score(case_file, ids, labels, bearing):
    with np.load(case_file) as data:
        np.testing.assert_array_equal(data["window_index"], ids)
        np.testing.assert_array_equal(data["y"], labels)
        logits = data["logits"]
        assert logits.shape == (400, 4) and np.isfinite(logits).all()
        pred = logits.argmax(1)
    counts = np.bincount(4*labels + pred, minlength=16).reshape(4, 4)
    np.testing.assert_array_equal(counts.sum(1), [100]*4)
    recall = counts.diagonal()/counts.sum(1)
    f1 = (2*counts.diagonal()/(counts.sum(1)+counts.sum(0))).mean()
    return np.array((recall.mean(), (labels == pred).mean(), f1, *recall, *[
        (labels[bearing == code] == pred[bearing == code]).mean() for code in FIELDS[-4:]]))


def historical_scores():
    path = REFERENCE / "report/per_condition.csv"
    expected = read(REFERENCE / "report/AUDIT_COMPLETE.json")["results_sha256"][path.name]
    assert sha(path) == expected
    with path.open(encoding="utf-8-sig", newline="") as stream:
        values = {(r["source"], r["target"], r["method"], int(r["seed"]), r["case"]):
                  np.array([float(r[k]) for k in FIELDS])
                  for r in csv.DictReader(stream) if r["method"] != METHOD}
    assert len(values) == 3780
    return values, expected


def collect(out, splits, y, codes, historical):
    new, case_rows, seed_rows, data = {}, [], [], {}
    for src, dst in parent.ROUTES:
        ids = splits[dst]["test"]
        for method in METHODS:
            for seed in SEEDS:
                groups = {"clean": [], "noisy": []}
                folder = out / "predictions" / src / dst / method / str(seed)
                if method == METHOD:
                    assert {p.stem for p in folder.glob("*.npz")} == {c[0] for c in CASES}, folder
                for case, snr, _ in CASES:
                    if method == METHOD:
                        path = folder / (case + ".npz")
                        new[path.relative_to(out).as_posix()] = sha(path)
                        scores = score(path, ids, y[ids], codes[ids])
                    else:
                        scores = historical[src, dst, method, seed, case]
                    endpoint = "clean" if snr is None else "noisy"
                    case_rows.append(dict(source=src, target=dst, method=method, seed=seed,
                                          endpoint=endpoint, case=case, snr_db=snr, **dict(zip(FIELDS, scores))))
                    groups[endpoint].append(scores)
                for endpoint, scores in groups.items():
                    means = np.mean(scores, axis=0)
                    data[src, dst, method, seed, endpoint] = means
                    seed_rows.append(dict(source=src, target=dst, method=method, seed=seed,
                                          endpoint=endpoint, **dict(zip(FIELDS, means))))
    assert (len(new), len(case_rows), len(seed_rows)) == (1260, 5040, 480)
    return new, case_rows, seed_rows, data


def summary(data):
    routes = parent.ROUTES
    slices = [("overall", "all", routes),
              *(("route", f"{a} to {b}", ((a, b),)) for a, b in routes),
              *(("source", c, tuple(r for r in routes if r[0] == c)) for c in CONDITIONS),
              *(("target", c, tuple(r for r in routes if r[1] == c)) for c in CONDITIONS)]
    summaries, paired = [], []
    for group, key, rs in slices:
        for endpoint in ("clean", "noisy"):
            by_method = {m: np.stack([np.mean([data[a, b, m, s, endpoint] for a, b in rs], axis=0)
                                      for s in SEEDS]) for m in METHODS}
            case = dict(group=group, key=key, endpoint=endpoint)
            for method, rows in by_method.items():
                for metric, values in zip(FIELDS, rows.T):
                    summaries.append(dict(**case, method=method, metric=metric,
                                          mean=float(values.mean()), sd=float(values.std(ddof=1))))
            for ref in ("v2", "equal", "high34"):
                difference = by_method[METHOD] - by_method[ref]
                for metric, values in zip(FIELDS, difference.T):
                    paired.append(dict(**case, method=METHOD, reference=ref, metric=metric,
                                       delta=float(values.mean()), positive=int((values > 1e-12).sum()),
                                       **{f"seed_{s}": float(v) for s, v in zip(SEEDS, values)}))
    return summaries, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    state = read(out / "config.json")
    assert state["stage"] == "predictions_complete" and not state["preflight"]
    assert state["predictions"] == 1260 and len(state["models"]) == 20
    assert read(out / "evaluation_lock.json") == state["models"]
    anchor = verify_reference()
    frozen_audit = None
    if (out / "RUN_COMPLETE.json").is_file():
        mark = read(out / "RUN_COMPLETE.json")
        assert mark["stage"] == "complete" and mark["config_sha256"] == sha(out / "config.json")
        assert mark["audit_sha256"] == sha(out / "report/AUDIT_COMPLETE.json")
        frozen_audit = read(out / "report/AUDIT_COMPLETE.json")
    else:
        assert state["reference"] == anchor
        for filename in ("run_mean.py", "audit_mean.py", "PROTOCOL.md"):
            assert sha(HERE / filename) == sha(out / "source" / filename)
    windows, y, splits, _ = parent.metadata()
    audit_models(out, state, splits, y)
    history, history_sha = historical_scores()
    new, cases, seeds, values = collect(out, splits, y, np.array([r["bearing"] for r in windows]), history)
    if frozen_audit:
        assert new == frozen_audit["predictions_sha256"]
    results, deltas = summary(values)
    report = args.report_dir or out / "report"
    report.mkdir()
    for filename, rows in (("summary", results), ("paired_differences", deltas), ("per_seed", seeds),
                           ("per_condition", cases)):
        write_csv(report / (filename + ".csv"), rows)
    write_json(report / "AUDIT_COMPLETE.json", dict(
        stage="passed", models=20, updates=44000, new_inference=0, new_predictions=len(new),
        historical_metric_rows=len(history), historical_metrics_sha256=history_sha,
        historical_prediction_files_checked=0, portable=state.get("portable", False),
        config_sha256=sha(out / "config.json"),
        evaluation_lock_sha256=sha(out / "evaluation_lock.json"),
        results_sha256={p.name: sha(p) for p in report.glob("*.csv")},
        predictions_sha256=new))
    if frozen_audit is None and args.report_dir is None:
        write_json(out / "RUN_COMPLETE.json", dict(stage="complete",
            config_sha256=sha(out / "config.json"), audit_sha256=sha(report / "AUDIT_COMPLETE.json")))
    for row in results:
        if row["group"] == "overall" and row["metric"] == "BA":
            print(row["endpoint"], row["method"], round(100*row["mean"], 5), round(100*row["sd"], 5))


if __name__ == "__main__":
    main()
