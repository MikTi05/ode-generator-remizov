#!/usr/bin/env python3
"""Independent saved-only validation of all 1000 four-method rows; no solves."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from all_methods_dataset_loader import load_all_methods_equations
from diagnostics.four_method_common import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify(out):
    release = out / "release"
    plan = read(out / "plan.json")
    manifest = read(release / "release_manifest.json")
    assert digest(out / "plan.json") == manifest["plan_sha256"]
    for name, expected in manifest["artifact_sha256"].items():
        assert digest(release / name) == expected, name
    for name, expected in manifest["consumer_sha256"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in read(release / "source_sha256.json").items():
        assert digest(ROOT / name) == expected, name
    methods = tuple(plan["tie_order"])
    assert methods == ("remizov", "monte_carlo", "rk4", "product_integral")
    folder = ROOT / plan["dataset_dir"]
    equations = load_all_methods_equations(folder / "all_methods_equations_v1.csv",
                                         folder / "all_methods_equations_v1_metadata.jsonl")
    assert len(equations) == 1000 and [e.equation_id for e in equations] == plan["equation_ids"]
    with np.load(folder / "all_methods_equations_v1_ml.npz", allow_pickle=False) as z:
        initial_y, initial_v = z["y0"].copy(), z["v0"].copy()
        npz_ids = z["equation_id"].tolist()
    assert npz_ids == plan["equation_ids"]
    rows = table(release / "coverage_all1000.csv")
    assert [r["equation_id"] for r in rows] == plan["equation_ids"]
    assert len({r["equation_id"] for r in rows}) == 1000
    expected_records = {f"{method}/{pid}.json" for method in ("rk4", "product_integral") for pid in plan["equation_ids"]}
    actual_records = {str(p.relative_to(out / "records")) for p in (out / "records").glob("*/*.json")}
    assert actual_records == expected_records
    old_coverage = {r["equation_id"]: r for r in table(ROOT / plan["historical_sources"]["mean4_reference"])}
    errors = {method: {} for method in methods}
    primary_statuses = {method: Counter() for method in ("rk4", "product_integral")}
    true_labels = {}
    ties = []
    complete_ids = []
    initial_matches = 0
    scalar_probes = 0
    for index, (equation, row) in enumerate(zip(equations, rows)):
        pid, m, p = equation.equation_id, equation.metadata, equation.cauchy_problem
        exact = float(equation.y_exact(0.0))  # all numerical jobs are already committed
        assert p.y0 == initial_y[index] == m["y0"] and p.v0 == initial_v[index] == m["v0"]
        initial_matches += 1
        for name in ("a", "b", "c", "g"):
            for x in (-1.0, 0.0, 1.0):
                assert math.isfinite(getattr(p, name)(x))
                scalar_probes += 1
        probes = np.array([-1.0, 0.0, 1.0])
        rp = equation.remizov_problem
        np.testing.assert_allclose([p.c(float(x)) for x in probes], rp.c_remizov(probes) - rp.lambda_value)
        np.testing.assert_array_equal([p.g(float(x)) for x in probes], -rp.g_remizov(probes))
        for key in ("expression_hash", "family_id", "structure_signature", "complexity_level", "regime"):
            assert row[key] == m[key]
        assert row["split_original"] == m["split"] and row["dataset_sha256"] == plan["dataset_sha256"]
        for method in ("rk4", "product_integral"):
            path = out / "records" / method / (pid + ".json")
            record = read(path)
            body = {k: v for k, v in record.items() if k != "record_sha256"}
            encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
            assert record["record_sha256"] == hashlib.sha256(encoded).hexdigest()
            assert record["equation_id"] == pid and record["method"] == method
            assert record["plan_sha256"] == manifest["plan_sha256"] and record["config"] == plan["config"]
            assert record["inputs"] == {key: getattr(p, key) for key in ("x0", "x1", "y0", "v0")}
            primary_statuses[method][record["status"]] += 1
            if record["usable"]:
                result = record["result"]
                assert result["success"] and result["failure_reason"] == "" and result["equation_id"] == pid
                assert result["n_steps"] == 128 and result["h"] == 1 / 64
                assert len(result["y"]) == len(result["dy"]) == len(result["x"]) == 129
                np.testing.assert_array_equal(result["x"], np.linspace(-1, 1, 129))
                assert result["x"][64] == 0.0 and result["y"][0] == p.y0 and result["dy"][0] == p.v0
                assert np.isfinite(result["y"]).all() and np.isfinite(result["dy"]).all()
                err = abs(float(result["y"][64]) - exact)
                assert err == record["E0"] and exact == record["exact0"]
                exact_grid = equation.y_exact(np.linspace(-1, 1, 33))
                e33 = float(np.max(np.abs(np.asarray(result["y"])[::4] - exact_grid)))
                assert e33 == record["E_inf_grid33"]
                if method == "rk4":
                    assert result["nfev"] == 512
                else:
                    assert all(result[key] == 128 for key in
                               ("steps_requested", "steps_completed", "coefficient_points_evaluated", "expm_evals", "matvec_evals"))
                    assert result["scalar_coefficient_calls"] == 512 and result["max_invariant_drift"] <= 1e-8
                errors[method][pid] = err
                assert float(row["y0_" + method]) == result["y"][64]
            else:
                assert record["E0"] is None and record["y0_target"] is None and row["E_" + method] == ""
        old = old_coverage[pid]
        raw = read(ROOT / json.loads(old["remizov_sources"])[0])
        if raw["computation_status"] == "success":
            assert raw["result"]["x"][16] == 0.0 and raw["y_exact"][16] == exact
            errors["remizov"][pid] = abs(raw["result"]["f"][16] - exact)
        else:
            assert raw["result"] is None and row["E_remizov"] == ""
        mc_values = []
        for path in json.loads(old["mc_sources"]):
            raw = read(ROOT / path)
            assert raw["status"] == "completed" and raw["result"]["success"] and raw["result"]["x"] == [0.0]
            assert raw["y_exact"] == exact and raw["config"]["M"] == 1024
            mc_values.append(raw["result"]["f"][0])
        errors["monte_carlo"][pid] = abs(float(np.mean(np.asarray(mc_values, dtype=np.float64))) - exact)
        valid = [(method, errors[method][pid]) for method in methods if pid in errors[method]]
        expected_label = sorted(valid, key=lambda item: (item[1], methods.index(item[0])))[0][0] if valid else None
        assert row["best_method"] == (expected_label or "")
        tied = [method for method, error in valid if error == min(e for _, e in valid)] if valid else []
        assert json.loads(row["tied_methods"]) == (tied if len(tied) > 1 else [])
        complete = len(valid) == 4
        assert (row["eligible_four_method_ml"] == "True") == complete
        assert row["best_method_scope"] == ("all_four" if complete else "available_only")
        for method in methods:
            available = pid in errors[method]
            assert (row["usable_" + method] == "True") == available
            if available:
                assert float(row["E_" + method]) == errors[method][pid]
                assert (row["accuracy_met_" + method] == "True") == (errors[method][pid] <= 0.05)
            else:
                assert row["E_" + method] == row["accuracy_met_" + method] == ""
        if complete:
            complete_ids.append(pid)
            true_labels[pid] = expected_label
            if len(tied) > 1:
                ties.append(pid)
    complete_rows = table(release / "ml_four_methods.csv")
    incomplete_rows = table(release / "incomplete_coverage.csv")
    assert [r["equation_id"] for r in complete_rows] == complete_ids
    assert {r["equation_id"] for r in incomplete_rows} == set(plan["equation_ids"]) - set(complete_ids)
    assert {r["equation_id"] for r in table(release / "exact_ties.csv")} == set(ties)
    counts = Counter(true_labels.values())
    assert {r["method"]: int(r["wins"]) for r in table(release / "winners.csv")} == {method: counts[method] for method in methods}
    for stat in table(release / "method_statistics.csv"):
        values = np.asarray(list(errors[stat["method"]].values()))
        assert int(stat["usable"]) == len(values) and int(stat["failures"]) == 1000 - len(values)
        assert int(stat["accurate"]) == int(np.sum(values <= 0.05))
        for key, value in (("mean", np.mean(values)), ("median", np.median(values)), ("maximum", np.max(values))):
            assert float(stat[key]) == value
    groups = dict(complexity=["complexity_level"], regime=["regime"],
                  complexity_regime=["complexity_level", "regime"],
                  family=["family_id", "structure_signature", "complexity_level", "regime"],
                  original_split=["split_original"], latest_split=["split_latest"])
    for name, keys in groups.items():
        independent = defaultdict(Counter)
        for row in complete_rows:
            independent[tuple(row[key] for key in keys)][true_labels[row["equation_id"]]] += 1
        saved = table(release / ("winners_by_" + name + ".csv"))
        assert len(saved) == len(independent)
        for row in saved:
            count = independent[tuple(row[key] for key in keys)]
            assert int(row["n"]) == sum(count.values())
            assert all(int(row[method]) == count[method] for method in methods)
    original_splits = read(ROOT / plan["historical_sources"]["splits"])
    assert read(release / "previous_splits.json") == original_splits
    test_ids = read(release / "previous_test_ids.json")
    assert test_ids == original_splits["scenarios"]["old"]["test"]["ids"] == original_splits["scenarios"]["new"]["test"]["ids"]
    assert len(test_ids) == 150 and set(test_ids) == {r["equation_id"] for r in rows if r["split_original"] == "test"}
    for key in ("split_original", "split_latest"):
        families = defaultdict(set)
        topologies = defaultdict(set)
        for row in complete_rows:
            families[row["family_id"]].add(row[key])
            topologies[row["structure_signature"]].add(row[key])
        assert all(len(values) == 1 for values in list(families.values()) + list(topologies.values()))
    sessions = [read(p) for p in out.glob("session_*.json")]
    assert sum(s["new_solver_calls"] for s in sessions) == 2000
    assert any(s["new_solver_calls"] == 0 and s["skipped_valid_records"] == 2000 for s in sessions)
    report = dict(schema="independent_four_method_saved_verification_v1", equations=1000,
                  new_records_checked=2000, old_R_records_checked=1000, old_MC_seed_records_checked=4000,
                  initial_conditions_checked=initial_matches, finite_scalar_input_probes=scalar_probes,
                  full_numerical_grids_checked=2000, complete_rows=len(complete_ids),
                  incomplete_rows=len(incomplete_rows), winner_counts={method: counts[method] for method in methods},
                  exact_ties=len(ties), statuses={method: dict(value) for method, value in primary_statuses.items()},
                  group_tables_checked=list(groups), test_ids_preserved=150, family_leakage=False,
                  physical_solver_calls_in_series=2000, new_solver_calls_in_verifier=0,
                  R_or_MC_recomputations=0, resume_without_recomputation_verified=True, ml_training=False)
    atomic_json(out / "independent_verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
