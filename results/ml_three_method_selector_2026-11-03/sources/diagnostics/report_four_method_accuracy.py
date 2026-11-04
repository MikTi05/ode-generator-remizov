#!/usr/bin/env python3
"""Assemble four-method labels from saved answers only; never call a solver."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from diagnostics.four_method_common import (
    DATA_DIR, METHODS, atomic_json, read, sha, validate_record, winner,
)

SEEDS = [202610250101, 202610250102, 202610250103, 202610250104]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def measurement(status, usable, value=None, exact=None, seconds=None, reason="", **diagnostics):
    error = abs(value - exact) if usable else None
    if usable and (not math.isfinite(error) or error < 0):
        raise ValueError("invalid measured error")
    return dict(status=status, usable=usable, y0=value if usable else None, E=error,
                accuracy_met=(error <= 0.05) if usable else None,
                seconds=seconds, failure_reason=reason, diagnostics=diagnostics)


def historical(metadata, reference, load):
    pid = metadata["equation_id"]
    exact = float(reference["y_exact0"])
    rpath = json.loads(reference["remizov_sources"])
    assert rpath == [f"results/chernov_solve1000_2026-10-18/solutions/{pid}.json"]
    raw = load(rpath[0])
    assert raw["problem_id"] == pid and raw["config"]["n"] == 128 and raw["config"]["boundary"] == "taper"
    if raw["computation_status"] == "success":
        assert raw["metadata"]["expression_hash"] == metadata["expression_hash"]
        result = raw["result"]
        assert result["success"] and result["failure_reason"] == "" and result["n"] == 128
        assert result["x"][16] == 0.0 and len(result["f"]) == 33 and raw["y_exact"][16] == exact
        assert np.isfinite(result["f"]).all()
        remizov = measurement("success", True, float(result["f"][16]), exact,
                              result["runtime_seconds"], source=rpath, config=raw["config"],
                              n=result["n"], n_operator_applications=result["n_operator_applications"],
                              task_wall_seconds=raw["task_wall_seconds"], time_scope="saved result.runtime_seconds")
        assert remizov["E"] == float(reference["E_R"])
    else:
        assert raw["result"] is None and reference["E_R"] == ""
        remizov = measurement(raw["computation_status"], False, reason="missing_saved_remizov_answer",
                              source=rpath, config=raw["config"], task_wall_seconds=raw["task_wall_seconds"],
                              time_scope="method time absent; task time in diagnostics")
    mcpaths = json.loads(reference["mc_sources"])
    assert mcpaths == [f"results/mc1000_x0_2026-10-25/attempts/{pid}_seed{i}.json" for i in range(4)]
    values, times, steps = [], [], []
    for index, path in enumerate(mcpaths):
        raw = load(path)
        result = raw["result"]
        assert raw["problem_id"] == result["problem_id"] == pid and raw["seed_index"] == index
        assert raw["status"] == "completed" and result["success"] and result["failure_reason"] == ""
        assert raw["expression_hash"] == metadata["expression_hash"]
        assert raw["config"] == json.loads(reference["mc_configs"])[index]
        assert raw["config"]["seed"] == SEEDS[index] and raw["config"]["M"] == 1024
        assert raw["config"]["dt"] == 1 / 128 and result["x"] == [0.0]
        assert result["completed_paths"] == [1024] and raw["y_exact"] == exact
        value = float(result["f"][0])
        assert math.isfinite(value) and abs(value - exact) == raw["abs_error"]
        values.append(value)
        times.append(result["runtime_seconds"])
        steps.append(result["n_steps"])
    mean = float(np.mean(values, dtype=np.float64))
    mc = measurement("success", True, mean, exact, float(sum(times)), source=mcpaths,
                     aggregation="arithmetic_mean_of_four_saved_MC1024_answers", seeds=SEEDS,
                     seed_answers=values, seed_seconds=times, seed_steps=steps, M_per_seed=1024,
                     dt_requested=1 / 128, time_scope="sum of four saved result.runtime_seconds")
    assert mean == float(reference["mc_mean_f0"]) and mc["E"] == float(reference["E_MC"])
    return remizov, mc, exact


def statistics(rows):
    output = []
    for method in METHODS:
        values = np.array([r["E_" + method] for r in rows if r["usable_" + method]], dtype=float)
        output.append(dict(method=method, equations=len(rows), usable=len(values),
                           failures=len(rows) - len(values), accurate=int(np.sum(values <= 0.05)),
                           mean=float(np.mean(values)) if len(values) else None,
                           median=float(np.median(values)) if len(values) else None,
                           maximum=float(np.max(values)) if len(values) else None))
    return output


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [dict(zip(keys, values), n=len(items),
                 **{method: sum(r["best_method"] == method for r in items) for method in METHODS})
            for values, items in sorted(groups.items())]


def build(out):
    plan = read(out / "plan.json")
    plan_sha = sha(out / "plan.json")
    baseline = read(out / "before_sha256.json")
    sources = {}

    def load(name):
        path = ROOT / name
        digest = sha(path)
        if name in baseline:
            assert digest == baseline[name], "historical source changed: " + name
        sources[name] = digest
        return read(path)

    metadata = [json.loads(line) for line in (DATA_DIR / "all_methods_equations_v1_metadata.jsonl").read_text().splitlines()]
    assert [m["equation_id"] for m in metadata] == plan["equation_ids"]
    assert sha(DATA_DIR / "all_methods_equations_v1.csv") == plan["dataset_sha256"]
    reference = {r["equation_id"]: r for r in read_csv(ROOT / plan["historical_sources"]["mean4_reference"])}
    assert len(reference) == 1000 and set(reference) == set(plan["equation_ids"])
    splits = load(plan["historical_sources"]["splits"])
    old_test = splits["scenarios"]["old"]["test"]["ids"]
    assert old_test == splits["scenarios"]["new"]["test"]["ids"] and len(set(old_test)) == 150
    latest = {pid: name for name, value in splits["scenarios"]["new"].items() for pid in value["ids"]}
    assert len(latest) == 997
    rows = []
    for m in metadata:
        pid = m["equation_id"]
        ref = reference[pid]
        assert ref["expression_hash"] == m["expression_hash"] and ref["family_id"] == m["family_id"]
        assert ref["dataset_sha256"] == plan["dataset_sha256"] and ref["split"] == m["split"]
        remizov, mc, exact = historical(m, ref, load)
        measures = dict(remizov=remizov, monte_carlo=mc)
        for method in ("rk4", "product_integral"):
            name = str((out / "records" / method / (pid + ".json")).relative_to(ROOT))
            record = validate_record(load(name), plan_sha, method, m)
            assert record["dataset_sha256"] == plan["dataset_sha256"]
            if record["usable"]:
                assert record["exact0"] == exact
            measures[method] = measurement(record["status"], record["usable"], record["y0_target"], exact,
                                           record["solver_wall_seconds"],
                                           reason=(record.get("result") or {}).get("failure_reason", record.get("exception", "")),
                                           source=[name], config=record["config"],
                                           result_diagnostics={k: v for k, v in (record.get("result") or {}).items()
                                                               if k not in {"x", "y", "dy"}},
                                           E_inf_grid33=record["E_inf_grid33"],
                                           time_scope="wrapper wall time excluding error evaluation and IO")
            assert measures[method]["E"] == record["E0"]
        available = [method for method in METHODS if measures[method]["usable"]]
        best, tied = winner({method: measures[method]["E"] for method in METHODS}, plan["tie_order"])
        row = dict(equation_id=pid, dataset_sha256=plan["dataset_sha256"],
                   **{key: m[key] for key in ("expression_hash", "family_id", "structure_signature", "complexity_level", "regime")},
                   split_original=m["split"], split_latest=latest.get(pid), is_previous_test=pid in old_test,
                   eligible_four_method_ml=len(available) == 4, available_methods=available,
                   missing_methods=[method for method in METHODS if method not in available],
                   best_method=best, best_method_scope="all_four" if len(available) == 4 else "available_only",
                   tied_methods=tied if len(tied) > 1 else [])
        for method in METHODS:
            row["E_" + method] = measures[method]["E"]
            for key in ("status", "usable", "y0", "accuracy_met", "seconds", "failure_reason", "diagnostics"):
                row[key + "_" + method] = measures[method][key]
        rows.append(row)
    complete = [r for r in rows if r["eligible_four_method_ml"]]
    incomplete = [r for r in rows if not r["eligible_four_method_ml"]]
    assert len({r["equation_id"] for r in rows}) == 1000
    release = out / "release"
    release.mkdir(exist_ok=False)
    write_csv(release / "coverage_all1000.csv", rows)
    write_csv(release / "ml_four_methods.csv", complete, list(rows[0]))
    write_csv(release / "incomplete_coverage.csv", incomplete, list(rows[0]))
    write_csv(release / "exact_ties.csv", [r for r in complete if r["tied_methods"]], list(rows[0]))
    method_stats = statistics(rows)
    write_csv(release / "method_statistics.csv", method_stats)
    write_csv(release / "method_statistics_common.csv", statistics(complete))
    group_specs = dict(complexity=["complexity_level"], regime=["regime"],
                       complexity_regime=["complexity_level", "regime"],
                       family=["family_id", "structure_signature", "complexity_level", "regime"],
                       original_split=["split_original"], latest_split=["split_latest"])
    for name, keys in group_specs.items():
        write_csv(release / ("winners_by_" + name + ".csv"), grouped(complete, keys),
                  keys + ["n", *METHODS])
    wins = [dict(method=method, wins=sum(r["best_method"] == method for r in complete)) for method in METHODS]
    write_csv(release / "winners.csv", wins)
    problems = []
    for method in METHODS:
        for row in rows:
            if not row["usable_" + method]:
                problems.append(dict(equation_id=row["equation_id"], method=method, category="unusable",
                                     rank=None, E0=None, status=row["status_" + method],
                                     reason=row["failure_reason_" + method], family_id=row["family_id"],
                                     regime=row["regime"], complexity_level=row["complexity_level"]))
        if method in ("rk4", "product_integral"):
            largest = sorted((r for r in rows if r["usable_" + method]),
                             key=lambda r: (-r["E_" + method], r["equation_id"]))[:10]
            for rank, row in enumerate(largest, 1):
                problems.append(dict(equation_id=row["equation_id"], method=method, category="largest_E0",
                                     rank=rank, E0=row["E_" + method], status=row["status_" + method],
                                     reason="fixed budget; no refinement or corrective run", family_id=row["family_id"],
                                     regime=row["regime"], complexity_level=row["complexity_level"]))
    write_csv(release / "problem_equations.csv", problems)
    atomic_json(release / "previous_test_ids.json", old_test)
    atomic_json(release / "previous_splits.json", splits)
    summary = dict(schema="four_method_accuracy_release_v1", rows=len(rows), complete_rows=len(complete),
                   incomplete_rows=len(incomplete), method_statistics=method_stats, winners=wins,
                   exact_ties=sum(bool(r["tied_methods"]) for r in complete), tie_order=plan["tie_order"],
                   original_split=dict(Counter(r["split_original"] for r in complete)),
                   latest_split=dict(Counter(r["split_latest"] for r in complete)),
                   previous_test_ids_preserved=True, families=len({r["family_id"] for r in complete}),
                   primary_metric="absolute_error_at_x0", epsilon=0.05,
                   feature_generation=False, ml_training=False, new_R_or_MC_calls=0,
                   producer_sessions=[read(p) for p in sorted(out.glob("session_*.json"))])
    atomic_json(release / "summary.json", summary)
    atomic_json(release / "ml_manifest.json", dict(
        labels_file="ml_four_methods.csv", rows=len(complete), target="best_method",
        join_keys=["dataset_sha256", "equation_id", "expression_hash"],
        feature_builder="ml.remizov_features.load_inputs", feature_builder_sha256=sha(ROOT / "ml/remizov_features.py"),
        input_dataset=str(DATA_DIR.relative_to(ROOT)), feature_generation=False,
        permitted_npz_entries=["X", "x_grid", "equation_id", "family_id", "split"],
        coefficient_channels=["a", "b", "c", "g"],
        coefficient_metadata_fields=["a_lower_cert", "a_upper_cert", "b_abs_cert", "remizov_c_abs_cert", "remizov_lambda"],
        permitted_ast_roles=["a_raw", "b_raw", "c_raw"],
        forbidden_as_features="all E_, y0_method, status_, usable_, accuracy_met_, seconds_, failure_reason_, diagnostics_, best_method, exact solutions, solver answers, solution AST; IDs/splits are join/group fields",
        split_policy="existing original and latest membership retained; no new split; previous Test already disclosed",
        previous_test_ids="previous_test_ids.json", ml_training=False))
    atomic_json(release / "source_sha256.json", sources)
    # Consumers are versioned here independently of the immutable numerical plan.
    atomic_json(release / "release_manifest.json", dict(
        plan_sha256=plan_sha,
        consumer_sha256={name: sha(ROOT / name) for name in
                         ("diagnostics/report_four_method_accuracy.py", "diagnostics/verify_four_method_accuracy.py")},
        artifact_sha256={str(p.relative_to(release)): sha(p) for p in sorted(release.iterdir()) if p.is_file()}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
