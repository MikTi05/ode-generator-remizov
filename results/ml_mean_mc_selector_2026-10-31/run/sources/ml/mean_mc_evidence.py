"""Re-label saved x0 evidence using the mean of four existing MC answers.

Only the completed median-error experiment is read. No equation evaluator,
numerical solver, trajectory generation, or artifact writing is used here.
Four equally sized 1024-path runs form a pooled 4096-path estimator, with the
cost of all four saved runs; this is not a newly executed single MC4096 run.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_RUN = ROOT / "results/ml_accuracy_selector_2026-10-31/run"
EPSILON = 0.05
SEEDS = [202610250101, 202610250102, 202610250103, 202610250104]
METHODS = ("remizov", "monte_carlo")


def _check(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(rows):
    return dict(pairs=len(rows), winners={m:sum(r["best_method"] == m for r in rows) for m in METHODS},
                remizov_success=sum(r["remizov_success"] for r in rows),
                mc_success=sum(r["mc_success"] for r in rows),
                oracle_success=sum(r["chosen_success"] for r in rows))


def _relabel(row):
    row = copy.deepcopy(row)
    values = np.asarray(row.get("mc_seed_f0"), dtype=float)
    exact = row.get("y_exact0")
    seeds = row.get("mc_seeds")
    configs = row.get("mc_configs")
    _check(values.shape == (4,) and np.isfinite(values).all() and
           isinstance(exact, (int, float)) and math.isfinite(exact),
           "four finite saved MC values and one finite saved exact value required")
    _check(row.get("mc_seed_count") == 4 and seeds == SEEDS and len(set(seeds)) == 4 and
           isinstance(configs, list) and len(configs) == 4,
           "frozen four independent MC seeds required")
    for seed, config in zip(seeds, configs):
        _check(config.get("seed") == seed and config.get("M") == 1024 and
               config.get("dt") == 1/128 and config.get("tail_eps") == 1e-4 and
               config.get("stream_id") == 0, "saved MC physical profile changed")
    physical = [{key:value for key, value in c.items() if key != "seed"} for c in configs]
    _check(all(c == physical[0] for c in physical), "MC seeds have different physical parameters")
    seed_errors = np.abs(values - exact)
    _check(np.array_equal(seed_errors, np.asarray(row.get("mc_seed_errors"), dtype=float)),
           "saved MC absolute seed errors disagree with saved values")
    previous_error = float(np.median(seed_errors))
    _check(row.get("epsilon") == EPSILON and row.get("E_MC") == previous_error and
           row.get("mc_success") == (previous_error <= EPSILON),
           "previous median-error target is inconsistent")
    _check(len(row.get("mc_sources", [])) == 4 and len(set(row["mc_sources"])) == 4,
           "MC evidence contains duplicate seed sources")

    mean_f0 = float(np.mean(values))
    error = abs(mean_f0 - exact)
    previous_label = row.get("best_method")
    row.update(E_MC_median_seed_error=previous_error,
               best_method_median_seed_error=previous_label,
               mc_success_median_seed_error=bool(previous_error <= EPSILON),
               mc_mean_f0=mean_f0, E_MC=error, mc_success=bool(error <= EPSILON),
               mc_estimator="arithmetic_mean_of_four_saved_1024_path_answers",
               mc_total_paths=4096,
               mc_execution_semantics="four existing independent runs of 1024 paths; pooled mean and total four-run cost",
               cohort="original_mass_mean4_x0",
               context="mass_R128_taper_vs_MC_mean4_equal_1024_path_runs_x0")
    er, fr = row.get("E_R"), row.get("remizov_f0")
    _check(row.get("remizov_config", {}).get("n") == 128 and
           row["remizov_config"].get("boundary") == "taper",
           "primary Remizov profile must remain the saved mass R128/taper")
    if er is None:
        _check(fr is None and previous_label is None and row.get("pair_available") is False and
               row.get("remizov_success") is None and row.get("chosen_success") is None and
               bool(row.get("missing_reason")), "missing Remizov answer has a synthetic target")
        row.update(best_method=None, chosen_success=None)
    else:
        _check(isinstance(fr, (int, float)) and math.isfinite(fr) and
               math.isfinite(er) and er >= 0 and er == abs(fr - exact) and
               row.get("pair_available") is True, "saved Remizov value/error mismatch")
        _check(previous_label == ("remizov" if er <= previous_error else "monte_carlo") and
               row.get("remizov_success") == (er <= EPSILON) and
               row.get("chosen_success") == (min(er, previous_error) <= EPSILON),
               "previous pair label disagrees with continuous errors")
        row.update(best_method="remizov" if er <= error else "monte_carlo",
                   chosen_success=bool(min(er, error) <= EPSILON))
    row["target_changed_from_median_seed_error"] = row["best_method"] != previous_label
    return row


def build_mean_evidence(previous_run=PREVIOUS_RUN):
    """Return main997/coverage1000 with the new mean-answer target and provenance.

    Source checks read existing bytes only. All source values and source model
    artifacts remain unchanged; only the returned in-memory rows are relabeled.
    """
    previous_run = Path(previous_run).resolve()
    _check(ROOT == previous_run or ROOT in previous_run.parents,
           "the saved previous run must be inside this local repository")
    pins = {}

    def pin(path):
        path = Path(path)
        _check(path.resolve() == ROOT or ROOT in path.resolve().parents, "source escaped repository")
        name = str(path.relative_to(ROOT))
        digest = _sha(path)
        _check(name not in pins or pins[name] == digest, "source changed during mean evidence read")
        pins[name] = digest
        return digest

    def read(path):
        pin(path)
        return json.loads(Path(path).read_text())

    artifact_path = previous_run / "artifact_sha256.json"
    if artifact_path.exists():
        manifest = read(artifact_path)
        for name, expected in manifest.items():
            _check(pin(previous_run / name) == expected, "previous immutable artifact changed: " + name)
    evidence = read(previous_run / "evidence.json")
    registry = read(previous_run / "model/registry.json")
    metrics = read(previous_run / "metrics.json")
    for name in ("input_features.npz", "predictions.csv", "model/model.cbm", "example_input.json"):
        pin(previous_run / name)
    _check(registry.get("epsilon") == EPSILON and registry.get("methods") == list(METHODS) and
           registry.get("comparison_context", {}).get("remizov", {}).get("n") == 128 and
           registry.get("comparison_context", {}).get("monte_carlo", {}).get("estimator") ==
           "median(abs(f_seed(0)-exact(0)))", "unsupported previous experiment target")
    _check(pin(previous_run / "model/model.cbm") == registry["model_sha256"],
           "previous saved model hash mismatch")

    # Numerical/data source hashes in the old evidence are strict when files
    # remain available. Historical code hashes stay in the immutable source
    # manifest and snapshots; new research may extend the current source tree.
    unavailable = []
    old_pins = evidence.get("source_sha256", {})
    _check(isinstance(old_pins, dict) and bool(old_pins), "previous evidence lacks provenance")
    for name, expected in old_pins.items():
        path = ROOT / name
        if path.exists():
            _check(pin(path) == expected, "previous numerical evidence source changed: " + name)
        else:
            unavailable.append(name)
    provenance_path = previous_run / "source_sha256.json"
    historical_code_changes = []
    if provenance_path.exists():
        old_source_manifest = read(provenance_path)
        for name, expected in old_source_manifest.items():
            if name in old_pins:
                _check(expected == old_pins[name], "previous source manifests disagree")
            elif (ROOT / name).exists():
                current_sha = _sha(ROOT / name)
                if current_sha == expected:
                    pin(ROOT / name)
                else:
                    snapshot = previous_run / "sources" / name
                    _check(snapshot.exists() and pin(snapshot) == expected,
                           "changed historical code has no verified snapshot: " + name)
                    historical_code_changes.append(name)

    old_main = evidence.get("main_rows", [])
    old_coverage = evidence.get("coverage_rows", [])
    _check(len(old_main) == 997 and len(old_coverage) == 1000, "previous comparable coverage changed")
    coverage_by_id = {r["equation_id"]: r for r in old_coverage}
    main_by_id = {r["equation_id"]: r for r in old_main}
    _check(len(coverage_by_id) == 1000 and len(main_by_id) == 997 and
           set(main_by_id) <= set(coverage_by_id), "duplicate or unknown equation IDs")
    _check(all(coverage_by_id[pid] == row for pid, row in main_by_id.items()),
           "previous main/coverage rows disagree")
    dataset_shas = {r.get("dataset_sha256") for r in old_coverage}
    _check(len(dataset_shas) == 1 and next(iter(dataset_shas)) ==
           evidence["audit"]["canonical_dataset_sha256"], "mixed canonical dataset identity")
    identities, family_splits, topology_splits = set(), {}, {}
    for row in old_coverage:
        identity = (row["dataset_sha256"], row["equation_id"], row["expression_hash"])
        _check(len(row["expression_hash"]) == 64 and identity not in identities,
               "invalid or duplicate expression-qualified identity")
        identities.add(identity)
        for groups, key in ((family_splits, row["family_id"]),
                            (topology_splits, row["structure_signature"])):
            _check(groups.setdefault(key, row["split"]) == row["split"], "family split leakage")
        _check(all(source in old_pins for source in row["mc_sources"] + row["remizov_sources"]),
               "row source lacks original evidence hash")
    _check(Counter(r["split"] for r in old_coverage) == dict(train=700, validation=150, test=150),
           "original split changed")
    coverage = [_relabel(row) for row in old_coverage]
    main = [r for r in coverage if r["pair_available"]]
    _check({r["equation_id"] for r in main} == set(main_by_id) and
           Counter(r["split"] for r in main) == dict(train=697, validation=150, test=150),
           "mean target changed comparable equation coverage")
    changes = [dict(equation_id=r["equation_id"], split=r["split"],
                    previous=r["best_method_median_seed_error"], current=r["best_method"],
                    E_R=r["E_R"], E_MC_median_seed_error=r["E_MC_median_seed_error"], E_MC=r["E_MC"])
               for r in main if r["target_changed_from_median_seed_error"]]
    audit = dict(
        schema="saved_mc_mean4_x0_evidence_v1", epsilon=EPSILON,
        previous_run=str(previous_run.relative_to(ROOT)),
        previous_evidence_sha256=pins[str((previous_run / "evidence.json").relative_to(ROOT))],
        paired_summary=_summary(main),
        by_split={s:_summary([r for r in main if r["split"] == s]) for s in ("train", "validation", "test")},
        previous_paired_summary=_summary(old_main),
        target_changes=len(changes), target_change_rows=changes,
        target_changes_by_split=dict(Counter(r["split"] for r in changes)),
        target_change_directions=dict(Counter(r["previous"] + "->" + r["current"] for r in changes)),
        original_split=dict(train=700, validation=150, test=150),
        comparable_split=dict(train=697, validation=150, test=150),
        missing_pairs=[r["equation_id"] for r in coverage if not r["pair_available"]],
        original_x0_missing_pair_count=1000-len(main),
        standalone_mc_success=sum(r["mc_success"] for r in coverage),
        canonical_dataset_sha256=next(iter(dataset_shas)),
        exact_values_source="unchanged values saved in the previous completed evidence; no exact evaluation",
        numerical_solver_calls=0, trajectory_generation_calls=0, exact_evaluation_calls=0,
        numerical_parameters_changed=False, continuous_errors_preserved=True,
        comparison_without_rounding=True, tie_rule="exact error equality chooses remizov",
        estimator="abs(mean(f_seed(0), four equal 1024-path runs)-saved_exact(0))",
        paths_per_seed=1024, independent_seeds=SEEDS, total_paths=4096,
        computational_cost="sum of four existing independent 1024-path runs",
        not_a_new_single_4096_path_experiment=True,
        previous_registry=registry, previous_metrics=metrics,
        previous_evidence_audit=evidence["audit"],
        previous_source_files_unavailable=unavailable,
        changed_current_code_with_verified_historical_snapshots=historical_code_changes,
        previous_artifacts_unchanged=True, source_files=len(pins),
        scope="x=0 only; historical n256 and grid33 comparisons remain separate and are not relabeled here",
    )
    return dict(main_rows=main, coverage_rows=coverage, source_sha256=pins, audit=audit)
