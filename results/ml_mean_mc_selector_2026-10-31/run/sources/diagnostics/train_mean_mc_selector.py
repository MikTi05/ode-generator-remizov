"""One frozen CatBoost fit: saved R128 versus the mean of four saved MC1024.

No numerical methods, equation generation, exact evaluation, parameter search,
class weighting, or validation/test tuning. The former release is immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from diagnostics.train_accuracy_selector import write_json, write_csv, summary
from ml.mean_mc_evidence import build_mean_evidence, PREVIOUS_RUN
from ml.mean_mc_selector import (MeanMCMethodSelector, AGGREGATION, CATBOOST_CONFIG,
                                EPSILON, METHODS, SCHEMA, selection_metrics, sha)
from ml.remizov_features import VERSION


def read(path):
    return json.loads(Path(path).read_text())


def run(out, plan_path):
    from catboost import CatBoostClassifier, Pool

    started = time.perf_counter()
    out = Path(out).resolve()
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    if (plan["schema"] != SCHEMA or plan["epsilon"] != EPSILON
            or plan["catboost"] != CATBOOST_CONFIG or plan["fit_count"] != 1
            or plan["monte_carlo"]["aggregation"] != AGGREGATION
            or any(plan[k] for k in ("hyperparameter_search", "class_balancing", "early_stopping", "test_tuning"))):
        raise ValueError("the pre-fit frozen protocol is required")
    previous = (ROOT / plan["source"]).resolve()
    if previous != PREVIOUS_RUN:
        raise ValueError("use the existing completed local experiment")
    if previous == out or previous in out.parents:
        raise ValueError("the previous release is immutable")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(plan_path, out / "plan.json")
    evidence = build_mean_evidence(previous)
    rows = evidence["main_rows"]
    coverage = evidence["coverage_rows"]
    shutil.copy2(previous / "input_features.npz", out / "input_features.npz")
    with np.load(out / "input_features.npz", allow_pickle=False) as archive:
        X = archive["X"].copy()
        names = archive["feature_names"].tolist()
        ids = archive["equation_id"].tolist()
        families = archive["family_id"].tolist()
        splits = archive["split"].tolist()
    if X.shape != (1000, 253) or not np.isfinite(X).all() or len(set(ids)) != 1000:
        raise ValueError("unchanged finite input-only features required")
    lookup = {pid:i for i, pid in enumerate(ids)}
    by_id = {r["equation_id"]:r for r in coverage}
    if set(ids) != set(by_id) or Counter(splits) != dict(train=700, validation=150, test=150):
        raise ValueError("original input identities or splits changed")
    family_groups, topology_groups = {}, {}
    for i, pid in enumerate(ids):
        row = by_id[pid]
        if (families[i], splits[i]) != (row["family_id"], row["split"]):
            raise ValueError("saved feature/evidence identity mismatch")
        for groups, key in ((family_groups, row["family_id"]), (topology_groups, row["structure_signature"])):
            if groups.setdefault(key, row["split"]) != row["split"]:
                raise ValueError("family or topology leakage")
    old_registry = read(previous / "model/registry.json")
    if names != old_registry["feature_names"]:
        raise ValueError("original feature schema changed")
    train_rows = [r for r in rows if r["split"] == "train"]
    train_indices = [lookup[r["equation_id"]] for r in train_rows]
    labels = [r["best_method"] for r in train_rows]
    if len(train_rows) != 697:
        raise ValueError("train coverage changed")
    fit_config = copy.deepcopy(plan["catboost"])
    model = CatBoostClassifier(**fit_config)
    fit_started = time.perf_counter()
    model.fit(Pool(X[train_indices], label=labels, feature_names=names))
    fit_seconds = time.perf_counter() - fit_started
    model_dir = out / "model"
    model_dir.mkdir()
    model.save_model(str(model_dir / "model.cbm"))
    context = dict(remizov=train_rows[0]["remizov_config"],
                   monte_carlo=dict(aggregation=AGGREGATION,
                       estimator="abs(mean(f_seed(0))-exact(0))", M=1024, M_per_seed=1024, aggregate_paths=4096,
                       dt=1/128, tail_eps=1e-4, seeds=plan["monte_carlo"]["seeds"],
                       cost="four saved runs of 1024 paths; no new single4096 run"),
                   metric="absolute_error_at_x0", domain="bounded solution on R")
    registry = dict(schema=SCHEMA, epsilon=EPSILON, methods=list(METHODS),
                    feature_version=VERSION, feature_names=names,
                    feature_code_sha256=sha(ROOT / "ml/remizov_features.py"),
                    input_grid_sha256=old_registry["input_grid_sha256"],
                    input_features_sha256=sha(out / "input_features.npz"),
                    model_sha256=sha(model_dir / "model.cbm"), catboost=fit_config, fit_count=1,
                    observed_training_classes=dict(Counter(labels)), single_class_training=len(set(labels)) == 1,
                    train_equations=[r["equation_id"] for r in train_rows],
                    train_families=sorted({r["family_id"] for r in train_rows}),
                    train_topologies=sorted({r["structure_signature"] for r in train_rows}),
                    comparison_context=context, class_balancing=False,
                    validation_test_used_for_fit_or_tuning=False, numerical_calls=0,
                    previous_run=str(previous.relative_to(ROOT)),
                    retrospective_reanalysis=True, prospective_independent_confirmation=False,
                    frozen_plan_sha256=sha(plan_path))
    write_json(model_dir / "registry.json", registry)
    restored = MeanMCMethodSelector(model_dir)
    metrics, previous_scores, by_split, prediction_rows = {}, {}, {}, []
    with (previous / "predictions.csv").open(newline="") as stream:
        previous_predictions = {r["equation_id"]:r for r in csv.DictReader(stream)}
    for split in ("train", "validation", "test"):
        own = [r for r in rows if r["split"] == split]
        values = X[[lookup[r["equation_id"]] for r in own]]
        choices, probabilities = restored.predict_features(values, names)
        np.testing.assert_array_equal(probabilities, model.predict_proba(Pool(values, feature_names=names), thread_count=1))
        metrics[split] = {name:selection_metrics(own, selected) for name, selected in (
            ("ML", choices), ("always_remizov", [METHODS[0]]*len(own)),
            ("always_monte_carlo", [METHODS[1]]*len(own)), ("oracle", [r["best_method"] for r in own]))}
        previous_scores[split] = selection_metrics(own, [previous_predictions[r["equation_id"]]["selected_method"] for r in own])
        by_split[split] = dict(original=Counter(splits)[split], summary=summary(own))
        for row, choice, probability in zip(own, choices, probabilities):
            selected_error = row["E_R"] if choice == METHODS[0] else row["E_MC"]
            prediction_rows.append(dict(equation_id=row["equation_id"], expression_hash=row["expression_hash"],
                family_id=row["family_id"], split=split, best_method=row["best_method"],
                E_R=row["E_R"], E_MC=row["E_MC"], mc_mean_f0=row["mc_mean_f0"],
                E_MC_median_seed_error=row["E_MC_median_seed_error"], selected_method=choice,
                E_selected=selected_error, selected_success=selected_error <= EPSILON,
                regret=selected_error-min(row["E_R"],row["E_MC"]),
                p_remizov=float(probability[0]), p_monte_carlo=float(probability[1])))
    importance = model.get_feature_importance()
    feature_rows = sorted((dict(name=name, importance=float(value)) for name,value in zip(names, importance)),
                          key=lambda r:r["importance"], reverse=True)
    prediction_by_id = {r["equation_id"]:r for r in prediction_rows}
    outliers = []
    for pid in ("AMV1_F0141_S00", "AMV1_F0180_S04"):
        row, prediction, prior = by_id[pid], prediction_by_id[pid], previous_predictions[pid]
        features = []
        for f in feature_rows[:10]:
            column = names.index(f["name"])
            value = float(X[lookup[pid], column])
            features.append(dict(f, value=value, train_percentile=float(100*np.mean(X[train_indices, column] <= value))))
        outliers.append(dict(prediction, y_exact0=row["y_exact0"], remizov_f0=row["remizov_f0"],
            mc_seed_f0=row["mc_seed_f0"], previous_selected_method=prior["selected_method"],
            previous_p_monte_carlo=float(prior["p_monte_carlo"]),
            prediction_scope="in-sample training recognition; not generalization" if row["split"] == "train"
                             else "retrospective held-family test; no tuning allowed",
            risk_interpretation="p_MC estimates lower aggregate error, not probability of catastrophic R error",
            features=features, diagnostic_features_used_for_selection_or_refit=False))
    audit = dict(paired_summary=summary(rows), by_split=by_split, fit_count=1,
                 catboost_fit_seconds=fit_seconds, numerical_calls=0, exact_evaluations=0,
                 feature_count=len(names), feature_source=str((previous / "input_features.npz").relative_to(ROOT)),
                 features_copied_byte_for_byte=sha(out / "input_features.npz") == sha(previous / "input_features.npz"),
                 family_topology_split_verified=True, retrospective_reanalysis=True,
                 train_classes=dict(Counter(labels)), previous_release_unchanged=True,
                 no_automatic_followup_experiments=True)
    write_json(out / "evidence.json", evidence)
    write_json(out / "evidence_audit.json", evidence["audit"])
    write_json(out / "metrics.json", metrics)
    write_json(out / "data_audit.json", audit)
    write_json(out / "previous_model_comparison.json", dict(
        old_metrics_under_old_target=read(previous / "metrics.json"),
        old_policy_under_mean_target=previous_scores, new_policy_under_mean_target=metrics,
        comparison_note="Old decisions reused unchanged; new target errors applied to both policies"))
    write_json(out / "outliers.json", outliers)
    write_csv(out / "feature_importance.csv", feature_rows)
    write_csv(out / "comparable_x0.csv", rows)
    write_csv(out / "coverage.csv", coverage)
    write_csv(out / "predictions.csv", prediction_rows)
    write_json(out / "confusion_matrix_test.json", dict(class_order=list(METHODS), rows="true winner", columns="ML choice",
        n=metrics["test"]["ML"]["n"], matrix=metrics["test"]["ML"]["confusion_matrix"]))
    shutil.copy2(previous / "example_input.json", out / "example_input.json")
    payload = read(out / "example_input.json")
    write_json(out / "example_prediction.json", restored.predict(**payload))
    pins = dict(evidence["source_sha256"])
    pins[str(plan_path.relative_to(ROOT))] = sha(plan_path)
    sources = ["ml/mean_mc_selector.py", "ml/mean_mc_evidence.py", "ml/accuracy_selector.py", "ml/remizov_features.py",
               "diagnostics/train_mean_mc_selector.py", "diagnostics/predict_mean_mc_selector.py",
               "diagnostics/report_mean_mc_selector.py", "diagnostics/verify_mean_mc_selector.py",
               "diagnostics/train_accuracy_selector.py", "tests/test_mean_mc_selector.py", "requirements-ml.txt"]
    for name in sources:
        source = ROOT / name
        if not source.is_file():
            raise ValueError("expected frozen source missing: " + name)
        dest = out / "sources" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        pins[name] = sha(source)
    write_json(out / "source_sha256.json", pins)
    from diagnostics.report_mean_mc_selector import build_report
    build_report(out)
    write_json(out / "environment.json", dict(python=platform.python_version(), platform=platform.platform(),
        versions={p:importlib.metadata.version(p) for p in ("numpy", "catboost", "scikit-learn")},
        thread_count=1, fit_count=1, numerical_calls=0, wall_seconds=time.perf_counter()-started))
    write_json(out / "artifact_sha256.json", {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps(dict(out=str(out), main_pairs=len(rows), train_classes=dict(Counter(labels)), test=metrics["test"],
                         outliers=[{k:r[k] for k in ("equation_id","split","selected_method","p_monte_carlo","E_R","E_MC","E_selected","selected_success")} for r in outliers]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    run(args.out, args.plan)
