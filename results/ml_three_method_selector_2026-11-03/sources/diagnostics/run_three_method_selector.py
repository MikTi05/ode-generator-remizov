"""Saved-only three-method experiment: prepare, train, cv, final, report.

No ODE solver, equation generator, or exact evaluator is called. Historical
answers are validated with the existing four-method consumer contracts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from diagnostics.four_method_common import validate_record
from diagnostics.report_four_method_accuracy import historical
from ml.remizov_features import load_inputs
from ml.selector_research_features import additional_coefficient_features, FEATURE_NAMES
from ml.three_method_selector import (METHODS, CONFIGS, WEIGHTS, SEED, EPSILON,
    winner, metrics, decide, selection_key, policy_scan, assert_separated,
    fit_bundle, score_bundle, load_bundle, calibration, ThreeMethodSelector)

DATA = ROOT / "data/all_methods_v1_1000_2026-10-18"
FOUR = ROOT / "results/four_method_accuracy_1000_2026-11-01"
PREVIOUS = ROOT / "results/ml_selector_hypotheses_2026-10-31"
MEAN = ROOT / "results/ml_mean_mc_selector_2026-10-31/run"
NEW_SOURCES = ["ml/three_method_selector.py", "diagnostics/run_three_method_selector.py"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    # Rewriting progress/ledger files is confined to this new experiment.
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def csv_read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text("")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def summary(rows):
    counts = Counter(r["best_method"] for r in rows)
    return dict(n=len(rows), families=len({r["family_id"] for r in rows}), classes={m: counts[m] for m in METHODS})


def check_pins(out):
    pins = read(out / "preserved_sha256.json")
    changed = [p for p, value in pins.items() if not (ROOT / p).is_file() or sha(ROOT / p) != value]
    if changed:
        raise ValueError("previous files changed: " + str(changed[:10]))
    return dict(checked_files=len(pins), changed_files=changed)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "PROJECT_STATUS.md", out / "PROJECT_STATUS.before.md")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    pins = {p: sha(ROOT / p) for p in tracked if p and p != "PROJECT_STATUS.md" and (ROOT / p).is_file()}
    write(out / "preserved_sha256.json", pins)
    sources = {}

    def pin_read(path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        sources[str(path.relative_to(ROOT))] = sha(path)
        return read(path)

    old = {r["equation_id"]: r for r in csv_read(MEAN / "coverage.csv")}
    release = {r["equation_id"]: r for r in csv_read(FOUR / "release/coverage_all1000.csv")}
    for p in (MEAN / "coverage.csv", FOUR / "release/coverage_all1000.csv"):
        sources[str(p.relative_to(ROOT))] = sha(p)
    plan4 = pin_read(FOUR / "plan.json")
    old_splits = pin_read(PREVIOUS / "splits.json")
    metas = [json.loads(s) for s in (DATA / "all_methods_equations_v1_metadata.jsonl").read_text().splitlines()]
    sources[str((DATA / "all_methods_equations_v1_metadata.jsonl").relative_to(ROOT))] = sha(DATA / "all_methods_equations_v1_metadata.jsonl")
    latest = {pid: split for split, group in old_splits["scenarios"]["new"].items() for pid in group["ids"]}
    coverage = []
    for m in metas:
        pid = m["equation_id"]
        r, mc, exact = historical(m, old[pid], pin_read)
        pi_path = FOUR / "records/product_integral" / (pid + ".json")
        pi = validate_record(pin_read(pi_path), sha(FOUR / "plan.json"), "product_integral", m)
        if pi["usable"]:
            if pi["exact0"] != exact:
                raise ValueError("methods have different analytic targets")
            pi_value = pi["result"]["y"][64]
            pi_error = abs(pi_value - exact)
            if pi_error != pi["E0"]:
                raise ValueError("PI error mismatch")
        else:
            pi_value = pi_error = None
        row = dict(**{k: m[k] for k in ("equation_id", "family_id", "structure_signature", "expression_hash", "regime", "complexity_level")},
            dataset_sha256=plan4["dataset_sha256"], split_original=m["split"], split_latest=latest.get(pid),
            y_exact0=exact, f_remizov=r["y0"], f_monte_carlo=mc["y0"], f_product_integral=pi_value,
            E_remizov=r["E"], E_monte_carlo=mc["E"], E_product_integral=pi_error,
            eligible=bool(r["usable"] and mc["usable"] and pi["usable"]),
            status={"remizov": r["status"], "monte_carlo": mc["status"], "product_integral": pi["status"]},
            sources={"remizov": r["diagnostics"]["source"], "monte_carlo": mc["diagnostics"]["source"], "product_integral": [str(pi_path.relative_to(ROOT))]},
            config={"remizov": r["diagnostics"]["config"], "monte_carlo": dict(aggregation="arithmetic_mean_of_four_saved_MC1024_answers", seeds=mc["diagnostics"]["seeds"], configs=json.loads(old[pid]["mc_configs"])), "product_integral": pi["config"]},
            best_method=None)
        if row["eligible"]:
            row["best_method"] = winner([row["E_" + method] for method in METHODS])
        row["missing_methods"] = [method for method in METHODS if row["E_" + method] is None]
        for method in METHODS:
            expected = release[pid]["E_" + method]
            if row["E_" + method] != (float(expected) if expected else None):
                raise ValueError("four-method release error changed")
        coverage.append(row)
    rows = [r for r in coverage if r["eligible"]]
    if len(coverage) != 1000 or len(rows) != 997:
        raise ValueError("saved result eligibility changed")
    scenarios = {}
    for scenario, key in (("old", "split_original"), ("new", "split_latest")):
        scenarios[scenario] = {split: dict(ids=[r["equation_id"] for r in rows if r[key] == split], **summary([r for r in rows if r[key] == split])) for split in ("train", "validation", "test")}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            assert_separated([r for r in rows if r[key] == a], [r for r in rows if r[key] == b])
        for split in scenarios[scenario]:
            if scenarios[scenario][split]["ids"] != old_splits["scenarios"][scenario][split]["ids"]:
                raise ValueError("historical split membership changed")
    inputs = load_inputs(DATA)
    lookup = {str(pid): i for i, pid in enumerate(inputs["ids"])}
    indices = [lookup[r["equation_id"]] for r in rows]
    X = inputs["X"][indices]
    with np.load(MEAN / "input_features.npz", allow_pickle=False) as z:
        old_lookup = {pid: i for i, pid in enumerate(z["equation_id"].tolist())}
        if not np.array_equal(X, z["X"][[old_lookup[r["equation_id"]] for r in rows]]) or inputs["feature_names"] != z["feature_names"].tolist():
            raise ValueError("original253 feature implementation changed")
    with np.load(DATA / "all_methods_equations_v1_ml.npz", allow_pickle=False) as z:
        raw_lookup = {pid: i for i, pid in enumerate(z["equation_id"].tolist())}
        raw, x = z["X"], z["x_grid"]
        extra = np.array([list(additional_coefficient_features(raw[raw_lookup[r["equation_id"]]], x).values()) for r in rows])
    np.savez_compressed(out / "features.npz", X=X, augmented=np.c_[X, extra],
        feature_names=inputs["feature_names"], augmented_names=inputs["feature_names"] + list(FEATURE_NAMES),
        equation_id=[r["equation_id"] for r in rows], family_id=[r["family_id"] for r in rows])
    for name in ("all_methods_equations_v1.csv", "all_methods_equations_v1_ml.npz"):
        sources[str((DATA / name).relative_to(ROOT))] = sha(DATA / name)
    sources[str((MEAN / "input_features.npz").relative_to(ROOT))] = sha(MEAN / "input_features.npz")
    write(out / "dataset.json", dict(rows=rows, coverage=coverage, source_sha256=sources,
        summary=summary(rows), numerical_calls=0, new_equations=0))
    csv_write(out / "labels.csv", rows)
    csv_write(out / "coverage_all1000.csv", coverage)
    csv_write(out / "incomplete_coverage.csv", [r for r in coverage if not r["eligible"]])
    write(out / "splits.json", dict(seed=SEED, scenarios=scenarios, source=str((PREVIOUS / "splits.json").relative_to(ROOT)), family_leakage=False, topology_leakage=False))
    plan = dict(schema="three_method_selector_v1", seed=SEED, methods=list(METHODS), epsilon=EPSILON,
        primary_scenario="new", sensitivity_scenario="old", classifier_configs=CONFIGS, weights=WEIGHTS,
        class_weights="[1,w,w], no resampling or modification of labels/errors", features=["original253", "augmented258"],
        classifier_base=dict(iterations=120, depth=2, l2_leaf_reg=10, learning_rate=.05, loss_function="MultiClass", random_seed=SEED, bootstrap_type="No", random_strength=0, thread_count=1),
        regression="two RMSE heads delta_MC=E_R-E_MC, delta_PI=E_R-E_PI; raw and asinh(delta/train_median_abs_delta)",
        policy="argmax(scores+[0,bias_MC,bias_PI]); classifier scores log raw probabilities; regression scores [0,delta_MC,delta_PI]; ties R,MC,PI",
        policy_search="old binary tau=.05:.05:.50 mapped to log((1-tau)/tau), 100 classifier coarse pairs; regression coordinate-boundary quantiles; 2 starts, <=3 coordinate partition sweeps, boundaries and both nextafter neighbours; local search, not certified global 2D optimum",
        selection="minimum Validation mean_regret, higher success, higher fixed-three-class Macro-F1, fewer non-R predictions, stable candidate ID/policy; latest primary fixed before fit, original sensitivity",
        cv="5 outer StratifiedGroupKFold within each Train, first 4-fold inner holdout; baseline original253 six weights+raw+asinh+three constants; inner policy/spec selected independently then corresponding outer refit evaluated",
        fit_budget=dict(main_specifications=96, main_estimator_fits=120, cv_estimator_fits=200, total_estimator_fits=320),
        numerical_calls=0, new_equations=0, test_tuning=False, retrospective=True, prospective_independent_test=False,
        resources=dict(processes=1, catboost_threads=1, blas_threads=1),
        input_grid_sha256=inputs["grid_sha256"], data_sha256=sha(out / "dataset.json"), splits_sha256=sha(out / "splits.json"),
        feature_sha256=sha(out / "features.npz"), source_sha256={p: sha(ROOT / p) for p in NEW_SOURCES + ["ml/remizov_features.py", "ml/selector_research_features.py", "ml/accuracy_selector.py", "diagnostics/report_four_method_accuracy.py", "diagnostics/four_method_common.py"]})
    write(out / "plan.json", plan)
    write(out / "environment.json", dict(python=sys.version, platform=platform.platform(), versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "sympy", "catboost", "scikit-learn", "pandas")}, threads={k: v for k, v in os.environ.items() if "THREAD" in k}))
    write(out / "fit_ledger.json", [])
    for folder in ("models", "cv_models", "policy_search", "sources"):
        (out / folder).mkdir()
    for path in plan["source_sha256"]:
        dest = out / "sources" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, dest)
    print("PREPARED", summary(rows), {s: {k: v["n"] for k, v in groups.items()} for s, groups in scenarios.items()}, flush=True)


def load_data(out, development=False):
    rows = read(out / "dataset.json")["rows"]
    if development:
        rows = [r for r in rows if r["split_original"] != "test"]
    with np.load(out / "features.npz", allow_pickle=False) as z:
        lookup = {pid: i for i, pid in enumerate(z["equation_id"].tolist())}
        indices = [lookup[r["equation_id"]] for r in rows]
        matrices = {key: z[key][indices] for key in ("X", "augmented")}
        names = dict(X=z["feature_names"].tolist(), augmented=z["augmented_names"].tolist())
    return rows, matrices, names


def fit_save(out, stage, scenario, candidate, X, rows, names, fold=None):
    began = time.perf_counter()
    models, scales = fit_bundle(candidate["kind"], candidate["config"], candidate["weight"], X, rows, names)
    paths = []
    ledger = read(out / "fit_ledger.json")
    directory = "models" if stage == "main" else "cv_models"
    for head, model in enumerate(models):
        name = f'{directory}/{candidate["id"]}_head{head}.cbm'
        if (out / name).exists():
            raise ValueError("refusing to replace a fitted model")
        model.save_model(str(out / name))
        paths.append(name)
        ledger.append(dict(stage=stage, scenario=scenario, candidate_id=candidate["id"], fold=fold, head=head, model_path=name, train_ids=[r["equation_id"] for r in rows], classes=summary(rows)["classes"]))
    write(out / "fit_ledger.json", ledger)
    candidate.update(models=paths, model_sha256={p: sha(out / p) for p in paths}, scales=scales, fit_seconds=time.perf_counter() - began)
    return models


def candidate_scores(candidate, out, X, names, models=None):
    if candidate["kind"] == "constant":
        scores = np.zeros((len(X), 3))
        scores[:, METHODS.index(candidate["constant"])] = 1.
        return scores, None
    return score_bundle(models if models is not None else load_bundle(candidate, out), candidate["kind"], candidate["scales"], X, names)


def prediction_rows(rows, scores, probabilities, biases, **identity):
    chosen = decide(scores, biases)
    for i, (r, method) in enumerate(zip(rows, chosen)):
        value = r["E_" + method]
        yield dict(**identity, equation_id=r["equation_id"], family_id=r["family_id"], best_method=r["best_method"], selected_method=method,
            scores=scores[i].tolist(), probabilities=None if probabilities is None else probabilities[i].tolist(), biases=list(biases),
            E_selected=value, regret=value - min(r["E_" + m] for m in METHODS))


def constants(scenario):
    return [dict(id=f"{scenario}__always_{m}", scenario=scenario, kind="constant", constant=m, feature_set="X", config={}, weight=1, models=[], scales=[]) for m in METHODS]


def train(out):
    if (out / "TRAIN_STARTED.json").exists():
        raise ValueError("training already started; use a new experiment directory")
    check_pins(out)
    write(out / "TRAIN_STARTED.json", dict(started=datetime.now(timezone.utc).isoformat(), plan_sha256=sha(out / "plan.json")))
    rows, matrices, names = load_data(out, development=True)
    candidates, policies, predictions, importance, calibration_rows = [], [], [], [], []
    for scenario, split_key in (("old", "split_original"), ("new", "split_latest")):
        ti = [i for i, r in enumerate(rows) if r[split_key] == "train"]
        vi = [i for i, r in enumerate(rows) if r[split_key] == "validation"]
        tr, va = [rows[i] for i in ti], [rows[i] for i in vi]
        for c in constants(scenario):
            candidates.append(c)
            scores, proba = candidate_scores(c, out, matrices["X"][vi], names["X"])
            policies.append(dict(candidate_id=c["id"], scenario=scenario, policy="constant", biases=[0., 0.], **metrics(va, decide(scores))))
            predictions.extend(prediction_rows(va, scores, proba, [0., 0.], candidate_id=c["id"], scenario=scenario, policy="constant", split="validation"))
        for feature_set in ("X", "augmented"):
            X, own_names = matrices[feature_set], names[feature_set]
            for kind in ("classifier", "raw", "asinh"):
                for weight in (WEIGHTS if kind == "classifier" else [1]):
                    for config in CONFIGS:
                        cid = f'{scenario}__{feature_set}__{kind}__w{weight}__{config["name"]}'
                        c = dict(id=cid, scenario=scenario, kind=kind, feature_set=feature_set, config=config, weight=weight, constant=None)
                        models = fit_save(out, "main", scenario, c, X[ti], tr, own_names)
                        scores, probabilities = candidate_scores(c, out, X[vi], own_names, models)
                        table, best = policy_scan(va, scores, kind)
                        write(out / "policy_search" / (cid + ".json"), dict(candidate_id=cid, visited=table, best=best, validation_ids=[r["equation_id"] for r in va]))
                        candidates.append(c)
                        for policy, biases in (("default", [0., 0.]), ("validation_optimal", best["biases"])):
                            result = metrics(va, decide(scores, biases))
                            policies.append(dict(candidate_id=cid, scenario=scenario, policy=policy, biases=biases, **result))
                            predictions.extend(prediction_rows(va, scores, probabilities, biases, candidate_id=cid, scenario=scenario, policy=policy, split="validation"))
                        values = np.mean([m.get_feature_importance() for m in models], axis=0)
                        importance.extend(dict(candidate_id=cid, feature=n, importance=float(v)) for n, v in zip(own_names, values))
                        calibration_rows.append(dict(candidate_id=cid, scenario=scenario, split="validation", calibration=calibration(va, probabilities)))
                        print("FIT", cid, "regret", best["mean_regret"], "F1", best["macro_f1"], flush=True)
    write(out / "candidates.json", candidates)
    write(out / "validation_metrics.json", policies)
    csv_write(out / "validation_metrics.csv", policies)
    csv_write(out / "validation_predictions.csv", predictions)
    csv_write(out / "feature_importance.csv", importance)
    write(out / "validation_calibration.json", calibration_rows)
    selected = {scenario: min((p for p in policies if p["scenario"] == scenario), key=lambda p: selection_key(p) + (p["candidate_id"], p["policy"])) for scenario in ("old", "new")}
    main_fits = sum(e["stage"] == "main" for e in read(out / "fit_ledger.json"))
    if main_fits != 120:
        raise ValueError("unexpected main estimator fit count")
    write(out / "selection_lock.json", dict(primary_scenario="new", selected=selected["new"], selected_by_scenario=selected,
        test_evaluated=False, test_used_for_selection=False, primary_scenario_fixed_before_fit=True,
        main_estimator_fits=main_fits, main_specifications=96,
        hashes={name: sha(out / name) for name in ("plan.json", "dataset.json", "features.npz", "splits.json", "candidates.json", "validation_metrics.json")}))
    print("LOCKED", {s: (p["candidate_id"], p["biases"]) for s, p in selected.items()}, flush=True)


def group_split(rows, n_splits):
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(
        np.zeros((len(rows), 1)), [METHODS.index(r["best_method"]) for r in rows], [r["family_id"] for r in rows])


def cv(out):
    if (out / "CV_STARTED.json").exists() or (out / "FINAL_STARTED.json").exists():
        raise ValueError("CV must run once before final evaluation")
    write(out / "CV_STARTED.json", dict(seed=SEED, expected_estimator_fits=200))
    rows, matrices, names = load_data(out, development=True)
    X, own_names = matrices["X"], names["X"]
    folds, records, predictions, inner_records, cv_candidates = [], [], [], [], []
    specs = [dict(name=f"classifier_w{w}", kind="classifier", weight=w) for w in WEIGHTS] + [dict(name=k, kind=k, weight=1) for k in ("raw", "asinh")]
    for scenario, split_key in (("old", "split_original"), ("new", "split_latest")):
        indices = np.asarray([i for i, r in enumerate(rows) if r[split_key] == "train"])
        own = [rows[i] for i in indices]
        for fold, (local_t, local_v) in enumerate(group_split(own, 5), 1):
            ti, vi = indices[local_t], indices[local_v]
            tr, va = [rows[i] for i in ti], [rows[i] for i in vi]
            inner_t, inner_v = next(iter(group_split(tr, 4)))
            it, iv = ti[inner_t], ti[inner_v]
            itr, iva = [rows[i] for i in it], [rows[i] for i in iv]
            assert_separated(tr, va)
            assert_separated(itr, iva)
            folds.append(dict(scenario=scenario, fold=fold, outer_train_ids=[r["equation_id"] for r in tr], outer_validation_ids=[r["equation_id"] for r in va], inner_train_ids=[r["equation_id"] for r in itr], inner_validation_ids=[r["equation_id"] for r in iva]))
            outer_scores, choices_inner = {}, []
            for spec in specs:
                for scope, a, b, fit_rows, valid_rows in (("outer", ti, vi, tr, va), ("inner", it, iv, itr, iva)):
                    cid = f'cv_{scenario}_f{fold}_{scope}_{spec["name"]}'
                    c = dict(id=cid, scenario=scenario, kind=spec["kind"], feature_set="X", config=CONFIGS[0], weight=spec["weight"], constant=None)
                    models = fit_save(out, "cv_" + scope, scenario, c, X[a], fit_rows, own_names, fold)
                    cv_candidates.append(c)
                    scores, probabilities = candidate_scores(c, out, X[b], own_names, models)
                    if scope == "outer":
                        outer_scores[spec["name"]] = (scores, probabilities, cid)
                        records.append(dict(scenario=scenario, fold=fold, policy=spec["name"], chosen_spec=spec["name"], **metrics(va, decide(scores))))
                        predictions.extend(prediction_rows(va, scores, probabilities, [0., 0.], candidate_id=cid, scenario=scenario, fold=fold, policy=spec["name"], split="outer_validation"))
                    else:
                        table, best = policy_scan(iva, scores, spec["kind"])
                        choices_inner.append(dict(spec=spec["name"], candidate_id=cid, **best))
                        write(out / "policy_search" / (cid + ".json"), dict(candidate_id=cid, visited=table, best=best, validation_ids=[r["equation_id"] for r in iva]))
                        # Save the actual inner scores so selection is independently auditable.
                        inner_records.extend(prediction_rows(iva, scores, probabilities, best["biases"], candidate_id=cid, scenario=scenario, fold=fold, policy=spec["name"], split="inner_validation"))
            for c in constants(scenario):
                sid = "always_" + c["constant"]
                scores, probabilities = candidate_scores(c, out, X[vi], own_names)
                outer_scores[sid] = (scores, probabilities, c["id"])
                records.append(dict(scenario=scenario, fold=fold, policy=sid, chosen_spec=sid, **metrics(va, decide(scores))))
                predictions.extend(prediction_rows(va, scores, probabilities, [0., 0.], candidate_id=c["id"], scenario=scenario, fold=fold, policy=sid, split="outer_validation"))
                choices_inner.append(dict(spec=sid, candidate_id=c["id"], biases=[0., 0.], **metrics(iva, [c["constant"]] * len(iva))))
            best = min(choices_inner, key=lambda p: selection_key(p) + (p["spec"],))
            scores, probabilities, cid = outer_scores[best["spec"]]
            records.append(dict(scenario=scenario, fold=fold, policy="nested_selection", chosen_spec=best["spec"], biases=best["biases"], inner_candidates=choices_inner, **metrics(va, decide(scores, best["biases"]))))
            predictions.extend(prediction_rows(va, scores, probabilities, best["biases"], candidate_id=cid, scenario=scenario, fold=fold, policy="nested_selection", split="outer_validation"))
            write(out / "cv_progress.json", dict(completed_folds=len(folds), estimator_fits=len(read(out / "fit_ledger.json")) - 120))
            print("CV", scenario, fold, best["spec"], records[-1]["mean_regret"], flush=True)
    cv_fits = sum(e["stage"].startswith("cv_") for e in read(out / "fit_ledger.json"))
    if cv_fits != 200:
        raise ValueError("unexpected CV estimator count")
    write(out / "cv_candidates.json", cv_candidates)
    write(out / "cv_splits.json", dict(seed=SEED, folds=folds))
    write(out / "cv_folds.json", records)
    csv_write(out / "cv_folds.csv", records)
    csv_write(out / "cv_predictions.csv", predictions)
    csv_write(out / "cv_inner_predictions.csv", inner_records)
    summaries = []
    by_id = {r["equation_id"]: r for r in rows}
    for scenario, policy in sorted({(r["scenario"], r["policy"]) for r in records}):
        own = [r for r in records if r["scenario"] == scenario and r["policy"] == policy]
        pred = [r for r in predictions if r["scenario"] == scenario and r["policy"] == policy]
        if len({r["equation_id"] for r in pred}) != len(pred):
            raise ValueError("duplicate OOF predictions")
        pooled = metrics([by_id[r["equation_id"]] for r in pred], [r["selected_method"] for r in pred])
        variability = {key: dict(mean=float(np.mean([r[key] for r in own])), sd=float(np.std([r[key] for r in own], ddof=1))) for key in ("accuracy", "macro_f1", "mean_error", "mean_regret", "success_rate")}
        summaries.append(dict(scenario=scenario, policy=policy, pooled=pooled, fold_variability=variability))
    write(out / "cv_summary.json", dict(fit_count=cv_fits, outer_folds=5, inner_holdout="first SGKF 4-fold validation", policy_summary=summaries, test_labels_accessed=False, global_selected_hyperparameters_used=False, numerical_calls=0))


def final(out):
    if (out / "FINAL_STARTED.json").exists():
        raise ValueError("one-shot final stage already started")
    lock = read(out / "selection_lock.json")
    if not (out / "cv_summary.json").exists():
        raise ValueError("complete CV before final Test")
    for path, expected in lock["hashes"].items():
        if sha(out / path) != expected:
            raise ValueError("locked artifact changed")
    check_pins(out)
    write(out / "FINAL_STARTED.json", dict(selection_lock_sha256=sha(out / "selection_lock.json"), time=datetime.now(timezone.utc).isoformat(), all_policies_frozen=True))
    rows, matrices, names = load_data(out)
    policies = read(out / "validation_metrics.json")
    candidates = read(out / "candidates.json")
    test_metrics, train_metrics, test_predictions, train_predictions, calibrations = [], [], [], [], []
    for c in candidates:
        own_policies = [p for p in policies if p["candidate_id"] == c["id"]]
        split_key = "split_original" if c["scenario"] == "old" else "split_latest"
        for split, output, pred_output in (("test", test_metrics, test_predictions), ("train", train_metrics, train_predictions)):
            indices = [i for i, r in enumerate(rows) if r[split_key] == split]
            own = [rows[i] for i in indices]
            scores, probabilities = candidate_scores(c, out, matrices[c["feature_set"]][indices], names[c["feature_set"]])
            calibrations.append(dict(candidate_id=c["id"], split=split, calibration=calibration(own, probabilities)))
            for p in own_policies:
                chosen = decide(scores, p["biases"])
                selected = all(p[k] == lock["selected_by_scenario"][c["scenario"]][k] for k in ("candidate_id", "policy", "biases"))
                output.append(dict(candidate_id=c["id"], scenario=c["scenario"], policy=p["policy"], biases=p["biases"], selected_on_validation=selected, **metrics(own, chosen)))
                # Test: every frozen policy. Train: only preselected policies (in-sample diagnostics).
                if split == "test" or selected:
                    pred_output.extend(prediction_rows(own, scores, probabilities, p["biases"], candidate_id=c["id"], scenario=c["scenario"], policy=p["policy"], split=split))
    test_rows = [r for r in rows if r["split_original"] == "test"]
    selected = {s: next(p for p in test_metrics if p["scenario"] == s and p["selected_on_validation"]) for s in ("old", "new")}
    write(out / "test_metrics.json", dict(policies=test_metrics, selected_by_scenario=selected, selected=selected["new"], oracle=metrics(test_rows, [r["best_method"] for r in test_rows]), retrospective=True, independent_test=False, test_evaluation_passes=1))
    write(out / "train_metrics.json", dict(policies=train_metrics, interpretation="in-sample, not generalization evidence"))
    csv_write(out / "test_metrics.csv", test_metrics)
    csv_write(out / "train_metrics.csv", train_metrics)
    csv_write(out / "test_predictions.csv", test_predictions)
    csv_write(out / "train_predictions.csv", train_predictions)
    write(out / "calibration.json", calibrations)
    # Export the preselected latest candidate as a self-contained reloadable bundle.
    chosen = next(c for c in candidates if c["id"] == lock["selected"]["candidate_id"])
    folder = out / "selected_model"
    folder.mkdir()
    exported = dict(chosen)
    exported["models"] = []
    for path in chosen["models"]:
        dest = folder / Path(path).name
        shutil.copy2(out / path, dest)
        exported["models"].append(dest.name)
    registry = dict(methods=list(METHODS), epsilon=EPSILON, candidate=exported, biases=lock["selected"]["biases"],
        feature_names=names[chosen["feature_set"]], input_grid_sha256=read(out / "plan.json")["input_grid_sha256"],
        model_sha256={name: sha(folder / name) for name in exported["models"]}, selection_lock_sha256=sha(out / "selection_lock.json"),
        training_only=True, test_used_for_selection=False, feature_builder="ml.remizov_features.coefficient_features + optional existing additional_coefficient_features", numerical_calls=0)
    write(folder / "registry.json", registry)
    restored = ThreeMethodSelector(folder)
    indices = [i for i, r in enumerate(rows) if r["split_original"] == "test"]
    reloaded, _ = restored.predict_features(matrices[chosen["feature_set"]][indices], names[chosen["feature_set"]])
    expected = [p["selected_method"] for p in test_predictions if p["scenario"] == "new" and p["candidate_id"] == chosen["id"] and p["policy"] == lock["selected"]["policy"]]
    if reloaded != expected:
        raise ValueError("exported model predictions changed")
    write(out / "reload_verification.json", dict(test_predictions_checked=len(expected), exact_match=True))
    write(out / "preservation_verification.json", check_pins(out))
    print("FINAL", selected, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "cv", "final"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.out.resolve())
