"""Independently verify the saved three-method ML study, without any fitting.

Raw labels are reconstructed from the archived R128, four MC1024 answers and
PI trajectories. Metrics and bias decisions use NumPy/sklearn independently of
the training driver's policy helpers. Only verification.json is written.
"""
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
from sklearn.metrics import (confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support, roc_auc_score)

METHODS = ("remizov", "monte_carlo", "product_integral")
EXPECTED_MISSING = {"AMV1_F0134_S03", "AMV1_F0136_S01", "AMV1_F0139_S00"}
DATA = ROOT / "data/all_methods_v1_1000_2026-10-18"
FOUR = ROOT / "results/four_method_accuracy_1000_2026-11-01"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, label):
    if not condition:
        raise AssertionError(label)


def close(actual, expected, label):
    """Compare the independently known subset of an artifact contract."""
    if isinstance(expected, dict):
        check(isinstance(actual, dict), label + ": expected mapping")
        for key, value in expected.items():
            check(key in actual, label + ": missing " + key)
            close(actual[key], value, label + "." + key)
    elif isinstance(expected, (list, tuple)):
        check(len(actual) == len(expected), label + ": length")
        for index, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, label + "." + str(index))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        check(np.isclose(actual, expected, rtol=2e-11, atol=2e-12),
              label + ": " + repr(actual) + " != " + repr(expected))
    else:
        check(actual == expected, label + ": value")


def decoded_csv(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for row in rows:
        decoded = {}
        for key, value in row.items():
            if value == "":
                decoded[key] = None
            elif value in {"True", "False"}:
                decoded[key] = value == "True"
            else:
                try:
                    decoded[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    decoded[key] = value
        output.append(decoded)
    return output


def independent_metrics(rows, predictions):
    check(len(rows) == len(predictions) and len(rows) > 0, "metric support")
    truth = [r["best_method"] for r in rows]
    check(set(predictions) <= set(METHODS), "prediction outside three methods")
    matrix = np.array([[r["E_" + m] for m in METHODS] for r in rows])
    chosen = np.array([METHODS.index(m) for m in predictions])
    errors = matrix[np.arange(len(rows)), chosen]
    regret = errors - matrix.min(axis=1)
    cm = confusion_matrix(truth, predictions, labels=list(METHODS))
    precision, recall, fscore, support = precision_recall_fscore_support(
        truth, predictions, labels=list(METHODS), zero_division=0)
    available_recall = recall[support > 0]
    counts = {m: predictions.count(m) for m in METHODS}
    return dict(n=len(rows), class_order=list(METHODS), confusion_matrix=cm.tolist(),
                accuracy=float(np.mean(np.array(truth) == np.array(predictions))),
                balanced_accuracy=float(np.mean(available_recall)),
                macro_f1=float(f1_score(truth, predictions, labels=list(METHODS),
                                        average="macro", zero_division=0)),
                success_rate=float(np.mean(errors <= .05)),
                success_count=int(np.sum(errors <= .05)),
                mean_error=float(errors.mean()), median_error=float(np.median(errors)),
                max_error=float(errors.max()), mean_regret=float(regret.mean()),
                median_regret=float(np.median(regret)), max_regret=float(regret.max()),
                true_class_counts={m: truth.count(m) for m in METHODS},
                selection_counts=counts,
                selection_fractions={m: counts[m] / len(rows) for m in METHODS},
                non_R_predictions=int(np.sum(chosen != 0)),
                selection_errors=int(np.sum(np.array(truth) != np.array(predictions))),
                precision={m: float(precision[i]) for i, m in enumerate(METHODS)},
                recall={m: float(recall[i]) if support[i] else None for i, m in enumerate(METHODS)},
                f1={m: float(fscore[i]) for i, m in enumerate(METHODS)})


def metric_subset(actual, expected, label):
    required = ("n", "accuracy", "macro_f1", "confusion_matrix", "mean_error",
                "mean_regret", "success_rate")
    for key in required:
        check(key in actual, label + ": required metric " + key)
    close(actual, {k: v for k, v in expected.items() if k in actual}, label)


def rank(metric):
    non_r = metric.get("non_R_predictions", metric["n"] - metric["selection_counts"][METHODS[0]])
    return (metric["mean_regret"], -metric["success_rate"], -metric["macro_f1"],
            non_r, metric.get("candidate_id", ""), metric.get("policy", ""))


def verify_raw(out):
    from all_methods_dataset_loader import load_all_methods_equations

    dataset = read(out / "dataset.json")
    rows, coverage = dataset["rows"], dataset["coverage"]
    check(len(rows) == 997 and len(coverage) == 1000, "997/1000 coverage")
    by_id = {r["equation_id"]: r for r in rows}
    all_by_id = {r["equation_id"]: r for r in coverage}
    check(len(by_id) == 997 and len(all_by_id) == 1000, "unique equation IDs")
    check(len({r["expression_hash"] for r in rows}) == 997, "unique expressions")
    check(set(all_by_id) - set(by_id) == EXPECTED_MISSING, "only historical missing R excluded")
    for name, digest in dataset["source_sha256"].items():
        check(sha(ROOT / name) == digest, "source changed: " + name)

    equations = load_all_methods_equations(DATA / "all_methods_equations_v1.csv",
                                         DATA / "all_methods_equations_v1_metadata.jsonl")
    four_plan_sha = sha(FOUR / "plan.json")
    dataset_sha = sha(DATA / "all_methods_equations_v1.csv")
    checked = Counter()
    labels, errors, ties, large = Counter(), {}, [], []
    for e in equations:
        pid, meta = e.equation_id, e.metadata
        row = all_by_id[pid]
        for key in ("family_id", "structure_signature", "expression_hash"):
            check(row[key] == meta[key], "raw identity " + key)
        exact = float(e.y_exact(0.0))
        check(row["y_exact0"] == exact, "canonical analytic exact(0)")
        check(row["dataset_sha256"] == dataset_sha,
              "input dataset hash")
        check(row["split_original"] == meta["split"], "original split")
        measured = {}
        source_paths = []

        path = ROOT / row["sources"]["remizov"][0]
        source_paths.append(path)
        raw = read(path)
        checked["remizov_records"] += 1
        check(raw["problem_id"] == pid and raw["config"]["n"] == 128
              and raw["config"]["boundary"] == "taper", "R128 profile")
        check(row["config"]["remizov"] == raw["config"], "saved R config")
        if pid in EXPECTED_MISSING:
            check(raw["computation_status"] == "worker_exception" and raw["result"] is None,
                  "historical R missing status")
            check(row["E_remizov"] is None and row["f_remizov"] is None and not row["eligible"],
                  "missing R has synthetic value/label")
        else:
            result = raw["result"]
            check(raw["computation_status"] == "success" and result["success"]
                  and result["failure_reason"] == "", "R admissibility")
            check(raw["metadata"]["expression_hash"] == meta["expression_hash"], "R identity")
            check(result["x"] == np.linspace(-1, 1, 33).tolist()
                  and len(result["f"]) == 33 and np.isfinite(result["f"]).all(), "R grid/finite")
            check(raw["y_exact"][16] == exact, "R analytic target")
            measured["remizov"] = float(result["f"][16])

        mc = []
        paths = row["sources"]["monte_carlo"]
        check(len(paths) == 4 and len(set(paths)) == 4, "four distinct MC sources")
        for index, name in enumerate(paths):
            path = ROOT / name
            source_paths.append(path)
            raw = read(path)
            result = raw["result"]
            checked["MC_seed_records"] += 1
            check(raw["problem_id"] == result["problem_id"] == pid
                  and raw["expression_hash"] == meta["expression_hash"], "MC identity")
            check(raw["seed_index"] == index and raw["config"]["seed"] == 202610250101 + index
                  and raw["config"]["M"] == 1024 and raw["config"]["dt"] == 1 / 128,
                  "MC frozen profile")
            check(raw["status"] == "completed" and result["success"]
                  and result["failure_reason"] == "" and result["x"] == [0.0]
                  and result["completed_paths"] == [1024], "MC admissibility")
            check(raw["y_exact"] == exact and math.isfinite(result["f"][0]), "MC target")
            check(abs(result["f"][0] - exact) == raw["abs_error"], "MC per-seed error")
            mc.append(result["f"][0])
        measured["monte_carlo"] = float(np.mean(np.asarray(mc, dtype=np.float64)))

        path = ROOT / row["sources"]["product_integral"][0]
        source_paths.append(path)
        raw = read(path)
        checked["PI_records"] += 1
        body = {k: v for k, v in raw.items() if k != "record_sha256"}
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()
        check(hashlib.sha256(encoded).hexdigest() == raw["record_sha256"], "PI record seal")
        check(raw["plan_sha256"] == four_plan_sha and raw["method"] == "product_integral"
              and raw["expression_hash"] == meta["expression_hash"], "PI provenance")
        check(raw["config"] == row["config"]["product_integral"], "PI config")
        check(raw["inputs"] == {k: getattr(e.cauchy_problem, k) for k in ("x0", "x1", "y0", "v0")},
              "PI initial conditions")
        result = raw["result"]
        check(raw["usable"] and raw["status"] == "success" and result["success"]
              and result["failure_reason"] == "", "PI admissibility")
        check(result["x"] == [-1 + i / 64 for i in range(129)]
              and all(len(result[k]) == 129 and np.isfinite(result[k]).all()
                      for k in ("x", "y", "dy")), "PI grid/finite")
        check(raw["exact0"] == exact and raw["y0_target"] == result["y"][64], "PI target")
        measured["product_integral"] = float(result["y"][64])
        check(raw["E0"] == abs(measured["product_integral"] - exact), "PI raw error")
        for path in source_paths:
            check(str(path.relative_to(ROOT)) in dataset["source_sha256"], "unhashed raw source")
        for method, value in measured.items():
            check(row["f_" + method] == value, "saved " + method + " value")
            check(row["E_" + method] == abs(value - exact), "saved " + method + " error")
        if pid not in EXPECTED_MISSING:
            check(row["eligible"] and by_id[pid] == row, "eligible rows/coverage differ")
            minimum = min(row["E_" + m] for m in METHODS)
            winners = [m for m in METHODS if row["E_" + m] == minimum]
            check(row["best_method"] == winners[0], "three-method minimum label")
            labels[winners[0]] += 1
            if len(winners) > 1:
                ties.append(pid)
            if row["E_remizov"] > .05:
                large.append(pid)
            errors[pid] = measured
    check(dict(labels) == {"remizov": 826, "monte_carlo": 106, "product_integral": 65},
          "independent observed class census")
    check(len(large) == 15 and "AMV1_F0141_S00" in large, "large finite R answers were removed")
    return by_id, dict(checked, classes=dict(labels), exact_ties=len(ties),
                       large_R_errors_retained=len(large), missing_R=sorted(EXPECTED_MISSING))


def verify_features(out, by_id):
    from ml.remizov_features import load_inputs
    from ml.selector_research_features import additional_coefficient_features, FEATURE_NAMES

    inputs = load_inputs(DATA)
    with np.load(out / "features.npz", allow_pickle=False) as saved:
        ids = saved["equation_id"].tolist()
        check(len(ids) == 997 and set(ids) == set(by_id), "feature IDs")
        lookup = {str(pid): i for i, pid in enumerate(inputs["ids"])}
        matrix = inputs["X"][[lookup[pid] for pid in ids]]
        np.testing.assert_array_equal(saved["X"], matrix)
        check(saved["feature_names"].tolist() == inputs["feature_names"], "253 feature names")
        with np.load(DATA / "all_methods_equations_v1_ml.npz", allow_pickle=False) as z:
            coefficients, x, coefficient_ids = z["X"], z["x_grid"], z["equation_id"].tolist()
        c_lookup = {pid: i for i, pid in enumerate(coefficient_ids)}
        extra = np.array([list(additional_coefficient_features(coefficients[c_lookup[pid]], x).values())
                          for pid in ids])
        np.testing.assert_array_equal(saved["augmented"], np.column_stack([matrix, extra]))
        check(saved["augmented_names"].tolist() == inputs["feature_names"] + list(FEATURE_NAMES),
              "258 feature names")
        check(saved["X"].shape == (997, 253) and saved["augmented"].shape == (997, 258),
              "253/258 feature shapes")
        result = {"equation_id": ids, "X": saved["X"].copy(),
                  "augmented": saved["augmented"].copy(),
                  "feature_names": saved["feature_names"].tolist(),
                  "augmented_names": saved["augmented_names"].tolist()}
    historical = ROOT / "results/ml_mean_mc_selector_2026-10-31/run/input_features.npz"
    with np.load(historical, allow_pickle=False) as z:
        old_lookup = {pid: i for i, pid in enumerate(z["equation_id"].tolist())}
        np.testing.assert_array_equal(matrix, z["X"][[old_lookup[pid] for pid in ids]])
    return result, dict(rows=997, base_features=253, augmented_features=258,
                        reconstructed_values=997 * 258, historical253_identical=True)


def verify_splits(out, by_id):
    manifest = read(out / "splits.json")
    scenarios = manifest.get("scenarios", manifest)
    old_splits = read(FOUR / "release/previous_splits.json")["scenarios"]
    for scenario, source_key in (("old", "split_original"), ("new", "split_latest")):
        union = set()
        for split in ("train", "validation", "test"):
            part = scenarios[scenario][split]
            ids = part["ids"]
            check(len(ids) == len(set(ids)) and not union.intersection(ids), "split duplicate/overlap")
            check(set(ids) == {pid for pid, row in by_id.items() if row[source_key] == split},
                  "historical " + scenario + " membership")
            check(set(ids) == set(old_splits[scenario][split]["ids"]), "historical split IDs")
            union.update(ids)
            own = [by_id[pid] for pid in ids]
            close(part, dict(n=len(ids), classes=dict(Counter(r["best_method"] for r in own)),
                             families=len({r["family_id"] for r in own})), "split summary")
        check(union == set(by_id), "split full997 partition")
        for key in ("family_id", "structure_signature"):
            owner = {}
            for split in ("train", "validation", "test"):
                for pid in scenarios[scenario][split]["ids"]:
                    check(owner.setdefault(by_id[pid][key], split) == split, "split " + key + " leakage")
    check(scenarios["old"]["test"]["ids"] == scenarios["new"]["test"]["ids"], "preserved Test")
    return scenarios


def independent_scores(candidate, out, features):
    """Reload CBMs independently and reproduce score construction, never fit."""
    kind = candidate["kind"]
    own = features[candidate["feature_set"]]
    names = features["augmented_names" if candidate["feature_set"] == "augmented" else "feature_names"]
    if kind == "constant":
        values = np.zeros((len(own), 3))
        values[:, METHODS.index(candidate["constant"])] = 1.
        return values, None
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    pool = Pool(own, feature_names=names)
    models = []
    for name in candidate["models"]:
        model = CatBoostClassifier() if kind == "classifier" else CatBoostRegressor()
        model.load_model(str(out / name))
        check(list(model.feature_names_) == names, "CBM input feature names")
        models.append(model)
    if kind == "classifier":
        check(len(models) == 1, "one classifier per candidate")
        raw = models[0].predict_proba(pool, thread_count=1)
        probabilities = np.zeros((len(own), 3))
        for index, label in enumerate(models[0].classes_):
            probabilities[:, METHODS.index(str(label))] = raw[:, index]
        check(np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1),
              "CBM three-class probabilities")
        return np.log(np.maximum(probabilities, np.finfo(float).tiny)), probabilities
    check(kind in {"raw", "asinh"} and len(models) == len(candidate["scales"]) == 2,
          "two regression heads per candidate")
    output = [np.asarray(model.predict(pool, thread_count=1), dtype=float) for model in models]
    if kind == "asinh":
        output = [scale * np.sinh(prediction) for scale, prediction in zip(candidate["scales"], output)]
    return np.column_stack([np.zeros(len(own)), *output]), None


def independent_calibration(rows, probabilities):
    if probabilities is None:
        return None
    truth = np.asarray([METHODS.index(r["best_method"]) for r in rows])
    indicators = np.eye(3)[truth]
    confidence, predictions = probabilities.max(axis=1), probabilities.argmax(axis=1)
    bins = []
    for index in range(10):
        mask = (confidence >= index / 10) & ((confidence < (index + 1) / 10)
                 if index < 9 else confidence <= 1)
        if mask.any():
            bins.append(dict(lower=index / 10, n=int(mask.sum()),
                             confidence=float(confidence[mask].mean()),
                             accuracy=float(np.mean(predictions[mask] == truth[mask]))))
    return dict(brier_multiclass=float(np.mean(np.sum((probabilities - indicators) ** 2, axis=1))),
                logloss=float(log_loss(truth, probabilities, labels=[0, 1, 2])),
                ece=float(sum(b["n"] * abs(b["confidence"] - b["accuracy"]) for b in bins) / len(rows)),
                roc_auc_ovr={m: float(roc_auc_score(indicators[:, j], probabilities[:, j]))
                             if len(np.unique(indicators[:, j])) == 2 else None
                             for j, m in enumerate(METHODS)}, bins=bins)


def verify_predictions(out, by_id, features, scenarios):
    candidates = read(out / "candidates.json")
    if isinstance(candidates, dict):
        candidates = candidates["candidates"]
    by_candidate = {r["id"]: r for r in candidates}
    check(len(by_candidate) == len(candidates), "unique candidate IDs")
    expected_configs = {"baseline": dict(name="baseline", iterations=120, depth=2, l2_leaf_reg=10),
                        "medium": dict(name="medium", iterations=240, depth=3, l2_leaf_reg=10),
                        "deeper": dict(name="deeper", iterations=300, depth=4, l2_leaf_reg=20)}
    expected_grid = {(scenario, feature_set, kind, config, weight)
        for scenario in ("old", "new") for feature_set in ("X", "augmented")
        for kind in ("classifier", "raw", "asinh") for config in expected_configs
        for weight in ([1, 2, 3, 5, 8, 12] if kind == "classifier" else [1])}
    actual_grid = {(c["scenario"], c["feature_set"], c["kind"], c["config"]["name"], c["weight"])
                   for c in candidates if c["kind"] != "constant"}
    check(actual_grid == expected_grid and len([c for c in candidates if c["kind"] != "constant"]) == 96,
          "complete96 main hyperparameter specifications")
    for candidate in candidates:
        if candidate["kind"] != "constant":
            check(candidate["config"] == expected_configs[candidate["config"]["name"]], "unchanged CatBoost grid")
    check({(c["scenario"], c["constant"]) for c in candidates if c["kind"] == "constant"}
          == {(s, m) for s in ("old", "new") for m in METHODS} and len(candidates) == 102,
          "six constant baselines plus96 trained specifications")
    lookup = {pid: i for i, pid in enumerate(features["equation_id"])}
    caches = {}

    def scores_for(candidate_id):
        if candidate_id not in caches:
            caches[candidate_id] = independent_scores(by_candidate[candidate_id], out, features)
        return caches[candidate_id]

    metric_sets = {}
    prediction_sets = {}
    counts = {}
    for split in ("validation", "test", "train"):
        path = out / (split + "_predictions.csv")
        check(path.is_file(), "saved " + split + " predictions required")
        predictions = decoded_csv(path)
        grouped = defaultdict(list)
        for record in predictions:
            candidate_id = record["candidate_id"]
            check(candidate_id in by_candidate, "prediction references unknown candidate")
            candidate = by_candidate[candidate_id]
            check(record["scenario"] == candidate["scenario"] and record["split"] == split,
                  "prediction scenario/split identity")
            grouped[(candidate_id, record["scenario"], record["policy"])].append(record)
        for key, own in grouped.items():
            candidate_id, scenario, policy = key
            ids = [record["equation_id"] for record in own]
            check(ids == scenarios[scenario][split]["ids"], "saved prediction ID order/support")
            all_scores, all_probabilities = scores_for(candidate_id)
            indices = [lookup[pid] for pid in ids]
            expected_scores = all_scores[indices]
            expected_probabilities = None if all_probabilities is None else all_probabilities[indices]
            for i, record in enumerate(own):
                row = by_id[record["equation_id"]]
                check(record["best_method"] == row["best_method"], "prediction true label")
                close(record["scores"], expected_scores[i].tolist(), "reloaded model scores")
                if expected_probabilities is None:
                    check(record["probabilities"] is None, "regression/constant has probabilities")
                else:
                    close(record["probabilities"], expected_probabilities[i].tolist(), "reloaded probabilities")
                offsets = np.r_[0., record["biases"]]
                chosen = METHODS[int(np.argmax(np.asarray(record["scores"]) + offsets))]
                check(record["selected_method"] == chosen, "independent bias argmax")
                close(record["E_selected"], row["E_" + chosen], "selected absolute error")
                close(record["regret"], row["E_" + chosen] - min(row["E_" + m] for m in METHODS),
                      "selected regret")
            metric_sets[(split,) + key] = independent_metrics([by_id[pid] for pid in ids],
                                                            [r["selected_method"] for r in own])
            prediction_sets[(split,) + key] = own
        counts[split + "_prediction_rows"] = len(predictions)

    validation = read(out / "validation_metrics.json")
    if isinstance(validation, dict):
        validation = validation.get("policies", validation.get("metrics"))
    test = read(out / "test_metrics.json")
    check(isinstance(validation, list) and isinstance(test["policies"], list), "saved policy metric lists")
    for split, metrics in (("validation", validation), ("test", test["policies"])):
        for record in metrics:
            key = (split, record["candidate_id"], record["scenario"], record["policy"])
            check(key in metric_sets, "metric missing corresponding predictions")
            metric_subset(record, metric_sets[key], split + " policy metrics")
            own = prediction_sets[key]
            check(all(r["biases"] == record["biases"] for r in own), "fixed policy biases")
    lock = read(out / "selection_lock.json")
    check(lock["primary_scenario"] == "new", "predeclared latest split primary")
    for scenario in ("old", "new"):
        own = [r for r in validation if r["scenario"] == scenario]
        selected = min(own, key=rank)
        check(lock["selected_by_scenario"][scenario] == selected, "Validation-only model selection")
        actual = next(r for r in test["policies"] if r["candidate_id"] == selected["candidate_id"]
                      and r["scenario"] == scenario and r["policy"] == selected["policy"])
        close(test["selected_by_scenario"][scenario], actual, "locked Test policy")
    check(lock["selected"] == lock["selected_by_scenario"]["new"], "primary selected from new only")
    close(test["selected"], test["selected_by_scenario"]["new"], "primary Test result")
    train_report = {}
    train_metric_path = out / "train_metrics.json"
    if train_metric_path.is_file():
        records = read(train_metric_path)
        records = records.get("policies", records) if isinstance(records, dict) else records
        for record in records:
            key = ("train", record["candidate_id"], record["scenario"], record["policy"])
            if key in metric_sets:
                expected = metric_sets[key]
            else:
                ids = scenarios[record["scenario"]]["train"]["ids"]
                scores = scores_for(record["candidate_id"])[0][[lookup[pid] for pid in ids]]
                choices = np.argmax(scores + np.r_[0., record["biases"]], axis=1)
                expected = independent_metrics([by_id[pid] for pid in ids], [METHODS[i] for i in choices])
            metric_subset(record, expected, "in-sample Train policy")
        train_report["train_metrics_checked"] = len(records)
    for path in (out / "validation_calibration.json", out / "calibration.json"):
        for record in read(path):
            candidate_id, split = record["candidate_id"], record["split"]
            scenario = by_candidate[candidate_id]["scenario"]
            ids = scenarios[scenario][split]["ids"]
            probabilities = scores_for(candidate_id)[1]
            expected = independent_calibration([by_id[pid] for pid in ids],
                       None if probabilities is None else probabilities[[lookup[pid] for pid in ids]])
            close(record["calibration"], expected, "independent raw-score calibration")
    for path, expected in lock["hashes"].items():
        check(sha(out / path) == expected, "Validation lock changed: " + path)
    check(lock["test_evaluated"] is False and lock["test_used_for_selection"] is False
          and lock["primary_scenario_fixed_before_fit"] is True, "Test selection contract")
    final_started = read(out / "FINAL_STARTED.json")
    check(final_started["selection_lock_sha256"] == sha(out / "selection_lock.json")
          and final_started["all_policies_frozen"] is True, "frozen one-shot final stage")
    test_ids = scenarios["new"]["test"]["ids"]
    metric_subset(test["oracle"], independent_metrics([by_id[pid] for pid in test_ids],
                  [by_id[pid]["best_method"] for pid in test_ids]), "three-method oracle")
    cv_report = verify_cv_and_fits(out, by_id, features, scenarios)
    return dict(counts, candidate_bundles_reloaded=len(caches), validation_policies=len(validation),
                test_policies=len(test["policies"]), selected_before_Test=True,
                primary_scenario="new", **train_report, cv=cv_report)


def verify_policy_search(path, candidate, scores, ids, by_id):
    """Audit every evaluated policy; this does not claim a global 2D optimum."""
    table = read(path)
    check(table["candidate_id"] == candidate["id"] and table["validation_ids"] == ids,
          "policy search candidate/validation membership")
    visited = table["visited"]
    biases = [tuple(r["biases"]) for r in visited]
    check(len(biases) == len(set(biases)) and (0., 0.) in biases, "policy search unique/default biases")
    rows = [by_id[pid] for pid in ids]
    errors = np.array([[r["E_" + method] for method in METHODS] for r in rows])
    truth = np.argmin(errors, axis=1)
    support = np.bincount(truth, minlength=3)
    oracle = errors.min(axis=1)
    for begin in range(0, len(visited), 512):
        chunk = visited[begin:begin + 512]
        offsets = np.column_stack([np.zeros(len(chunk)), [r["biases"] for r in chunk]])
        selected = np.argmax(scores[None, :, :] + offsets[:, None, :], axis=2)
        loss = errors[np.arange(len(ids))[None, :], selected]
        class_counts = np.column_stack([(selected == i).sum(axis=1) for i in range(3)])
        true_positives = np.column_stack([((selected == i) & (truth[None, :] == i)).sum(axis=1)
                                         for i in range(3)])
        denominator = support + class_counts
        fscore = np.divide(2 * true_positives, denominator,
                          out=np.zeros_like(true_positives, dtype=float), where=denominator != 0)
        for index, record in enumerate(chunk):
            close(record, dict(mean_regret=float(np.mean(loss[index] - oracle)),
                  success_rate=float(np.mean(loss[index] <= .05)), macro_f1=float(fscore[index].mean()),
                  non_R_predictions=int(class_counts[index, 1:].sum())), "policy search measured objective")
    best = min(visited, key=lambda r: (r["mean_regret"], -r["success_rate"], -r["macro_f1"],
               r["non_R_predictions"], sum(abs(v) for v in r["biases"]), *r["biases"]))
    check(table["best"]["biases"] == best["biases"], "policy search best among visited")
    decisions = np.argmax(scores + np.r_[0., best["biases"]], axis=1)
    metric_subset(table["best"], independent_metrics(rows, [METHODS[i] for i in decisions]),
                  "policy search best independent metrics")
    return len(visited)


def verify_cv_and_fits(out, by_id, features, scenarios):
    from sklearn.model_selection import StratifiedGroupKFold

    candidates = read(out / "candidates.json")
    cv_candidates = read(out / "cv_candidates.json")
    bundles = {r["id"]: r for r in candidates + cv_candidates}
    check(len(cv_candidates) == 160, "160 actual outer/inner CV bundles")
    lookup = {pid: i for i, pid in enumerate(features["equation_id"])}
    all_scores = {}

    def get_scores(candidate_id, ids):
        if candidate_id not in all_scores:
            all_scores[candidate_id] = independent_scores(bundles[candidate_id], out, features)
        scores, probabilities = all_scores[candidate_id]
        indices = [lookup[pid] for pid in ids]
        return scores[indices], None if probabilities is None else probabilities[indices]

    manifests = read(out / "cv_splits.json")["folds"]
    check(len(manifests) == 10, "ten outer CV folds")
    keyed_manifests = {(r["scenario"], r["fold"]): r for r in manifests}
    check(len(keyed_manifests) == 10, "unique CV fold IDs")
    for scenario in ("old", "new"):
        original = scenarios[scenario]["train"]["ids"]
        own = [by_id[pid] for pid in original]
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2488)
        # sklearn's tie ordering is tied to integer labels in the driver.
        partitions = splitter.split(np.zeros((len(own), 1)),
                                    [METHODS.index(r["best_method"]) for r in own],
                                    [r["family_id"] for r in own])
        for fold, (ti, vi) in enumerate(partitions, 1):
            manifest = keyed_manifests[(scenario, fold)]
            expected_t = [original[i] for i in ti]
            expected_v = [original[i] for i in vi]
            check(manifest["outer_train_ids"] == expected_t
                  and manifest["outer_validation_ids"] == expected_v, "seeded family outer CV reproduction")
            tr = [by_id[pid] for pid in expected_t]
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=2488)
            it, iv = next(iter(inner.split(np.zeros((len(tr), 1)),
                                [METHODS.index(r["best_method"]) for r in tr],
                                [r["family_id"] for r in tr])))
            check(manifest["inner_train_ids"] == [expected_t[i] for i in it]
                  and manifest["inner_validation_ids"] == [expected_t[i] for i in iv],
                  "seeded family inner CV reproduction")
            for left, right in ((expected_t, expected_v),
                                (manifest["inner_train_ids"], manifest["inner_validation_ids"])):
                for key in ("equation_id", "family_id", "structure_signature"):
                    check(not {by_id[pid][key] for pid in left}.intersection(by_id[pid][key] for pid in right),
                          "CV " + key + " overlap")
                check(not set(left + right).intersection(scenarios[scenario]["test"]["ids"]),
                      "CV used historical Test")

    ledger = read(out / "fit_ledger.json")
    check(len(ledger) == 320 and Counter(r["stage"] for r in ledger)
          == {"main": 120, "cv_outer": 100, "cv_inner": 100}, "320 actual estimator fit ledger")
    check(len({r["model_path"] for r in ledger}) == 320, "one path per actual estimator fit")
    all_model_paths = {p for candidate in candidates + cv_candidates for p in candidate["models"]}
    check(all_model_paths == {r["model_path"] for r in ledger}, "all320 estimators have ledger provenance")
    for record in ledger:
        candidate = bundles[record["candidate_id"]]
        check(record["scenario"] == candidate["scenario"] and record["model_path"] in candidate["models"],
              "fit ledger candidate/model identity")
        check(sha(out / record["model_path"]) == candidate["model_sha256"][record["model_path"]],
              "trained model checksum")
        if record["stage"] == "main":
            expected_ids = scenarios[record["scenario"]]["train"]["ids"]
        else:
            manifest = keyed_manifests[(record["scenario"], record["fold"])]
            expected_ids = manifest["outer_train_ids" if record["stage"] == "cv_outer" else "inner_train_ids"]
        check(record["train_ids"] == expected_ids, "fit used only declared Train family rows")
        check(not set(expected_ids).intersection(scenarios[record["scenario"]]["test"]["ids"]),
              "estimator fitted Test rows")
        close(record["classes"], dict(Counter(by_id[pid]["best_method"] for pid in expected_ids)),
              "fit ledger observed classes")
        if candidate["kind"] == "asinh":
            expected_scales = [max(float(np.median([abs(by_id[pid]["E_remizov"] - by_id[pid]["E_" + m])
                                                   for pid in expected_ids])), 1e-12)
                               for m in METHODS[1:]]
            close(candidate["scales"], expected_scales, "Train-only asinh scaling")

    outer_predictions = decoded_csv(out / "cv_predictions.csv")
    inner_predictions = decoded_csv(out / "cv_inner_predictions.csv")
    records = read(out / "cv_folds.json")
    check(len(records) == 120, "12 policies per CV fold")
    outer_groups, inner_groups = defaultdict(list), defaultdict(list)
    for prediction in outer_predictions:
        outer_groups[(prediction["scenario"], prediction["fold"], prediction["policy"])].append(prediction)
    for prediction in inner_predictions:
        inner_groups[(prediction["scenario"], prediction["fold"], prediction["candidate_id"])].append(prediction)

    def verify_group(predictions, ids, scope):
        check([r["equation_id"] for r in predictions] == ids, "CV saved prediction order/support")
        cid = predictions[0]["candidate_id"]
        scores, probabilities = get_scores(cid, ids)
        for index, prediction in enumerate(predictions):
            row = by_id[prediction["equation_id"]]
            check(prediction["split"] == scope and prediction["best_method"] == row["best_method"],
                  "CV prediction split/label")
            close(prediction["scores"], scores[index].tolist(), "CV reloaded scores")
            close(prediction["probabilities"], None if probabilities is None else probabilities[index].tolist(),
                  "CV reloaded probabilities")
            chosen = METHODS[int(np.argmax(np.asarray(prediction["scores"]) + np.r_[0., prediction["biases"]]))]
            check(prediction["selected_method"] == chosen, "CV independent argmax")
            close(prediction["E_selected"], row["E_" + chosen], "CV selected error")
            close(prediction["regret"], row["E_" + chosen] - min(row["E_" + m] for m in METHODS),
                  "CV selected regret")
        return independent_metrics([by_id[pid] for pid in ids], [p["selected_method"] for p in predictions])

    policy_evaluations = 0
    for candidate in candidates:
        if candidate["kind"] == "constant":
            continue
        ids = scenarios[candidate["scenario"]]["validation"]["ids"]
        scores, _ = get_scores(candidate["id"], ids)
        policy_path = out / "policy_search" / (candidate["id"] + ".json")
        policy_evaluations += verify_policy_search(policy_path,
                                                  candidate, scores, ids, by_id)
        validation = read(out / "validation_metrics.json")
        chosen = next(r for r in validation if r["candidate_id"] == candidate["id"]
                      and r["policy"] == "validation_optimal")
        check(chosen["biases"] == read(policy_path)["best"]["biases"], "main policy uses audited search result")
        default = next(r for r in validation if r["candidate_id"] == candidate["id"]
                       and r["policy"] == "default")
        check(default["biases"] == [0., 0.], "unmodified default policy")
    for manifest in manifests:
        scenario, fold = manifest["scenario"], manifest["fold"]
        own_records = [r for r in records if r["scenario"] == scenario and r["fold"] == fold]
        check(len(own_records) == 12, "CV policies complete")
        for record in own_records:
            predictions = outer_groups[(scenario, fold, record["policy"])]
            expected = verify_group(predictions, manifest["outer_validation_ids"], "outer_validation")
            metric_subset(record, expected, "CV outer metrics")
        nested = next(r for r in own_records if r["policy"] == "nested_selection")
        check(len(nested["inner_candidates"]) == 11, "eleven independently selected inner candidates")
        for inner in nested["inner_candidates"]:
            if inner["spec"].startswith("always_"):
                choices = [bundles[inner["candidate_id"]]["constant"]] * len(manifest["inner_validation_ids"])
                expected = independent_metrics([by_id[pid] for pid in manifest["inner_validation_ids"]], choices)
            else:
                predictions = inner_groups[(scenario, fold, inner["candidate_id"])]
                expected = verify_group(predictions, manifest["inner_validation_ids"], "inner_validation")
                candidate = bundles[inner["candidate_id"]]
                scores, _ = get_scores(candidate["id"], manifest["inner_validation_ids"])
                policy_evaluations += verify_policy_search(out / "policy_search" / (candidate["id"] + ".json"),
                    candidate, scores, manifest["inner_validation_ids"], by_id)
                check(inner["biases"] == read(out / "policy_search" / (candidate["id"] + ".json"))["best"]["biases"],
                      "inner policy uses audited search result")
                check(all(r["biases"] == inner["biases"] for r in predictions), "inner fixed selected biases")
            metric_subset(inner, expected, "independent inner selection metrics")
        best = min(nested["inner_candidates"], key=lambda p: (p["mean_regret"], -p["success_rate"],
                         -p["macro_f1"], p["non_R_predictions"], p["spec"]))
        check(nested["chosen_spec"] == best["spec"] and nested["biases"] == best["biases"],
              "nested policy chose only inner Validation")
        nested_predictions = outer_groups[(scenario, fold, "nested_selection")]
        check(all(r["biases"] == best["biases"] for r in nested_predictions), "nested applied frozen inner biases")
        selected_id = nested_predictions[0]["candidate_id"]
        expected_candidate = best["candidate_id"].replace("_inner_", "_outer_")
        check(selected_id == expected_candidate, "nested corresponding outer refit")

    summary = read(out / "cv_summary.json")
    check(summary["fit_count"] == 200 and summary["test_labels_accessed"] is False
          and summary["global_selected_hyperparameters_used"] is False, "CV independence contract")
    for policy in summary["policy_summary"]:
        own = [p for p in outer_predictions if p["scenario"] == policy["scenario"] and p["policy"] == policy["policy"]]
        ids = [p["equation_id"] for p in own]
        check(len(ids) == len(set(ids)) and set(ids) == set(scenarios[policy["scenario"]]["train"]["ids"]),
              "CV pooled complete Train OOF coverage")
        metric_subset(policy["pooled"], independent_metrics([by_id[pid] for pid in ids],
                      [p["selected_method"] for p in own]), "CV pooled OOF metrics")
        folds = [r for r in records if r["scenario"] == policy["scenario"] and r["policy"] == policy["policy"]]
        check(len(folds) == 5, "five pooled folds")
        for key, value in policy["fold_variability"].items():
            actual = np.asarray([r[key] for r in folds])
            close(value, dict(mean=float(actual.mean()), sd=float(actual.std(ddof=1))), "CV fold variability")
    return dict(outer_folds=len(manifests), outer_policy_metrics=len(records),
                outer_prediction_rows=len(outer_predictions), inner_prediction_rows=len(inner_predictions),
                estimator_fits=320, main_estimator_fits=120, CV_estimator_fits=200,
                policy_objectives_recomputed=policy_evaluations,
                policy_search_scope="best among saved evaluated policies; no global 2D optimality claim")


def verify(out):
    out = Path(out).resolve()
    check((out / "test_metrics.json").is_file(), "final test must complete before verification")
    by_id, raw_report = verify_raw(out)
    features, feature_report = verify_features(out, by_id)
    scenarios = verify_splits(out, by_id)
    # The remaining stages are deliberately independent of the training driver's
    # metric and policy functions; model loading performs inference only.
    prediction_report = verify_predictions(out, by_id, features, scenarios)
    registry = read(out / "selected_model/registry.json")
    lock = read(out / "selection_lock.json")
    check(registry["methods"] == list(METHODS) and registry["test_used_for_selection"] is False,
          "exported model three-method/input-only contract")
    check(registry["selection_lock_sha256"] == sha(out / "selection_lock.json")
          and registry["biases"] == lock["selected"]["biases"], "exported model frozen policy")
    for name, digest in registry["model_sha256"].items():
        check(sha(out / "selected_model" / name) == digest, "exported model checksum")
    scores, _ = independent_scores(registry["candidate"], out / "selected_model", features)
    predicted = [METHODS[int(i)] for i in np.argmax(scores + np.r_[0., registry["biases"]], axis=1)]
    lookup = dict(zip(features["equation_id"], predicted))
    chosen_test = [p for p in decoded_csv(out / "test_predictions.csv")
                   if p["scenario"] == "new" and p["candidate_id"] == lock["selected"]["candidate_id"]
                   and p["policy"] == lock["selected"]["policy"]]
    check(len(chosen_test) == 150 and all(lookup[p["equation_id"]] == p["selected_method"] for p in chosen_test),
          "self-contained exported model Test replay")
    pins = read(out / "preserved_sha256.json")
    for name, expected in pins.items():
        check((ROOT / name).is_file() and sha(ROOT / name) == expected,
              "pre-existing artifact changed: " + name)
    report = dict(schema="independent_three_method_saved_verification_v1", passed=True,
                  raw=raw_report, features=feature_report, family_leakage=False,
                  test_ids_preserved=150, predictions=prediction_report,
                  exported_model_predictions_checked=150, preserved_files=len(pins),
                  numerical_solver_calls=0, generator_calls=0, training_calls=0)
    from diagnostics.four_method_common import atomic_json
    atomic_json(out / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
