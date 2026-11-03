"""Saved-artifact analysis only: no training, generators, or ODE solver imports."""
from __future__ import annotations

import csv
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import sys
from collections import Counter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"


class NoExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("remizov_solver", "chernov_solver", "mc_solver", "mc_optimized",
                   "src.numerical_methods", "src.all_methods_dataset_v1",
                   "generate_all_methods_dataset_v1", "diagnostics.run_selector_hypotheses")
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise RuntimeError("Experiment import forbidden in evidence collection: " + fullname)
        return None


sys.meta_path.insert(0, NoExperiments())
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from ml.remizov_features import coefficient_features, CERT_FIELDS
from ml.selector_research_features import additional_coefficient_features, FEATURE_NAMES


def forbidden_fit(*args, **kwargs):
    raise RuntimeError("fit is forbidden: this task only reads existing models")


CatBoostClassifier.fit = forbidden_fit
CatBoostRegressor.fit = forbidden_fit
HYP = ROOT / "results/ml_selector_hypotheses_2026-10-31"
FOUR = ROOT / "results/four_method_accuracy_1000_2026-11-01"
DATA = ROOT / "data/all_methods_v1_1000_2026-10-18"
DIAGNOSTIC = "new__augmented__classifier__w1__baseline"
METHODS = ["remizov", "monte_carlo"]
BASELINE = json.loads((OUT / "before_sha256.json").read_text())
used = {}


def pin(path):
    path = Path(path).resolve()
    name = str(path.relative_to(ROOT))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert name in BASELINE and BASELINE[name] == digest, name
    used[name] = digest
    return path


def read(path):
    return json.loads(pin(path).read_text())


def frame(path):
    return pd.read_csv(pin(path), float_precision="round_trip", keep_default_na=False)


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def csv_out(name, rows):
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(OUT / "tables" / name, index=False)
        return
    pd.DataFrame(rows).to_csv(OUT / "tables" / name, index=False)


def metrics(er, em, predictions):
    er, em = np.asarray(er, float), np.asarray(em, float)
    truth = (em < er).astype(int)
    prediction = np.asarray(predictions, int)
    cm = np.zeros((2, 2), int)
    np.add.at(cm, (truth, prediction), 1)
    tn, fp, fn, tp = cm.ravel()
    selected = np.where(prediction == 1, em, er)
    regret = selected - np.minimum(er, em)
    recalls = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros(2), where=cm.sum(axis=1) != 0)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * np.diag(cm), den, out=np.zeros(2), where=den != 0)
    return dict(n=len(er), confusion_matrix=cm.tolist(), accuracy=float(np.trace(cm) / len(er)),
                balanced_accuracy=float(np.mean(recalls)), macro_f1=float(np.mean(f1)),
                mc_precision=float(tp / (tp + fp)) if tp + fp else 0.,
                mc_recall=float(tp / (tp + fn)) if tp + fn else 0.,
                mc_f1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.,
                mc_correct=int(tp), mc_false=int(fp), mc_missed=int(fn), mc_predictions=int(tp + fp),
                success_count=int(np.sum(selected <= .05)), success_rate=float(np.mean(selected <= .05)),
                mean_error=float(np.mean(selected)), median_error=float(np.median(selected)),
                mean_regret=float(np.mean(regret)), median_regret=float(np.median(regret)),
                max_regret=float(np.max(regret)))


def main():
    pin(ROOT / "ml/remizov_features.py")
    pin(ROOT / "ml/selector_research_features.py")
    pin(ROOT / "ml/accuracy_selector.py")
    pin(ROOT / "ml/selector_hypotheses.py")
    pin(ROOT / "diagnostics/run_selector_hypotheses.py")
    pin(ROOT / "diagnostics/analyze_selector_hypotheses.py")
    coverage = frame(ROOT / "results/ml_mean_mc_selector_2026-10-31/run/coverage.csv")
    pairs = coverage[coverage.pair_available == True].copy()
    pairs = pairs.set_index("equation_id", drop=False)
    candidates = read(HYP / "candidates.json")
    candidate_by_id = {c["id"]: c for c in candidates}
    splits = read(HYP / "splits.json")
    lock = read(HYP / "selection_lock.json")
    plan = read(HYP / "plan.json")
    val_metrics = frame(HYP / "validation_metrics.csv")
    test_metrics = frame(HYP / "test_metrics.csv")
    val_predictions = frame(HYP / "validation_predictions.csv")
    test_predictions = frame(HYP / "test_predictions.csv")
    with np.load(pin(HYP / "features.npz"), allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    ids = arrays["equation_id"].tolist()
    lookup = {pid: i for i, pid in enumerate(ids)}
    names = arrays["augmented_names"].tolist()
    assert len(ids) == len(set(ids)) == 997 and len(names) == len(set(names)) == 258
    assert len(arrays["feature_names"]) == 253 and names[-5:] == list(FEATURE_NAMES)
    matrices = dict(X=arrays["X"], augmented=arrays["augmented"])
    feature_names = dict(X=arrays["feature_names"].tolist(), augmented=names)
    metadata = [json.loads(line) for line in pin(DATA / "all_methods_equations_v1_metadata.jsonl").read_text().splitlines()]
    metadata = {m["equation_id"]: m for m in metadata}
    with np.load(pin(DATA / "all_methods_equations_v1_ml.npz"), allow_pickle=False) as z:
        raw = z["X"].copy()
        x = z["x_grid"].copy()
        raw_lookup = {pid: i for i, pid in enumerate(z["equation_id"].tolist())}
    recomputed = []
    for pid in ids:
        m = metadata[pid]
        whitelist = {k: m[k] for k in CERT_FIELDS}
        whitelist["serialized_ast"] = {k: m["serialized_ast"][k] for k in ("a_raw", "b_raw", "c_raw")}
        base = coefficient_features(raw[raw_lookup[pid]], x, whitelist)
        extra = additional_coefficient_features(raw[raw_lookup[pid]], x)
        recomputed.append([base[name] for name in feature_names["X"]] + [extra[name] for name in FEATURE_NAMES])
    recomputed = np.asarray(recomputed)
    difference = np.abs(recomputed - arrays["augmented"])
    assert np.array_equal(recomputed, arrays["augmented"])
    write("checks/feature_values.json", dict(rows=997, columns=258, exact_values_checked=997 * 258,
                                             max_absolute_difference=float(difference.max()),
                                             feature_names=names, solver_calls=0, model_fits=0))

    score_by_candidate = {}
    score_audits = []
    for c in candidates:
        if c["kind"] == "constant":
            score_by_candidate[c["id"]] = np.full(997, float(c["constant"] == "monte_carlo"))
            continue
        path = pin(HYP / c["model"])
        assert used[str(path.relative_to(ROOT))] == c["model_sha256"]
        model = CatBoostClassifier() if c["kind"] == "classifier" else CatBoostRegressor()
        model.load_model(str(path))
        own_names = feature_names[c["feature_set"]]
        assert model.feature_names_ == own_names
        pool = Pool(matrices[c["feature_set"]], feature_names=own_names)
        if c["kind"] == "classifier":
            scores = model.predict_proba(pool, thread_count=1)[:, list(model.classes_).index("monte_carlo")]
        else:
            scores = model.predict(pool, thread_count=1)
            if c["kind"] == "asinh":
                scores = c["scale"] * np.sinh(scores)
        score_by_candidate[c["id"]] = scores
        own_val = val_predictions[val_predictions.candidate_id == c["id"]]
        own_test = test_predictions[test_predictions.candidate_id == c["id"]].drop_duplicates("equation_id")
        dv = np.max(np.abs(scores[[lookup[p] for p in own_val.equation_id]] - own_val.score.to_numpy(float)))
        dt = np.max(np.abs(scores[[lookup[p] for p in own_test.equation_id]] - own_test.score.to_numpy(float)))
        assert dv == 0 and dt == 0, (c["id"], dv, dt)
        score_audits.append(dict(candidate_id=c["id"], validation_max_difference=float(dv), test_max_difference=float(dt)))
        if c["id"] == DIAGNOSTIC:
            write("model258_all_parameters.json", model.get_all_params())
            write("model258_metadata.json", dict(candidate=c, tree_count=model.tree_count_,
                                                  feature_names=model.feature_names_, classes=list(model.classes_)))
    assert len(score_audits) == 96
    write("checks/model_scores.json", dict(saved_models_loaded=96, prediction_calls=96, fit_calls=0,
                                          policy_selection_changed=False, score_audits=score_audits))

    computed = []
    metric_checks = 0
    for saved in val_metrics.to_dict("records"):
        cid, scenario, policy = saved["candidate_id"], saved["scenario"], saved["policy"]
        c = candidate_by_id[cid]
        threshold = float(saved["threshold"])
        for split in ("train", "validation", "test"):
            own_ids = splits["scenarios"][scenario][split]["ids"]
            own = pairs.loc[own_ids]
            scores = score_by_candidate[cid][[lookup[p] for p in own_ids]]
            prediction = (scores > threshold).astype(int) if c["kind"] != "constant" else np.full(len(own_ids), int(c["constant"] == "monte_carlo"))
            result = metrics(own.E_R.to_numpy(float), own.E_MC.to_numpy(float), prediction)
            result.update(candidate_id=cid, scenario=scenario, split=split, policy=policy, threshold=threshold,
                          feature_set=c["feature_set"], kind=c["kind"], weight=c["weight"],
                          source="new inference of saved CBM; in-sample Train" if split == "train" else "recomputed from existing scores; checked against saved metrics")
            if split != "train":
                expected = saved if split == "validation" else test_metrics[(test_metrics.candidate_id == cid) & (test_metrics.policy == policy)].iloc[0].to_dict()
                for key in ("n", "accuracy", "balanced_accuracy", "macro_f1", "mean_error", "mean_regret", "median_error",
                            "success_count", "mc_correct", "mc_false", "mc_missed", "mc_precision", "mc_recall", "mc_f1"):
                    assert np.isclose(result[key], float(expected[key]), rtol=2e-14, atol=1e-15), (cid, split, policy, key)
                assert result["confusion_matrix"] == json.loads(expected["confusion_matrix"])
                metric_checks += 1
            computed.append(result)
    pd.DataFrame([{**r, "confusion_matrix": json.dumps(r["confusion_matrix"])} for r in computed]).to_csv(OUT / "tables/model_policy_metrics_all.csv", index=False)
    assert metric_checks == 1688
    write("checks/metrics.json", dict(policies=844, split_policy_rows=len(computed), saved_val_test_metric_rows_checked=metric_checks,
                                      train_in_sample_derived=True, fit_calls=0, new_solver_calls=0))
    csv_out("models_96.csv", [dict(candidate_id=c["id"], scenario=c["scenario"], kind=c["kind"],
                                   feature_set=c["feature_set"], features=258 if c["feature_set"] == "augmented" else 253,
                                   weight=c["weight"], threshold=c["threshold"], scale=c["scale"],
                                   **c["config"], model=c["model"], model_sha256=c["model_sha256"])
                              for c in candidates if c["kind"] != "constant"])

    diagnostic = candidate_by_id[DIAGNOSTIC]
    threshold = diagnostic["threshold"]
    diagnostic_rows = []
    new_map = {pid: split for split, info in splits["scenarios"]["new"].items() for pid in info["ids"]}
    for pid in ids:
        row = pairs.loc[pid]
        er, em = float(row.E_R), float(row.E_MC)
        score = float(score_by_candidate[DIAGNOSTIC][lookup[pid]])
        truth = "monte_carlo" if em < er else "remizov"
        selected = "monte_carlo" if score > threshold else "remizov"
        category = "TP" if selected == truth == "monte_carlo" else "FP" if selected == "monte_carlo" else "FN" if truth == "monte_carlo" else "TN"
        selected_error = em if selected == "monte_carlo" else er
        diagnostic_rows.append(dict(equation_id=pid, family_id=row.family_id, split=new_map[pid],
                                    category=category, true_method=truth, selected_method=selected,
                                    score=score, threshold=threshold, E_R=er, E_MC=em, delta_R_minus_MC=er - em,
                                    E_selected=selected_error, regret=selected_error - min(er, em),
                                    selected_minus_always_R=selected_error - er,
                                    R_accurate=er <= .05, MC_accurate=em <= .05,
                                    chosen_accurate=selected_error <= .05,
                                    harms_epsilon=er <= .05 < selected_error,
                                    rescues_epsilon=selected_error <= .05 < er))
    diagnostic_frame = pd.DataFrame(diagnostic_rows)
    csv_out("diagnostic_258_all_splits.csv", diagnostic_frame)
    cases = diagnostic_frame[(diagnostic_frame.split == "test") & (diagnostic_frame.category != "TN")]
    cases = cases.sort_values(["category", "equation_id"])
    assert Counter(cases.category) == dict(TP=6, FP=14, FN=3)
    csv_out("diagnostic_258_cases.csv", cases)
    test = diagnostic_frame[diagnostic_frame.split == "test"]
    no_outlier = test[test.equation_id != "AMV1_F0180_S04"]
    case_summary = dict(candidate_id=DIAGNOSTIC, threshold=threshold,
                        confusion_matrix=[[127, 14], [3, 6]], cases=dict(Counter(cases.category)),
                        false_positive_sum_regret=float(cases[cases.category == "FP"].regret.sum()),
                        false_negative_sum_regret=float(cases[cases.category == "FN"].regret.sum()),
                        true_positive_sum_saving=float(cases[cases.category == "TP"].delta_R_minus_MC.sum()),
                        epsilon_harmed_ids=test[test.harms_epsilon].equation_id.tolist(),
                        epsilon_rescued_ids=test[test.rescues_epsilon].equation_id.tolist(),
                        both_inaccurate_true_positive_ids=test[(test.category == "TP") & ~test.chosen_accurate].equation_id.tolist(),
                        total_excess_error=float(test.regret.sum()),
                        sensitivity_excluding_only_largest_R_error=dict(n=149, omitted_id="AMV1_F0180_S04",
                            always_R_mean=float(no_outlier.E_R.mean()), selected_mean=float(no_outlier.E_selected.mean()),
                            scope="post-hoc sensitivity only; omitted row remains in all primary metrics"))
    write("checks/case_summary.json", case_summary)

    comparison_specs = [
        ("old__X__classifier__w1__baseline", "default", "baseline253 old"),
        ("new__X__classifier__w1__baseline", "default", "baseline253 new"),
        ("new__X__classifier__w1__baseline", "grid_0.25", "selected253 new"),
        (DIAGNOSTIC, "validation_optimal", "diagnostic258 new"),
        ("old__augmented__classifier__w12__baseline", "validation_optimal", "selected258 old"),
    ]
    comparison = []
    for cid, policy, label in comparison_specs:
        for r in computed:
            if r["candidate_id"] == cid and r["policy"] == policy:
                comparison.append(dict(label=label, **r))
    for scenario in ("old", "new"):
        for split, info in splits["scenarios"][scenario].items():
            own = pairs.loc[info["ids"]]
            er, em = own.E_R.to_numpy(float), own.E_MC.to_numpy(float)
            for label, pred in (("always R", np.zeros(len(own), int)), ("always MC", np.ones(len(own), int)), ("oracle", (em < er).astype(int))):
                comparison.append(dict(label=label, candidate_id=label, scenario=scenario, split=split,
                                       policy="reference", threshold=None, **metrics(er, em, pred)))
    csv_out("main_model_comparisons.csv", [{**r, "confusion_matrix": json.dumps(r["confusion_matrix"])} for r in comparison])

    four = frame(FOUR / "release/coverage_all1000.csv")
    common = four[four.eligible_four_method_ml == True].copy()
    assert len(four) == 1000 and len(common) == 997
    for method in ("remizov", "monte_carlo", "rk4", "product_integral"):
        common["E_" + method] = common["E_" + method].astype(float)
    assert (common.best_method == "rk4").all()
    independent = read(FOUR / "independent_verification.json")
    assert independent["winner_counts"]["rk4"] == 997
    distribution_rows = []
    grouping_rows = []
    for scope, data in (("available", four), ("common997", common)):
        for method in ("remizov", "monte_carlo", "rk4", "product_integral"):
            values = data.loc[data["usable_" + method] == True, "E_" + method].astype(float).to_numpy()
            qs = np.quantile(values, [0, .01, .05, .25, .5, .75, .9, .95, .99, 1])
            distribution_rows.append(dict(scope=scope, method=method, n=len(values),
                mean=float(values.mean()), std_population=float(values.std()), accurate=int(np.sum(values <= .05)),
                zero_count=int(np.sum(values == 0)), **dict(zip(["min", "q01", "q05", "q25", "median", "q75", "q90", "q95", "q99", "max"], map(float, qs)))))
    for column in ("complexity_level", "regime", "family_id"):
        for group, data in common.groupby(column):
            for method in ("remizov", "monte_carlo", "rk4", "product_integral"):
                values = data["E_" + method].to_numpy(float)
                grouping_rows.append(dict(group_type=column, group=group, method=method, n=len(values),
                    mean=float(values.mean()), median=float(np.median(values)), maximum=float(values.max()),
                    q25=float(np.quantile(values,.25)), q75=float(np.quantile(values,.75)),
                    accurate=int(np.sum(values <= .05)), success_rate=float(np.mean(values <= .05))))
    csv_out("four_method_distributions.csv", distribution_rows)
    csv_out("four_method_group_distributions.csv", grouping_rows)
    csv_out("four_method_common997.csv", common)

    legacy_folders = [
        "remizov_ml_v1_2026-10-25", "ml_selector_observed_2026-10-26_v2",
        "ml_probability_selector_2026-10-26", "ml_probability_validation_2026-10-26",
        "ml_probability_selector_audit_2026-10-26", "ml_nested_calibration_2026-10-26",
        "ml_switch5_r_timing_2026-10-26", "ml_selector_generalization_plan_2026-10-26",
        "ml_development8_2026-10-26", "ml_train_revision_2026-10-27",
        "ml_selector_final_2026-10-27", "ml_selector_final140_protocol_2026-10-27",
        "ml_accuracy_selector_2026-10-31/run", "ml_mean_mc_selector_2026-10-31/run",
    ]
    ledger = []
    legacy_sources = []
    def flatten(value, pointer, name):
        if isinstance(value, dict):
            for key, own in value.items():
                flatten(own, pointer + "/" + key, name)
        elif isinstance(value, list):
            if len(value) <= 40 and all(not isinstance(v, (dict, list)) for v in value):
                ledger.append(dict(source=name, json_pointer=pointer, value=json.dumps(value)))
        elif value is None or isinstance(value, (str, int, float, bool)):
            ledger.append(dict(source=name, json_pointer=pointer, value=value))
    for folder in legacy_folders:
        root = ROOT / "results" / folder
        for name in ("REPORT.md", "FINAL_REPORT.md", "metrics.json", "run_summary.json", "diagnostic_cv.json",
                     "policy_diagnostics.json", "report_summary.json", "final_summary.json", "result_summary.json",
                     "final_policy_selection.json", "protocol.json", "training_protocol.json", "family_cv.json"):
            path = root / name
            if path.is_file():
                pin(path)
                legacy_sources.append(str(path.relative_to(ROOT)))
                if path.suffix == ".json":
                    flatten(read(path), "", str(path.relative_to(ROOT)))
    csv_out("historical_metrics_ledger.csv", ledger)
    write("historical_sources.json", legacy_sources)
    for name in ("cv_summary.json", "calibration.json", "feature_analysis.json", "independent_verification.json", "baseline_independent_reproduction.json"):
        read(HYP / name)
    frame(HYP / "cv_folds.csv")
    frame(HYP / "cv_predictions.csv")
    frame(HYP / "feature_importance.csv")
    frame(HYP / "feature_distributions.csv")
    write("used_sources.json", used)
    write("evidence_summary.json", dict(diagnostic_candidate=diagnostic, diagnostic_cases=case_summary,
        selected_on_validation=lock["selected"], plan=plan,
        two_method_pairs=997, diagnostic_model_train_n=747, validation_n=100, test_n=150,
        feature_count=258, all_policy_metric_rows=len(computed), models_loaded=96, model_prediction_calls=96,
        training_calls=0, numerical_solver_calls=0, generator_calls=0,
        four_method_common=997, four_method_winners={"remizov":0,"monte_carlo":0,"rk4":997,"product_integral":0},
        limitations=["Train metrics are in-sample derivations from already saved models", "Test previously disclosed; highlighted 258 model is diagnostic, not validation-selected", "historical objectives and solver profiles differ"]))
    print(json.dumps(dict(models_loaded=96, prediction_calls=96, training_calls=0, solver_calls=0,
                          features_checked=int(difference.size), policies_checked=1688,
                          diagnostic_cases=dict(Counter(cases.category))), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
