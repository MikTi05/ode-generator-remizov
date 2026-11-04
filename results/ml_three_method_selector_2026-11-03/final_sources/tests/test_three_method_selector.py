"""Independent three-method label, policy and evaluation checks; no ODE solves."""
from collections import Counter
import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from ml.three_method_selector import (
    EPSILON, METHODS, assert_separated, decide, fit_bundle, load_bundle, metrics,
    policy_scan, score_bundle, selection_key, winner,
)


ROOT = Path(__file__).resolve().parents[1]
ORDER = ("remizov", "monte_carlo", "product_integral")


def equation(pid, errors, family=None):
    """Build truth without calling the implementation under test."""
    family = family or "family_" + pid
    chosen = min(range(3), key=lambda i: errors[i])
    return dict(equation_id=pid, family_id=family,
                structure_signature="topology_" + family,
                best_method=ORDER[chosen],
                **{"E_" + method: float(error) for method, error in zip(ORDER, errors)})


def test_three_real_methods_and_exact_unrounded_ties_exclude_rk4():
    assert METHODS == ORDER and EPSILON == .05
    assert winner([.1, .1, .1]) == "remizov"
    assert winner([.2, .1, .1]) == "monte_carlo"
    assert winner([.1, .2, .1]) == "remizov"
    assert winner([.2, .2, .1]) == "product_integral"
    assert winner([np.nextafter(.1, np.inf), .1, .2]) == "monte_carlo"
    row = equation("rk4_is_forbidden", [.2, .1, .3])
    row["E_rk4"] = 0.
    result = metrics([row], ["monte_carlo"])
    assert result["accuracy"] == 1. and result["mean_error"] == .1
    with pytest.raises(ValueError):
        metrics([row], ["rk4"])


@pytest.mark.parametrize("errors", [
    [], [.1, .2], [.1, .2, .3, .4], [.1, None, .3],
    [.1, float("nan"), .3], [.1, float("inf"), .3], [.1, -.01, .3],
])
def test_missing_nonfinite_or_negative_errors_cannot_become_labels(errors):
    with pytest.raises((TypeError, ValueError)):
        winner(errors)


def test_metrics_match_sklearn_and_direct_selected_errors():
    rows = [equation(str(i), errors) for i, errors in enumerate([
        [.01, .04, .05], [.3, .01, .02], [.4, .3, .02],
        [.05, .1, .2], [.3, .04, .2], [.3, .2, .06],
    ])]
    predictions = ["monte_carlo", "monte_carlo", "remizov",
                   "product_integral", "product_integral", "product_integral"]
    truth = [row["best_method"] for row in rows]
    actual = np.array([row["E_" + method] for row, method in zip(rows, predictions)])
    oracle = np.array([min(row["E_" + method] for method in ORDER) for row in rows])
    result = metrics(rows, predictions)
    assert result["confusion_matrix"] == confusion_matrix(truth, predictions, labels=ORDER).tolist()
    assert result["accuracy"] == accuracy_score(truth, predictions)
    assert result["macro_f1"] == pytest.approx(f1_score(truth, predictions, labels=ORDER, average="macro", zero_division=0))
    assert result["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(truth, predictions))
    assert result["mean_error"] == pytest.approx(actual.mean())
    assert result["median_error"] == pytest.approx(np.median(actual))
    assert result["mean_regret"] == pytest.approx((actual - oracle).mean())
    assert result["median_regret"] == pytest.approx(np.median(actual - oracle))
    assert result["max_regret"] == pytest.approx((actual - oracle).max())
    assert result["success_count"] == int(np.sum(actual <= .05))
    assert result["success_rate"] == pytest.approx(np.mean(actual <= .05))
    assert result["non_R_predictions"] == sum(method != "remizov" for method in predictions)


def test_success_boundary_and_large_r128_errors_are_retained():
    boundary = equation("boundary", [.05, .2, .3])
    assert metrics([boundary], ["remizov"])["success_count"] == 1
    above = equation("above", [np.nextafter(.05, np.inf), .2, .3])
    assert metrics([above], ["remizov"])["success_count"] == 0
    catastrophe = equation("large_but_usable", [842.0865810144144, .08240083666033149, .2])
    result = metrics([catastrophe], ["remizov"])
    assert result["n"] == 1 and result["mean_error"] == 842.0865810144144
    assert result["mean_regret"] == pytest.approx(842.0865810144144 - .08240083666033149)
    oracle = metrics([catastrophe], ["monte_carlo"])
    assert oracle["accuracy"] == 1. and oracle["success_count"] == 0


@pytest.mark.parametrize("mutation", ["missing", "nan", "negative", "wrong_label"])
def test_metrics_reject_incomplete_or_inconsistent_measured_rows(mutation):
    row = equation("bad", [.2, .1, .3])
    if mutation == "missing":
        row.pop("E_product_integral")
    elif mutation == "nan":
        row["E_remizov"] = float("nan")
    elif mutation == "negative":
        row["E_product_integral"] = -.1
    else:
        row["best_method"] = "product_integral"
    with pytest.raises((KeyError, TypeError, ValueError)):
        metrics([row], ["monte_carlo"])


def test_empty_split_is_unavailable_and_prediction_lengths_are_checked():
    result = metrics([], [])
    assert result["n"] == 0 and result["accuracy"] is None
    assert result["mean_error"] is None and result["success_rate"] is None
    assert result["confusion_matrix"] == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    with pytest.raises(ValueError):
        metrics([equation("a", [.1, .2, .3])], [])


def test_decision_biases_and_every_partial_tie_use_method_order():
    scores = np.array([[0., 0., 0.], [-1., 1., 1.], [1., -1., 1.], [-1., 0., 1.]])
    assert decide(scores) == ["remizov", "monte_carlo", "remizov", "product_integral"]
    assert decide(np.zeros((1, 3)), (2., 2.)) == ["monte_carlo"]
    assert decide(np.zeros((1, 3)), (1., 2.)) == ["product_integral"]
    assert decide(np.zeros((1, 3)), (-1., -2.)) == ["remizov"]


@pytest.mark.parametrize("scores,biases", [
    ([[0., 1.]], (0., 0.)), ([[0., 1., np.nan]], (0., 0.)),
    ([[0., 1., 2.]], (0.,)), ([[0., 1., 2.]], (0., np.nan)),
])
def test_decision_rejects_malformed_or_nonfinite_scores(scores, biases):
    with pytest.raises(ValueError):
        decide(scores, biases)


@pytest.mark.parametrize("kind", ["classifier", "raw", "asinh"])
def test_policy_search_minimizes_real_error_rather_than_label_accuracy(kind):
    rows = [equation("a", [1000., .01, .02]),
            equation("b", [0., 1., 2.]), equation("c", [0., 1., 2.])]
    # The rare costly MC winner is ranked below both harmless non-winners.
    # Selecting MC everywhere lowers accuracy but reduces mean regret to 2/3.
    scores = np.array([[0., -3., -100.], [0., -2., -100.], [0., -1., -100.]])
    table, best = policy_scan(rows, scores, kind)
    default = metrics(rows, decide(scores))
    independently_recomputed = metrics(rows, decide(scores, best["biases"]))
    assert table and any(np.array_equal(entry["biases"], [0., 0.]) for entry in table)
    assert independently_recomputed["mean_regret"] == pytest.approx(2 / 3)
    assert independently_recomputed["accuracy"] < default["accuracy"]
    assert best["mean_regret"] == independently_recomputed["mean_regret"]
    assert selection_key(best) <= selection_key(default)


@pytest.mark.parametrize("key", ["equation_id", "family_id", "structure_signature"])
def test_cv_separation_rejects_each_independent_leakage_key(key):
    left, right = [equation("left", [.1, .2, .3])], [equation("right", [.3, .2, .1])]
    assert_separated(left, right)
    right[0][key] = left[0][key]
    with pytest.raises(ValueError):
        assert_separated(left, right)


def test_saved_three_method_distribution_includes_all_usable_outliers():
    path = ROOT / "results/four_method_accuracy_1000_2026-11-01/release/ml_four_methods.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    labels = [winner([float(row["E_" + method]) for method in ORDER]) for row in rows]
    assert len(rows) == 997
    assert Counter(labels) == {"remizov": 826, "monte_carlo": 106, "product_integral": 65}
    catastrophe = next(row for row in rows if row["equation_id"] == "AMV1_F0141_S00")
    assert float(catastrophe["E_remizov"]) == 842.0865810144144


def _tiny_training_data():
    rows = [equation(str(i), errors) for i, errors in enumerate([
        [.01, .1, .3], [.02, .08, .2], [.03, .06, .1],
        [.3, .01, .02], [.4, .2, .01], [.2, .3, .1],
    ])]
    X = np.array([[i, i * i] for i in range(len(rows))], dtype=float)
    config = dict(name="unit_only", iterations=3, depth=2, l2_leaf_reg=10)
    return rows, X, ["input_x", "input_x_squared"], config


def test_tiny_three_class_bundle_survives_serialization_without_class_reordering(tmp_path):
    rows, X, names, config = _tiny_training_data()
    models, scales = fit_bundle("classifier", config, 1, X, rows, names)
    scores, probabilities = score_bundle(models, "classifier", scales, X, names)
    assert list(models[0].classes_) == list(ORDER)
    assert probabilities.shape == scores.shape == (6, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1., rtol=0., atol=1e-14)
    assert decide(scores) == [ORDER[i] for i in probabilities.argmax(axis=1)]
    models[0].save_model(str(tmp_path / "tiny.cbm"))
    restored = load_bundle(dict(kind="classifier", models=["tiny.cbm"]), tmp_path)
    after_scores, after_probabilities = score_bundle(restored, "classifier", scales, X, names)
    np.testing.assert_array_equal(after_probabilities, probabilities)
    np.testing.assert_array_equal(after_scores, scores)


def test_asinh_dual_head_scales_come_from_training_errors_and_survive_reload(tmp_path):
    rows, X, names, config = _tiny_training_data()
    models, scales = fit_bundle("asinh", config, 1, X, rows, names)
    # Independent hand calculation: medians of |ER-EMC| and |ER-EPI|.
    np.testing.assert_allclose(scales, [.095, .23], rtol=0., atol=1e-14)
    assert len(models) == 2
    probe = np.r_[X, [[10000., 1.e8]]]
    before, probabilities = score_bundle(models, "asinh", scales, probe, names)
    assert probabilities is None and before.shape == (7, 3)
    np.testing.assert_array_equal(before[:, 0], np.zeros(7))
    files = []
    for i, model in enumerate(models):
        files.append("head_" + str(i) + ".cbm")
        model.save_model(str(tmp_path / files[-1]))
    restored = load_bundle(dict(kind="asinh", models=files), tmp_path)
    after, _ = score_bundle(restored, "asinh", scales, probe, names)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_allclose(scales, [.095, .23], rtol=0., atol=1e-14)
