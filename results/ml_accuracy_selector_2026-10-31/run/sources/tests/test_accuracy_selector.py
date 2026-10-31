"""Error-choice contract and saved-data integration; no numerical method runs."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ml.accuracy_selector import (AccuracyMethodSelector, EPSILON, METHODS,
                                  accuracy_target, selection_metrics)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/ml_accuracy_selector_2026-10-31/run"


def read(path):
    return json.loads(Path(path).read_text())


def test_median_of_errors_is_not_error_of_mean_solution():
    # Exact=0, two independent predictions -1,+1 have mean zero but error one.
    row = accuracy_target(.2, [1., 1.])
    assert row["E_MC"] == 1. and row["best_method"] == "remizov"


def test_exact_tie_and_success_boundary_without_rounding():
    row = accuracy_target(.05, [.04, .06])
    assert row["best_method"] == "remizov"
    assert row["chosen_success"] and row["remizov_success"] and row["mc_success"]
    above = np.nextafter(EPSILON, np.inf)
    row = accuracy_target(above, [above, above])
    assert row["best_method"] == "remizov"
    assert not row["chosen_success"] and not row["remizov_success"] and not row["mc_success"]


def test_both_fail_still_have_real_winner():
    row = accuracy_target(.2, [.1, .12])
    assert row["best_method"] == "monte_carlo" and not row["chosen_success"]


@pytest.mark.parametrize("er,errors", [(0.1, []), (float("nan"), [.1]), (-1., [.1]), (.1, [float("inf")]), (.1, [-.1])])
def test_missing_invalid_values_never_receive_labels(er, errors):
    with pytest.raises(ValueError):
        accuracy_target(er, errors)


def test_metrics_from_continuous_selected_errors():
    rows = [accuracy_target(.02, [.1]), accuracy_target(.3, [.01])]
    scores = selection_metrics(rows, ["remizov", "remizov"])
    assert scores["confusion_matrix"] == [[1, 0], [1, 0]]
    assert scores["accuracy"] == scores["balanced_accuracy"] == .5
    assert scores["macro_f1"] == pytest.approx(1/3)
    assert scores["success_rate"] == .5 and scores["mean_error"] == pytest.approx(.16)
    assert scores["mean_regret"] == pytest.approx(.145)
    oracle = selection_metrics(rows, [r["best_method"] for r in rows])
    assert oracle["mean_regret"] == 0 and oracle["success_rate"] == 1
    with pytest.raises(ValueError, match="label disagrees"):
        selection_metrics([dict(rows[0], best_method="monte_carlo")], ["remizov"])


def test_single_class_and_missing_split_metrics_are_explicit():
    scores = selection_metrics([accuracy_target(.01, [.1])], ["remizov"])
    assert scores["accuracy"] == scores["balanced_accuracy"] == 1
    assert scores["macro_f1"] == .5 and scores["recall"]["monte_carlo"] is None
    empty = selection_metrics([], [])
    assert empty["n"] == 0 and empty["accuracy"] is None and empty["success_rate"] is None
    assert empty["confusion_matrix"] == [[0, 0], [0, 0]]


@pytest.fixture(scope="module")
def evidence():
    from ml.accuracy_evidence import build_evidence
    return build_evidence()


def test_main_saved_pair_census_and_independent_audit(evidence):
    rows = evidence["main_rows"]
    assert len(rows) == 997 and len(evidence["coverage_rows"]) == 1000
    assert len(evidence["mc_seed_rows"]) == 4000
    expected = {"train": (697, 686, 11, 689, 632, 691),
                "validation": (150, 144, 6, 146, 135, 146),
                "test": (150, 146, 4, 147, 138, 148)}
    for split, counts in expected.items():
        own = [r for r in rows if r["split"] == split]
        assert (len(own), sum(r["best_method"] == "remizov" for r in own),
                sum(r["best_method"] == "monte_carlo" for r in own),
                sum(r["remizov_success"] for r in own), sum(r["mc_success"] for r in own),
                sum(r["chosen_success"] for r in own)) == counts
    groups = [{r["structure_signature"] for r in rows if r["split"] == s}
              for s in ("train", "validation", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert all(r["remizov_config"]["n"] == 128 and r["mc_seed_count"] == 4 for r in rows)
    for row in rows:
        assert row["E_R"] == abs(row["remizov_f0"] - row["y_exact0"])
        assert row["E_MC"] == float(np.median(row["mc_seed_errors"]))
        assert row["best_method"] == ("remizov" if row["E_R"] <= row["E_MC"] else "monte_carlo")


def test_failures_are_missing_pairs_and_outliers_remain(evidence):
    rows = evidence["coverage_rows"]
    missing = [r for r in rows if r["E_R"] is None]
    assert {r["equation_id"] for r in missing} == {"AMV1_F0134_S03", "AMV1_F0136_S01", "AMV1_F0139_S00"}
    assert all(r["best_method"] is None for r in missing)
    largest = max(evidence["main_rows"], key=lambda r:r["E_R"])
    assert largest["equation_id"] == "AMV1_F0141_S00" and largest["E_R"] > 800
    assert len(evidence["supplementary_rows"]) == 48
    assert all(r["remizov_config"]["n"] == 256 for r in evidence["supplementary_rows"])


@pytest.fixture(scope="module")
def selector():
    return AccuracyMethodSelector(RUN / "model")


def test_fit_scope_reload_metrics_and_nonleaking_inference(selector):
    r = selector.registry
    assert r["fit_count"] == 1 and len(r["train_equations"]) == 697
    assert set(r["observed_training_classes"]) == set(METHODS)
    assert not r["validation_test_used_for_fit_or_tuning"] and not r["class_balancing"]
    assert len(r["feature_names"]) == 253 and r["comparison_context"]["remizov"]["n"] == 128
    payload = read(RUN / "example_input.json")
    first = selector.predict(**payload)
    changed = copy.deepcopy(payload)
    changed["coefficient_metadata"].update(y_exact=1e90, error=0, runtime_seconds=0, success=True)
    changed["coefficient_metadata"]["serialized_ast"]["y_shape"] = {"op": "unsafe_solution"}
    assert first == selector.predict(**changed)
    assert first["selected_method"] in METHODS and not first["accuracy_certificate"]
    assert first["epsilon"] == .05 and first["criterion"] == "lower_absolute_error_at_x0"
    metrics = read(RUN / "metrics.json")
    assert metrics["test"]["ML"]["n"] == 150
    assert metrics["test"]["always_remizov"]["accuracy"] == 146/150
    assert metrics["test"]["always_monte_carlo"]["success_rate"] == 138/150
    assert metrics["test"]["oracle"]["success_rate"] == 148/150


def test_cli_and_model_integrity(selector, tmp_path):
    output = tmp_path / "prediction.json"
    command = [sys.executable, "-B", str(ROOT / "diagnostics/predict_accuracy_selector.py"),
               "--model", str(RUN / "model"), "--input", str(RUN / "example_input.json"), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert read(output) == read(RUN / "example_prediction.json")
    import shutil
    dest = tmp_path / "model"
    shutil.copytree(RUN / "model", dest)
    with (dest / "model.cbm").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        AccuracyMethodSelector(dest)
