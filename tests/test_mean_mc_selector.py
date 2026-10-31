"""Mean-four estimates contract and saved binary-model checks; no fit or solvers."""
import ast
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ml.accuracy_selector import AccuracyMethodSelector, selection_metrics, sha
from ml.mean_mc_selector import MeanMCMethodSelector, mean_accuracy_target

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "results/ml_accuracy_selector_2026-10-31/run"
RUN = ROOT / "results/ml_mean_mc_selector_2026-10-31/run"


def read(path):
    return json.loads(Path(path).read_text())


def test_average_solutions_then_absolute_error():
    row = mean_accuracy_target(.1, [-.1, .1, -.2, .2], 0.)
    assert row["mc_mean_f0"] == row["E_MC"] == 0
    assert row["best_method"] == "monte_carlo"
    # Mean aggregation is not guaranteed to improve on median absolute errors.
    other = mean_accuracy_target(1., [0., 0., 0., 100.], 0.)
    assert other["E_MC"] == 25. and other["best_method"] == "remizov"


def test_equal_error_tie_and_unrounded_threshold():
    row = mean_accuracy_target(.05, [.05]*4, 0.)
    assert row["best_method"] == "remizov" and row["chosen_success"]
    above = np.nextafter(.05, np.inf)
    row = mean_accuracy_target(above, [above]*4, 0.)
    assert row["best_method"] == "remizov" and not row["chosen_success"]
    assert not row["remizov_success"] and not row["mc_success"]


@pytest.mark.parametrize("values,exact,er", [([],0.,.1), ([.1]*3,0.,.1), ([.1]*5,0.,.1),
                                             ([.1,float('nan'),.2,.3],0.,.1), ([.1]*4,float('inf'),.1),
                                             ([.1]*4,0.,-1.)])
def test_incomplete_or_nonfinite_values_do_not_receive_target(values, exact, er):
    with pytest.raises(ValueError):
        mean_accuracy_target(er, values, exact)


def test_no_numerical_import_calls_and_single_fixed_fit_in_source():
    forbidden = {"solve", "solve_remizov", "solve_monte_carlo", "solve_monte_carlo_optimized", "compile_backend", "y_exact"}
    for name in ("ml/mean_mc_evidence.py", "ml/mean_mc_selector.py", "diagnostics/train_mean_mc_selector.py"):
        tree = ast.parse((ROOT/name).read_text())
        modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not set(modules) & {"chernov_solver", "remizov_solver", "mc_solver", "mc_optimized", "all_methods_dataset_loader"}
        calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))]
        assert not set(calls) & forbidden
        if name.endswith("train_mean_mc_selector.py"):
            assert calls.count("fit") == 1
    plan = read(RUN.parent / "plan.json")
    assert plan["fit_count"] == 1 and not plan["class_balancing"] and not plan["test_tuning"]


@pytest.fixture(scope="module")
def evidence():
    from ml.mean_mc_evidence import build_mean_evidence
    return build_mean_evidence()


def test_saved_mean_labels_and_unchanged_coverage(evidence):
    rows = evidence["main_rows"]
    assert len(rows) == 997 and len(evidence["coverage_rows"]) == 1000
    expected = {"train":(697,608,89,689,655,690), "validation":(150,132,18,146,139,146), "test":(150,141,9,147,141,147)}
    for split, counts in expected.items():
        own = [r for r in rows if r["split"] == split]
        assert (len(own), sum(r["best_method"] == "remizov" for r in own),
                sum(r["best_method"] == "monte_carlo" for r in own),
                sum(r["remizov_success"] for r in own),sum(r["mc_success"] for r in own),sum(r["chosen_success"] for r in own)) == counts
    for row in rows:
        assert row["E_MC"] == abs(float(np.mean(row["mc_seed_f0"]))-row["y_exact0"])
        assert row["mc_seed_count"] == 4 and row["remizov_config"]["n"] == 128
        assert row["E_R"] == abs(row["remizov_f0"]-row["y_exact0"])
    assert evidence["audit"]["target_changes"] == 103
    missing = [r for r in evidence["coverage_rows"] if not r["pair_available"]]
    assert len(missing) == 3 and all(r["best_method"] is None and r["E_R"] is None for r in missing)


def test_catastrophes_are_kept_and_mc_can_still_fail_epsilon(evidence):
    rows = {r["equation_id"]:r for r in evidence["main_rows"]}
    for pid, split, er, em in (("AMV1_F0141_S00","train",842.0865810144144,.08240083666033149),
                               ("AMV1_F0180_S04","test",248.32307672784063,.10308096210860278)):
        row = rows[pid]
        assert row["split"] == split and row["E_R"] == er and row["E_MC"] == em
        assert row["best_method"] == "monte_carlo" and not row["chosen_success"]


def test_old_release_and_feature_bytes_are_preserved():
    assert all(sha(PREVIOUS/name) == digest for name,digest in read(PREVIOUS/'artifact_sha256.json').items())
    assert sha(PREVIOUS/'input_features.npz') == sha(RUN/'input_features.npz')
    assert read(RUN/'model/registry.json')["fit_count"] == 1
    assert read(RUN/'model/registry.json')["retrospective_reanalysis"]


def test_model_inference_and_forbidden_metadata_invariance():
    selector = MeanMCMethodSelector(RUN/'model')
    payload = read(RUN/'example_input.json')
    before = selector.predict(**payload)
    changed = copy.deepcopy(payload)
    changed['coefficient_metadata'].update(y_exact=1e100,E_R=1e100,E_MC=0.,success=True,runtime_seconds=0.)
    changed['coefficient_metadata']['serialized_ast']['y_shape'] = {'op':'forbidden_solution'}
    assert selector.predict(**changed) == before
    assert before['comparison_context']['monte_carlo']['aggregate_paths'] == 4096
    assert not before['accuracy_certificate']
    with pytest.raises(ValueError,match='contract'):
        AccuracyMethodSelector(RUN/'model')
    with pytest.raises(ValueError,match='contract'):
        MeanMCMethodSelector(PREVIOUS/'model')


def test_saved_predictions_reproduce_all_selected_error_metrics():
    evidence = read(RUN/'evidence.json')
    with (RUN/'predictions.csv').open() as stream:
        predictions = {r['equation_id']:r['selected_method'] for r in csv.DictReader(stream)}
    metrics = read(RUN/'metrics.json')
    for split in ('train','validation','test'):
        own = [r for r in evidence['main_rows'] if r['split']==split]
        assert selection_metrics(own,[predictions[r['equation_id']] for r in own]) == metrics[split]['ML']
    assert metrics['test']['always_remizov']['success_rate'] == 147/150
    assert metrics['test']['always_monte_carlo']['success_rate'] == 141/150
    assert metrics['test']['oracle']['success_rate'] == 147/150


def test_actual_mean_model_cli(tmp_path):
    output = tmp_path/'prediction.json'
    result = subprocess.run([sys.executable,'-B',str(ROOT/'diagnostics/predict_mean_mc_selector.py'),
        '--model',str(RUN/'model'),'--input',str(RUN/'example_input.json'),'--output',str(output)],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    assert read(output) == read(RUN/'example_prediction.json')
