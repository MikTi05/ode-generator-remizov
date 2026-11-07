"""Saved-only checks for the grid33 release; never execute an ODE solver."""

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from diagnostics.build_saved_grid33 import (
    EPSILON, GRID33, GRID129, build, classify_row, grid_error,
)
from diagnostics.verify_saved_grid33 import verify


ROOT = Path(__file__).resolve().parents[1]
FIRST_ID = "AMV1_F0000_S00"
MISSING_R = ("AMV1_F0134_S03", "AMV1_F0136_S01", "AMV1_F0139_S00")


def read(path):
    return json.loads(path.read_text())


def reseal(record):
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    record["record_sha256"] = hashlib.sha256(payload).hexdigest()


@pytest.fixture
def saved_inputs():
    metadata_path = ROOT / "data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1_metadata.jsonl"
    metadata = {item["equation_id"]: item for item in
                map(json.loads, metadata_path.read_text().splitlines())}

    def load(pid=FIRST_ID):
        ivp = ROOT / "results/four_method_accuracy_1000_2026-11-01/records"
        return (
            copy.deepcopy(metadata[pid]),
            read(ROOT / f"results/chernov_solve1000_2026-10-18/solutions/{pid}.json"),
            read(ivp / "rk4" / f"{pid}.json"),
            read(ivp / "product_integral" / f"{pid}.json"),
            [read(ROOT / f"results/mc1000_x0_2026-10-25/attempts/{pid}_seed{i}.json")
             for i in range(4)],
        )

    return load


def test_common_grid_is_exact_subgrid_and_contains_zero():
    assert list(GRID33) == [-1 + i / 16 for i in range(33)]
    assert list(GRID129)[::4] == list(GRID33)
    assert GRID33[16] == 0


def test_error_uses_all_33_nodes_and_accepts_exact_threshold():
    exact = np.zeros(33)
    values = np.zeros(33)
    values[-1] = -0.125
    assert grid_error(values, exact) == 0.125
    assert values[16] == exact[16]
    values[-1] = EPSILON
    assert grid_error(values, exact) == EPSILON == 0.05


@pytest.mark.parametrize("bad", [
    [], np.zeros(32), np.zeros(34), np.zeros((33, 1)),
    np.zeros(33, dtype=bool),
    [float("nan")] + [0.0] * 32,
    [float("inf")] + [0.0] * 32,
])
@pytest.mark.parametrize("argument", ["values", "exact"])
def test_invalid_arrays_cannot_become_accuracy_measurements(bad, argument):
    args = dict(values=np.zeros(33), exact=np.zeros(33))
    args[argument] = bad
    with pytest.raises(ValueError):
        grid_error(**args)


def test_saved_trajectories_are_recomputed_without_mutation(saved_inputs):
    args = saved_inputs()
    before = copy.deepcopy(args)
    row = classify_row(*args)
    _, remizov, rk4, pi, _ = args
    for method, values in (("R", remizov["result"]["f"]),
                           ("RK4", rk4["result"]["y"][::4]),
                           ("PI", pi["result"]["y"][::4])):
        expected = max(abs(a - b) for a, b in zip(values, rk4["exact_grid33"]))
        assert row[method + "_E33"] == expected
        assert row[method + "_accuracy_met"] == (expected <= 0.05)
    assert args == before
    assert row["inferred_unique_four_method_winner"] == "RK4"
    assert row["winner_scope"] == "all_four_inferred"
    assert row["MC_E33"] is None and row["MC_accuracy_met"] is None


def test_accuracy_flag_includes_the_exact_threshold(saved_inputs):
    args = saved_inputs()
    _, remizov, rk4, pi, _ = args
    # An in-memory boundary fixture keeps all saved exact arrays consistent.
    # Using exact=0 at this node makes the float error exactly EPSILON.
    for record in (rk4, pi):
        record["exact_grid33"][1] = 0.0
        record["result"]["y"][4] = 0.0
        record["E_inf_grid33"] = max(
            abs(a - b) for a, b in
            zip(record["result"]["y"][::4], record["exact_grid33"]))
        reseal(record)
    remizov["y_exact"][1] = 0.0
    remizov["result"]["f"] = list(remizov["y_exact"])
    remizov["result"]["f"][1] = EPSILON
    remizov["max_abs_error"] = EPSILON
    row = classify_row(*args)
    assert row["R_E33"] == 0.05
    assert row["R_accuracy_met"] is True


@pytest.mark.parametrize("pid", MISSING_R)
def test_missing_remizov_is_unknown_and_never_a_four_method_win(saved_inputs, pid):
    row = classify_row(*saved_inputs(pid))
    assert row["R_status"] == "worker_exception"
    assert row["R_available"] is False
    assert row["R_E33"] is None and row["R_accuracy_met"] is None
    assert row["margin_R"] is None
    assert row["inferred_unique_four_method_winner"] is None
    assert row["winner_scope"] == "available_only"
    assert row["best_available_method"] == "RK4"


def test_mc_lower_bound_is_error_of_mean_not_mean_error(saved_inputs):
    args = saved_inputs()
    exact0 = args[2]["exact_grid33"][16]
    for index, record in enumerate(args[-1]):
        value = exact0 + (0.1 if index % 2 == 0 else -0.1)
        record["result"]["f"] = [value]
        record["abs_error"] = abs(value - exact0)
        record["accuracy_met"] = False
    row = classify_row(*args)
    assert row["MC_E0_lower_bound"] < 1e-15
    assert sum(item["abs_error"] for item in args[-1]) / 4 > 0.09
    assert row["MC_mean4_0"] == pytest.approx(exact0, abs=1e-15)
    assert row["MC_threshold_evidence"] == "unknown"
    assert row["MC_E33"] is None and row["MC_accuracy_met"] is None
    assert row["inferred_unique_four_method_winner"] is None


def test_mc_lower_bound_can_disprove_threshold_but_is_not_full_error(saved_inputs):
    args = saved_inputs()
    exact0 = args[2]["exact_grid33"][16]
    for record in args[-1]:
        record["result"]["f"] = [exact0 + 0.125]
        record["abs_error"] = 0.125
        record["accuracy_met"] = False
    row = classify_row(*args)
    assert row["MC_E0_lower_bound"] == 0.125
    assert row["MC_threshold_evidence"] == "fails_by_lower_bound"
    assert row["MC_E33"] is None and row["MC_accuracy_met"] is None


def test_equal_measured_errors_cannot_prove_a_strict_four_method_winner(saved_inputs):
    args = saved_inputs()
    _, remizov, rk4, _, _ = args
    remizov["result"]["f"] = rk4["result"]["y"][::4]
    remizov["max_abs_error"] = rk4["E_inf_grid33"]
    row = classify_row(*args)
    assert row["R_E33"] == row["RK4_E33"]
    assert row["margin_R"] == 0
    assert row["inferred_unique_four_method_winner"] is None
    assert row["winner_scope"] == "available_only"


@pytest.mark.parametrize("mutation", [
    "r_grid", "r_length", "r_hash", "r_status", "r_error",
    "ivp_seal", "ivp_grid", "ivp_exact", "ivp_hash", "ivp_schema",
    "mc_grid", "mc_hash", "mc_seed", "mc_nonfinite",
])
def test_inconsistent_raw_records_are_rejected_even_when_resealed(saved_inputs, mutation):
    args = saved_inputs()
    _, remizov, rk4, _, mc = args
    if mutation == "r_grid":
        remizov["result"]["x"][1] += 0.001
    elif mutation == "r_length":
        remizov["result"]["f"].pop()
    elif mutation == "r_hash":
        remizov["metadata"]["expression_hash"] = "wrong"
    elif mutation == "r_status":
        remizov["result"]["success"] = False
    elif mutation == "r_error":
        remizov["max_abs_error"] = 0.0
    elif mutation == "ivp_seal":
        rk4["record_sha256"] = "wrong"
    elif mutation == "ivp_grid":
        rk4["result"]["x"][4] += 0.001
        reseal(rk4)
    elif mutation == "ivp_exact":
        rk4["exact_grid33"][1] += 0.001
        reseal(rk4)
    elif mutation == "ivp_hash":
        rk4["expression_hash"] = "wrong"
        reseal(rk4)
    elif mutation == "ivp_schema":
        rk4["schema"] = "unrecognized_record_version"
        reseal(rk4)
    elif mutation == "mc_grid":
        mc[0]["result"]["x"] = [0.001]
    elif mutation == "mc_hash":
        mc[0]["expression_hash"] = "wrong"
    elif mutation == "mc_seed":
        mc[0]["config"]["seed"] += 99
    elif mutation == "mc_nonfinite":
        mc[0]["result"]["f"] = [float("nan")]
    with pytest.raises(ValueError):
        classify_row(*args)


@pytest.mark.parametrize("with_existing_file", [False, True])
def test_existing_output_is_refused_before_any_source_is_read(tmp_path, with_existing_file):
    out = tmp_path / "existing_output"
    out.mkdir()
    if with_existing_file:
        (out / "preserved.txt").write_text("original bytes")
    with pytest.raises(ValueError, match="output directory already exists"):
        build(root=tmp_path / "nonexistent_source_root", out=out)
    assert sorted(p.name for p in out.iterdir()) == (
        ["preserved.txt"] if with_existing_file else [])
    if with_existing_file:
        assert (out / "preserved.txt").read_text() == "original bytes"


@pytest.mark.parametrize("corruption", ["numeric_cell", "duplicate_id", "missing_id"])
def test_independent_verifier_rejects_a_corrupted_release(tmp_path, corruption):
    release = ROOT / "results/saved_grid33_1000_2026-11-04"
    for name in ("grid33_all1000.csv", "missing_remizov.csv", "summary.json",
                 "input_sha256.json", "REPORT.md", "reproduce.json"):
        shutil.copy2(release / name, tmp_path / name)
    path = tmp_path / "grid33_all1000.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if corruption == "numeric_cell":
        rows[0]["RK4_E33"] = str(float(rows[0]["RK4_E33"]) + 0.125)
        message = "numeric cell RK4_E33"
    elif corruption == "duplicate_id":
        rows[1]["equation_id"] = rows[0]["equation_id"]
        message = "CSV IDs/count/duplicates"
    else:
        rows.pop()
        message = "CSV IDs/count/duplicates"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=message):
        verify(tmp_path, root=ROOT)
    assert not (tmp_path / "verification.json").exists()


@pytest.mark.parametrize("module", ["build_saved_grid33", "verify_saved_grid33"])
def test_saved_only_tools_do_not_load_solvers_or_ml(module):
    script = """
import builtins
original = builtins.__import__
blocked = ('remizov_solver', 'chernov_solver', 'mc_solver', 'mc_optimized',
           'all_methods_dataset_loader', 'ml', 'src.numerical_methods',
           'catboost', 'sklearn', 'torch')
def guarded(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked):
        raise AssertionError('saved-only script imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
"""
    script += "\nimport diagnostics." + module + "\n"
    subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, check=True,
                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                   capture_output=True, text=True)
