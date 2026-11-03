"""Release integrity and labels, using only saved data; no solver calls or fits."""

import copy
import csv
import json

import pytest

from diagnostics.four_method_common import (
    CONFIG, DATA_DIR, METHODS, RECORD_SCHEMA, ROOT, atomic_json, read,
    sealed, validate_record, verify_seal, winner,
)


def test_winner_exact_tie_and_missing_results():
    assert winner(dict(remizov=None, monte_carlo=0.1, rk4=0.0, product_integral=0.0)) == (
        "rk4", ["rk4", "product_integral"])
    assert winner(dict.fromkeys(METHODS, 0.05)) == ("remizov", list(METHODS))
    assert winner(dict.fromkeys(METHODS)) == (None, [])
    assert winner(dict(remizov=0.05, rk4=0.049999999999999996))[0] == "rk4"


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -1.0, True])
def test_fabricated_or_invalid_errors_are_rejected(bad):
    with pytest.raises(ValueError):
        winner(dict(rk4=bad))


def test_atomic_commit_is_exclusive_and_checksum_detects_corruption(tmp_path):
    path = tmp_path / "record.json"
    record = sealed(dict(equation_id="saved_data", E0=0.1))
    atomic_json(path, record)
    assert read(path) == record
    verify_seal(read(path))
    with pytest.raises(FileExistsError):
        atomic_json(path, dict(E0=0.0))
    assert read(path) == record
    assert not list(tmp_path.glob("*.tmp"))
    record["E0"] = 0.0
    with pytest.raises(ValueError, match="checksum"):
        verify_seal(record)


@pytest.fixture
def saved_record():
    old = read(ROOT / "results/rk4_product_integral_integration_2026-11-01/integration.json")["attempts"][0]
    metadata = next(json.loads(line) for line in
                    (DATA_DIR / "all_methods_equations_v1_metadata.jsonl").read_text().splitlines()
                    if json.loads(line)["equation_id"] == old["equation_id"])
    record = dict(schema=RECORD_SCHEMA, plan_sha256="plan", method="rk4", config=CONFIG,
                  **{key: metadata[key] for key in ("equation_id", "expression_hash", "family_id", "structure_signature")},
                  inputs={key: metadata[key] for key in ("x0", "x1", "y0", "v0")},
                  status="success", usable=True, result=old["result"],
                  exact0=old["exact_at_zero"], y0_target=old["y_at_zero"],
                  E0=old["absolute_error_at_zero"], accuracy_met=old["accuracy_met"])
    return record, metadata


def test_resume_accepts_an_existing_correct_terminal_record(saved_record):
    record, metadata = saved_record
    assert validate_record(sealed(record), "plan", "rk4", metadata)["usable"]


@pytest.mark.parametrize("mutation", ["grid", "error", "input", "plan"])
def test_resume_rejects_bad_record_even_with_recomputed_checksum(saved_record, mutation):
    original, metadata = saved_record
    record = copy.deepcopy(original)
    if mutation == "grid":
        record["result"]["x"][64] = 1e-12
    elif mutation == "error":
        record["E0"] = 0.0
    elif mutation == "input":
        record["inputs"]["y0"] = 0.0
    else:
        record["plan_sha256"] = "wrong"
    with pytest.raises(ValueError):
        validate_record(sealed(record), "plan", "rk4", metadata)


def test_failed_record_keeps_absent_error_instead_of_synthetic_ranking(saved_record):
    record, metadata = saved_record
    record.update(status="solver_exception", usable=False, result=None,
                  E0=None, y0_target=None, accuracy_met=None, exact0=None)
    assert not validate_record(sealed(record), "plan", "rk4", metadata)["usable"]
    record["E0"] = 0.0
    with pytest.raises(ValueError):
        validate_record(sealed(record), "plan", "rk4", metadata)


@pytest.mark.parametrize("pid", ["AMV1_F0134_S03", "AMV1_F0136_S01", "AMV1_F0139_S00"])
def test_historical_worker_exception_without_metadata_remains_missing(pid):
    from diagnostics.report_four_method_accuracy import historical
    metadata = next(json.loads(line) for line in
                    (DATA_DIR / "all_methods_equations_v1_metadata.jsonl").read_text().splitlines()
                    if json.loads(line)["equation_id"] == pid)
    path = ROOT / "results/ml_mean_mc_selector_2026-10-31/run/coverage.csv"
    with path.open() as handle:
        reference = next(row for row in csv.DictReader(handle) if row["equation_id"] == pid)
    r, mc, exact = historical(metadata, reference, lambda name: read(ROOT / name))
    assert r["status"] == "worker_exception" and not r["usable"] and r["E"] is None
    assert r["y0"] is None and r["accuracy_met"] is None and r["seconds"] is None
    assert mc["usable"] and mc["E"] == float(reference["E_MC"])
    assert exact == float(reference["y_exact0"])
