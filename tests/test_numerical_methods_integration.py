"""Production IVP contracts at a fixed budget; no accuracy-driven tuning."""

import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from all_methods_dataset_loader import load_all_methods_equations
from diagnostics.check_numerical_methods_integration import CASE_IDS, run_checks

ROOT = Path(__file__).resolve().parents[1]
TRANSFER = ROOT / "migration_input/rk4_product_integral_transfer"


@pytest.mark.parametrize("name", ["rk4_solver.py", "product_integral_solver.py"])
def test_solver_copy_only_adapts_type_import_and_documentation(name):
    original = (TRANSFER / name).read_text()
    expected = original.replace("from dataset_loader import CauchyProblem",
                                "from src.problem_types import CauchyProblem")
    expected = expected.replace("Convenience wrapper accepting a problem returned by dataset_loader.",
                                "Convenience wrapper accepting src.problem_types.CauchyProblem.")
    expected = expected.replace(":class:`dataset_loader.CauchyProblem`",
                                ":class:`src.problem_types.CauchyProblem`")
    assert (ROOT / "src/numerical_methods" / name).read_text() == expected


@pytest.mark.parametrize("name", ["test_rk4_solver.py", "test_product_integral_solver.py"])
def test_transferred_unit_tests_only_adapt_imports_and_patch_targets(name):
    expected = (TRANSFER / "tests" / name).read_text()
    expected = expected.replace("from rk4_solver import", "from src.numerical_methods.rk4_solver import")
    expected = expected.replace("from product_integral_solver import",
                                "from src.numerical_methods.product_integral_solver import")
    expected = expected.replace('"product_integral_solver.scipy.linalg.expm"',
                                '"src.numerical_methods.product_integral_solver.scipy.linalg.expm"')
    assert (ROOT / "tests" / name).read_text() == expected


@pytest.fixture(scope="module")
def production_check():
    return run_checks()


def test_production_selection_and_source_integrity(production_check):
    report = production_check
    assert report["loaded_equations"] == 1000
    assert report["solved_equations"] == 5
    assert tuple(case["equation_id"] for case in report["cases"]) == CASE_IDS
    assert {c["split"] for c in report["cases"]} == {"train"}
    assert len({c["regime"] for c in report["cases"]}) == 5
    assert {c["complexity_level"] for c in report["cases"]} == {"L1", "L2", "L3", "L4"}
    assert report["sources_unchanged"]
    assert not report["parameter_tuning"] and not report["ml_training"]
    assert not report["mass_benchmark"] and report["retries"] == 0


@pytest.mark.parametrize("index", range(10))
def test_real_ivp_trajectory_contains_zero_and_reports_exact_error(production_check, index):
    row = production_check["attempts"][index]
    assert row["status"] == "success", row
    result = row["result"]
    assert result["equation_id"] == row["equation_id"]
    assert result["failure_reason"] == ""
    assert result["n_steps"] == 128 and result["h"] == 1 / 64
    assert len(result["x"]) == len(result["y"]) == len(result["dy"]) == 129
    assert result["x"][0] == -1.0 and result["x"][-1] == 1.0
    assert result["y"][0] == row["initial_conditions"]["y0"]
    assert result["dy"][0] == row["initial_conditions"]["v0"]
    assert row["x0_index"] == 64 and result["x"][64] == 0.0
    assert row["y_at_zero"] == result["y"][64]
    error = abs(result["y"][64] - row["exact_at_zero"])
    assert math.isfinite(error) and row["absolute_error_at_zero"] == error
    assert row["accuracy_met"] == (error <= 0.05)
    assert row["E_inf_grid33"] == max(abs(y - exact) for y, exact in
                                       zip(row["y_grid33"], row["exact_grid33"]))
    # No assertion that this arbitrary fixed budget achieves epsilon.
    if row["method"] == "rk4":
        assert result["nfev"] == 512
    else:
        assert result["steps_completed"] == result["expm_evals"] == result["matvec_evals"] == 128
        assert result["max_invariant_drift"] <= 1.0e-8


@pytest.mark.parametrize("index", [0, 1])
def test_guard_failure_preserves_real_initial_point(production_check, index):
    row = production_check["expected_guard_failures"][index]
    result = row["result"]
    assert row["status"] == "solver_failure" and not result["success"]
    assert result["failure_reason"] == "a_too_small"
    assert result["x"] == [-1.0]
    assert result["y"] == [row["initial_conditions"]["y0"]]
    assert result["dy"] == [row["initial_conditions"]["v0"]]
    assert 0.0 not in result["x"]


@pytest.mark.parametrize("module_name,wrapper_name,core_name", [
    ("rk4_solver", "solve_problem_rk4", "solve_rk4"),
    ("product_integral_solver", "solve_problem_product_integral_left_exponential",
     "solve_product_integral_left_exponential"),
])
def test_wrapper_forwards_original_ivp_without_exact_or_resolvent_substitution(
    monkeypatch, production_equation, module_name, wrapper_name, core_name,
):
    equation = production_equation
    problem = equation.cauchy_problem
    module = importlib.import_module("src.numerical_methods." + module_name)
    sentinel = object()

    def inspect_inputs(**kwargs):
        assert set(kwargs) == {"equation_id", "a", "b", "c", "g", "x0", "x1",
                               "y0", "v0", "n_steps", "a_min"}
        for name in ("a", "b", "c", "g"):
            assert kwargs[name] is getattr(problem, name)
        for name in ("x0", "x1", "y0", "v0", "equation_id"):
            assert kwargs[name] == getattr(problem, name) == equation.metadata[name]
        assert kwargs["n_steps"] == 128 and kwargs["a_min"] == 0.001
        return sentinel

    monkeypatch.setattr(module, core_name, inspect_inputs)
    assert getattr(module, wrapper_name)(problem, n_steps=128) is sentinel
    x = np.array([-1.0, 0.0, 1.0])
    r = equation.remizov_problem
    np.testing.assert_allclose([problem.c(float(t)) for t in x], r.c_remizov(x) - r.lambda_value)
    np.testing.assert_array_equal([problem.g(float(t)) for t in x], -r.g_remizov(x))


@pytest.fixture(scope="module")
def production_equation():
    folder = ROOT / "data/all_methods_v1_1000_2026-10-18"
    equations = load_all_methods_equations(folder / "all_methods_equations_v1.csv",
                                         folder / "all_methods_equations_v1_metadata.jsonl")
    return next(e for e in equations if e.equation_id == CASE_IDS[0])


def test_qualified_package_creates_no_forbidden_top_level_modules():
    import src.numerical_methods as methods
    assert methods.solve_rk4.__module__ == "src.numerical_methods.rk4_solver"
    assert methods.solve_product_integral_left_exponential.__module__ == "src.numerical_methods.product_integral_solver"
    assert {"rk4_solver", "product_integral_solver", "dataset_loader"}.isdisjoint(sys.modules)
    manifest = json.loads((ROOT / "MIGRATION_MANIFEST.json").read_text())
    assert "src/all_methods_dataset_v1.py" in manifest["unchanged_mathematical_core"]
