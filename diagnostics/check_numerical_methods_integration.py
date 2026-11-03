#!/usr/bin/env python3
"""Fixed five-equation integration check; never dispatch a mass benchmark.

The original IVPs are integrated on [-1, 1] with 128 steps. Exact evaluators
are used only after a solver returns. No tolerance-based retry is performed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy

from all_methods_dataset_loader import load_all_methods_equations
from src.numerical_methods import (
    solve_problem_product_integral_left_exponential,
    solve_problem_rk4,
)

DATASET_DIR = ROOT / "data/all_methods_v1_1000_2026-10-18"
CASE_IDS = (
    "AMV1_F0001_S00",
    "AMV1_F0060_S00",
    "AMV1_F0122_S00",
    "AMV1_F0134_S03",
    "AMV1_F0192_S00",
)
N_STEPS = 128
A_MIN = 1.0e-3
EPSILON = 0.05
EVALUATION_GRID = np.linspace(-1.0, 1.0, 33)
SOLVERS = (
    ("rk4", solve_problem_rk4),
    ("product_integral_left_exponential", solve_problem_product_integral_left_exponential),
)


def _hashes():
    paths = [
        "MIGRATION_MANIFEST.json",
        "all_methods_dataset_loader.py",
        "src/problem_types.py",
        "src/all_methods_dataset_v1.py",
        "src/all_methods_expressions.py",
        "remizov_solver.py",
        "chernov_solver.py",
        "mc_solver.py",
        "mc_optimized.py",
        "diagnostics/check_numerical_methods_integration.py",
    ]
    paths.extend(str(p.relative_to(ROOT)) for p in sorted(DATASET_DIR.glob("*")) if p.is_file())
    for folder in ("src/numerical_methods", "migration_input/rk4_product_integral_transfer"):
        paths.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / folder).glob("*.py")))
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def _peak_rss_mib():
    divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def _attempt(method, solver, problem, a_min):
    started = time.perf_counter()
    record = dict(method=method, equation_id=problem.equation_id,
                  config=dict(n_steps=N_STEPS, a_min=a_min),
                  interval=[problem.x0, problem.x1],
                  initial_conditions=dict(y0=problem.y0, v0=problem.v0))
    try:
        result = solver(problem, n_steps=N_STEPS, a_min=a_min)
    except Exception:
        record.update(status="exception", result=None, exception=traceback.format_exc())
    else:
        record.update(status="success" if result.success else "solver_failure",
                      result=asdict(result))
    record.update(wall_seconds=time.perf_counter() - started,
                  process_peak_rss_mib=_peak_rss_mib())
    return record


def run_checks():
    """Return the fixed check in memory; loading 1000 records does not solve them."""
    started = time.perf_counter()
    before = _hashes()
    load_started = time.perf_counter()
    equations = {e.equation_id: e for e in load_all_methods_equations(
        DATASET_DIR / "all_methods_equations_v1.csv",
        DATASET_DIR / "all_methods_equations_v1_metadata.jsonl",
    )}
    load_seconds = time.perf_counter() - load_started
    attempts = []
    cases = []
    for equation_id in CASE_IDS:
        equation = equations[equation_id]
        problem = equation.cauchy_problem
        if (problem.x0, problem.x1) != (-1.0, 1.0):
            raise ValueError("the fixed check requires the original production interval")
        cases.append({key: equation.metadata[key] for key in
                      ("equation_id", "regime", "complexity_level", "split")})
        for method, solver in SOLVERS:
            record = _attempt(method, solver, problem, A_MIN)
            record.update(x0_index=None, y_at_zero=None, exact_at_zero=None,
                          absolute_error_at_zero=None, accuracy_met=None,
                          E_inf_grid33=None, relative_E_inf_grid33=None)
            if record["status"] == "success":
                result = record["result"]
                # Exact equality is intentional: no nearest-neighbour lookup
                # or interpolation of the even-sized NPZ input grid is used.
                if result["x"].count(0.0) != 1:
                    raise ValueError("the numerical trajectory must contain x=0 exactly once")
                index = result["x"].index(0.0)
                indices = np.arange(0, N_STEPS + 1, N_STEPS // 32)
                np.testing.assert_array_equal(np.asarray(result["x"])[indices], EVALUATION_GRID)
                # First public exact invocation for this attempt is AFTER solve.
                exact = np.asarray(equation.y_exact(EVALUATION_GRID))
                exact0 = float(equation.y_exact(0.0))
                values = np.asarray(result["y"])[indices]
                error0 = abs(result["y"][index] - exact0)
                error_grid = float(np.max(np.abs(values - exact)))
                amplitude = float(np.max(np.abs(exact)))
                record.update(x0_index=index, y_at_zero=result["y"][index],
                              exact_at_zero=exact0, absolute_error_at_zero=error0,
                              accuracy_met=error0 <= EPSILON,
                              evaluation_grid=EVALUATION_GRID.tolist(),
                              y_grid33=values.tolist(), exact_grid33=exact.tolist(),
                              E_inf_grid33=error_grid,
                              relative_E_inf_grid33=error_grid / amplitude if amplitude else None)
            attempts.append(record)

    # Exercise the guard using an unchanged real IVP. The elevated threshold
    # is a deliberately invalid applicability configuration, not a new equation.
    equation = equations[CASE_IDS[0]]
    guard_a_min = float(equation.metadata["a_upper_cert"]) + 1.0
    failure_probes = [_attempt(method, solver, equation.cauchy_problem, guard_a_min)
                      for method, solver in SOLVERS]
    after = _hashes()
    return dict(schema="rk4_product_integral_integration_v1", cases=cases,
                dataset_dir=str(DATASET_DIR.relative_to(ROOT)),
                loaded_equations=len(equations), solved_equations=len(CASE_IDS),
                config=dict(n_steps=N_STEPS, a_min=A_MIN, interval=[-1.0, 1.0],
                            h=2.0 / N_STEPS, target_x=[0.0], epsilon=EPSILON),
                exact_policy="stored y0/v0 are inputs; public exact evaluators only after solve",
                attempts=attempts, expected_guard_failures=failure_probes,
                source_sha256_before=before, source_sha256_after=after,
                sources_unchanged=before == after,
                load_seconds=load_seconds, wall_seconds=time.perf_counter() - started,
                process_peak_rss_mib=_peak_rss_mib(),
                environment=dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, platform=platform.platform(),
                                 threads={key: os.environ.get(key) for key in
                                          ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}),
                retries=0, parameter_tuning=False, ml_training=False, mass_benchmark=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON path; existing files are never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    report = run_checks()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("solved_equations", "config", "sources_unchanged", "wall_seconds")}, indent=2))
    return 0 if report["sources_unchanged"] and all(
        a["status"] == "success" for a in report["attempts"]
    ) and all(a["status"] == "solver_failure" and a["result"]["failure_reason"] == "a_too_small"
              for a in report["expected_guard_failures"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
