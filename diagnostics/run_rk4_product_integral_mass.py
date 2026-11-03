#!/usr/bin/env python3
"""One resumable fixed-N run of RK4 and left-exponential PI on production1000."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from all_methods_dataset_loader import load_all_methods_equations
from diagnostics.four_method_common import (
    CONFIG, DATA_DIR, METHODS, RECORD_SCHEMA, atomic_json, read, sealed, sha, validate_record,
)
from src.numerical_methods import (
    solve_problem_product_integral_left_exponential, solve_problem_rk4,
)

SOLVERS = {"rk4": solve_problem_rk4,
           "product_integral": solve_problem_product_integral_left_exponential}
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def utc():
    return datetime.now(timezone.utc).isoformat()


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)


def prepare(out):
    # Include the already integrated, still uncommitted files in preservation.
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
    ).decode().split("\0")[:-1]
    own_sources = {"diagnostics/four_method_common.py", "diagnostics/run_rk4_product_integral_mass.py",
                   "diagnostics/report_four_method_accuracy.py", "diagnostics/verify_four_method_accuracy.py",
                   "tests/test_four_method_accuracy.py"}
    before = {name: sha(ROOT / name) for name in sorted(set(names))
              if name not in own_sources and (ROOT / name).is_file()}
    metadata = [json.loads(line) for line in (DATA_DIR / "all_methods_equations_v1_metadata.jsonl").read_text().splitlines()]
    ids = [m["equation_id"] for m in metadata]
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError("production dataset must contain exactly 1000 unique IDs")
    paths = ["all_methods_dataset_loader.py", "src/problem_types.py", "src/all_methods_expressions.py",
             "src/all_methods_dataset_v1.py", "src/numerical_methods/__init__.py",
             "src/numerical_methods/rk4_solver.py", "src/numerical_methods/product_integral_solver.py",
             "diagnostics/four_method_common.py", "diagnostics/run_rk4_product_integral_mass.py",
             "results/ml_mean_mc_selector_2026-10-31/run/coverage.csv",
             "results/ml_selector_hypotheses_2026-10-31/splits.json"]
    paths.extend(str(p.relative_to(ROOT)) for p in sorted(DATA_DIR.iterdir()) if p.is_file())
    out.mkdir(exist_ok=False)
    (out / "records").mkdir()
    for method in SOLVERS:
        (out / "records" / method).mkdir()
    atomic_json(out / "before_sha256.json", before)
    (out / "PROJECT_STATUS.before.md").write_bytes((ROOT / "PROJECT_STATUS.md").read_bytes())
    atomic_json(out / "plan.json", dict(
        schema="rk4_product_integral_mass_v1", created_utc=utc(), date="2026-11-01",
        dataset_dir=str(DATA_DIR.relative_to(ROOT)), equation_ids=ids, config=CONFIG,
        methods=list(SOLVERS), tie_order=list(METHODS), tie_equality="exact equality of unrounded measured E0",
        source_sha256={name: sha(ROOT / name) for name in paths},
        dataset_sha256=sha(DATA_DIR / "all_methods_equations_v1.csv"),
        historical_sources=dict(remizov="results/chernov_solve1000_2026-10-18/solutions",
                                monte_carlo="results/mc1000_x0_2026-10-25/attempts",
                                mean4_reference="results/ml_mean_mc_selector_2026-10-31/run/coverage.csv",
                                splits="results/ml_selector_hypotheses_2026-10-31/splits.json"),
        exact_policy="public exact only after solver return; original loader validation is retained",
        resume_policy="validate checksum, plan, identity, configuration and metrics; skip all valid terminal records including failures",
        corrupt_policy="stop and preserve corrupt file; never silently overwrite or recompute it",
        interrupted_policy="only in-flight uncommitted work may be repeated on resume",
        resources=dict(processes=1, blas_threads=1, expected_rk4_rhs_evals=512000,
                       expected_pi_expm_evals=128000, saved_points_per_success=129,
                       external_time_limit=None, external_memory_limit=None),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "sympy")},
                         threads={name: os.environ.get(name) for name in THREADS}),
        ml_training=False, new_equations=False, refinements=False, parameter_search=False,
        initial_git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
    ))
    print(json.dumps(dict(prepared=str(out), equations=1000, tasks=2000, config=CONFIG), indent=2), flush=True)


def attempt(equation, method, plan, plan_sha):
    problem = equation.cauchy_problem
    metadata = equation.metadata
    if (problem.x0, problem.x1) != (-1.0, 1.0):
        raise ValueError("unexpected IVP interval")
    for key in ("x0", "x1", "y0", "v0"):
        if getattr(problem, key) != metadata[key]:
            raise ValueError("input differs from stored metadata")
    record = dict(schema=RECORD_SCHEMA, plan_sha256=plan_sha, dataset_sha256=plan["dataset_sha256"],
                  method=method, config=CONFIG, started_utc=utc(),
                  **{key: metadata[key] for key in ("equation_id", "expression_hash", "family_id", "structure_signature")},
                  inputs={key: getattr(problem, key) for key in ("x0", "x1", "y0", "v0")},
                  usable=False, E0=None, y0_target=None, exact0=None, accuracy_met=None,
                  E_inf_grid33=None, relative_E_inf_grid33=None, result=None)
    started = time.perf_counter()
    try:
        result = SOLVERS[method](problem, n_steps=128, a_min=0.001)
    except Exception:
        record.update(status="solver_exception", exception=traceback.format_exc())
    else:
        record["result"] = asdict(result)
        record["status"] = "solver_failure" if not result.success else "invalid_target"
    record["solver_wall_seconds"] = time.perf_counter() - started
    if record["result"] and record["result"]["success"]:
        result = record["result"]
        try:
            grid = np.asarray(result["x"], dtype=float)
            values = np.asarray(result["y"], dtype=float)
            derivatives = np.asarray(result["dy"], dtype=float)
            if (grid.shape != (129,) or values.shape != (129,) or derivatives.shape != (129,)
                    or not np.array_equal(grid, np.linspace(-1, 1, 129)) or grid[64] != 0.0
                    or not np.isfinite(values).all() or not np.isfinite(derivatives).all()):
                raise ValueError("incomplete or nonfinite target trajectory")
            # No exact evaluator has been passed to or called by this driver
            # during the solver invocation. g is the original manufactured input.
            evaluation_start = time.perf_counter()
            exact = np.asarray(equation.y_exact(np.linspace(-1, 1, 33)), dtype=float)
            exact0 = float(equation.y_exact(0.0))
            if exact.shape != (33,) or not np.isfinite(exact).all() or exact[16] != exact0:
                raise ValueError("invalid exact evaluator output")
            error = abs(float(values[64]) - exact0)
            e33 = float(np.max(np.abs(values[::4] - exact)))
            if not math.isfinite(error) or not math.isfinite(e33):
                raise ValueError("nonfinite error is not an admissible measurement")
            amplitude = float(np.max(np.abs(exact)))
            record.update(status="success", usable=True, y0_target=float(values[64]), exact0=exact0,
                          E0=error, accuracy_met=error <= 0.05, E_inf_grid33=e33,
                          relative_E_inf_grid33=e33 / amplitude if amplitude else None,
                          exact_grid33=exact.tolist(), evaluation_wall_seconds=time.perf_counter() - evaluation_start)
        except Exception:
            record.update(status="evaluation_failure", exception=traceback.format_exc())
    record.update(finished_utc=utc(), process_peak_rss_mib=peak_rss())
    return sealed(record)


def run(out):
    plan = read(out / "plan.json")
    if plan["config"] != CONFIG or plan["tie_order"] != list(METHODS):
        raise ValueError("fixed plan changed")
    if any(sha(ROOT / name) != digest for name, digest in plan["source_sha256"].items()):
        raise ValueError("a pinned source changed since preparation")
    if any(os.environ.get(name) != "1" for name in THREADS):
        raise ValueError("set all three planned thread variables to 1")
    if platform.python_version() != plan["environment"]["python"] or any(
        importlib.metadata.version(name) != version for name, version in plan["environment"]["versions"].items()
    ):
        raise ValueError("execution environment changed")
    plan_sha = sha(out / "plan.json")
    equations = load_all_methods_equations(DATA_DIR / "all_methods_equations_v1.csv",
                                         DATA_DIR / "all_methods_equations_v1_metadata.jsonl")
    if [e.equation_id for e in equations] != plan["equation_ids"]:
        raise ValueError("loaded ID order differs from plan")
    # Check every existing terminal record before doing any new numerical work.
    expected = {(method, e.equation_id): e.metadata for e in equations for method in SOLVERS}
    existing = {}
    for path in sorted((out / "records").glob("*/*.json")):
        key = (path.parent.name, path.stem)
        if key not in expected:
            raise ValueError("unexpected saved ID or method: " + str(path))
        existing[key] = validate_record(read(path), plan_sha, key[0], expected[key])
    started = time.perf_counter()
    written = 0
    for equation in equations:
        for method in SOLVERS:
            key = (method, equation.equation_id)
            if key in existing:
                continue
            record = attempt(equation, method, plan, plan_sha)
            validate_record(record, plan_sha, method, equation.metadata)
            atomic_json(out / "records" / method / (equation.equation_id + ".json"), record)
            written += 1
            if written % 50 == 0:
                print(json.dumps(dict(saved=len(existing) + written, total=2000,
                                      elapsed_seconds=round(time.perf_counter() - started, 2),
                                      last_equation=equation.equation_id, method=method,
                                      status=record["status"], peak_rss_mib=peak_rss())), flush=True)
    session = dict(finished_utc=utc(), new_solver_calls=written, skipped_valid_records=len(existing),
                   total_records=len(existing) + written, wall_seconds=time.perf_counter() - started,
                   peak_rss_mib=peak_rss(), complete=len(existing) + written == 2000)
    atomic_json(out / ("session_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"), session)
    print(json.dumps(session), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == "prepare":
        prepare(out)
    else:
        with (out / "run.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(out)


if __name__ == "__main__":
    main()
