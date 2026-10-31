#!/usr/bin/env python3
"""Run the serial verification studies for the Remizov solver.

This program intentionally exercises only the translation-based Remizov
implementation and the manufactured Remizov benchmark.  It neither loads the
production equations dataset nor invokes an IVP solver.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Iterable, List, Mapping, Sequence

import numpy as np

from create_remizov_benchmark import DEFAULT_OUTPUT as DEFAULT_BENCHMARK
from remizov_benchmark_loader import (
    load_remizov_benchmark,
    make_verification_target_x,
)
from remizov_solver import RemizovConfig, RemizovProblem, RemizovResult, solve_remizov


DEFAULT_REPORT = Path("data/remizov_solver_verification_report.md")
R1_N_VALUES = (1, 2, 4, 8, 16, 32)
VARIABLE_CASE_N_VALUES = (4, 8)
MANUAL_ORACLES = {1: 1.48, 2: 1.1728395061728395}

# These dictionaries deliberately spell out the scientific-study budgets.  In
# particular, the study must not silently inherit or adapt the two refinement
# caps from the public library defaults.
VERIFICATION_CONFIG_KWARGS: Mapping[str, float | int] = {
    "tail_eps": 1.0e-8,
    "nt_initial": 8,
    "quadrature_rtol": 1.0e-7,
    "quadrature_atol": 1.0e-9,
    "max_quadrature_refinements": 8,
    "dx_initial": 0.05,
    "spatial_rtol": 1.0e-7,
    "spatial_atol": 1.0e-9,
    "max_spatial_refinements": 5,
    "max_work_grid_points": 100000,
}

MANUAL_CONFIG_KWARGS: Mapping[str, float | int] = {
    "tail_eps": 1.0e-10,
    "nt_initial": 8,
    "quadrature_rtol": 1.0e-9,
    "quadrature_atol": 1.0e-11,
    "max_quadrature_refinements": 8,
    "dx_initial": 0.05,
    "spatial_rtol": 1.0e-9,
    "spatial_atol": 1.0e-11,
    "max_spatial_refinements": 5,
    "max_work_grid_points": 100000,
}


@dataclass(frozen=True)
class StudyRun:
    label: str
    phase: str
    problem_id: str
    config: RemizovConfig
    result: RemizovResult
    max_error: float = math.nan
    error_at_zero: float = math.nan
    manual_oracle: float = math.nan
    manual_abs_error: float = math.nan


def make_verification_config(n: int) -> RemizovConfig:
    """Return the single explicit configuration used by Stage A and Stage B."""

    return RemizovConfig(n=n, **VERIFICATION_CONFIG_KWARGS)


def make_manual_oracle_config(n: int) -> RemizovConfig:
    """Return the strict, fixed-budget configuration used by both R1 oracles."""

    return RemizovConfig(n=n, **MANUAL_CONFIG_KWARGS)


def _finite_or_nan(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else math.nan


def _exact_solution(points: np.ndarray) -> np.ndarray:
    return np.exp(-(np.asarray(points, dtype=float) ** 2))


def _solution_errors(result: RemizovResult) -> tuple[float, float]:
    if not result.success or result.f.shape != result.x.shape:
        return math.nan, math.nan
    errors = np.abs(result.f - _exact_solution(result.x))
    zero_index = int(np.argmin(np.abs(result.x)))
    if not math.isclose(float(result.x[zero_index]), 0.0, abs_tol=1.0e-14):
        return float(np.max(errors)), math.nan
    return float(np.max(errors)), float(errors[zero_index])


def _run(
    problem: RemizovProblem,
    target_x: np.ndarray,
    config: RemizovConfig,
    *,
    label: str,
    phase: str,
    manual_oracle: float = math.nan,
) -> StudyRun:
    print(
        f"starting {label}: n={config.n}, target_points={target_x.size}, "
        f"tail_eps={config.tail_eps:.1e}",
        flush=True,
    )
    result = solve_remizov(problem, target_x, config)
    max_error, error_at_zero = _solution_errors(result)
    manual_abs_error = (
        abs(float(result.f[0]) - manual_oracle)
        if result.success and result.f.size == 1 and math.isfinite(manual_oracle)
        else math.nan
    )
    status = "success" if result.success else result.failure_reason
    print(
        f"finished {label}: {status}; dx={_format_number(result.final_dx)}, "
        f"Nt={result.final_nt}; runtime={result.runtime_seconds:.3f}s",
        flush=True,
    )
    return StudyRun(
        label=label,
        phase=phase,
        problem_id=problem.problem_id,
        config=config,
        result=result,
        max_error=max_error,
        error_at_zero=error_at_zero,
        manual_oracle=manual_oracle,
        manual_abs_error=manual_abs_error,
    )


def run_verification(
    benchmark_path: Path | str = DEFAULT_BENCHMARK,
) -> tuple[List[StudyRun], float]:
    """Run manual oracles, Stage A, and Stage B serially.

    Every reportable run is attempted exactly once.  Stage A/B use the same
    explicit verification configuration and both manual oracles use the same
    explicit strict configuration; a failure is reported without a
    problem-dependent second attempt.
    """

    started = time.perf_counter()
    problems = {problem.problem_id: problem for problem in load_remizov_benchmark(benchmark_path)}
    expected_ids = {"R1", "R2", "R3", "R4", "R5"}
    if set(problems) != expected_ids:
        raise RuntimeError("verification benchmark must contain exactly R1 through R5")

    studies: List[StudyRun] = []
    r1 = problems["R1"]

    for n, oracle in MANUAL_ORACLES.items():
        studies.append(
            _run(
                r1,
                np.array([0.0]),
                make_manual_oracle_config(n),
                label=f"manual R1 n={n}",
                phase="manual",
                manual_oracle=oracle,
            )
        )

    r1_target = make_verification_target_x(r1, 257)
    for n in R1_N_VALUES:
        studies.append(
            _run(
                r1,
                r1_target,
                make_verification_config(n),
                label=f"Stage A R1 n={n}",
                phase="A",
            )
        )

    for problem_id in ("R2", "R3", "R4", "R5"):
        problem = problems[problem_id]
        target_x = make_verification_target_x(problem, 257)
        for n in VARIABLE_CASE_N_VALUES:
            studies.append(
                _run(
                    problem,
                    target_x,
                    make_verification_config(n),
                    label=f"Stage B {problem_id} n={n}",
                    phase="B",
                )
            )

    return studies, time.perf_counter() - started


def _format_number(value: float, digits: int = 8) -> str:
    value = _finite_or_nan(value)
    return "—" if math.isnan(value) else f"{value:.{digits}g}"


def _format_seconds(value: float) -> str:
    return f"{float(value):.3f} s"


def _status(result: RemizovResult) -> str:
    return "success" if result.success else result.failure_reason


def _arrow(values: Iterable[object]) -> str:
    return " → ".join(str(value) for value in values)


def _number_series(values: Iterable[float]) -> str:
    materialized = list(values)
    return ", ".join(_format_number(value, 6) for value in materialized) if materialized else "—"


def _observed_orders(runs: Sequence[StudyRun]) -> dict[int, float]:
    by_n = {run.result.n: run for run in runs}
    orders: dict[int, float] = {}
    for n in R1_N_VALUES:
        current = by_n.get(n)
        refined = by_n.get(2 * n)
        if (
            current is not None
            and refined is not None
            and current.max_error > 0.0
            and refined.max_error > 0.0
            and math.isfinite(current.max_error)
            and math.isfinite(refined.max_error)
        ):
            orders[n] = math.log2(current.max_error / refined.max_error)
        else:
            orders[n] = math.nan
    return orders


def _append_config_table(
    lines: List[str], config: RemizovConfig, value_heading: str
) -> None:
    lines.extend(
        [
            f"| Parameter | {value_heading} |",
            "|---|---:|",
            f"| `tail_eps` | `{config.tail_eps:.0e}` |",
            f"| `nt_initial` | `{config.nt_initial}` |",
            f"| `quadrature_rtol` | `{config.quadrature_rtol:.0e}` |",
            f"| `quadrature_atol` | `{config.quadrature_atol:.0e}` |",
            f"| `max_quadrature_refinements` | `{config.max_quadrature_refinements}` |",
            f"| `dx_initial` | `{config.dx_initial:g}` |",
            f"| `spatial_rtol` | `{config.spatial_rtol:.0e}` |",
            f"| `spatial_atol` | `{config.spatial_atol:.0e}` |",
            f"| `max_spatial_refinements` | `{config.max_spatial_refinements}` |",
            f"| `max_work_grid_points` | `{config.max_work_grid_points}` |",
            "",
        ]
    )


def _last_quadrature_threshold(result: RemizovResult) -> float:
    for history in reversed(result.quadrature_history):
        if history.convergence_thresholds:
            return _finite_or_nan(history.convergence_thresholds[-1])
    return math.nan


def _spatial_thresholds(result: RemizovResult) -> Sequence[float]:
    """Read the public paired spatial thresholds added to solver diagnostics."""

    return getattr(result, "spatial_convergence_thresholds", ())


def _last_spatial_threshold(result: RemizovResult) -> float:
    thresholds = _spatial_thresholds(result)
    return _finite_or_nan(thresholds[-1]) if thresholds else math.nan


def compute_control_indicator(run: StudyRun) -> float:
    """Return the refinement diagnostic, or NaN when it is not defined."""

    if not run.result.success:
        return math.nan
    components = (
        float(run.result.tail_eps),
        float(run.result.quadrature_error_estimate),
        float(run.result.spatial_error_estimate),
    )
    if not all(math.isfinite(value) for value in components):
        return math.nan
    return max(components)


def compute_control_to_method_error_ratio(run: StudyRun) -> float:
    """Return control_indicator/max_error when both quantities are defined."""

    indicator = compute_control_indicator(run)
    if (
        not math.isfinite(indicator)
        or not math.isfinite(run.max_error)
        or run.max_error <= 0.0
    ):
        return math.nan
    return indicator / run.max_error


def build_report(
    studies: Sequence[StudyRun],
    total_runtime_seconds: float,
    benchmark_path: Path | str,
) -> str:
    manual = [run for run in studies if run.phase == "manual"]
    stage_a = [run for run in studies if run.phase == "A"]
    stage_b = [run for run in studies if run.phase == "B"]
    all_runs = list(studies)
    defaults = RemizovConfig(n=1)
    verification_config = make_verification_config(1)
    manual_config = make_manual_oracle_config(1)
    lines: List[str] = [
        "# Remizov solver verification report",
        "",
        "This is a verification implementation. It is not yet the production comparison implementation.",
        "",
        "The spatial interpolation and Gauss-Legendre quadrature are numerical implementation choices and are not part of the mathematical statement of Remizov's Theorem 6.",
        "",
        f"Benchmark source: `{Path(benchmark_path)}`. All runs were executed serially. Total verification-script runtime: **{_format_seconds(total_runtime_seconds)}**.",
        "",
        "## Implemented operator",
        "",
        "The implementation uses exactly",
        "",
        r"$$",
        r"(S(\tau)u)(x)=\frac14u\!\left(x+2\sqrt{a(x)\tau}\right)+\frac14u\!\left(x-2\sqrt{a(x)\tau}\right)+\frac12u\!\left(x+2b(x)\tau\right)+\tau c_R(x)u(x).",
        r"$$",
        "",
        r"For fixed $n$ and $t$, $\tau=t/n$, $u_0=g_R$, and $u_n=S(t/n)^n g_R$. The reported approximation is",
        "",
        r"$$f_{n,T}(x)=\int_0^T e^{-\lambda t}\,u_n(x;t)\,dt,$$",
        "",
        r"with $T=\max\{0,\,\log(2/((\lambda-C_{\mathrm{bound}})\,\varepsilon_{\mathrm{tail}}))/(\lambda-C_{\mathrm{bound}})\}$. The parameter $\lambda$ is not included inside $S(\tau)$.",
        "",
        "## Computational configurations",
        "",
        "### A. Library safety defaults",
        "",
        "These are the unchanged defaults of the public `RemizovConfig` API. They are computational safety defaults; this report does not claim that they are sufficient for every verification problem.",
        "",
    ]
    _append_config_table(lines, defaults, "Library default")
    lines.extend(
        [
            "### B. Verification study configuration (Stage A and Stage B)",
            "",
            "Every Stage A/B case is run exactly once with this same configuration. Refinement stops as soon as its numerical criterion is met.",
            "",
        ]
    )
    _append_config_table(lines, verification_config, "Study value")
    lines.extend(
        [
            "The quadrature comparison budget permits `8 → 16 → 32 → 64 → 128 → 256 → 512 → 1024 → 2048`. The spatial comparison budget permits `0.05 → 0.025 → 0.0125 → 0.00625 → 0.003125 → 0.0015625`. Levels after convergence are not evaluated.",
            "",
            "### C. Manual-oracle configuration",
            "",
            "Both strict R1 oracle cases are run exactly once with this common configuration.",
            "",
        ]
    )
    _append_config_table(lines, manual_config, "Manual value")
    lines.extend(
        [
            "Additional fixed verification choices: `CubicSpline(extrapolate=False)`, Gauss-Legendre quadrature, at least four points per uniform work grid, a 257-point target grid on `[-2, 2]`, and serial evaluation of every quadrature node. The target grid is independent of the work-grid `dx`; clipping, extrapolation, and zero fill outside a work interval are not used.",
            "",
            "The truncation and work-domain geometry use the benchmark's analytically certified bounds. Runtime coefficient checks are only numerical sanity checks and are not treated as proofs of global bounds.",
            "",
            "## Numerical-control interpretation",
            "",
            "For a successful run with finite refinement estimates, `control_indicator = max(tail_eps, quadrature_error_estimate, spatial_error_estimate)` and `control_to_method_error_ratio = control_indicator / max_error` when `max_error > 0`.",
            "",
            "**control_indicator is a numerical refinement diagnostic, not a rigorous total error bound.** Quadrature and spatial estimates are differences between refinement levels, not proven mathematical error bounds. No hard assertion is imposed on their ratio to the finite-n Chernoff error.",
            "",
            "## Manual oracle: R1 at x = 0",
            "",
            "The references below are infinite-integral finite-Chernoff values, not the exact manufactured solution `f(0)=1`.",
            "",
            "| n | status | numerical truncated value | infinite-integral oracle | absolute difference | method max error | control indicator | control/method error | T | final dx | final Nt | runtime |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in sorted(manual, key=lambda item: item.result.n):
        value = float(run.result.f[0]) if run.result.success and run.result.f.size == 1 else math.nan
        lines.append(
            f"| {run.result.n} | `{_status(run.result)}` | {_format_number(value, 12)} | "
            f"{_format_number(run.manual_oracle, 12)} | {_format_number(run.manual_abs_error, 6)} | "
            f"{_format_number(run.max_error)} | "
            f"{_format_number(compute_control_indicator(run))} | "
            f"{_format_number(compute_control_to_method_error_ratio(run))} | "
            f"{_format_number(run.result.T, 9)} | {_format_number(run.result.final_dx, 7)} | "
            f"{run.result.final_nt} | {_format_seconds(run.result.runtime_seconds)} |"
        )
    lines.append("")

    orders = _observed_orders(stage_a)
    lines.extend(
        [
            "## Stage A: R1 convergence on 257 target points",
            "",
            "Observed order is `log2(E_n/E_2n)` where both maximum errors are positive. It is reported descriptively; no assertion that low-n values must equal one is made.",
            "",
            "| n | status | max error | error at zero | observed order to 2n | control indicator | control/method error | T | final dx | final Nt | runtime |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in sorted(stage_a, key=lambda item: item.result.n):
        result = run.result
        lines.append(
            f"| {result.n} | `{_status(result)}` | {_format_number(run.max_error)} | "
            f"{_format_number(run.error_at_zero)} | {_format_number(orders[result.n], 6)} | "
            f"{_format_number(compute_control_indicator(run))} | "
            f"{_format_number(compute_control_to_method_error_ratio(run))} | "
            f"{_format_number(result.T, 9)} | {_format_number(result.final_dx, 7)} | "
            f"{result.final_nt} | {_format_seconds(result.runtime_seconds)} |"
        )
    lines.extend(
        [
            "",
            "## Stage B: variable-coefficient benchmarks",
            "",
            "| problem | n | status | max error | error at zero | control indicator | control/method error | T | final dx | final Nt | quadrature error / threshold | spatial error / threshold | runtime |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in sorted(stage_b, key=lambda item: (item.problem_id, item.result.n)):
        result = run.result
        lines.append(
            f"| {run.problem_id} | {result.n} | `{_status(result)}` | "
            f"{_format_number(run.max_error)} | {_format_number(run.error_at_zero)} | "
            f"{_format_number(compute_control_indicator(run))} | "
            f"{_format_number(compute_control_to_method_error_ratio(run))} | "
            f"{_format_number(result.T, 9)} | {_format_number(result.final_dx, 7)} | "
            f"{result.final_nt} | {_format_number(result.quadrature_error_estimate)} / "
            f"{_format_number(_last_quadrature_threshold(result))} | "
            f"{_format_number(result.spatial_error_estimate)} / "
            f"{_format_number(_last_spatial_threshold(result))} | "
            f"{_format_seconds(result.runtime_seconds)} |"
        )

    lines.extend(
        [
            "",
            "### Previously budget-limited cases",
            "",
            "These cases are highlighted without changing their common Stage B budget. The two error/threshold pairs are the last quadrature and spatial refinement comparisons, respectively.",
            "",
            "| problem | n | status | final dx | final Nt | quadrature error | quadrature threshold | spatial error | spatial threshold | max method error |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    highlighted = {
        ("R3", 8),
        ("R4", 4),
        ("R4", 8),
    }
    for run in sorted(stage_b, key=lambda item: (item.problem_id, item.result.n)):
        if (run.problem_id, run.result.n) not in highlighted:
            continue
        result = run.result
        lines.append(
            f"| {run.problem_id} | {result.n} | `{_status(result)}` | "
            f"{_format_number(result.final_dx, 7)} | {result.final_nt} | "
            f"{_format_number(result.quadrature_error_estimate)} | "
            f"{_format_number(_last_quadrature_threshold(result))} | "
            f"{_format_number(result.spatial_error_estimate)} | "
            f"{_format_number(_last_spatial_threshold(result))} | "
            f"{_format_number(run.max_error)} |"
        )

    lines.extend(
        [
            "",
            "## Refinement histories",
            "",
            "Every full Gauss-Legendre convergence run is listed. Error and criterion entries correspond pairwise to consecutive Nt levels.",
            "",
            "| case | dx limit | Nt levels | quadrature errors | convergence criteria | converged |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for run in all_runs:
        if not run.result.quadrature_history:
            lines.append(f"| {run.label} | — | — | — | — | no |")
            continue
        for history in run.result.quadrature_history:
            lines.append(
                f"| {run.label} | {_format_number(history.dx_limit, 7)} | "
                f"{_arrow(history.nt_levels)} | {_number_series(history.error_estimates)} | "
                f"{_number_series(history.convergence_thresholds)} | "
                f"{'yes' if history.converged else 'no'} |"
            )
    lines.extend(
        [
            "",
            "Spatial refinement uses direct comparisons on the fixed target grid. Error and threshold entries correspond pairwise; each threshold is `spatial_atol + spatial_rtol * max(abs(F_fine))`.",
            "",
            "| case | dx levels attempted | spatial comparison errors | spatial thresholds | paired lengths | final spatial estimate | final threshold | refinements |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for run in all_runs:
        result = run.result
        thresholds = _spatial_thresholds(result)
        paired = (
            "yes"
            if len(result.spatial_error_history) == len(thresholds)
            else f"no ({len(result.spatial_error_history)}/{len(thresholds)})"
        )
        lines.append(
            f"| {run.label} | {_arrow(_format_number(value, 7) for value in result.spatial_dx_levels) or '—'} | "
            f"{_number_series(result.spatial_error_history)} | {_number_series(thresholds)} | "
            f"{paired} | {_format_number(result.spatial_error_estimate)} | "
            f"{_format_number(_last_spatial_threshold(result))} | {result.spatial_refinements} |"
        )

    lines.extend(
        [
            "",
            "## Work counters",
            "",
            "Counters are actual work accumulated across every attempted Nt and dx level, including failed attempts.",
            "",
            "| case | q comparisons | spatial comparisons | q nodes | S applications | splines | interpolation calls | interpolated scalar values | destination points processed | runtime |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in all_runs:
        result = run.result
        lines.append(
            f"| {run.label} | {result.quadrature_refinements} | "
            f"{result.spatial_refinements} | {result.n_quadrature_node_evaluations} | "
            f"{result.n_operator_applications} | {result.n_spline_builds} | "
            f"{result.n_interpolation_calls} | {result.n_interpolated_values} | "
            f"{result.spatial_grid_points_total_processed} | {_format_seconds(result.runtime_seconds)} |"
        )
    lines.extend(
        [
            "",
            "| case | peak grid points | max required points | leftmost work boundary | rightmost work boundary | max work width |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in all_runs:
        result = run.result
        lines.append(
            f"| {run.label} | {result.spatial_grid_points_peak} | "
            f"{result.work_grid_points_required_max} | "
            f"{_format_number(result.work_domain_left_min, 9)} | "
            f"{_format_number(result.work_domain_right_max, 9)} | "
            f"{_format_number(result.work_domain_width_max, 9)} |"
        )

    failures = Counter(run.result.failure_reason for run in all_runs if not run.result.success)
    lines.extend(["", "## Failure diagnostics", ""])
    if failures:
        lines.extend(["| failure status | count | cases |", "|---|---:|---|"])
        for reason, count in sorted(failures.items()):
            labels = ", ".join(run.label for run in all_runs if run.result.failure_reason == reason)
            lines.append(f"| `{reason}` | {count} | {labels} |")
        lines.extend(
            [
                "",
                "| problem | n | failure status | final dx | final Nt | last quadrature error | last quadrature threshold | last spatial error | last spatial threshold | peak grid | required max grid | runtime |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run in (item for item in all_runs if not item.result.success):
            result = run.result
            lines.append(
                f"| {run.problem_id} | {result.n} | `{result.failure_reason}` | "
                f"{_format_number(result.final_dx, 7)} | {result.final_nt} | "
                f"{_format_number(result.quadrature_error_estimate)} | "
                f"{_format_number(_last_quadrature_threshold(result))} | "
                f"{_format_number(result.spatial_error_estimate)} | "
                f"{_format_number(_last_spatial_threshold(result))} | "
                f"{result.spatial_grid_points_peak} | "
                f"{result.work_grid_points_required_max} | "
                f"{_format_seconds(result.runtime_seconds)} |"
            )
    else:
        lines.append("No failure status occurred in the verification studies.")

    if all_runs:
        peak_run = max(all_runs, key=lambda run: run.result.spatial_grid_points_peak)
        lines.extend(
            [
                "",
                "## Verification totals",
                "",
                f"Maximum peak work grid: **{peak_run.result.spatial_grid_points_peak} points** in `{peak_run.label}`. Maximum required grid recorded over all runs: **{max(run.result.work_grid_points_required_max for run in all_runs)} points**.",
                "",
                f"Full serial verification runtime: **{_format_seconds(total_runtime_seconds)}**.",
            ]
        )
    lines.extend(
        [
            "",
            "A failed result contains the requested target `x` but an empty final `f`; diagnostic approximations are never exposed as successful solutions.",
            "",
            "## Scope boundary",
            "",
            "No production equation, initial condition, RK4 comparison, other numerical method, ML workflow, multiprocessing, GPU path, or best-method label was used by this script.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    studies, elapsed = run_verification(args.benchmark)
    report = build_report(studies, elapsed, args.benchmark)
    write_report(args.report, report)
    successes = sum(run.result.success for run in studies)
    failures = sum(not run.result.success for run in studies)
    print(
        f"wrote {args.report}: {successes} successful reportable runs, "
        f"{failures} failed reportable runs; total runtime={elapsed:.3f}s",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
