"""Transparent serial verification implementation of Remizov's translation method.

This module implements only the finite Chernoff approximation and the numerical
truncation/interpolation/quadrature controls specified for the verification
benchmark.  It does not solve an initial-value problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline


ArrayFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RemizovProblem:
    problem_id: str
    a: ArrayFunction
    b: ArrayFunction
    c_remizov: ArrayFunction
    g_remizov: ArrayFunction
    lambda_value: float
    a_lower_bound: float
    a_upper_bound: float
    b_abs_upper_bound: float
    c_abs_upper_bound: float
    bounds_certification: str
    bounds_domain: str
    x_target_min: float
    x_target_max: float


@dataclass(frozen=True)
class RemizovConfig:
    n: int
    tail_eps: float = 1.0e-8
    nt_initial: int = 8
    quadrature_rtol: float = 1.0e-7
    quadrature_atol: float = 1.0e-9
    max_quadrature_refinements: int = 4
    dx_initial: float = 0.05
    spatial_rtol: float = 1.0e-7
    spatial_atol: float = 1.0e-9
    max_spatial_refinements: int = 4
    max_work_grid_points: int = 100000


@dataclass
class QuadratureRunDiagnostics:
    dx_limit: float
    nt_levels: List[int] = field(default_factory=list)
    error_estimates: List[float] = field(default_factory=list)
    convergence_thresholds: List[float] = field(default_factory=list)
    converged: bool = False
    final_nt: int = 0


@dataclass
class _WorkCounters:
    n_quadrature_node_evaluations: int = 0
    n_operator_applications: int = 0
    n_spline_builds: int = 0
    n_interpolation_calls: int = 0
    n_interpolated_values: int = 0
    spatial_grid_points_peak: int = 0
    spatial_grid_points_total_processed: int = 0
    work_grid_points_required_max: int = 0
    work_domain_left_min: float = math.inf
    work_domain_right_max: float = -math.inf
    work_domain_width_max: float = 0.0

    def record_domain(self, left: float, right: float, required_points: int) -> None:
        self.work_grid_points_required_max = max(
            self.work_grid_points_required_max, required_points
        )
        self.work_domain_left_min = min(self.work_domain_left_min, left)
        self.work_domain_right_max = max(self.work_domain_right_max, right)
        self.work_domain_width_max = max(self.work_domain_width_max, right - left)

    def record_allocated_grid(self, points: int) -> None:
        self.spatial_grid_points_peak = max(self.spatial_grid_points_peak, points)


@dataclass
class _SolveDiagnostics:
    counters: _WorkCounters = field(default_factory=_WorkCounters)
    final_dx: float = math.nan
    final_nt: int = 0
    quadrature_error_estimate: float = math.nan
    spatial_error_estimate: float = math.nan
    quadrature_refinements: int = 0
    spatial_refinements: int = 0
    quadrature_history: List[QuadratureRunDiagnostics] = field(default_factory=list)
    spatial_dx_levels: List[float] = field(default_factory=list)
    spatial_error_history: List[float] = field(default_factory=list)
    spatial_convergence_thresholds: List[float] = field(default_factory=list)


@dataclass
class RemizovResult:
    problem_id: str
    success: bool
    failure_reason: str
    x: np.ndarray
    f: np.ndarray
    n: int
    T: float
    tail_eps: float
    lambda_value: float
    lambda_c_gap: float
    final_dx: float
    final_nt: int
    quadrature_error_estimate: float
    spatial_error_estimate: float
    quadrature_refinements: int
    spatial_refinements: int
    n_quadrature_node_evaluations: int
    n_operator_applications: int
    n_spline_builds: int
    n_interpolation_calls: int
    n_interpolated_values: int
    spatial_grid_points_peak: int
    spatial_grid_points_total_processed: int
    work_grid_points_required_max: int
    work_domain_left_min: float
    work_domain_right_max: float
    work_domain_width_max: float
    runtime_seconds: float
    quadrature_history: List[QuadratureRunDiagnostics]
    spatial_dx_levels: List[float]
    spatial_error_history: List[float]
    spatial_convergence_thresholds: List[float]


class _RemizovFailure(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_config(config: RemizovConfig) -> None:
    if not _is_integer(config.n) or config.n <= 0:
        raise ValueError("n must be a positive integer and bool is not allowed")
    if not _is_integer(config.nt_initial) or config.nt_initial <= 0:
        raise ValueError("nt_initial must be a positive integer")
    for name in ("max_quadrature_refinements", "max_spatial_refinements"):
        value = getattr(config, name)
        if not _is_integer(value) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not _is_integer(config.max_work_grid_points) or config.max_work_grid_points < 4:
        raise ValueError("max_work_grid_points must be an integer >= 4")
    positive_finite = (
        "tail_eps",
        "quadrature_rtol",
        "quadrature_atol",
        "dx_initial",
        "spatial_rtol",
        "spatial_atol",
    )
    for name in positive_finite:
        try:
            value = float(getattr(config, name))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite real number") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")


def _problem_conditions_valid(problem: RemizovProblem) -> bool:
    if not all(callable(function) for function in (problem.a, problem.b, problem.c_remizov, problem.g_remizov)):
        raise ValueError("a, b, c_remizov, and g_remizov must be callable")
    try:
        numeric = tuple(
            float(value)
            for value in (
                problem.lambda_value,
                problem.a_lower_bound,
                problem.a_upper_bound,
                problem.b_abs_upper_bound,
                problem.c_abs_upper_bound,
                problem.x_target_min,
                problem.x_target_max,
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("problem bounds and interval endpoints must be real scalars") from exc
    if not all(math.isfinite(value) for value in numeric):
        return False
    lambda_value, a_lower, a_upper, b_upper, c_upper, x_min, x_max = numeric
    return bool(
        problem.bounds_certification == "analytic"
        and problem.bounds_domain == "all_real_x"
        and a_lower > 0.0
        and a_upper >= a_lower
        and b_upper >= 0.0
        and c_upper >= 0.0
        and lambda_value > c_upper
        and x_max > x_min
    )


def _validate_target_x(problem: RemizovProblem, target_x: Sequence[float]) -> np.ndarray:
    try:
        values = np.asarray(target_x, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("target_x must be a real numeric array") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError("target_x must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("target_x must contain only finite values")
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise ValueError("target_x must be strictly increasing")
    if values[0] < problem.x_target_min or values[-1] > problem.x_target_max:
        raise ValueError("target_x lies outside the problem target interval")
    return values.copy()


def compute_truncation_T(problem: RemizovProblem, tail_eps: float) -> float:
    """Compute exactly the specified Theorem 6 conservative truncation formula."""

    try:
        tail_eps = float(tail_eps)
        gap = float(problem.lambda_value) - float(problem.c_abs_upper_bound)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tail_eps, lambda, and C bound must be real scalars") from exc
    if not math.isfinite(tail_eps) or tail_eps <= 0.0:
        raise ValueError("tail_eps must be finite and strictly positive")
    if not math.isfinite(gap) or gap <= 0.0:
        raise _RemizovFailure("invalid_remizov_conditions")
    # This is algebraically the prescribed log(2 / (gap * tail_eps)),
    # evaluated in the log domain so valid extreme inputs cannot overflow or
    # underflow before the logarithm is taken.
    logarithm = math.log(2.0) - math.log(gap) - math.log(tail_eps)
    return max(0.0, logarithm / gap)


def _bound_tolerance(problem: RemizovProblem) -> float:
    return 1.0e-12 * (
        1.0
        + max(
            abs(problem.a_lower_bound),
            abs(problem.a_upper_bound),
            problem.b_abs_upper_bound,
            problem.c_abs_upper_bound,
        )
    )


def _as_values(
    function: ArrayFunction,
    points: np.ndarray,
    failure_reason: str,
) -> np.ndarray:
    try:
        raw = np.asarray(function(points), dtype=float)
        if raw.ndim == 0:
            values = np.full(points.shape, float(raw), dtype=float)
        else:
            values = np.broadcast_to(raw, points.shape).astype(float, copy=False)
    except Exception as exc:
        raise _RemizovFailure(failure_reason) from exc
    if not np.all(np.isfinite(values)):
        raise _RemizovFailure(failure_reason)
    return values


def _evaluate_coefficients(
    problem: RemizovProblem,
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_values = _as_values(problem.a, points, "nonfinite_coefficient")
    b_values = _as_values(problem.b, points, "nonfinite_coefficient")
    c_values = _as_values(problem.c_remizov, points, "nonfinite_coefficient")
    tolerance = _bound_tolerance(problem)
    if (
        np.any(a_values < problem.a_lower_bound - tolerance)
        or np.any(a_values > problem.a_upper_bound + tolerance)
        or np.any(np.abs(b_values) > problem.b_abs_upper_bound + tolerance)
        or np.any(np.abs(c_values) > problem.c_abs_upper_bound + tolerance)
        or np.any(a_values <= 0.0)
    ):
        raise _RemizovFailure("certified_bound_violation")
    return a_values, b_values, c_values


def _uniform_grid(
    left: float,
    right: float,
    dx_limit: float,
    max_points: int,
    counters: _WorkCounters,
) -> np.ndarray:
    width = right - left
    if not math.isfinite(width):
        # A finite allocation cannot represent this domain.  The sentinel is
        # a lower bound on the required size; no allocation is attempted.
        counters.record_domain(left, right, max_points + 1)
        raise _RemizovFailure("work_domain_too_large")
    if width <= 0.0:
        raise RuntimeError("internal work interval must have positive width")
    intervals_float = width / dx_limit
    if not math.isfinite(intervals_float):
        counters.record_domain(left, right, max_points + 1)
        raise _RemizovFailure("work_domain_too_large")
    intervals = max(3, int(math.ceil(intervals_float)))
    required_points = intervals + 1
    counters.record_domain(left, right, required_points)
    if required_points > max_points:
        raise _RemizovFailure("work_domain_too_large")
    grid = np.linspace(left, right, required_points)
    counters.record_allocated_grid(required_points)
    actual_dx = width / intervals
    if actual_dx > dx_limit * (1.0 + 32.0 * np.finfo(float).eps):
        raise RuntimeError("internal grid construction error")
    return grid


def _coordinates_inside_source(
    coordinates: Sequence[np.ndarray],
    source_x: np.ndarray,
) -> bool:
    left = float(source_x[0])
    right = float(source_x[-1])
    if any(not np.all(np.isfinite(item)) for item in coordinates):
        return False
    coordinate_scale = max(
        1.0,
        abs(left),
        abs(right),
        *(float(np.max(np.abs(item))) for item in coordinates if item.size),
    )
    tolerance = 64.0 * np.finfo(float).eps * coordinate_scale
    return all(
        bool(np.min(item) >= left - tolerance and np.max(item) <= right + tolerance)
        for item in coordinates
    )


def _apply_operator_on_grid(
    problem: RemizovProblem,
    source_x: np.ndarray,
    source_u: np.ndarray,
    destination_x: np.ndarray,
    tau: float,
    counters: _WorkCounters,
) -> np.ndarray:
    """Apply the specified S(tau) once using four vectorized spline calls."""

    counters.n_operator_applications += 1
    counters.spatial_grid_points_total_processed += destination_x.size
    counters.record_allocated_grid(destination_x.size)
    if (
        source_x.ndim != 1
        or source_u.shape != source_x.shape
        or source_x.size < 4
        or not np.all(np.isfinite(source_x))
        or not np.all(np.diff(source_x) > 0.0)
        or not np.all(np.isfinite(source_u))
    ):
        raise _RemizovFailure("interpolation_failure")
    if destination_x.ndim != 1 or destination_x.size == 0 or not np.all(np.isfinite(destination_x)):
        raise _RemizovFailure("interpolation_failure")
    counters.n_spline_builds += 1
    try:
        spline = CubicSpline(source_x, source_u, extrapolate=False)
    except Exception as exc:
        raise _RemizovFailure("interpolation_failure") from exc

    a_values, b_values, c_values = _evaluate_coefficients(problem, destination_x)
    shift_a = 2.0 * np.sqrt(a_values * tau)
    xi_plus = destination_x + shift_a
    xi_minus = destination_x - shift_a
    xi_b = destination_x + 2.0 * b_values * tau
    requested = (xi_plus, xi_minus, xi_b, destination_x)
    if not _coordinates_inside_source(requested, source_x):
        raise _RemizovFailure("interpolation_out_of_bounds")

    interpolated: List[np.ndarray] = []
    for coordinates in requested:
        counters.n_interpolation_calls += 1
        counters.n_interpolated_values += coordinates.size
        try:
            values = np.asarray(spline(coordinates), dtype=float)
        except Exception as exc:
            raise _RemizovFailure("interpolation_failure") from exc
        if values.shape != coordinates.shape or not np.all(np.isfinite(values)):
            raise _RemizovFailure("interpolation_failure")
        interpolated.append(values)
    u_plus, u_minus, u_b, u_center = interpolated
    with np.errstate(all="ignore"):
        u_next = (
            0.25 * u_plus
            + 0.25 * u_minus
            + 0.50 * u_b
            + tau * c_values * u_center
        )
    if not np.all(np.isfinite(u_next)):
        raise _RemizovFailure("nonfinite_state")
    return u_next


def _compute_chernoff_at_t(
    problem: RemizovProblem,
    target_x: np.ndarray,
    n: int,
    t: float,
    T: float,
    dx_limit: float,
    max_work_grid_points: int,
    counters: _WorkCounters,
) -> np.ndarray:
    tau = t / n
    d_max = max(
        2.0 * math.sqrt(problem.a_upper_bound * T / n),
        2.0 * problem.b_abs_upper_bound * T / n,
    )

    def interval(level: int) -> Tuple[float, float]:
        remaining = n - level
        return (
            problem.x_target_min - remaining * d_max,
            problem.x_target_max + remaining * d_max,
        )

    left, right = interval(0)
    source_x = _uniform_grid(
        left, right, dx_limit, max_work_grid_points, counters
    )
    source_u = _as_values(problem.g_remizov, source_x, "nonfinite_state")
    for level in range(n):
        if level + 1 < n:
            destination_left, destination_right = interval(level + 1)
            destination_x = _uniform_grid(
                destination_left,
                destination_right,
                dx_limit,
                max_work_grid_points,
                counters,
            )
        else:
            destination_x = target_x
        destination_u = _apply_operator_on_grid(
            problem,
            source_x,
            source_u,
            destination_x,
            tau,
            counters,
        )
        source_x, source_u = destination_x, destination_u
    return source_u


def _integrate_fixed_nt(
    problem: RemizovProblem,
    target_x: np.ndarray,
    config: RemizovConfig,
    T: float,
    dx_limit: float,
    nt: int,
    counters: _WorkCounters,
) -> np.ndarray:
    if T == 0.0:
        return np.zeros_like(target_x)
    nodes, weights = np.polynomial.legendre.leggauss(nt)
    t_nodes = 0.5 * T * (nodes + 1.0)
    accumulator = np.zeros_like(target_x)
    for node, weight in zip(t_nodes, weights):
        counters.n_quadrature_node_evaluations += 1
        values = _compute_chernoff_at_t(
            problem,
            target_x,
            config.n,
            float(node),
            T,
            dx_limit,
            config.max_work_grid_points,
            counters,
        )
        with np.errstate(all="ignore"):
            accumulator += float(weight) * math.exp(-problem.lambda_value * float(node)) * values
        if not np.all(np.isfinite(accumulator)):
            raise _RemizovFailure("nonfinite_state")
    result = 0.5 * T * accumulator
    if not np.all(np.isfinite(result)):
        raise _RemizovFailure("nonfinite_state")
    return result


def _converge_quadrature(
    problem: RemizovProblem,
    target_x: np.ndarray,
    config: RemizovConfig,
    T: float,
    dx_limit: float,
    diagnostics: _SolveDiagnostics,
) -> np.ndarray:
    record = QuadratureRunDiagnostics(dx_limit=dx_limit)
    diagnostics.quadrature_history.append(record)
    nt = config.nt_initial
    record.nt_levels.append(nt)
    diagnostics.final_nt = nt
    try:
        previous = _integrate_fixed_nt(
            problem, target_x, config, T, dx_limit, nt, diagnostics.counters
        )
        for refinement in range(1, config.max_quadrature_refinements + 1):
            nt *= 2
            record.nt_levels.append(nt)
            diagnostics.final_nt = nt
            current = _integrate_fixed_nt(
                problem, target_x, config, T, dx_limit, nt, diagnostics.counters
            )
            error = float(np.max(np.abs(current - previous)))
            threshold = config.quadrature_atol + config.quadrature_rtol * float(
                np.max(np.abs(current))
            )
            record.error_estimates.append(error)
            record.convergence_thresholds.append(threshold)
            diagnostics.quadrature_error_estimate = error
            diagnostics.quadrature_refinements += 1
            record.final_nt = nt
            if error <= threshold:
                record.converged = True
                return current
            previous = current
    except _RemizovFailure:
        record.final_nt = diagnostics.final_nt
        raise
    record.final_nt = nt
    raise _RemizovFailure("quadrature_not_converged")


def _converge_spatial(
    problem: RemizovProblem,
    target_x: np.ndarray,
    config: RemizovConfig,
    T: float,
    diagnostics: _SolveDiagnostics,
) -> np.ndarray:
    dx_limit = config.dx_initial
    diagnostics.final_dx = dx_limit
    diagnostics.spatial_dx_levels.append(dx_limit)
    previous = _converge_quadrature(
        problem, target_x, config, T, dx_limit, diagnostics
    )
    for refinement in range(1, config.max_spatial_refinements + 1):
        dx_limit *= 0.5
        diagnostics.final_dx = dx_limit
        # Count an attempted spatial refinement even when its complete
        # quadrature run subsequently fails; result diagnostics describe the
        # work actually attempted, not only successful comparisons.
        diagnostics.spatial_refinements = refinement
        diagnostics.spatial_dx_levels.append(dx_limit)
        current = _converge_quadrature(
            problem, target_x, config, T, dx_limit, diagnostics
        )
        error = float(np.max(np.abs(current - previous)))
        threshold = config.spatial_atol + config.spatial_rtol * float(
            np.max(np.abs(current))
        )
        diagnostics.spatial_error_estimate = error
        diagnostics.spatial_error_history.append(error)
        diagnostics.spatial_convergence_thresholds.append(threshold)
        if error <= threshold:
            return current
        previous = current
    raise _RemizovFailure("spatial_not_converged")


def _make_result(
    problem: RemizovProblem,
    target_x: np.ndarray,
    config: RemizovConfig,
    diagnostics: _SolveDiagnostics,
    success: bool,
    failure_reason: str,
    values: Optional[np.ndarray],
    T: float,
    gap: float,
    runtime_seconds: float,
) -> RemizovResult:
    counters = diagnostics.counters
    left = counters.work_domain_left_min
    right = counters.work_domain_right_max
    return RemizovResult(
        problem_id=problem.problem_id,
        success=success,
        failure_reason=failure_reason,
        x=target_x.copy(),
        f=values.copy() if success and values is not None else np.array([], dtype=float),
        n=config.n,
        T=T,
        tail_eps=config.tail_eps,
        lambda_value=problem.lambda_value,
        lambda_c_gap=gap,
        final_dx=diagnostics.final_dx,
        final_nt=diagnostics.final_nt,
        quadrature_error_estimate=diagnostics.quadrature_error_estimate,
        spatial_error_estimate=diagnostics.spatial_error_estimate,
        quadrature_refinements=diagnostics.quadrature_refinements,
        spatial_refinements=diagnostics.spatial_refinements,
        n_quadrature_node_evaluations=counters.n_quadrature_node_evaluations,
        n_operator_applications=counters.n_operator_applications,
        n_spline_builds=counters.n_spline_builds,
        n_interpolation_calls=counters.n_interpolation_calls,
        n_interpolated_values=counters.n_interpolated_values,
        spatial_grid_points_peak=counters.spatial_grid_points_peak,
        spatial_grid_points_total_processed=counters.spatial_grid_points_total_processed,
        work_grid_points_required_max=counters.work_grid_points_required_max,
        work_domain_left_min=left if math.isfinite(left) else math.nan,
        work_domain_right_max=right if math.isfinite(right) else math.nan,
        work_domain_width_max=counters.work_domain_width_max,
        runtime_seconds=runtime_seconds,
        quadrature_history=diagnostics.quadrature_history,
        spatial_dx_levels=diagnostics.spatial_dx_levels,
        spatial_error_history=diagnostics.spatial_error_history,
        spatial_convergence_thresholds=diagnostics.spatial_convergence_thresholds,
    )


def solve_remizov(
    problem: RemizovProblem,
    target_x: Sequence[float],
    config: RemizovConfig,
) -> RemizovResult:
    """Compute the verified finite translation/Chernoff approximation.

    On every failure ``x`` remains the requested target grid and ``f`` is an
    empty array, so a diagnostic approximation cannot be mistaken for a valid
    solution.
    """

    started = time.perf_counter()
    _validate_config(config)
    diagnostics = _SolveDiagnostics()
    try:
        basic_target = np.asarray(target_x, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("target_x must be a real numeric array") from exc
    if basic_target.ndim != 1 or basic_target.size == 0 or not np.all(np.isfinite(basic_target)):
        raise ValueError("target_x must be a nonempty finite one-dimensional array")
    if basic_target.size > 1 and not np.all(np.diff(basic_target) > 0.0):
        raise ValueError("target_x must be strictly increasing")
    if not _problem_conditions_valid(problem):
        elapsed = time.perf_counter() - started
        return _make_result(
            problem,
            basic_target,
            config,
            diagnostics,
            False,
            "invalid_remizov_conditions",
            None,
            math.nan,
            float(problem.lambda_value) - float(problem.c_abs_upper_bound),
            elapsed,
        )
    checked_target = _validate_target_x(problem, basic_target)
    gap = float(problem.lambda_value) - float(problem.c_abs_upper_bound)
    try:
        T = compute_truncation_T(problem, config.tail_eps)
        if T == 0.0:
            diagnostics.final_dx = 0.0
            diagnostics.final_nt = 0
            diagnostics.quadrature_error_estimate = 0.0
            diagnostics.spatial_error_estimate = 0.0
            values = np.zeros_like(checked_target)
        else:
            values = _converge_spatial(problem, checked_target, config, T, diagnostics)
    except _RemizovFailure as failure:
        elapsed = time.perf_counter() - started
        return _make_result(
            problem,
            checked_target,
            config,
            diagnostics,
            False,
            failure.reason,
            None,
            locals().get("T", math.nan),
            gap,
            elapsed,
        )
    elapsed = time.perf_counter() - started
    return _make_result(
        problem,
        checked_target,
        config,
        diagnostics,
        True,
        "",
        values,
        T,
        gap,
        elapsed,
    )


__all__ = [
    "QuadratureRunDiagnostics",
    "RemizovConfig",
    "RemizovProblem",
    "RemizovResult",
    "compute_truncation_T",
    "solve_remizov",
]
