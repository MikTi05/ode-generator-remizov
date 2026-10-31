"""Fixed-window adaptation of the supplied Chern.py.

Both schemes solve (lambda - L)y = g_remizov; ``solve_streaming`` is a
different time discretization, not a faster evaluation of fixed-n Chernoff.
Success means finite computation, not certified accuracy. Refine dx, the
window, quadrature and n (or n_steps) independently before trusting results.
The original solver and the supplied source in comparison_inputs are retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing as mp
import time
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline

from remizov_solver import (
    RemizovProblem, _RemizovFailure, _as_values, _evaluate_coefficients,
    _problem_conditions_valid, _validate_target_x, compute_truncation_T,
)


@dataclass(frozen=True)
class ChernovConfig:
    n: int = 100
    num_t: int = 257
    n_steps: int = 2000
    dx: float = 0.025
    padding: float = 8.0
    tail_eps: float = 1e-8
    T: float | None = None
    t_grading: str = "quadratic"
    interp: str = "cubic"
    boundary: str = "taper"
    taper_width: float | None = None
    max_work_grid_points: int = 100000
    n_jobs: int = 1
    start_method: str = "spawn"

    def validate(self) -> None:
        for name, minimum in (("n", 1), ("num_t", 3), ("n_steps", 2),
                              ("max_work_grid_points", 4), ("n_jobs", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("dx", "padding", "tail_eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.T is not None and (not math.isfinite(self.T) or self.T < 0):
            raise ValueError("T must be finite and nonnegative")
        if self.taper_width is not None and (
            not math.isfinite(self.taper_width) or self.taper_width <= 0
        ):
            raise ValueError("taper_width must be finite and positive")
        if self.interp not in ("cubic", "linear"):
            raise ValueError("interp must be cubic or linear")
        if self.boundary not in ("taper", "zero"):
            raise ValueError("boundary must be taper or zero")
        if self.interp == "linear" and self.boundary != "zero":
            raise ValueError("linear interpolation requires boundary=zero")
        if self.t_grading not in ("uniform", "quadratic"):
            raise ValueError("t_grading must be uniform or quadratic")
        if self.start_method not in mp.get_all_start_methods():
            raise ValueError("unsupported multiprocessing start_method")


@dataclass
class ChernovResult:
    problem_id: str
    scheme: str
    success: bool
    failure_reason: str
    x: np.ndarray
    f: np.ndarray
    T: float
    n: int
    n_steps: int
    time_nodes: int
    actual_dx: float
    work_grid_points: int
    work_domain: tuple[float, float]
    runtime_seconds: float
    preparation_seconds: float
    n_operator_applications: int
    n_spline_builds: int
    outside_query_values: int
    boundary_state_max_abs: float
    total_error_certified: bool = False
    refinement_controls_performed: bool = False


@dataclass
class _Counts:
    operators: int = 0
    splines: int = 0
    outside: int = 0
    boundary_max: float = 0.0

    def merge(self, other: _Counts) -> None:
        self.operators += other.operators
        self.splines += other.splines
        self.outside += other.outside
        self.boundary_max = max(self.boundary_max, other.boundary_max)


@dataclass
class _State:
    # Numerical arrays only: spawn workers never pickle user functions/ASTs.
    x: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    g: np.ndarray
    lam: float
    n: int
    interp: str
    boundary: str
    taper_width: float


def _simpson_weights(t: np.ndarray) -> np.ndarray:
    """Composite nonuniform Simpson, identical panels to scipy.simpson.

    Odd number of points; accumulate weights instead of retaining an m by N
    table. Quadratic grading may produce negative endpoint weights.
    """
    h0, h1 = np.diff(t)[::2], np.diff(t)[1::2]
    total = h0 + h1
    weights = np.zeros(t.size)
    weights[:-2:2] += total / 6 * (2 - h1 / h0)
    weights[1::2] += total**3 / (6 * h0 * h1)
    weights[2::2] += total / 6 * (2 - h0 / h1)
    return weights


def _geometry(state: _State, tau: float) -> tuple[np.ndarray, ...]:
    shift = 2 * np.sqrt(state.a * tau)
    return state.x + shift, state.x - shift, state.x + 2 * state.b * tau


def _apply(state: _State, u: np.ndarray, tau: float,
           queries: tuple[np.ndarray, ...], counts: _Counts) -> np.ndarray:
    counts.operators += 1
    counts.boundary_max = max(counts.boundary_max, abs(float(u[0])), abs(float(u[-1])))
    spline = None
    if state.interp == "cubic":
        counts.splines += 1
        spline = CubicSpline(state.x, u, extrapolate=False)
    values = []
    for q in queries:
        left, right = q < state.x[0], q > state.x[-1]
        outside = left | right
        counts.outside += int(np.count_nonzero(outside))
        if spline is None:
            v = np.interp(q, state.x, u, left=0, right=0)
        else:
            v = spline(q)
            v[outside] = 0
            if state.boundary == "taper":
                for mask, edge, sign in ((left, 0, -1), (right, -1, 1)):
                    if np.any(mask):
                        d = sign * (q[mask] - state.x[edge])
                        s = np.clip(1 - d / state.taper_width, 0, 1)
                        v[mask] = (u[edge] + sign * float(spline(state.x[edge], 1)) * d
                                   ) * s * s * (3 - 2 * s)
        values.append(v)
    with np.errstate(all="ignore"):
        result = .25 * values[0] + .25 * values[1] + .5 * values[2] + tau * state.c * u
    if not np.all(np.isfinite(result)):
        raise _RemizovFailure("nonfinite_state")
    return result


def _at_t(state: _State, t: float) -> tuple[np.ndarray, _Counts]:
    counts = _Counts()
    u = state.g.copy()
    if t > 0:
        tau = t / state.n
        queries = _geometry(state, tau)
        for _ in range(state.n):
            u = _apply(state, u, tau, queries, counts)
    return math.exp(-state.lam * t) * u, counts


_POOL_STATE = None


def _init_pool(state: _State) -> None:
    global _POOL_STATE
    _POOL_STATE = state


def _pool_at_t(t: float) -> tuple[np.ndarray, _Counts]:
    return _at_t(_POOL_STATE, t)


class ChernovSolver:
    """Use existing RemizovProblem callables and certified global bounds.

    Each call includes fresh preparation in runtime_seconds. padding defines
    the fixed window around the *problem* target interval. It must be checked
    by an independent window enlargement; boundary amplitude is not an error
    bound. n_steps controls streaming; n/num_t/t_grading do not control it.
    """

    def __init__(self, problem: RemizovProblem, config: ChernovConfig | None = None):
        self.problem = problem
        self.config = config or ChernovConfig()
        self.config.validate()

    def solve(self, target_x: Sequence[float]) -> ChernovResult:
        """Fixed n in every independent Simpson node."""
        return self._run(target_x, "solve")

    def solve_streaming(self, target_x: Sequence[float]) -> ChernovResult:
        """One fixed-step chain, with online composite Simpson accumulation."""
        return self._run(target_x, "streaming")

    def _run(self, target_x: Sequence[float], scheme: str) -> ChernovResult:
        began = time.perf_counter()
        p, c = self.problem, self.config
        # Programming errors remain exceptions, numerical failures use f=[].
        x = np.asarray(target_x, dtype=float)
        if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            raise ValueError("target_x must be a finite, nonempty, strictly increasing vector")
        conditions_valid = _problem_conditions_valid(p)
        if conditions_valid:
            x = _validate_target_x(p, x)
        counts = _Counts()
        T, actual_dx = math.nan, math.nan
        domain = (math.nan, math.nan)
        points, time_nodes, steps, prep_seconds = 0, 0, 0, 0.0
        failure = ""
        f = np.array([], dtype=float)
        try:
            if not conditions_valid:
                raise _RemizovFailure("invalid_remizov_conditions")
            T = compute_truncation_T(p, c.tail_eps) if c.T is None else float(c.T)
            if T == 0:
                f = np.zeros_like(x)
            else:
                domain = (p.x_target_min - c.padding, p.x_target_max + c.padding)
                ratio = (domain[1] - domain[0]) / c.dx
                if not math.isfinite(ratio) or ratio > c.max_work_grid_points - 1:
                    raise _RemizovFailure("work_domain_too_large")
                intervals = max(3, math.ceil(ratio))
                points = intervals + 1
                grid = np.linspace(*domain, points)
                actual_dx = (domain[1] - domain[0]) / intervals
                a, b, potential = _evaluate_coefficients(p, grid)
                forcing = _as_values(p.g_remizov, grid, "nonfinite_state")
                state = _State(grid, a, b, potential, forcing, p.lambda_value,
                               c.n, c.interp, c.boundary,
                               c.taper_width or .1 * (domain[1] - domain[0]))
                prep_seconds = time.perf_counter() - began
                if scheme == "solve":
                    time_nodes = c.num_t + (c.num_t % 2 == 0)
                    s = np.linspace(0, 1, time_nodes)
                    t = T * (s**2 if c.t_grading == "quadratic" else s)
                    weights = _simpson_weights(t)
                    integral = np.zeros_like(grid)
                    if c.n_jobs == 1:
                        rows = (_at_t(state, float(ti)) for ti in t)
                        for weight, (values, work) in zip(weights, rows):
                            integral += weight * values
                            counts.merge(work)
                    else:
                        with mp.get_context(c.start_method).Pool(
                            c.n_jobs, initializer=_init_pool, initargs=(state,)
                        ) as pool:
                            for weight, (values, work) in zip(weights, pool.imap(_pool_at_t, t)):
                                integral += weight * values
                                counts.merge(work)
                else:
                    steps = c.n_steps + c.n_steps % 2
                    time_nodes = steps + 1
                    tau = T / steps
                    queries = _geometry(state, tau)
                    u = forcing.copy()
                    integral = u.copy()  # Simpson endpoint weight = 1
                    for k in range(1, steps + 1):
                        u = _apply(state, u, tau, queries, counts)
                        weight = 1 if k == steps else (4 if k % 2 else 2)
                        integral += weight * math.exp(-p.lambda_value * k * tau) * u
                    integral *= tau / 3
                counts.boundary_max = max(counts.boundary_max,
                                          abs(float(integral[0])), abs(float(integral[-1])))
                if not np.all(np.isfinite(integral)):
                    raise _RemizovFailure("nonfinite_state")
                if c.interp == "cubic":
                    f = CubicSpline(grid, integral, extrapolate=False)(x)
                    counts.splines += 1  # final projection to the requested grid
                else:
                    f = np.interp(x, grid, integral)
                if not np.all(np.isfinite(f)):
                    raise _RemizovFailure("interpolation_failure")
        except _RemizovFailure as exc:
            failure, f = exc.reason, np.array([], dtype=float)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            # Bad target_x was validated before this numerical section.
            failure, f = "interpolation_failure", np.array([], dtype=float)
        return ChernovResult(
            p.problem_id, scheme, not failure, failure, x.copy(), f.copy(), T,
            c.n, steps, time_nodes, actual_dx, points, domain,
            time.perf_counter() - began, prep_seconds, counts.operators,
            counts.splines, counts.outside, counts.boundary_max,
        )


__all__ = ["ChernovConfig", "ChernovResult", "ChernovSolver"]
