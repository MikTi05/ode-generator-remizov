"""Feynman--Kac Monte Carlo for the bounded whole-line Remizov problem.

Euler--Maruyama, trapezoids for the potential and time integral, independent
paths. A normal interval describes E[Z_(h,T)], not the full ODE error. Global
bounds are trusted analytical input certificates, never inferred from a grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from numbers import Integral, Real
import struct
import time
from typing import Sequence

import numpy as np

from remizov_solver import RemizovProblem, _problem_conditions_valid, _validate_target_x


@dataclass(frozen=True)
class MCForcingBound:
    """Caller-supplied proof certificate; source must identify its derivation.

    As for RemizovProblem, the solver checks consistency at visited points but
    cannot prove a certificate for arbitrary Python callables.
    """
    abs_upper_bound: float
    source: str
    bounds_certification: str = "analytic"
    bounds_domain: str = "all_real_x"

    def validate(self) -> None:
        _real(self.abs_upper_bound, "forcing bound", minimum=0.)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("forcing bound needs an analytical source")
        if (self.bounds_certification != "analytic"
                or self.bounds_domain != "all_real_x"):
            raise ValueError("forcing bound must be analytic on all_real_x")


@dataclass(frozen=True)
class MCConfig:
    M: int = 20000
    dt: float = .005
    T: float | None = None
    tail_eps: float = 1e-4
    seed: int | None = 2488
    stream_id: int = 0
    chunk: int = 20000
    max_steps: int = 100000
    max_path_steps: int = 100000000
    max_batch_paths: int = 200000
    max_runtime_seconds: float | None = None

    def validate(self) -> None:
        for name, minimum in (("M", 2), ("chunk", 1), ("max_steps", 1),
                              ("max_path_steps", 1), ("max_batch_paths", 1), ("stream_id", 0)):
            _integer(getattr(self, name), name, minimum)
        if self.seed is not None:
            _integer(self.seed, "seed", 0)
        _real(self.dt, "dt", strictly_positive=True)
        _real(self.tail_eps, "tail_eps", strictly_positive=True)
        if self.T is not None:
            _real(self.T, "T", minimum=0.)
        if self.max_runtime_seconds is not None:
            _real(self.max_runtime_seconds, "max_runtime_seconds", strictly_positive=True)


@dataclass
class MCResult:
    problem_id: str
    success: bool
    failure_reason: str
    message: str
    x: np.ndarray
    f: np.ndarray
    stderr: np.ndarray
    ci95: np.ndarray                 # pointwise approximate normal half-width
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    M: int                          # requested paths per point
    completed_paths: np.ndarray     # complete samples merged into statistics
    dt_requested: float
    dt_actual: float
    T: float | None
    n_steps: int
    lambda_c_gap: float | None
    forcing_abs_upper_bound: float | None
    forcing_bound_source: str | None
    tail_eps: float
    tail_error_bound: float | None
    tail_tolerance_met: bool | None
    seed_entropy: int
    stream_id: int
    point_spawn_keys: list[list[int]]
    runtime_seconds: float
    preparation_seconds: float
    n_path_steps: int
    n_normal_draws: int
    n_batches: int
    coefficient_calls: dict[str, int]
    coefficient_values: dict[str, int]
    peak_batch_paths: int
    applicability_status: str
    confidence_level: float = .95
    interval_kind: str = "pointwise_asymptotic_normal_for_discrete_truncated_mean"
    rng_scheme: str = "PCG64/SeedSequence/point_address_v1"
    discretization_error_bound: float | None = None
    total_error_certified: bool = False


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; bool is not allowed")
    return int(value)


def _real(value: object, name: str, *, minimum: float | None = None,
          strictly_positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    number = float(value)
    if (not math.isfinite(number) or (minimum is not None and number < minimum)
            or (strictly_positive and number <= 0)):
        raise ValueError(f"invalid {name}: expected finite real in the allowed range")
    return number


def auto_T(lambda_value: float, c_abs_upper_bound: float,
           g_abs_upper_bound: float, tail_eps: float) -> float:
    """T with G exp(-(lambda-C)T)/(lambda-C) <= tail_eps.

    C and G must be proven global bounds. Log arithmetic avoids overflow of
    G/(gap*eps). This bounds only the continuous infinite-integral tail.
    """
    lam = _real(lambda_value, "lambda_value")
    C = _real(c_abs_upper_bound, "C", minimum=0.)
    G = _real(g_abs_upper_bound, "G", minimum=0.)
    eps = _real(tail_eps, "tail_eps", strictly_positive=True)
    gap = lam - C
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError("need a finite lambda-C > 0")
    if G == 0:
        return 0.
    T = max(0., (math.log(G) - math.log(gap) - math.log(eps)) / gap)
    T = math.nextafter(T, math.inf) if T > 0 else 0.
    if not math.isfinite(T):
        raise OverflowError("certified horizon is not representable")
    return T


def _tail_bound(G: float, gap: float, T: float) -> float:
    if G == 0:
        return 0.
    logarithm = math.log(G) - math.log(gap) - gap * T
    if logarithm > math.log(np.finfo(float).max):
        return math.inf
    # A positive analytical bound must not be reported as exact zero on underflow.
    return max(math.exp(logarithm), math.nextafter(0., 1.))


class _MCFailure(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class _Run:
    deadline: float | None = None
    n_path_steps: int = 0
    n_normal_draws: int = 0
    n_batches: int = 0
    peak_batch_paths: int = 0
    calls: dict[str, int] = field(default_factory=lambda: dict.fromkeys(("a", "b", "c_R", "g_R"), 0))
    values: dict[str, int] = field(default_factory=lambda: dict.fromkeys(("a", "b", "c_R", "g_R"), 0))

    def check_time(self) -> None:
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise _MCFailure("time_limit_exceeded", "MC cooperative runtime budget exhausted")


def _values(func, x: np.ndarray, role: str, run: _Run) -> np.ndarray:
    run.calls[role] += 1
    run.values[role] += x.size
    try:
        raw = np.asarray(func(x))
        if np.iscomplexobj(raw):
            raise _MCFailure("invalid_coefficient", f"{role} returned complex values")
        values = np.broadcast_to(np.asarray(raw, dtype=float), x.shape)
    except _MCFailure:
        raise
    except Exception as exc:
        raise _MCFailure("coefficient_evaluation_failure", f"{role}: {type(exc).__name__}: {exc}") from exc
    if not np.all(np.isfinite(values)):
        raise _MCFailure("nonfinite_coefficient", f"{role} returned NaN or Inf")
    return values


def _forcing(p: RemizovProblem, x: np.ndarray, bound: MCForcingBound | None,
             run: _Run) -> np.ndarray:
    values = _values(p.g_remizov, x, "g_R", run)
    if bound is not None and np.any(np.abs(values) > bound.abs_upper_bound):
        raise _MCFailure("certified_bound_violation", "g_R exceeds its declared global bound")
    return values


def _potential(p: RemizovProblem, x: np.ndarray, run: _Run) -> np.ndarray:
    values = _values(p.c_remizov, x, "c_R", run)
    if np.any(np.abs(values) > p.c_abs_upper_bound):
        raise _MCFailure("certified_bound_violation", "c_R exceeds its declared global bound")
    return values


def _path_contributions(p: RemizovProblem, x0: float, T: float, N: int,
                        m: int, rng: np.random.Generator,
                        bound: MCForcingBound | None, run: _Run) -> np.ndarray:
    """Original EM and trapezoids; store log weight as integral of c_R-lambda."""
    h = T / N
    x = np.full(m, x0, dtype=float)
    log_weight = np.zeros(m)
    c_prev = _potential(p, x, run).copy()
    total = (.5 * h) * _forcing(p, x, bound, run)
    run.peak_batch_paths = max(run.peak_batch_paths, m)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        for step in range(1, N + 1):
            run.check_time()
            a = _values(p.a, x, "a", run)
            b = _values(p.b, x, "b", run)
            if np.any(a <= 0):
                raise _MCFailure("nonpositive_diffusion", "a(x) must be strictly positive")
            if (np.any(a < p.a_lower_bound) or np.any(a > p.a_upper_bound)
                    or np.any(np.abs(b) > p.b_abs_upper_bound)):
                raise _MCFailure("certified_bound_violation", "a or b exceeds its declared global bound")
            # sqrt(a)*sqrt(h)*sqrt(2) avoids overflowing the intermediate 2*a.
            normal = rng.standard_normal(m)
            run.n_normal_draws += m
            x = x + b * h + (np.sqrt(a) * math.sqrt(h) * math.sqrt(2.)) * normal
            run.n_path_steps += m
            if not np.all(np.isfinite(x)):
                raise _MCFailure("nonfinite_state", "trajectory contains NaN or Inf")
            c_now = _potential(p, x, run)
            log_weight += h * (.5 * (c_prev - p.lambda_value)
                               + .5 * (c_now - p.lambda_value))
            c_prev = c_now.copy()
            weight = .5 * h if step == N else h
            total += weight * np.exp(log_weight) * _forcing(p, x, bound, run)
        if not np.all(np.isfinite(total)):
            raise _MCFailure("nonfinite_state", "path integral contains NaN or Inf")
    return total


@dataclass
class _Moments:
    count: int = 0
    mean: float = 0.
    m2: float = 0.

    def add(self, values: np.ndarray) -> None:
        """Centered batch moments, Chan merge; never subtract E[Z^2]-E[Z]^2."""
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            count = values.size
            anchor = float(values[0])
            mean = anchor + float(np.mean(values - anchor))
            centered = values - mean
            m2 = float(np.dot(centered, centered))
            if self.count:
                total = self.count + count
                delta = mean - self.mean
                self.m2 += m2 + delta * delta * (self.count * (count / total))
                self.mean += delta * (count / total)
            else:
                self.mean, self.m2 = mean, m2
            self.count += count
        if not math.isfinite(self.mean) or not math.isfinite(self.m2) or self.m2 < 0:
            raise _MCFailure("nonfinite_statistics", "centered sample moments are not representable")

    def stderr(self) -> float:
        if self.count < 2:
            raise _MCFailure("insufficient_samples", "sample variance needs at least two paths")
        return math.sqrt((self.m2 / (self.count - 1)) / self.count)


def _spawn_key(problem_id: str, x: float, stream_id: int) -> list[int]:
    # Stable across scheduling/order/adding other points. Chunk remains part of
    # the reproducibility configuration: its change changes path/draw grouping.
    address = b"mc_point_v1\0" + problem_id.encode("utf-8") + struct.pack("<d", 0. if x == 0 else x)
    words = struct.unpack("<8I", hashlib.sha256(address).digest())
    # Arbitrarily large nonnegative stream ids are encoded without uint32 loss.
    chunks = [0] if stream_id == 0 else []
    while stream_id:
        chunks.append(stream_id & 0xffffffff)
        stream_id >>= 32
    return [*words, *chunks]


def solve_monte_carlo(problem: RemizovProblem, target_x: Sequence[float],
                      config: MCConfig | None = None, *,
                      forcing_bound: MCForcingBound | None = None) -> MCResult:
    """Return values on target_x, with no initial conditions or derivative.

    No G certificate: explicit T is required, tail bound remains unknown.
    Success means completed finite sampling, never certified total accuracy.
    All callable regularity/SDE hypotheses remain mathematical prerequisites.
    On numerical failure f and all interval arrays are empty; work is retained.
    Runtime checks are cooperative; a blocking callable needs an outer watchdog.
    """
    began = time.perf_counter()
    cfg = config or MCConfig()
    cfg.validate()
    if not isinstance(problem.problem_id, str) or not problem.problem_id:
        raise ValueError("problem_id must be a nonempty string")
    if forcing_bound is not None:
        forcing_bound.validate()
    raw_x = np.asarray(target_x)
    if np.iscomplexobj(raw_x) or raw_x.dtype.kind == "b":
        raise ValueError("target_x must contain real coordinates")
    x = np.asarray(target_x, dtype=float)
    if (x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x))
            or np.any(np.diff(x) <= 0)):
        raise ValueError("target_x must be a finite nonempty strictly increasing vector")
    for name in ("lambda_value", "a_lower_bound", "a_upper_bound", "b_abs_upper_bound",
                 "c_abs_upper_bound", "x_target_min", "x_target_max"):
        _real(getattr(problem, name), name)
    conditions = _problem_conditions_valid(problem)
    if conditions:
        x = _validate_target_x(problem, x)
    entropy = int(np.random.SeedSequence(cfg.seed).entropy)
    keys = [_spawn_key(problem.problem_id, float(p), int(cfg.stream_id)) for p in x]
    run = _Run(deadline=None if cfg.max_runtime_seconds is None else began + cfg.max_runtime_seconds)
    complete = np.zeros(x.size, dtype=np.int64)
    f, errors = [], []
    T, h, N, gap, tail = cfg.T, math.nan, 0, None, None
    failure, message = "", ""
    applicability = "bounds_certified_regularity_assumed" if conditions else "invalid_global_bounds"
    preparation, prepared = 0., False
    try:
        if not conditions:
            raise _MCFailure("invalid_remizov_conditions", "need analytical whole-line coefficient bounds and lambda>C")
        gap = problem.lambda_value - problem.c_abs_upper_bound
        if T is None:
            if forcing_bound is None:
                raise _MCFailure("missing_forcing_bound", "automatic T needs an analytical global G; otherwise pass explicit T")
            try:
                T = auto_T(problem.lambda_value, problem.c_abs_upper_bound,
                           forcing_bound.abs_upper_bound, cfg.tail_eps)
            except OverflowError as exc:
                raise _MCFailure("work_limit_exceeded", str(exc)) from exc
        T = float(T)
        if forcing_bound is not None:
            tail = _tail_bound(forcing_bound.abs_upper_bound, gap, T)
        ratio = T / cfg.dt
        if not math.isfinite(ratio) or ratio > cfg.max_steps:
            raise _MCFailure("work_limit_exceeded", "T/dt exceeds max_steps")
        N = math.ceil(ratio) if T > 0 else 0
        # h<=requested dt, including a possible floating-point rounding at ceil.
        if N and T / N > cfg.dt:
            N += 1
        if N > cfg.max_steps or int(cfg.M) * N * x.size > cfg.max_path_steps:
            raise _MCFailure("work_limit_exceeded", "planned path steps exceed the configured budget")
        if N and min(cfg.M, cfg.chunk) > cfg.max_batch_paths:
            raise _MCFailure("work_limit_exceeded", "batch size exceeds max_batch_paths")
        h = T / N if N else 0.
        preparation = time.perf_counter() - began
        prepared = True
        run.check_time()
        for index, point in enumerate(x):
            if N == 0:
                f.append(0.)
                errors.append(0.)       # the zero-horizon integral is exactly zero
                continue
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy, spawn_key=keys[index])))
            moments = _Moments()
            while moments.count < cfg.M:
                run.check_time()
                m = min(int(cfg.chunk), int(cfg.M) - moments.count)
                run.n_batches += 1
                values = _path_contributions(problem, float(point), T, N, m, rng, forcing_bound, run)
                moments.add(values)
                complete[index] = moments.count
            f.append(moments.mean)
            errors.append(moments.stderr())
        run.check_time()
        f, errors = np.asarray(f), np.asarray(errors)
        with np.errstate(over="raise", invalid="raise"):
            half = 1.96 * errors
            lower, upper = f - half, f + half
        if not all(np.all(np.isfinite(v)) for v in (f, errors, half, lower, upper)):
            raise _MCFailure("nonfinite_statistics", "sample estimates or intervals are not finite")
    except _MCFailure as exc:
        failure, message = exc.reason, str(exc)
    except (FloatingPointError, OverflowError) as exc:
        failure, message = "numerical_overflow", f"{type(exc).__name__}: {exc}"
    if failure:
        f, errors, half, lower, upper = (np.array([], dtype=float) for _ in range(5))
        if failure in ("certified_bound_violation", "nonpositive_diffusion"):
            applicability = "violated_on_visited_points"
    elif forcing_bound is None:
        message = "explicit T; tail error unknown without global G; time bias and total error unbounded"
    else:
        message = "tail bound available; intervals exclude time bias and tail; total error is not certified"
    runtime = time.perf_counter() - began
    if not prepared:
        preparation = runtime
    return MCResult(
        problem.problem_id, not failure, failure, message, x.copy(), f, errors, half, lower, upper,
        int(cfg.M), complete, float(cfg.dt), h, T, N, gap,
        None if forcing_bound is None else float(forcing_bound.abs_upper_bound),
        None if forcing_bound is None else forcing_bound.source,
        float(cfg.tail_eps), tail, None if tail is None else bool(tail <= cfg.tail_eps),
        entropy, int(cfg.stream_id), keys, runtime, preparation,
        run.n_path_steps, run.n_normal_draws, run.n_batches, run.calls, run.values,
        run.peak_batch_paths, applicability,
    )


def paths_for_tolerance(stderr: float, M: int, tol: float) -> int:
    """Pilot estimate of TOTAL paths for pointwise normal half-width <= tol.

    Only sampling uncertainty, not tail/time bias or guaranteed next-run width.
    A zero pilot variance is uninformative here; do not invent a variance floor.
    """
    se = _real(stderr, "stderr", strictly_positive=True)
    count = _integer(M, "M", 2)
    eps = _real(tol, "tol", strictly_positive=True)
    logarithm = math.log(count) + 2 * (math.log(1.96) + math.log(se) - math.log(eps))
    if logarithm > math.log(np.finfo(float).max):
        raise OverflowError("estimated path count is not representable")
    return max(2, math.ceil(math.exp(logarithm)))


__all__ = ["MCConfig", "MCForcingBound", "MCResult", "auto_T",
           "paths_for_tolerance", "solve_monte_carlo"]
