"""Left-exponential product-integral solver for a second-order IVP.

The module solves

``a(x) y'' + b(x) y' + c(x) y = g(x)``

with prescribed ``y(x0)`` and ``y'(x0)``.  It uses the augmented state
``Z = (y, y', 1)^T`` and the matrix

``B(x) = [[0, 1, 0], [-c/a, -b/a, g/a], [0, 0, 0]]``.

On the fixed grid ``x_k = x0 + k h`` the implemented step is exactly

``Z_{k+1} = scipy.linalg.expm(h * B(x_k)) @ Z_k``.

Thus, this is the *left piecewise-constant exponential approximation of the
matrix product integral*.  The endpoint is deliberately the left endpoint;
there is no midpoint evaluation, adaptivity, Magnus correction, or separate
variation-of-constants approximation.  Later propagators act on the left, so
after ``N`` steps their order is
``exp(h B_{N-1}) ... exp(h B_1) exp(h B_0) Z_0`` and must not be rearranged.

For constant ``B`` the scheme reproduces the constant-coefficient augmented
system exactly up to floating-point and matrix-exponential error.  For a
sufficiently smooth variable ``B``, this particular left-endpoint
discretization has local truncation error ``O(h**2)`` and global error
``O(h)``.  This is a statement about the chosen discretization, not about
product integrals in general.

For ``N`` steps, the implementation performs ``N`` complete coefficient-point
evaluations, ``N`` 3-by-3 matrix exponentials, and ``N`` matrix-vector
multiplications on success.  With fixed matrix dimension its asymptotic
step-count complexity is ``O(N)``, although a matrix exponential has a much
larger constant cost than a simple scalar right-hand-side evaluation.  The
saved trajectory uses ``O(N)`` memory and the additional working memory is
``O(1)`` (excluding temporary storage internal to SciPy).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import TYPE_CHECKING, Callable, List, NoReturn, Tuple

import numpy as np
import scipy.linalg

if TYPE_CHECKING:
    from dataset_loader import CauchyProblem


Coefficient = Callable[[float], float]

METHOD_ID = "product_integral_left_exponential"
DEFAULT_A_MIN = 1.0e-3
INVARIANT_FAIL_TOL = 1.0e-8


@dataclass
class ProductIntegralResult:
    """Trajectory and method-specific diagnostics for one solver run.

    The trajectory contains only fully completed grid points after a failure.
    ``expm_evals`` and ``matvec_evals`` count attempts: an attempted operation
    is counted even if that operation raises or produces a rejected result.
    ``coefficient_points_evaluated`` counts only points at which all four
    scalar coefficient values were obtained and validated.  In contrast,
    ``scalar_coefficient_calls`` counts each individual callable invocation,
    including an invocation that raises or returns an invalid value.
    """

    equation_id: str
    success: bool
    failure_reason: str
    x: List[float]
    y: List[float]
    dy: List[float]
    n_steps: int
    h: float
    method_id: str
    status: str
    message: str
    steps_requested: int
    steps_completed: int
    coefficient_points_evaluated: int
    scalar_coefficient_calls: int
    expm_evals: int
    matvec_evals: int
    max_invariant_drift: float


@dataclass
class _Counters:
    coefficient_points_evaluated: int = 0
    scalar_coefficient_calls: int = 0
    expm_evals: int = 0
    matvec_evals: int = 0
    max_invariant_drift: float = 0.0


class _StepFailure(RuntimeError):
    """Internal exception carrying a public status and concise message."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _input_real(value: object, name: str) -> float:
    """Convert one scalar input without silently dropping an imaginary part."""

    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite real scalar ({type(exc).__name__})") from None
    if array.ndim != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = array.item()
    if np.iscomplexobj(array) or isinstance(scalar, complex):
        try:
            complex_value = complex(scalar)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a finite real scalar") from None
        if not math.isfinite(complex_value.real) or not math.isfinite(complex_value.imag):
            raise ValueError(f"{name} must be a finite real scalar")
        if complex_value.imag != 0.0:
            raise ValueError(f"{name} must be real")
        return float(complex_value.real)
    try:
        converted = float(scalar)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite real scalar") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _initial_trajectory(
    x0: object,
    y0: object,
    v0: object,
) -> Tuple[List[float], List[float], List[float]]:
    """Preserve a valid initial point in an invalid-input result when possible."""

    try:
        initial_x = _input_real(x0, "x0")
        initial_y = _input_real(y0, "y0")
        initial_v = _input_real(v0, "v0")
    except ValueError:
        return [], [], []
    return [initial_x], [initial_y], [initial_v]


def _result(
    *,
    equation_id: str,
    success: bool,
    status: str,
    message: str,
    x: List[float],
    y: List[float],
    dy: List[float],
    n_steps: int,
    h: float,
    counters: _Counters,
) -> ProductIntegralResult:
    return ProductIntegralResult(
        equation_id=equation_id,
        success=success,
        failure_reason="" if success else status,
        x=x,
        y=y,
        dy=dy,
        n_steps=n_steps,
        h=h,
        method_id=METHOD_ID,
        status=status,
        message=message,
        steps_requested=n_steps,
        steps_completed=len(x) - 1 if x else 0,
        coefficient_points_evaluated=counters.coefficient_points_evaluated,
        scalar_coefficient_calls=counters.scalar_coefficient_calls,
        expm_evals=counters.expm_evals,
        matvec_evals=counters.matvec_evals,
        max_invariant_drift=counters.max_invariant_drift,
    )


def _coefficient_value(
    function: Coefficient,
    name: str,
    x_k: float,
    step: int,
    counters: _Counters,
) -> float:
    counters.scalar_coefficient_calls += 1
    try:
        raw_value = function(x_k)
    except Exception as exc:
        raise _StepFailure(
            "coefficient_evaluation_failed",
            f"step {step}, x={x_k:.17g}: coefficient {name} raised {type(exc).__name__}",
        ) from None

    try:
        array = np.asarray(raw_value)
    except Exception as exc:
        raise _StepFailure(
            "coefficient_evaluation_failed",
            f"step {step}, x={x_k:.17g}: coefficient {name} is not a real scalar "
            f"({type(exc).__name__})",
        ) from None
    if array.ndim != 0:
        raise _StepFailure(
            "coefficient_evaluation_failed",
            f"step {step}, x={x_k:.17g}: coefficient {name} is not a real scalar",
        )

    scalar = array.item()
    if np.iscomplexobj(array) or isinstance(scalar, complex):
        try:
            complex_value = complex(scalar)
        except (TypeError, ValueError, OverflowError):
            raise _StepFailure(
                "coefficient_evaluation_failed",
                f"step {step}, x={x_k:.17g}: coefficient {name} is not a real scalar",
            ) from None
        if not math.isfinite(complex_value.real) or not math.isfinite(complex_value.imag):
            raise _StepFailure(
                "nonfinite_coefficient",
                f"step {step}, x={x_k:.17g}: coefficient {name} is non-finite",
            )
        if complex_value.imag != 0.0:
            raise _StepFailure(
                "coefficient_evaluation_failed",
                f"step {step}, x={x_k:.17g}: coefficient {name} is non-real",
            )
        value = float(complex_value.real)
    else:
        try:
            value = float(scalar)
        except (TypeError, ValueError, OverflowError):
            raise _StepFailure(
                "coefficient_evaluation_failed",
                f"step {step}, x={x_k:.17g}: coefficient {name} is not a real scalar",
            ) from None

    if not math.isfinite(value):
        raise _StepFailure(
            "nonfinite_coefficient",
            f"step {step}, x={x_k:.17g}: coefficient {name} is non-finite",
        )
    return value


def _raise(status: str, message: str) -> NoReturn:
    raise _StepFailure(status, message)


def solve_product_integral_left_exponential(
    a: Coefficient,
    b: Coefficient,
    c: Coefficient,
    g: Coefficient,
    x0: float,
    x1: float,
    y0: float,
    v0: float,
    n_steps: int,
    a_min: float = DEFAULT_A_MIN,
    equation_id: str = "",
) -> ProductIntegralResult:
    """Solve one IVP with fixed left-endpoint frozen matrix exponentials.

    Failures are represented by ``ProductIntegralResult`` rather than raised.
    Its trajectory contains the initial condition and every fully completed
    step.  As in :func:`rk4_solver.solve_rk4`, only forward intervals
    (``x1 > x0``) are accepted and the default divisor guard is ``1e-3``.
    """

    counters = _Counters()
    requested_steps = n_steps
    invalid_h = math.nan

    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps <= 0:
        initial_x, initial_y, initial_dy = _initial_trajectory(x0, y0, v0)
        return _result(
            equation_id=str(equation_id),
            success=False,
            status="invalid_input",
            message="n_steps must be a positive integer",
            x=initial_x,
            y=initial_y,
            dy=initial_dy,
            n_steps=requested_steps,
            h=invalid_h,
            counters=counters,
        )

    # Normalize NumPy integer inputs while continuing to reject bool values.
    n_steps = int(n_steps)
    requested_steps = n_steps
    try:
        x0_value = _input_real(x0, "x0")
        x1_value = _input_real(x1, "x1")
        a_min_value = _input_real(a_min, "a_min")
    except ValueError as exc:
        initial_x, initial_y, initial_dy = _initial_trajectory(x0, y0, v0)
        return _result(
            equation_id=str(equation_id),
            success=False,
            status="invalid_input",
            message=str(exc),
            x=initial_x,
            y=initial_y,
            dy=initial_dy,
            n_steps=requested_steps,
            h=invalid_h,
            counters=counters,
        )

    h = (x1_value - x0_value) / n_steps
    if x1_value <= x0_value:
        initial_x, initial_y, initial_dy = _initial_trajectory(x0, y0, v0)
        return _result(
            equation_id=str(equation_id),
            success=False,
            status="invalid_input",
            message="x1 must be greater than x0",
            x=initial_x,
            y=initial_y,
            dy=initial_dy,
            n_steps=requested_steps,
            h=h,
            counters=counters,
        )
    if a_min_value <= 0.0:
        initial_x, initial_y, initial_dy = _initial_trajectory(x0, y0, v0)
        return _result(
            equation_id=str(equation_id),
            success=False,
            status="invalid_input",
            message="a_min must be positive",
            x=initial_x,
            y=initial_y,
            dy=initial_dy,
            n_steps=requested_steps,
            h=h,
            counters=counters,
        )
    if not math.isfinite(h) or h <= 0.0:
        initial_x, initial_y, initial_dy = _initial_trajectory(x0, y0, v0)
        return _result(
            equation_id=str(equation_id),
            success=False,
            status="invalid_input",
            message="computed step size h must be finite and positive",
            x=initial_x,
            y=initial_y,
            dy=initial_dy,
            n_steps=requested_steps,
            h=h,
            counters=counters,
        )
    if not all(callable(function) for function in (a, b, c, g)):
        initial_x, initial_y, initial_dy = _initial_trajectory(x0, y0, v0)
        return _result(
            equation_id=str(equation_id),
            success=False,
            status="invalid_input",
            message="a, b, c, and g must be callable",
            x=initial_x,
            y=initial_y,
            dy=initial_dy,
            n_steps=requested_steps,
            h=h,
            counters=counters,
        )

    try:
        y0_value = _input_real(y0, "y0")
        v0_value = _input_real(v0, "v0")
    except ValueError as exc:
        # A non-finite/non-real initial augmented state is a state failure,
        # while malformed grid and guard parameters above are invalid inputs.
        return _result(
            equation_id=str(equation_id),
            success=False,
            status="nonfinite_state",
            message=f"initial state is invalid: {exc}",
            x=[],
            y=[],
            dy=[],
            n_steps=requested_steps,
            h=h,
            counters=counters,
        )

    try:
        grid = np.linspace(x0_value, x1_value, n_steps + 1, dtype=float)
    except (Exception, MemoryError) as exc:
        return _result(
            equation_id=str(equation_id),
            success=False,
            status="invalid_input",
            message=f"fixed grid construction failed ({type(exc).__name__})",
            x=[x0_value],
            y=[y0_value],
            dy=[v0_value],
            n_steps=requested_steps,
            h=h,
            counters=counters,
        )
    if grid.shape != (n_steps + 1,) or not np.all(np.isfinite(grid)):
        return _result(
            equation_id=str(equation_id),
            success=False,
            status="invalid_input",
            message="fixed grid is malformed or non-finite",
            x=[x0_value],
            y=[y0_value],
            dy=[v0_value],
            n_steps=requested_steps,
            h=h,
            counters=counters,
        )
    if n_steps > 1 and any(grid[index + 1] <= grid[index] for index in range(n_steps)):
        return _result(
            equation_id=str(equation_id),
            success=False,
            status="invalid_input",
            message="fixed grid cannot represent N distinct forward steps",
            x=[x0_value],
            y=[y0_value],
            dy=[v0_value],
            n_steps=requested_steps,
            h=h,
            counters=counters,
        )
    # np.linspace gives the endpoints directly; this assignment also makes
    # the endpoint contract explicit in case NumPy's implementation changes.
    grid[0] = x0_value
    grid[-1] = x1_value
    x_values: List[float] = [x0_value]
    y_values: List[float] = [y0_value]
    dy_values: List[float] = [v0_value]
    state = np.array([y0_value, v0_value, 1.0], dtype=float)

    for step in range(n_steps):
        x_k = float(grid[step])
        try:
            # A coefficient point is complete only after all four values have
            # been obtained.  The a guard follows immediately and, critically,
            # still precedes every division used to build B.
            a_k = _coefficient_value(a, "a", x_k, step, counters)
            b_k = _coefficient_value(b, "b", x_k, step, counters)
            c_k = _coefficient_value(c, "c", x_k, step, counters)
            g_k = _coefficient_value(g, "g", x_k, step, counters)
            counters.coefficient_points_evaluated += 1
            if abs(a_k) < a_min_value:
                _raise(
                    "a_too_small",
                    f"step {step}, x={x_k:.17g}: |a(x)|={abs(a_k):.17g} "
                    f"is below a_min={a_min_value:.17g}",
                )

            try:
                matrix = np.array(
                    [
                        [0.0, 1.0, 0.0],
                        [-c_k / a_k, -b_k / a_k, g_k / a_k],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=float,
                )
            except (ArithmeticError, ValueError, TypeError, OverflowError):
                _raise(
                    "nonfinite_matrix",
                    f"step {step}, x={x_k:.17g}: augmented matrix construction failed",
                )
            if not np.all(np.isfinite(matrix)):
                _raise(
                    "nonfinite_matrix",
                    f"step {step}, x={x_k:.17g}: augmented matrix is non-finite",
                )

            with np.errstate(over="ignore", invalid="ignore"):
                scaled_matrix = h * matrix
            if not np.all(np.isfinite(scaled_matrix)):
                _raise(
                    "nonfinite_matrix",
                    f"step {step}, x={x_k:.17g}: h * B(x) is non-finite",
                )

            counters.expm_evals += 1
            try:
                propagator = scipy.linalg.expm(scaled_matrix)
            except Exception as exc:
                _raise(
                    "matrix_exponential_failed",
                    f"step {step}, x={x_k:.17g}: scipy.linalg.expm raised "
                    f"{type(exc).__name__}",
                )
            try:
                propagator = np.asarray(propagator)
            except Exception as exc:
                _raise(
                    "matrix_exponential_failed",
                    f"step {step}, x={x_k:.17g}: matrix exponential result is invalid "
                    f"({type(exc).__name__})",
                )
            if propagator.shape != (3, 3):
                _raise(
                    "matrix_exponential_failed",
                    f"step {step}, x={x_k:.17g}: matrix exponential has shape "
                    f"{propagator.shape}, expected (3, 3)",
                )
            try:
                propagator_is_finite = bool(np.all(np.isfinite(propagator)))
            except (TypeError, ValueError, OverflowError) as exc:
                _raise(
                    "matrix_exponential_failed",
                    f"step {step}, x={x_k:.17g}: matrix exponential is not a numeric "
                    f"matrix ({type(exc).__name__})",
                )
            if np.iscomplexobj(propagator):
                if not propagator_is_finite:
                    _raise(
                        "nonfinite_matrix",
                        f"step {step}, x={x_k:.17g}: matrix exponential is non-finite",
                    )
                if np.any(np.imag(propagator) != 0.0):
                    _raise(
                        "nonfinite_matrix",
                        f"step {step}, x={x_k:.17g}: matrix exponential is non-real",
                    )
                propagator = np.real(propagator)
            if not propagator_is_finite:
                _raise(
                    "nonfinite_matrix",
                    f"step {step}, x={x_k:.17g}: matrix exponential is non-finite",
                )

            counters.matvec_evals += 1
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    next_state = propagator @ state
            except Exception as exc:
                _raise(
                    "nonfinite_state",
                    f"step {step}, x={x_k:.17g}: matrix-vector multiplication raised "
                    f"{type(exc).__name__}",
                )
            if next_state.shape != (3,) or not np.all(np.isfinite(next_state)):
                _raise(
                    "nonfinite_state",
                    f"step {step}, x={x_k:.17g}: propagated state is non-finite or malformed",
                )

            drift = abs(float(next_state[2]) - 1.0)
            counters.max_invariant_drift = max(counters.max_invariant_drift, drift)
            if drift > INVARIANT_FAIL_TOL:
                _raise(
                    "invariant_violation",
                    f"step {step}, x={x_k:.17g}: augmented-coordinate drift "
                    f"{drift:.17g} exceeds {INVARIANT_FAIL_TOL:.17g}",
                )
        except _StepFailure as failure:
            return _result(
                equation_id=str(equation_id),
                success=False,
                status=failure.status,
                message=failure.message,
                x=x_values,
                y=y_values,
                dy=dy_values,
                n_steps=requested_steps,
                h=h,
                counters=counters,
            )

        # A step becomes public only after every validation, including the
        # invariant check.  Never overwrite or reset next_state[2].
        state = next_state
        x_next = float(grid[step + 1])
        x_values.append(x_next)
        y_values.append(float(state[0]))
        dy_values.append(float(state[1]))

    return _result(
        equation_id=str(equation_id),
        success=True,
        status="success",
        message=f"completed {n_steps} fixed left-endpoint exponential steps",
        x=x_values,
        y=y_values,
        dy=dy_values,
        n_steps=requested_steps,
        h=h,
        counters=counters,
    )


def solve_problem_product_integral_left_exponential(
    problem: "CauchyProblem",
    n_steps: int,
    a_min: float = DEFAULT_A_MIN,
) -> ProductIntegralResult:
    """Solve a :class:`dataset_loader.CauchyProblem` with this method."""

    return solve_product_integral_left_exponential(
        equation_id=problem.equation_id,
        a=problem.a,
        b=problem.b,
        c=problem.c,
        g=problem.g,
        x0=problem.x0,
        x1=problem.x1,
        y0=problem.y0,
        v0=problem.v0,
        n_steps=n_steps,
        a_min=a_min,
    )


__all__ = [
    "DEFAULT_A_MIN",
    "INVARIANT_FAIL_TOL",
    "METHOD_ID",
    "ProductIntegralResult",
    "solve_product_integral_left_exponential",
    "solve_problem_product_integral_left_exponential",
]
