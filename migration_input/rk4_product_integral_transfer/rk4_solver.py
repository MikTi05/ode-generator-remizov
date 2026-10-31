"""Classical explicit fourth-order Runge--Kutta solver for one Cauchy problem."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Callable, List, Tuple

if TYPE_CHECKING:
    from dataset_loader import CauchyProblem


Coefficient = Callable[[float], float]


@dataclass
class RK4Result:
    equation_id: str
    success: bool
    failure_reason: str
    x: List[float]
    y: List[float]
    dy: List[float]
    n_steps: int
    h: float
    nfev: int


class _IntegrationFailure(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def solve_rk4(
    a: Coefficient,
    b: Coefficient,
    c: Coefficient,
    g: Coefficient,
    x0: float,
    x1: float,
    y0: float,
    v0: float,
    n_steps: int,
    a_min: float = 1.0e-3,
    equation_id: str = "",
) -> RK4Result:
    """Solve the second-order equation with fixed-step classical RK4.

    On a numerical failure, ``x``, ``y`` and ``dy`` contain only fully
    completed grid points.  ``nfev`` counts attempted evaluations of the
    two-component right-hand side, including the attempt that detects
    ``a_too_small`` or ``nonfinite_state``.  A successful run therefore has
    exactly ``4 * n_steps`` evaluations.
    """

    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    numeric_inputs = (x0, x1, a_min)
    if not all(math.isfinite(float(value)) for value in numeric_inputs):
        raise ValueError("x0, x1, and a_min must be finite")
    if x1 <= x0:
        raise ValueError("x1 must be greater than x0")
    if a_min <= 0.0:
        raise ValueError("a_min must be positive")

    x0 = float(x0)
    x1 = float(x1)
    y0 = float(y0)
    v0 = float(v0)
    h = (x1 - x0) / n_steps
    x_values = [x0]
    y_values = [y0]
    v_values = [v0]
    nfev = 0

    if not math.isfinite(y0) or not math.isfinite(v0):
        return RK4Result(
            equation_id, False, "nonfinite_state", x_values, y_values, v_values, n_steps, h, nfev
        )

    def rhs(x_value: float, y_value: float, v_value: float) -> Tuple[float, float]:
        nonlocal nfev
        nfev += 1
        if not all(math.isfinite(value) for value in (x_value, y_value, v_value)):
            raise _IntegrationFailure("nonfinite_state")
        try:
            a_value = float(a(x_value))
            if not math.isfinite(a_value):
                raise _IntegrationFailure("nonfinite_state")
            if abs(a_value) < a_min:
                raise _IntegrationFailure("a_too_small")
            b_value = float(b(x_value))
            c_value = float(c(x_value))
            g_value = float(g(x_value))
            if not all(math.isfinite(value) for value in (b_value, c_value, g_value)):
                raise _IntegrationFailure("nonfinite_state")
            acceleration = (g_value - b_value * v_value - c_value * y_value) / a_value
        except _IntegrationFailure:
            raise
        except (ArithmeticError, ValueError, TypeError):
            raise _IntegrationFailure("nonfinite_state") from None
        if not math.isfinite(acceleration):
            raise _IntegrationFailure("nonfinite_state")
        return v_value, acceleration

    for step in range(n_steps):
        x_n = x_values[-1]
        y_n = y_values[-1]
        v_n = v_values[-1]
        try:
            k1_y, k1_v = rhs(x_n, y_n, v_n)
            k2_y, k2_v = rhs(
                x_n + 0.5 * h,
                y_n + 0.5 * h * k1_y,
                v_n + 0.5 * h * k1_v,
            )
            k3_y, k3_v = rhs(
                x_n + 0.5 * h,
                y_n + 0.5 * h * k2_y,
                v_n + 0.5 * h * k2_v,
            )
            k4_y, k4_v = rhs(
                x_n + h,
                y_n + h * k3_y,
                v_n + h * k3_v,
            )
            next_y = y_n + (h / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
            next_v = v_n + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
            if not math.isfinite(next_y) or not math.isfinite(next_v):
                raise _IntegrationFailure("nonfinite_state")
        except (ArithmeticError, ValueError, TypeError):
            failure_reason = "nonfinite_state"
        except _IntegrationFailure as failure:
            failure_reason = failure.reason
        else:
            # Derive the grid point from its integer index to avoid cumulative drift.
            next_x = x1 if step + 1 == n_steps else x0 + (step + 1) * h
            x_values.append(next_x)
            y_values.append(next_y)
            v_values.append(next_v)
            continue
        return RK4Result(
            equation_id=equation_id,
            success=False,
            failure_reason=failure_reason,
            x=x_values,
            y=y_values,
            dy=v_values,
            n_steps=n_steps,
            h=h,
            nfev=nfev,
        )

    return RK4Result(
        equation_id=equation_id,
        success=True,
        failure_reason="",
        x=x_values,
        y=y_values,
        dy=v_values,
        n_steps=n_steps,
        h=h,
        nfev=nfev,
    )


def solve_problem_rk4(
    problem: "CauchyProblem",
    n_steps: int,
    a_min: float = 1.0e-3,
) -> RK4Result:
    """Convenience wrapper accepting a problem returned by dataset_loader."""

    return solve_rk4(
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
