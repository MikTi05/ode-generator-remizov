"""Load the symbolic Remizov benchmark as vectorized RemizovProblem objects."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List

import numpy as np
import sympy as sp

from create_remizov_benchmark import DEFAULT_OUTPUT
from remizov_solver import RemizovProblem
from validate_remizov_benchmark import X, load_benchmark


def _vectorized_function(expression: sp.Expr) -> Callable[[np.ndarray], np.ndarray]:
    compiled = sp.lambdify(X, expression, modules="numpy")

    def evaluate(points: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(points, dtype=float)
        raw = np.asarray(compiled(coordinates), dtype=float)
        if raw.ndim == 0:
            return np.full(coordinates.shape, float(raw), dtype=float)
        return np.broadcast_to(raw, coordinates.shape).astype(float, copy=False)

    return evaluate


def load_remizov_benchmark(
    path: Path | str = DEFAULT_OUTPUT,
) -> List[RemizovProblem]:
    """Load R1--R5 without transferring IVP-only benchmark metadata."""

    problems: List[RemizovProblem] = []
    for task in load_benchmark(path):
        problems.append(
            RemizovProblem(
                problem_id=task.benchmark_id,
                a=_vectorized_function(task.a_expr),
                b=_vectorized_function(task.b_expr),
                c_remizov=_vectorized_function(task.c_remizov_expr),
                g_remizov=_vectorized_function(task.g_remizov_expr),
                lambda_value=float(task.lambda_expr),
                a_lower_bound=float(task.a_lower_bound),
                a_upper_bound=float(task.a_upper_bound),
                b_abs_upper_bound=float(task.b_abs_upper_bound),
                c_abs_upper_bound=float(task.c_abs_upper_bound),
                bounds_certification=task.bounds_certification,
                bounds_domain=task.bounds_domain,
                x_target_min=float(task.eval_x0),
                x_target_max=float(task.eval_x1),
            )
        )
    return problems


def load_remizov_problem(
    problem_id: str,
    path: Path | str = DEFAULT_OUTPUT,
) -> RemizovProblem:
    for problem in load_remizov_benchmark(path):
        if problem.problem_id == problem_id:
            return problem
    raise KeyError(f"Remizov benchmark problem not found: {problem_id}")


def make_verification_target_x(
    problem: RemizovProblem,
    points: int = 257,
) -> np.ndarray:
    if isinstance(points, bool) or not isinstance(points, int) or points < 1:
        raise ValueError("points must be a positive integer")
    if points == 1:
        return np.array([0.5 * (problem.x_target_min + problem.x_target_max)])
    return np.linspace(problem.x_target_min, problem.x_target_max, points)


__all__ = [
    "load_remizov_benchmark",
    "load_remizov_problem",
    "make_verification_target_x",
]
