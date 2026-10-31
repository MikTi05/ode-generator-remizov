#!/usr/bin/env python3
"""Symbolically validate and numerically sanity-check the Remizov benchmark."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import sympy as sp

from create_remizov_benchmark import BENCHMARK_VERSION, DEFAULT_OUTPUT, FIELDS


X = sp.Symbol("x", real=True)
ALLOWED_FUNCTIONS: Mapping[str, object] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
}
NUMERICAL_RESIDUAL_TOLERANCE = 1.0e-10
BOUND_SANITY_TOLERANCE = 1.0e-12


class BenchmarkFormatError(ValueError):
    """Raised when benchmark CSV data is missing, unsafe, or inconsistent."""


def parse_symbolic_expression(text: str, label: str) -> sp.Expr:
    """Convert the small benchmark expression language to SymPy without eval."""

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise BenchmarkFormatError(f"{label}: invalid expression syntax") from exc

    def convert(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise BenchmarkFormatError(f"{label}: non-numeric constant")
            if isinstance(node.value, int):
                return sp.Integer(node.value)
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name):
            if node.id == "x":
                return X
            raise BenchmarkFormatError(f"{label}: forbidden name {node.id!r}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = convert(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left = convert(node.left)
            right = convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise BenchmarkFormatError(f"{label}: forbidden binary operation")
        if isinstance(node, ast.Call):
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in ALLOWED_FUNCTIONS
                or len(node.args) != 1
                or node.keywords
            ):
                raise BenchmarkFormatError(f"{label}: forbidden function call")
            function = ALLOWED_FUNCTIONS[node.func.id]
            return function(convert(node.args[0]))  # type: ignore[operator]
        raise BenchmarkFormatError(f"{label}: forbidden syntax {type(node).__name__}")

    return convert(tree)


@dataclass(frozen=True)
class RemizovBenchmarkTask:
    benchmark_id: str
    a_expr: sp.Expr
    b_expr: sp.Expr
    c_remizov_expr: sp.Expr
    lambda_expr: sp.Expr
    g_remizov_expr: sp.Expr
    exact_solution_expr: sp.Expr
    eval_x0: sp.Expr
    eval_x1: sp.Expr
    ivp_y0: sp.Expr
    ivp_v0: sp.Expr
    a_lower_bound: sp.Expr
    a_upper_bound: sp.Expr
    b_abs_upper_bound: sp.Expr
    c_abs_upper_bound: sp.Expr
    bounds_certification: str
    bounds_domain: str

    @property
    def q_expr(self) -> sp.Expr:
        """Cross-method coefficient q = c_R - lambda; intentionally not stored in CSV."""

        return self.c_remizov_expr - self.lambda_expr

    @property
    def r_expr(self) -> sp.Expr:
        """Cross-method forcing r = -g_R; intentionally not stored in CSV."""

        return -self.g_remizov_expr


@dataclass(frozen=True)
class NumericalSanityResult:
    grid_size: int
    finite_coefficients: bool
    finite_g: bool
    residual_max_abs: float
    sampled_a_min: float
    sampled_a_max: float
    sampled_b_abs_max: float
    sampled_c_abs_max: float
    maximum_bound_violation: float
    passed: bool


@dataclass(frozen=True)
class BenchmarkValidationResult:
    benchmark_id: str
    symbolic_residual_zero: bool
    exact_solution_matches: bool
    initial_conditions_match: bool
    a_lower_positive: bool
    lambda_dominates_c: bool
    certified_metadata_valid: bool
    numerical: NumericalSanityResult
    valid: bool


def _parse_row(row: Mapping[str, str]) -> RemizovBenchmarkTask:
    benchmark_id = row["benchmark_id"]

    def parsed(field: str) -> sp.Expr:
        return parse_symbolic_expression(row[field], f"{benchmark_id}:{field}")

    task = RemizovBenchmarkTask(
        benchmark_id=benchmark_id,
        a_expr=parsed("a_expr"),
        b_expr=parsed("b_expr"),
        c_remizov_expr=parsed("c_remizov_expr"),
        lambda_expr=parsed("lambda"),
        g_remizov_expr=parsed("g_remizov_expr"),
        exact_solution_expr=parsed("exact_solution_expr"),
        eval_x0=parsed("eval_x0"),
        eval_x1=parsed("eval_x1"),
        ivp_y0=parsed("ivp_y0"),
        ivp_v0=parsed("ivp_v0"),
        a_lower_bound=parsed("a_lower_bound"),
        a_upper_bound=parsed("a_upper_bound"),
        b_abs_upper_bound=parsed("b_abs_upper_bound"),
        c_abs_upper_bound=parsed("c_abs_upper_bound"),
        bounds_certification=row["bounds_certification"],
        bounds_domain=row["bounds_domain"],
    )
    scalar_fields = {
        "lambda": task.lambda_expr,
        "eval_x0": task.eval_x0,
        "eval_x1": task.eval_x1,
        "ivp_y0": task.ivp_y0,
        "ivp_v0": task.ivp_v0,
        "a_lower_bound": task.a_lower_bound,
        "a_upper_bound": task.a_upper_bound,
        "b_abs_upper_bound": task.b_abs_upper_bound,
        "c_abs_upper_bound": task.c_abs_upper_bound,
    }
    for field, expression in scalar_fields.items():
        if X in expression.free_symbols or expression.is_real is False:
            raise BenchmarkFormatError(f"{benchmark_id}:{field} must be a real scalar")
    return task


def load_benchmark(path: Path | str = DEFAULT_OUTPUT) -> List[RemizovBenchmarkTask]:
    benchmark_path = Path(path)
    with benchmark_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise BenchmarkFormatError("benchmark CSV has no header")
        missing = [field for field in FIELDS if field not in reader.fieldnames]
        if missing:
            raise BenchmarkFormatError(f"missing benchmark columns: {', '.join(missing)}")
        if "q_expr" in reader.fieldnames or "r_expr" in reader.fieldnames:
            raise BenchmarkFormatError("q and r must be derived, not duplicated in CSV")
        rows = list(reader)
    if not rows:
        raise BenchmarkFormatError("benchmark CSV is empty")
    identifiers = [row["benchmark_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise BenchmarkFormatError("duplicate benchmark_id")
    if identifiers != ["R1", "R2", "R3", "R4", "R5"]:
        raise BenchmarkFormatError("expected exactly the ordered benchmark IDs R1 through R5")
    for row in rows:
        if row["benchmark_version"] != BENCHMARK_VERSION:
            raise BenchmarkFormatError(
                f"{row['benchmark_id']}: unexpected benchmark_version"
            )
    return [_parse_row(row) for row in rows]


def _sample(expression: sp.Expr, grid: np.ndarray) -> np.ndarray:
    function = sp.lambdify(X, expression, modules="numpy")
    with np.errstate(all="ignore"):
        values = np.asarray(function(grid), dtype=float)
    if values.ndim == 0:
        values = np.full(grid.shape, float(values), dtype=float)
    else:
        values = np.broadcast_to(values, grid.shape).astype(float, copy=False)
    return values


def numerical_sanity_check(
    task: RemizovBenchmarkTask,
    grid_size: int = 20001,
) -> NumericalSanityResult:
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    grid = np.linspace(-10.0, 10.0, grid_size)
    a_values = _sample(task.a_expr, grid)
    b_values = _sample(task.b_expr, grid)
    c_values = _sample(task.c_remizov_expr, grid)
    g_values = _sample(task.g_remizov_expr, grid)
    exact_d1 = sp.diff(task.exact_solution_expr, X)
    exact_d2 = sp.diff(exact_d1, X)
    residual_expr = (
        task.a_expr * exact_d2
        + task.b_expr * exact_d1
        + task.q_expr * task.exact_solution_expr
        + task.g_remizov_expr
    )
    residual_values = _sample(residual_expr, grid)
    finite_coefficients = bool(
        np.all(np.isfinite(a_values))
        and np.all(np.isfinite(b_values))
        and np.all(np.isfinite(c_values))
    )
    finite_g = bool(np.all(np.isfinite(g_values)))
    residual_max_abs = (
        float(np.max(np.abs(residual_values)))
        if np.all(np.isfinite(residual_values))
        else math.inf
    )
    sampled_a_min = float(np.min(a_values))
    sampled_a_max = float(np.max(a_values))
    sampled_b_abs_max = float(np.max(np.abs(b_values)))
    sampled_c_abs_max = float(np.max(np.abs(c_values)))
    a_lower = float(task.a_lower_bound)
    a_upper = float(task.a_upper_bound)
    b_upper = float(task.b_abs_upper_bound)
    c_upper = float(task.c_abs_upper_bound)
    maximum_bound_violation = max(
        0.0,
        a_lower - sampled_a_min,
        sampled_a_max - a_upper,
        sampled_b_abs_max - b_upper,
        sampled_c_abs_max - c_upper,
    )
    passed = bool(
        finite_coefficients
        and finite_g
        and residual_max_abs <= NUMERICAL_RESIDUAL_TOLERANCE
        and maximum_bound_violation <= BOUND_SANITY_TOLERANCE
    )
    return NumericalSanityResult(
        grid_size=grid_size,
        finite_coefficients=finite_coefficients,
        finite_g=finite_g,
        residual_max_abs=residual_max_abs,
        sampled_a_min=sampled_a_min,
        sampled_a_max=sampled_a_max,
        sampled_b_abs_max=sampled_b_abs_max,
        sampled_c_abs_max=sampled_c_abs_max,
        maximum_bound_violation=maximum_bound_violation,
        passed=passed,
    )


def validate_task(
    task: RemizovBenchmarkTask,
    grid_size: int = 20001,
) -> BenchmarkValidationResult:
    exact_d1 = sp.diff(task.exact_solution_expr, X)
    exact_d2 = sp.diff(exact_d1, X)
    residual = sp.simplify(
        task.a_expr * exact_d2
        + task.b_expr * exact_d1
        + task.q_expr * task.exact_solution_expr
        + task.g_remizov_expr
    )
    symbolic_residual_zero = residual == 0
    expected_exact = sp.exp(-(X**2))
    exact_solution_matches = sp.simplify(task.exact_solution_expr - expected_exact) == 0
    initial_conditions_match = bool(
        sp.simplify(task.ivp_y0 - task.exact_solution_expr.subs(X, task.eval_x0)) == 0
        and sp.simplify(task.ivp_v0 - exact_d1.subs(X, task.eval_x0)) == 0
    )
    a_lower_positive = bool(task.a_lower_bound > 0)
    lambda_dominates_c = bool(task.lambda_expr > task.c_abs_upper_bound)
    certified_metadata_valid = bool(
        task.bounds_certification == "analytic"
        and task.bounds_domain == "all_real_x"
        and task.a_upper_bound >= task.a_lower_bound
        and task.b_abs_upper_bound >= 0
        and task.c_abs_upper_bound >= 0
    )
    numerical = numerical_sanity_check(task, grid_size=grid_size)
    valid = bool(
        symbolic_residual_zero
        and exact_solution_matches
        and initial_conditions_match
        and a_lower_positive
        and lambda_dominates_c
        and certified_metadata_valid
        and numerical.passed
    )
    return BenchmarkValidationResult(
        benchmark_id=task.benchmark_id,
        symbolic_residual_zero=symbolic_residual_zero,
        exact_solution_matches=exact_solution_matches,
        initial_conditions_match=initial_conditions_match,
        a_lower_positive=a_lower_positive,
        lambda_dominates_c=lambda_dominates_c,
        certified_metadata_valid=certified_metadata_valid,
        numerical=numerical,
        valid=valid,
    )


def validate_benchmark(
    path: Path | str = DEFAULT_OUTPUT,
    grid_size: int = 20001,
) -> List[BenchmarkValidationResult]:
    return [validate_task(task, grid_size=grid_size) for task in load_benchmark(path)]


def render_results(results: Sequence[BenchmarkValidationResult]) -> str:
    lines = [
        "Remizov verification benchmark validation",
        "",
        "The certified bounds are analytic benchmark metadata. The [-10, 10] grid is only a numerical sanity-check and is not a proof of a global bound.",
        "",
        "id symbolic IC a_lower>0 lambda>c_bound finite residual_max sampled_a[min,max] sampled_|b| sampled_|c| bound_violation valid",
    ]
    for result in results:
        sanity = result.numerical
        lines.append(
            f"{result.benchmark_id} {result.symbolic_residual_zero} "
            f"{result.initial_conditions_match} {result.a_lower_positive} "
            f"{result.lambda_dominates_c} "
            f"{sanity.finite_coefficients and sanity.finite_g} "
            f"{sanity.residual_max_abs:.3e} "
            f"[{sanity.sampled_a_min:.9g},{sanity.sampled_a_max:.9g}] "
            f"{sanity.sampled_b_abs_max:.9g} {sanity.sampled_c_abs_max:.9g} "
            f"{sanity.maximum_bound_violation:.3e} {result.valid}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-size", type=int, default=20001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = validate_benchmark(args.input, grid_size=args.grid_size)
    print(render_results(results), end="")
    if not all(result.valid for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
