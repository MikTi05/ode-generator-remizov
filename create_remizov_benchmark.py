#!/usr/bin/env python3
"""Create the five exact-solution verification tasks for a future Remizov solver."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import sympy as sp


BENCHMARK_VERSION = "remizov-verification-v1"
DEFAULT_OUTPUT = Path("data/remizov_benchmark.csv")
FIELDS = (
    "benchmark_version",
    "benchmark_id",
    "a_expr",
    "b_expr",
    "c_remizov_expr",
    "lambda",
    "g_remizov_expr",
    "exact_solution_expr",
    "eval_x0",
    "eval_x1",
    "ivp_y0",
    "ivp_v0",
    "a_lower_bound",
    "a_upper_bound",
    "b_abs_upper_bound",
    "c_abs_upper_bound",
    "bounds_certification",
    "bounds_domain",
)


def _text(expression: sp.Expr) -> str:
    return sp.sstr(expression)


def build_benchmark_rows() -> List[Dict[str, str]]:
    """Build deterministic rows and derive every g_R symbolically."""

    x = sp.Symbol("x", real=True)
    exact = sp.exp(-(x**2))
    exact_d1 = sp.diff(exact, x)
    exact_d2 = sp.diff(exact_d1, x)
    cases = (
        (
            "R1",
            sp.Integer(1),
            sp.Integer(0),
            sp.Integer(0),
            sp.Integer(1),
            sp.Integer(1),
            sp.Integer(1),
            sp.Integer(0),
            sp.Integer(0),
        ),
        (
            "R2",
            sp.Rational(3, 2) + sp.Rational(2, 5) * sp.sin(x),
            sp.Integer(0),
            sp.Integer(0),
            sp.Integer(1),
            sp.Rational(11, 10),
            sp.Rational(19, 10),
            sp.Integer(0),
            sp.Integer(0),
        ),
        (
            "R3",
            sp.Rational(3, 2) + sp.Rational(3, 10) * sp.sin(x),
            sp.Rational(1, 2) * sp.tanh(x),
            sp.Integer(0),
            sp.Integer(1),
            sp.Rational(6, 5),
            sp.Rational(9, 5),
            sp.Rational(1, 2),
            sp.Integer(0),
        ),
        (
            "R4",
            sp.Rational(3, 2) + sp.Rational(3, 10) * sp.sin(x),
            sp.Rational(1, 2) * sp.tanh(x),
            sp.Rational(3, 10) * sp.cos(x),
            sp.Integer(1),
            sp.Rational(6, 5),
            sp.Rational(9, 5),
            sp.Rational(1, 2),
            sp.Rational(3, 10),
        ),
        (
            "R5",
            sp.Rational(3, 2)
            + sp.Rational(1, 4) * sp.sin(sp.tanh(x) + sp.Rational(3, 10) * sp.cos(x)),
            sp.Rational(2, 5) * sp.cos(sp.sin(x))
            + sp.Rational(1, 5) * sp.tanh(x),
            sp.Rational(1, 4) * sp.sin(sp.cos(x))
            + sp.Rational(1, 10) * sp.exp(-(x**2)),
            sp.Integer(1),
            sp.Rational(5, 4),
            sp.Rational(7, 4),
            sp.Rational(3, 5),
            sp.Rational(7, 20),
        ),
    )
    rows: List[Dict[str, str]] = []
    for (
        benchmark_id,
        a_expr,
        b_expr,
        c_expr,
        lambda_expr,
        a_lower,
        a_upper,
        b_abs_upper,
        c_abs_upper,
    ) in cases:
        # This is the only construction of g_R: it is never handwritten.
        g_remizov = sp.factor_terms(
            sp.expand(
                -(
                    a_expr * exact_d2
                    + b_expr * exact_d1
                    + (c_expr - lambda_expr) * exact
                )
            )
        )
        rows.append(
            {
                "benchmark_version": BENCHMARK_VERSION,
                "benchmark_id": benchmark_id,
                "a_expr": _text(a_expr),
                "b_expr": _text(b_expr),
                "c_remizov_expr": _text(c_expr),
                "lambda": _text(lambda_expr),
                "g_remizov_expr": _text(g_remizov),
                "exact_solution_expr": _text(exact),
                "eval_x0": "-2",
                "eval_x1": "2",
                "ivp_y0": _text(exact.subs(x, -2)),
                "ivp_v0": _text(exact_d1.subs(x, -2)),
                "a_lower_bound": _text(a_lower),
                "a_upper_bound": _text(a_upper),
                "b_abs_upper_bound": _text(b_abs_upper),
                "c_abs_upper_bound": _text(c_abs_upper),
                "bounds_certification": "analytic",
                "bounds_domain": "all_real_x",
            }
        )
    return rows


def write_benchmark(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_benchmark_rows()
    write_benchmark(args.output, rows)
    print(f"wrote {len(rows)} benchmark tasks to {args.output}")


if __name__ == "__main__":
    main()
