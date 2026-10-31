"""Solve one existing dataset equation with either adapted Chern scheme."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from all_methods_dataset_loader import load_all_methods_equation
from chernov_solver import ChernovConfig, ChernovSolver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data/generated'))
    parser.add_argument('--equation-id', default='AMV1_F0005_S00')
    parser.add_argument('--scheme', choices=('solve', 'streaming'), default='solve')
    parser.add_argument('--points', type=int, default=33)
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--num-t', type=int, default=257)
    parser.add_argument('--n-steps', type=int, default=2000)
    parser.add_argument('--dx', type=float, default=.025)
    parser.add_argument('--padding', type=float, default=8)
    parser.add_argument('--tail-eps', type=float, default=1e-8)
    parser.add_argument('--T', type=float)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--interp', choices=('cubic', 'linear'), default='cubic')
    parser.add_argument('--boundary', choices=('taper', 'zero'), default='taper')
    parser.add_argument('--t-grading', choices=('uniform', 'quadratic'), default='quadratic')
    args = parser.parse_args()
    if args.points < 2:
        parser.error('--points must be >= 2')
    equation = load_all_methods_equation(args.equation_id,
        args.input_dir / 'all_methods_equations_v1.csv',
        args.input_dir / 'all_methods_equations_v1_metadata.jsonl')
    config = ChernovConfig(n=args.n, num_t=args.num_t, n_steps=args.n_steps,
        dx=args.dx, padding=args.padding, tail_eps=args.tail_eps, T=args.T,
        n_jobs=args.n_jobs, interp=args.interp, boundary=args.boundary,
        t_grading=args.t_grading)
    try:
        solver = ChernovSolver(equation.remizov_problem, config)
    except ValueError as exc:
        parser.error(str(exc))
    p = equation.remizov_problem
    x = np.linspace(p.x_target_min, p.x_target_max, args.points)
    result = solver.solve(x) if args.scheme == 'solve' else solver.solve_streaming(x)
    record = asdict(result)
    record['config'] = asdict(config)
    record['max_abs_error'] = float(np.max(np.abs(result.f-equation.y_exact(x)))) if result.success else None
    def clean(value):
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    print(json.dumps(clean(record), indent=2, allow_nan=False))
    return 0 if result.success else 1


if __name__ == '__main__':
    raise SystemExit(main())
