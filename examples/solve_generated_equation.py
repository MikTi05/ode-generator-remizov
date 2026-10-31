"""Load one generated equation and report a finite-n Remizov error."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from all_methods_dataset_loader import load_all_methods_equation
from remizov_solver import RemizovConfig, solve_remizov


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('data/generated'))
    parser.add_argument('--equation-id', default='AMV1_F0000_S00')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--points', type=int, default=33)
    parser.add_argument('--verification-budget', action='store_true')
    args = parser.parse_args()
    if args.points < 2:
        parser.error('--points must be at least 2')
    equation = load_all_methods_equation(
        args.equation_id,
        csv_path=args.input_dir / 'all_methods_equations_v1.csv',
        metadata_path=args.input_dir / 'all_methods_equations_v1_metadata.jsonl',
    )
    if args.verification_budget:
        from verify_remizov_solver import make_verification_config
        config = make_verification_config(args.n)
    else:
        config = RemizovConfig(n=args.n)
    problem = equation.remizov_problem
    points = np.linspace(problem.x_target_min, problem.x_target_max, args.points)
    result = solve_remizov(problem, points, config)
    error = (float(np.max(np.abs(result.f - equation.y_exact(result.x))))
             if result.success else None)
    summary = {
        'equation_id': equation.equation_id,
        'success': result.success,
        'failure_reason': result.failure_reason,
        'max_abs_error': error,
        'target_points': len(result.x),
        'solution_points': int(result.f.size),
        'config': asdict(config),
        'T': result.T if np.isfinite(result.T) else None,
        'final_dx': result.final_dx if np.isfinite(result.final_dx) else None,
        'final_nt': result.final_nt,
        'quadrature_error_estimate': (result.quadrature_error_estimate
            if np.isfinite(result.quadrature_error_estimate) else None),
        'spatial_error_estimate': (result.spatial_error_estimate
            if np.isfinite(result.spatial_error_estimate) else None),
        'runtime_seconds': result.runtime_seconds,
        'total_error_certified': False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.success else 1


if __name__ == '__main__':
    raise SystemExit(main())
