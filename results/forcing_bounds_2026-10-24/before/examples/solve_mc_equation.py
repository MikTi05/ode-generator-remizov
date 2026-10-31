"""Compute one existing all_methods_v1 equation with Feynman--Kac MC."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from all_methods_dataset_loader import load_all_methods_equation
from mc_solver import MCConfig, MCForcingBound, solve_monte_carlo


def json_ready(value):
    """Serialize diagnostics, preserving unknown/not-representable values as null."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'data/generated')
    parser.add_argument('--equation-id', default='AMV1_F0000_S00')
    parser.add_argument('--points', type=int, default=33)
    parser.add_argument('--M', type=int, default=20000)
    parser.add_argument('--dt', type=float, default=.005)
    parser.add_argument('--T', type=float)
    parser.add_argument('--tail-eps', type=float, default=1e-4)
    parser.add_argument('--g-bound', type=float)
    parser.add_argument('--g-bound-source', help='analytical derivation of a global |g_R| bound')
    parser.add_argument('--seed', type=int, default=2488)
    parser.add_argument('--stream-id', type=int, default=0)
    parser.add_argument('--chunk', type=int, default=20000)
    parser.add_argument('--max-steps', type=int, default=100000)
    parser.add_argument('--max-path-steps', type=int, default=100000000)
    parser.add_argument('--max-runtime-seconds', type=float)
    args = parser.parse_args()
    if args.points < 1:
        parser.error('--points must be positive')
    if args.g_bound is None and args.T is None:
        parser.error('pass explicit --T or an analytical --g-bound with --g-bound-source')
    if (args.g_bound is None) != (args.g_bound_source is None):
        parser.error('--g-bound and --g-bound-source must be supplied together')
    cfg = MCConfig(M=args.M, dt=args.dt, T=args.T, tail_eps=args.tail_eps,
        seed=args.seed, stream_id=args.stream_id, chunk=args.chunk,
        max_steps=args.max_steps, max_path_steps=args.max_path_steps,
        max_runtime_seconds=args.max_runtime_seconds)
    cert = None if args.g_bound is None else MCForcingBound(args.g_bound, args.g_bound_source)
    try:
        cfg.validate()
        if cert is not None:
            cert.validate()
    except ValueError as exc:
        parser.error(str(exc))
    equation = load_all_methods_equation(args.equation_id,
        args.input_dir/'all_methods_equations_v1.csv',
        args.input_dir/'all_methods_equations_v1_metadata.jsonl')
    p = equation.remizov_problem
    x = (np.array([.5*(p.x_target_min+p.x_target_max)]) if args.points == 1
         else np.linspace(p.x_target_min, p.x_target_max, args.points))
    result = solve_monte_carlo(p, x, cfg, forcing_bound=cert)
    exact = equation.y_exact(x)       # only after the numerical solve
    error = float(np.max(abs(result.f-exact))) if result.success else None
    record = dict(result=asdict(result), config=asdict(cfg), y_exact=exact,
        max_abs_error=error, forcing_bound=None if cert is None else asdict(cert),
        accuracy_reached=None)        # no requested total tolerance in this CLI
    print(json.dumps(json_ready(record), indent=2, allow_nan=False))
    return 0 if result.success else 1


if __name__ == '__main__':
    raise SystemExit(main())
