"""Check every public-loader equation against production NPZ; never solve ODEs.

Run the existing validate_all_methods_dataset_v1.py --expect-production first:
it authenticates all four artifacts, analytic constructions and seed/config.
This supplementary check exercises vector callables for every loaded equation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from all_methods_dataset_loader import load_all_methods_equations
from chernov_solver import ChernovConfig, ChernovSolver
from remizov_solver import _problem_conditions_valid
from src.all_methods_dataset_v1 import AllMethodsConfig, OUTPUT_FILENAMES


def check_dataset(directory: Path) -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    report = json.loads((directory / OUTPUT_FILENAMES['report']).read_text())
    if AllMethodsConfig(**report['configuration']) != AllMethodsConfig():
        raise ValueError('expected unchanged production configuration')
    equations = load_all_methods_equations(
        directory / OUTPUT_FILENAMES['csv'], directory / OUTPUT_FILENAMES['metadata'])
    if len(equations) != 1000:
        raise ValueError('expected all 1000 equations')
    families = defaultdict(list)
    rows = []
    with np.load(directory / OUTPUT_FILENAMES['ml'], allow_pickle=False) as data:
        np.testing.assert_array_equal(data['channel_order'], ['a', 'b', 'c', 'g'])
        np.testing.assert_array_equal(data['x_grid'], np.linspace(-1., 1., 256))
        x = data['x_grid']
        for index, equation in enumerate(equations):
            p = equation.remizov_problem
            ivp = equation.cauchy_problem
            m = equation.metadata
            if not _problem_conditions_valid(p):
                raise ValueError(f'{equation.equation_id}: invalid Remizov conditions')
            # Constructor only: neither solve nor solve_streaming is called.
            ChernovSolver(p, ChernovConfig())
            for key in ('equation_id', 'family_id', 'split', 'complexity_level', 'regime'):
                if str(data[key][index]) != str(m[key]):
                    raise ValueError(f'{equation.equation_id}: NPZ {key} mismatch')
            a, b = p.a(x), p.b(x)
            c_r, g_r = p.c_remizov(x), p.g_remizov(x)
            channels = np.stack((a, b, c_r - p.lambda_value, -g_r))
            y, dy, d2y = equation.y_exact(x), equation.dy_exact(x), equation.d2y_exact(x)
            for actual, expected in ((channels, data['X'][index]),
                                     (y, data['y_exact'][index]),
                                     (dy, data['dy_exact'][index])):
                if not np.all(np.isfinite(actual)):
                    raise ValueError(f'{equation.equation_id}: nonfinite loaded values')
                np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
            np.testing.assert_allclose([ivp.y0, ivp.v0],
                [equation.y_exact(ivp.x0), equation.dy_exact(ivp.x0)],
                rtol=2e-13, atol=2e-13)
            np.testing.assert_array_equal([ivp.y0, ivp.v0],
                                          [data['y0'][index], data['v0'][index]])
            residual = a * d2y + b * dy + channels[2] * y + g_r
            scale = max(1., float(np.max(np.abs(a*d2y))),
                float(np.max(np.abs(b*dy))), float(np.max(np.abs(channels[2]*y))),
                float(np.max(np.abs(g_r))))
            max_residual = float(np.max(np.abs(residual)))
            if not np.isfinite(max_residual) or max_residual > 1e-10 * scale:
                raise ValueError(f'{equation.equation_id}: loaded residual mismatch')
            families[m['family_id']].append(m)
            rows.append(dict(equation_id=equation.equation_id,
                family_id=m['family_id'], split=m['split'],
                complexity_level=m['complexity_level'], regime=m['regime'],
                a_lower=p.a_lower_bound, a_upper=p.a_upper_bound,
                b_abs=p.b_abs_upper_bound, c_R_abs=p.c_abs_upper_bound,
                lambda_value=p.lambda_value, gap=p.lambda_value-p.c_abs_upper_bound,
                y0=ivp.y0, v0=ivp.v0, max_loaded_residual=max_residual,
                remizov_conditions_valid=True, npz_functions_match=True))
    if len(families) != 200 or any(len(group) != 5 for group in families.values()):
        raise ValueError('expected 200 families with 5 samples each')
    if any(len({m['split'] for m in group}) != 1 for group in families.values()):
        raise ValueError('family split leakage')
    signatures = {group[0]['structure_signature'] for group in families.values()}
    if len(signatures) != 200:
        raise ValueError('duplicate family structure')
    split_counts = Counter(r['split'] for r in rows)
    if split_counts != {'train': 700, 'validation': 150, 'test': 150}:
        raise ValueError('incorrect production split')
    summary = dict(valid=True, total_equations=len(rows), total_families=len(families),
        samples_per_family=5, structure_signature_count=len(signatures),
        counts_by_split=dict(split_counts),
        family_counts_by_split=dict(Counter(group[0]['split'] for group in families.values())),
        counts_by_complexity=dict(Counter(r['complexity_level'] for r in rows)),
        counts_by_regime=dict(Counter(r['regime'] for r in rows)),
        family_split_leakage=False, all_npz_functions_match=True,
        all_initial_conditions_match=True, all_remizov_conditions_valid=True,
        chernov_constructors_checked=len(rows), numerical_solver_calls=0,
        max_loaded_residual=max(r['max_loaded_residual'] for r in rows),
        runtime_seconds=time.perf_counter()-started)
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if not args.output_dir.is_dir():
        parser.error('--output-dir must be an existing directory for new check files')
    outputs = [args.output_dir/name for name in ('loader_check.json', 'loaded_equations.csv')]
    if any(p.exists() for p in outputs):
        parser.error('refusing to overwrite previous check files')
    summary, rows = check_dataset(args.input_dir)
    with outputs[1].open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs[0].write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
