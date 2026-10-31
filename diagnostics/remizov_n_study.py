"""Six fixed-budget calls to the existing solver, with an external watchdog.

Run with the project's .venv/bin/python -B. No retries or solver modifications.
All pre-existing source/data files are hashed before and after the study.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
IDS = ('AMV1_F0000_S00', 'AMV1_F0005_S00')
NS = (1, 2, 4)
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
for key in THREAD_VARS:
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
WALL_LIMIT = 45.0
RSS_LIMIT_MIB = 512


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True,
                               allow_nan=False) + '\n', encoding='utf-8')


def snapshot(root):
    result = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in
                         {'.venv', '.git', '.pytest_cache', '__pycache__', 'results'})
        for name in sorted(files):
            if name == '.DS_Store' or name.endswith(('.pyc', '.pyo')):
                continue
            p = Path(base) / name
            result[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def load_equations():
    from all_methods_dataset_loader import load_all_methods_equations
    equations = load_all_methods_equations(
        ROOT / 'data/generated/all_methods_equations_v1.csv',
        ROOT / 'data/generated/all_methods_equations_v1_metadata.jsonl')
    return {e.equation_id: e for e in equations if e.equation_id in IDS}


def worker(equation_id, n, output):
    import numpy as np
    from remizov_solver import solve_remizov
    from verify_remizov_solver import make_verification_config
    e = load_equations()[equation_id]
    x = np.linspace(-1.0, 1.0, 33)
    config = make_verification_config(n)
    result = solve_remizov(e.remizov_problem, x, config)
    np.testing.assert_array_equal(result.x, x)
    exact = e.y_exact(x)
    record = {
        'equation_id': equation_id, 'n': n,
        'status': 'success' if result.success else result.failure_reason,
        'failure_reason': result.failure_reason,
        'max_abs_error': (float(np.max(np.abs(result.f - exact)))
                          if result.success else None),
        'runtime_seconds': result.runtime_seconds,
        'config': asdict(config), 'solver_result': asdict(result),
        'y_exact': exact, 'pointwise_error': result.f - exact if result.success else None,
        'max_error_x': float(x[np.argmax(np.abs(result.f - exact))]) if result.success else None,
        'total_error_certified': False,
    }
    write_json(output, record)


def input_checks(equations):
    """Independent SymPy derivatives, not a second numerical solver."""
    import numpy as np
    import sympy as sp
    from src.all_methods_expressions import deserialize_expression
    from remizov_solver import compute_truncation_T
    spec = importlib.util.spec_from_file_location(
        'existing_expression_oracle', ROOT / 'tests/test_all_methods_expressions.py')
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    symbol = sp.Symbol('x', real=True)
    checks, costs = {}, []
    for equation_id in IDS:
        e = equations[equation_id]
        p, m = e.remizov_problem, e.metadata
        construction = m['serialized_ast']['construction']
        mu, radius = construction['bump_mu'], construction['bump_radius']
        z = (symbol - sp.Float(mu)) / sp.Float(radius)
        shape = oracle._sympy_expression(
            deserialize_expression(m['serialized_ast']['y_shape']), symbol)
        interior_y = sp.Float(construction['solution_scale']) * sp.exp(1-1/(1-z*z)) * shape
        # Stay away from support boundaries for the independent derivative comparison.
        points = np.linspace(mu - .95*radius, mu + .95*radius, 513)
        values = [np.broadcast_to(np.asarray(sp.lambdify(symbol,
                    sp.diff(interior_y, symbol, order), 'numpy')(points), dtype=float),
                    points.shape) for order in range(3)]
        loaders = [e.y_exact(points), e.dy_exact(points), e.d2y_exact(points)]
        g_r_independent = -(p.a(points)*values[2] + p.b(points)*values[1]
                             + (p.c_remizov(points)-p.lambda_value)*values[0])
        x = np.linspace(-1, 1, 257)
        T = compute_truncation_T(p, 1e-8)
        checks[equation_id] = {
            'regime': m['regime'], 'complexity': m['complexity_level'],
            'construction': construction, 'support': [m['support_left'], m['support_right']],
            'lambda_value': p.lambda_value, 'c_abs_upper_bound': p.c_abs_upper_bound,
            'lambda_c_gap': p.lambda_value-p.c_abs_upper_bound, 'T': T,
            'actual_constant_coefficients': {key: float(function(0)) for key, function in
                [('a', p.a), ('b', p.b), ('c_R', p.c_remizov)]},
            'sympy_jet_max_abs_differences': [float(np.max(np.abs(a-b)))
                                              for a, b in zip(values, loaders)],
            'independent_g_R_max_abs_difference': float(np.max(np.abs(
                g_r_independent-p.g_remizov(points)))),
            'ordinary_vs_remizov_g_sign_max_difference': float(np.max(np.abs(
                np.array([e.cauchy_problem.g(float(t)) for t in x])+p.g_remizov(x)))),
            'ordinary_vs_remizov_c_shift_max_difference': float(np.max(np.abs(
                np.array([e.cauchy_problem.c(float(t)) for t in x])-
                (p.c_remizov(x)-p.lambda_value)))),
            'y0_difference': float(e.y_exact(-1)-m['y0']),
            'v0_difference': float(e.dy_exact(-1)-m['v0']),
            'sampled_target_amplitude': float(np.max(np.abs(e.y_exact(x)))),
            'metadata_target_amplitude': m['target_amplitude'],
            'sympy_points': points.tolist(), 'sympy_points_are_global_certificate': False,
        }
        for n in NS:
            d = max(2*math.sqrt(p.a_upper_bound*T/n), 2*p.b_abs_upper_bound*T/n)
            width = p.x_target_max-p.x_target_min+2*n*d
            costs.append({'equation_id': equation_id, 'n': n, 'work_width_bound': width,
                'initial_grid_points_bound': math.ceil(width/.05)+1,
                'finest_grid_points_bound': math.ceil(width/(.05/32))+1,
                'quadrature_nodes_all_levels_per_dx': 4088,
                'max_quadrature_nodes_all_dx': 6*4088,
                'max_operator_applications': n*6*4088})
    return checks, costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--worker-id', choices=IDS)
    parser.add_argument('--worker-n', type=int, choices=NS)
    parser.add_argument('--worker-output', type=Path)
    args = parser.parse_args()
    if args.worker_id:
        worker(args.worker_id, args.worker_n, args.worker_output)
        return
    if args.output_dir is None:
        parser.error('--output-dir is required')
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / 'results'):
        parser.error('output must be a new directory under project/results')
    out.mkdir(parents=True, exist_ok=False)
    before = snapshot(ROOT)
    original = ROOT.parent / 'generator_DE'
    original_before = snapshot(original)
    import numpy as np
    from verify_remizov_solver import make_verification_config
    checks, costs = input_checks(load_equations())
    summary = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'environment': {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'machine': platform.machine(),
            'cpu_count': os.cpu_count(), 'versions': {name: importlib.metadata.version(name)
             for name in ('numpy', 'scipy', 'sympy', 'pytest')},
            'thread_environment': {key: os.environ[key] for key in THREAD_VARS}},
        'common_config_n1': asdict(make_verification_config(1)),
        'x': np.linspace(-1, 1, 33).tolist(),
        'resource_policy': {'serial': True, 'per_worker_wall_seconds': WALL_LIMIT,
            'per_worker_rss_mib': RSS_LIMIT_MIB, 'poll_seconds': .25,
            'stop_series_on_external_failure': True, 'automatic_retries': 0,
            'limits_are_wrapper_not_native_solver': True},
        'input_checks': checks, 'cost_preflight': costs, 'runs': [],
    }
    write_json(out/'summary.json', summary)
    stop = False
    for n in NS:
        for equation_id in IDS:
            print(f'START {equation_id} n={n}', flush=True)
            target = out/f'{equation_id}_n{n}.json'
            started = time.perf_counter()
            peak_rss = 0
            reason = ''
            with (out/f'{equation_id}_n{n}.log').open('w') as log:
                child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                    '--worker-id', equation_id, '--worker-n', str(n),
                    '--worker-output', str(target)], cwd=ROOT, env=os.environ.copy(),
                    stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    elapsed = time.perf_counter()-started
                    rss = subprocess.run(['ps', '-o', 'rss=', '-p', str(child.pid)],
                                          capture_output=True, text=True).stdout.strip()
                    if rss:
                        peak_rss = max(peak_rss, int(rss))
                    if elapsed > WALL_LIMIT:
                        reason = 'external_time_limit'
                    elif peak_rss > RSS_LIMIT_MIB*1024:
                        reason = 'external_memory_limit'
                    if reason:
                        child.terminate()
                        try:
                            child.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        break
                    time.sleep(.25)
            elapsed = time.perf_counter()-started
            if reason or child.returncode != 0:
                reason = reason or 'worker_exception'
                record = {'equation_id': equation_id, 'n': n, 'status': reason,
                    'failure_reason': reason, 'max_abs_error': None,
                    'runtime_seconds': None, 'solver_result': None,
                    'config': asdict(make_verification_config(n))}
                stop = True
            else:
                record = json.loads(target.read_text())
            record.update(worker_wall_seconds=elapsed, worker_peak_rss_mib=peak_rss/1024)
            write_json(target, record)
            summary['runs'].append(record)
            write_json(out/'summary.json', summary)
            print(json.dumps({key: record[key] for key in ('equation_id', 'n', 'status',
                  'max_abs_error', 'runtime_seconds', 'worker_wall_seconds',
                  'worker_peak_rss_mib')}), flush=True)
            if stop:
                break
        if stop:
            break
    summary['not_executed'] = [{'equation_id': eid, 'n': n} for n in NS for eid in IDS
                              if not any(r['equation_id']==eid and r['n']==n
                                         for r in summary['runs'])]
    after, original_after = snapshot(ROOT), snapshot(original)
    summary['integrity'] = {
        'project_before_sha256': before,
        'preexisting_project_changed_or_removed': [p for p,h in before.items() if after.get(p)!=h],
        'original_before_sha256': original_before,
        'original_changed_or_removed': [p for p,h in original_before.items()
                                        if original_after.get(p)!=h],
        'original_added': sorted(set(original_after)-set(original_before)),
        'excluded': ['.venv', '.git', '.pytest_cache', '__pycache__', 'results', '.DS_Store'],
    }
    summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(out/'summary.json', summary)
    fields = ['equation_id', 'n', 'status', 'max_abs_error', 'runtime_seconds',
              'failure_reason', 'worker_wall_seconds', 'worker_peak_rss_mib']
    with (out/'runs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary['runs'])
    print(f'FINISHED {out}', flush=True)


if __name__ == '__main__':
    main()
