"""Controlled boundary diagnosis on four production tasks; never modify the solver.

Core phase reproduces the four baselines, traces actual production steps at
four real Simpson nodes, and runs cubic/zero. A separately requested linear
phase keeps the same grids, T, coefficients and quadrature. New files only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
for key in THREAD_VARS:
    os.environ[key] = '1'

import numpy as np
from scipy.interpolate import CubicSpline

from all_methods_dataset_loader import load_all_methods_equations
from chernov_solver import (ChernovConfig, ChernovSolver, _Counts, _State,
                            _apply, _geometry, _simpson_weights)
from remizov_solver import _as_values, _evaluate_coefficients, compute_truncation_T

IDS = ('AMV1_F0141_S00', 'AMV1_F0180_S04', 'AMV1_F0184_S01', 'AMV1_F0180_S03')
BATCH = ROOT / 'results/chernov_solve1000_2026-10-18'
DATA = ROOT / 'data/all_methods_v1_1000_2026-10-18'
CORE = ('src/all_methods_dataset_v1.py', 'src/all_methods_expressions.py',
        'all_methods_dataset_loader.py', 'remizov_solver.py', 'chernov_solver.py')


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(clean(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def max_abs(a):
    return float(np.max(np.abs(a))) if np.size(a) else 0.0


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**2 if sys.platform == 'darwin' else 1024)


class ResourceGuard:
    """Experiment-specific guard: six times historical wall cost and 768 MiB.

    SIGINT leaves the current incomplete attempt marked as stopped. Completed
    attempt files are already durable. No child compute processes are started.
    """
    def __init__(self, output, seconds):
        self.output, self.seconds = Path(output), seconds
        self.stop = threading.Event()
        self.began = time.perf_counter()

    def __enter__(self):
        def monitor():
            with self.output.open('x') as handle:
                while not self.stop.is_set():
                    elapsed, rss = time.perf_counter() - self.began, peak_mib()
                    handle.write(json.dumps(dict(wall_seconds=elapsed, peak_rss_mib=rss))+'\n')
                    handle.flush()
                    if elapsed > self.seconds or rss > 768:
                        os.kill(os.getpid(), signal.SIGINT)
                        return
                    self.stop.wait(1.)
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def protected_snapshot(output):
    paths = {ROOT / name for name in CORE}
    paths.update(ROOT / name for name in ('PROJECT_STATUS.md', 'MIGRATION_MANIFEST.json'))
    for directory in ('data', 'results', 'comparison_inputs'):
        paths.update(p for p in (ROOT / directory).rglob('*')
                     if p.is_file() and output not in p.parents and p.name != '.DS_Store')
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def prepare(problem, config, expected_T):
    """Mirror production grid preparation exactly; assert effective T identity."""
    T = compute_truncation_T(problem, config.tail_eps) if config.T is None else config.T
    if T != expected_T:
        raise ValueError('effective T differs from saved production result')
    domain = (problem.x_target_min - config.padding, problem.x_target_max + config.padding)
    intervals = max(3, math.ceil((domain[1] - domain[0]) / config.dx))
    x = np.linspace(*domain, intervals + 1)
    a, b, c = _evaluate_coefficients(problem, x)
    g = _as_values(problem.g_remizov, x, 'nonfinite_state')
    state = _State(x, a, b, c, g, problem.lambda_value, config.n,
                   config.interp, config.boundary,
                   config.taper_width or .1 * (domain[1] - domain[0]))
    s = np.linspace(0, 1, config.num_t + (config.num_t % 2 == 0))
    t = T * (s**2 if config.t_grading == 'quadratic' else s)
    return state, t, _simpson_weights(t)


def fingerprints(state, t, weights, target):
    return {name: array_digest(a) for name, a in dict(
        grid=state.x, a=state.a, b=state.b, c_R=state.c, g_R=state.g,
        t=t, weights=weights, target_x=target).items()}


def preflight(output):
    plan = read_json(BATCH / 'plan.json')
    identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    protected = {name: h for name, h in plan['protected_sha256'].items()
                 if name in CORE or name.startswith('data/')}
    checks = {name: digest(ROOT / name) == h for name, h in protected.items()}
    if not all(checks.values()):
        raise ValueError(f'production source/data hash mismatch: {checks}')
    versions = {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'sympy')}
    expected = plan['environment']['versions']
    if any(versions[k] != expected[k] for k in versions):
        raise ValueError('numerical package version mismatch')
    if platform.python_version() != plan['environment']['python'].split()[0]:
        raise ValueError('Python version mismatch')
    config = ChernovConfig(**plan['config'])
    required = dict(n=128, num_t=513, t_grading='quadratic', dx=.0125,
                    padding=12., interp='cubic', boundary='taper', tail_eps=1e-8,
                    n_jobs=1, T=None, taper_width=None)
    if any(getattr(config, k) != v for k, v in required.items()):
        raise ValueError('unexpected production configuration')
    equations = {e.equation_id: e for e in load_all_methods_equations(
        DATA / 'all_methods_equations_v1.csv', DATA / 'all_methods_equations_v1_metadata.jsonl')}
    tasks = []
    for pid in IDS:
        path = BATCH / 'solutions' / f'{pid}.json'
        saved = read_json(path)
        if (saved['plan_sha256'] != identity or saved['config'] != plan['config']
                or saved['metadata'] != equations[pid].metadata
                or saved['implementation'] != 'chernov_solver.ChernovSolver.solve'):
            raise ValueError(f'{pid}: identity mismatch')
        raw = BATCH / 'attempts' / saved['attempt_id'] / 'result.json'
        if path.read_bytes() != raw.read_bytes():
            raise ValueError(f'{pid}: primary attempt differs')
        target, f, exact = map(np.asarray, (saved['target_x'], saved['result']['f'], saved['y_exact']))
        np.testing.assert_array_equal(target, np.linspace(-1, 1, 33))
        np.testing.assert_array_equal(exact, equations[pid].y_exact(target))
        if max_abs(f - exact) != saved['max_abs_error']:
            raise ValueError(f'{pid}: stored metric differs')
        state, t, weights = prepare(equations[pid].remizov_problem, config, saved['result']['T'])
        assert (state.x[0], state.x[-1], state.x.size, state.taper_width) == (-13., 13., 2081, 2.6)
        tasks.append(dict(problem_id=pid, baseline_error=saved['max_abs_error'],
                          baseline_boundary_max=saved['result']['boundary_state_max_abs'],
                          T=saved['result']['T'], source_result_sha256=digest(path),
                          lambda_value=state.lam, effective_taper_width=state.taper_width,
                          fingerprints=fingerprints(state, t, weights, target)))
    return equations, dict(schema='chernov_boundary_diagnosis_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), production_plan_sha256=identity,
        source_data_hash_checks=checks, source_data_sha256=protected,
        environment=dict(python=sys.version, platform=platform.platform(), versions=versions,
                         thread_environment={k: os.environ[k] for k in THREAD_VARS}),
        baseline_config=asdict(config), tasks=tasks,
        changes=dict(cubic_zero=['boundary'], linear_zero=['interp', 'boundary']),
        trace_fractions=[.25, .5, .75, 1.],
        trace_onset_definitions=dict(significant_global_growth='first ||u_k|| > 2||g_grid||',
            step_bound_violation='first ||u_(k+1)|| > (1+tau*C)||u_k||*(1+1e-10)',
            target_boundary_effect='first ||u_taper-u_zero|| on [-1,1] > 0.01',
            weighted_target_boundary_effect='same difference times exp(-lambda*t*k/n) > 0.01'),
        resource_policy=dict(compute_processes=1, blas_threads=1, memory_mib=768,
            stage_wall_limit='six times max saved task wall, at least 60 s',
            expected_core_full_solves=8, expected_core_solver_seconds=128,
            linear_phase_conditional=True), numerical_refinement=False)


def inspect_step(state, u, tau, queries, k, counts):
    """Observe production cubic step; independently reconstruct and compare bits.

    Actual chain uses _apply from unmodified chernov_solver.py. Values outside
    are also separated by their destination, so a large extension is not
    mistaken for an equally large target error. No dense matrix is formed.
    """
    spline = CubicSpline(state.x, u, extrapolate=False)
    values, outside_terms = [], []
    inside_max = outside_max = outside_left = outside_right = 0.
    for weight, q in zip((.25, .25, .5), queries):
        left, right = q < state.x[0], q > state.x[-1]
        outside = left | right
        v = spline(q)
        inside_max = max(inside_max, max_abs(v[~outside]))
        v[outside] = 0
        if state.boundary == 'taper':
            for mask, edge, sign in ((left, 0, -1), (right, -1, 1)):
                if np.any(mask):
                    d = sign * (q[mask] - state.x[edge])
                    s = np.clip(1 - d / state.taper_width, 0, 1)
                    v[mask] = (u[edge] + sign * float(spline(state.x[edge], 1))*d) * s*s*(3-2*s)
        outside_max = max(outside_max, max_abs(v[outside]))
        outside_left = max(outside_left, max_abs(v[left]))
        outside_right = max(outside_right, max_abs(v[right]))
        outside_terms.append(weight * np.where(outside, v, 0.))
        values.append(v)
    direct = .25*values[0]+.25*values[1]+.5*values[2]+tau*state.c*u
    actual = _apply(state, u, tau, queries, counts)
    np.testing.assert_array_equal(actual, direct)
    outside_contribution = sum(outside_terms)
    target = np.abs(state.x) <= 1
    distributed_weight = math.exp(-state.lam*tau*k)
    after_weight = math.exp(-state.lam*tau*(k+1))
    before_norm, after_norm = max_abs(u), max_abs(actual)
    row = dict(k_before=k, k_after=k+1, global_before=before_norm,
        target_before=max_abs(u[target]), edge_left=float(u[0]), edge_right=float(u[-1]),
        slope_left=float(spline(state.x[0], 1)), slope_right=float(spline(state.x[-1], 1)),
        inside_query_max=inside_max, outside_extension_max=outside_max,
        outside_left_max=outside_left, outside_right_max=outside_right,
        outside_contribution_max=max_abs(outside_contribution),
        outside_contribution_target=max_abs(outside_contribution[target]),
        zero_extension_same_input_after_max=max_abs(actual-outside_contribution),
        global_after=after_norm, target_after=max_abs(actual[target]),
        edge_left_after=float(actual[0]), edge_right_after=float(actual[-1]),
        max_after_x=float(state.x[np.argmax(np.abs(actual))]),
        weighted_global_before=distributed_weight*before_norm,
        weighted_target_before=distributed_weight*max_abs(u[target]),
        weighted_global_after=after_weight*after_norm,
        weighted_target_after=after_weight*max_abs(actual[target]),
        one_step_gain=after_norm/before_norm if before_norm else 0.)
    return actual, row


def summarize_trace(rows, initial_norm, C, tau):
    def first(predicate):
        return next((r['k_after'] for r in rows if predicate(r)), None)
    return dict(first_global_gt_twice_initial=first(lambda r: r['global_after'] > 2*initial_norm),
        first_step_bound_violation=first(lambda r: r['global_after'] >
            (1+tau*C)*r['global_before']*(1+1e-10)),
        first_target_difference_gt_001=first(lambda r: r['target_pair_difference'] > .01),
        first_weighted_target_difference_gt_001=first(
            lambda r: r['weighted_target_pair_difference'] > .01),
        max_global=max(r['global_after'] for r in rows),
        max_target=max(r['target_after'] for r in rows),
        max_extension=max(r['outside_extension_max'] for r in rows),
        max_weighted_global=max(r['weighted_global_after'] for r in rows),
        max_weighted_target=max(r['weighted_target_after'] for r in rows),
        max_target_pair_difference=max(r['target_pair_difference'] for r in rows),
        max_weighted_target_pair_difference=max(r['weighted_target_pair_difference'] for r in rows),
        final=rows[-1])


def trace_task(equation, saved, config, output):
    state, times, weights = prepare(equation.remizov_problem, config, saved['result']['T'])
    zero = replace(state, boundary='zero')
    target = np.abs(state.x) <= 1
    summaries = []
    for fraction in (.25, .5, .75, 1.):
        j = int(np.argmin(np.abs(times/times[-1]-fraction)))
        t, tau = float(times[j]), float(times[j]/state.n)
        queries = _geometry(state, tau)
        u, v = state.g.copy(), state.g.copy()
        rows = {'taper': [], 'zero': []}
        counts = {'taper': _Counts(), 'zero': _Counts()}
        path = output / 'traces' / f'{equation.equation_id}_j{j:03d}.jsonl'
        with path.open('x') as handle:
            for k in range(state.n):
                u, a = inspect_step(state, u, tau, queries, k, counts['taper'])
                v, b = inspect_step(zero, v, tau, queries, k, counts['zero'])
                difference, target_difference = max_abs(u-v), max_abs((u-v)[target])
                weight = math.exp(-state.lam*tau*(k+1))
                for mode, row in (('taper', a), ('zero', b)):
                    row.update(mode=mode, problem_id=equation.equation_id, time_node_index=j,
                        t=t, tau=tau, t_over_T=t/times[-1], quadrature_weight=float(weights[j]),
                        global_pair_difference=difference, target_pair_difference=target_difference,
                        weighted_global_pair_difference=weight*difference,
                        weighted_target_pair_difference=weight*target_difference)
                    rows[mode].append(row)
                    handle.write(json.dumps(clean(row), sort_keys=True, allow_nan=False)+'\n')
        summary = dict(problem_id=equation.equation_id, time_node_index=j, t=t,
            t_over_T=t/times[-1], tau=tau, quadrature_weight=float(weights[j]),
            initial_global=max_abs(state.g), coefficient_C=equation.remizov_problem.c_abs_upper_bound,
            initial_edge_values=[float(state.g[0]), float(state.g[-1])],
            reconstruction_bitwise_equal=True,
            modes={mode: summarize_trace(rr, max_abs(state.g),
                equation.remizov_problem.c_abs_upper_bound, tau) for mode, rr in rows.items()})
        summaries.append(summary)
    write_json(output / 'traces' / f'{equation.equation_id}_summary.json', summaries)
    return summaries


def solve_task(equation, saved, config, mode, output, task_plan):
    state, t, weights = prepare(equation.remizov_problem, config, saved['result']['T'])
    target = np.asarray(saved['target_x'])
    assert fingerprints(state, t, weights, target) == task_plan['fingerprints']
    limits = max(60., 6*saved['task_wall_seconds'])
    telemetry = output / 'runs' / f'{equation.equation_id}_{mode}.resources.jsonl'
    with ResourceGuard(telemetry, limits):
        result = ChernovSolver(equation.remizov_problem, config).solve(target)
    if not result.success:
        raise RuntimeError(f'{equation.equation_id}/{mode}: {result.failure_reason}')
    assert result.T == saved['result']['T']
    assert result.work_domain == (-13., 13.) and result.work_grid_points == 2081
    assert result.n_operator_applications == 65536
    exact = equation.y_exact(target)
    np.testing.assert_array_equal(exact, saved['y_exact'])
    error = max_abs(result.f-exact)
    baseline_difference = max_abs(result.f-np.asarray(saved['result']['f']))
    record = dict(problem_id=equation.equation_id, mode=mode, config=asdict(config),
        effective_taper_width=state.taper_width, fingerprints=task_plan['fingerprints'],
        result=asdict(result), y_exact=exact, max_abs_error=error,
        baseline_max_abs_difference=baseline_difference,
        baseline_bitwise_equal=np.array_equal(result.f, saved['result']['f']),
        accuracy_reached=error <= .01, peak_rss_mib=peak_mib(),
        production_result_sha256=task_plan['source_result_sha256'])
    write_json(output / 'runs' / f'{equation.equation_id}_{mode}.json', record)
    if mode == 'cubic_taper' and baseline_difference > 1e-10*max(1., max_abs(saved['result']['f'])):
        raise ValueError('baseline not reproduced: stopping before intervention')
    print(json.dumps(dict(problem_id=equation.equation_id, mode=mode, error=error,
        seconds=result.runtime_seconds, boundary_max=result.boundary_state_max_abs,
        baseline_bitwise_equal=record['baseline_bitwise_equal'])), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--phase', choices=('core', 'linear'), default='core')
    parser.add_argument('--reason', help='required explanation before conditional linear phase')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.phase == 'core':
        output.mkdir(exist_ok=False)
        (output/'runs').mkdir()
        (output/'traces').mkdir()
        write_json(output/'before_sha256.json', protected_snapshot(output))
        (output/'PROJECT_STATUS.before.md').write_bytes((ROOT/'PROJECT_STATUS.md').read_bytes())
    elif not args.reason:
        parser.error('--reason is required for the conditional linear phase')
    try:
        equations, plan = preflight(output)
        if args.phase == 'core':
            write_json(output/'plan.json', plan)
        else:
            original = read_json(output/'plan.json')
            if plan['tasks'] != original['tasks'] or plan['source_data_sha256'] != original['source_data_sha256']:
                raise ValueError('inputs changed between phases')
            write_json(output/'linear_decision.json', dict(reason=args.reason,
                created_utc=datetime.now(timezone.utc).isoformat(),
                planned_ids=IDS, changed_fields_from_cubic_zero=['interp'],
                expected_full_solves=4, same_T_grid_quadrature=True))
        base = ChernovConfig(**plan['baseline_config'])
        for task in plan['tasks']:
            pid = task['problem_id']
            saved = read_json(BATCH/'solutions'/f'{pid}.json')
            if args.phase == 'core':
                solve_task(equations[pid], saved, base, 'cubic_taper', output, task)
                with ResourceGuard(output/'traces'/f'{pid}.resources.jsonl',
                                   max(60., 6*saved['task_wall_seconds'])):
                    trace_task(equations[pid], saved, base, output)
                solve_task(equations[pid], saved, replace(base, boundary='zero'),
                           'cubic_zero', output, task)
            else:
                solve_task(equations[pid], saved, replace(base, interp='linear', boundary='zero'),
                           'linear_zero', output, task)
        write_json(output/f'{args.phase}_completion.json', dict(status='complete',
            source_script_sha256=digest(__file__), peak_rss_mib=peak_mib(),
            completed_utc=datetime.now(timezone.utc).isoformat()))
    except BaseException:
        write_json(output/f'{args.phase}_stopped.json', dict(status='stopped', traceback=traceback.format_exc()))
        raise
    finally:
        before = read_json(output/'before_sha256.json')
        changed = [name for name, h in before.items() if not (ROOT/name).exists() or digest(ROOT/name) != h]
        write_json(output/f'{args.phase}_integrity.json', dict(checked_files=len(before),
            changed_files=changed, protected_files_unchanged=not changed))
        if changed:
            raise ValueError(f'protected files changed: {changed}')


if __name__ == '__main__':
    main()
