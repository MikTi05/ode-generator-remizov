"""One nested-grid padding=24 control against saved F0180/S04 cubic/zero.

Uses the unchanged production _apply, _geometry, Simpson weights and CubicSpline.
The grid's old [-13,13] coordinates are retained bitwise: two independent
linspace calls otherwise differ by a few ulps. Every actual step is observed;
no production baseline or parameter refinement is recalculated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# This import sets single-thread BLAS environment before importing numpy/scipy.
from diagnostics.chernov_boundary_diagnosis import (
    CORE, DATA, BATCH, ResourceGuard, THREAD_VARS, array_digest, clean, digest,
    fingerprints, max_abs, peak_mib, prepare, protected_snapshot, read_json, write_json,
)
import os
import numpy as np
from scipy.interpolate import CubicSpline
from all_methods_dataset_loader import load_all_methods_equation
from chernov_solver import ChernovConfig, _Counts, _State, _apply, _geometry, _simpson_weights
from remizov_solver import _as_values, _evaluate_coefficients

PID = 'AMV1_F0180_S04'
PRIOR = ROOT/'results/chernov_boundary_diagnosis_2026-10-24'
BASE = PRIOR/'runs'/f'{PID}_cubic_zero.json'


def nested_grid(old):
    native = np.linspace(-25., 25., 4001)
    grid = native.copy()
    grid[960:3041] = old
    np.testing.assert_array_equal(grid[960:3041], old)
    assert grid.size == 4001 and grid[0] == -25. and grid[-1] == 25.
    assert np.all(np.diff(grid) > 0)
    np.testing.assert_allclose(np.diff(grid), .0125, rtol=0., atol=8e-15)
    return grid, dict(previous_grid_subset_start=960, previous_grid_subset_stop=3041,
        all_2081_old_nodes_bitwise_retained=True, work_grid_points=4001,
        nominal_dx=.0125, min_float_spacing=float(np.min(np.diff(grid))),
        max_float_spacing=float(np.max(np.diff(grid))),
        native_linspace_common_node_differences=int(np.count_nonzero(native[960:3041] != old)),
        native_linspace_max_common_difference=max_abs(native[960:3041]-old),
        grid_sha256=array_digest(grid), previous_grid_sha256=array_digest(old))


def observed_chain(state, t, C, node_index):
    """Production chain exactly; reductions do not modify its states."""
    counts = _Counts()
    u = state.g.copy()
    initial = max_abs(u)
    summary = dict(time_node_index=node_index, t=float(t), tau=float(t/state.n),
        initial_global=initial, max_global=initial, max_target=max_abs(u[np.abs(state.x)<=1]),
        max_boundary=max(abs(float(u[0])), abs(float(u[-1]))),
        max_weighted_global=initial, max_step_ratio_to_bound=0., violations=0,
        first_violation=None, max_global_location=None)
    operator_seconds = 0.
    if t > 0:
        tau = float(t/state.n)
        q = _geometry(state, tau)
        target = np.abs(state.x) <= 1
        for k in range(state.n):
            before = max_abs(u)
            started = time.perf_counter()
            u = _apply(state, u, tau, q, counts)
            operator_seconds += time.perf_counter()-started
            after = max_abs(u)
            ratio = after/((1+tau*C)*before) if before else (0. if after == 0 else math.inf)
            summary['max_step_ratio_to_bound'] = max(summary['max_step_ratio_to_bound'], ratio)
            if ratio > 1+1e-10:
                summary['violations'] += 1
                if summary['first_violation'] is None:
                    summary['first_violation'] = dict(k_after=k+1, before=before, after=after,
                        ratio_to_bound=ratio, x=float(state.x[np.argmax(np.abs(u))]))
            if after > summary['max_global']:
                summary['max_global'] = after
                summary['max_global_location'] = dict(k_after=k+1,
                    x=float(state.x[np.argmax(np.abs(u))]))
            summary['max_target'] = max(summary['max_target'], max_abs(u[target]))
            summary['max_boundary'] = max(summary['max_boundary'], abs(float(u[0])), abs(float(u[-1])))
            summary['max_weighted_global'] = max(summary['max_weighted_global'],
                math.exp(-state.lam*tau*(k+1))*after)
    weighted = math.exp(-state.lam*t)*u
    summary.update(final_global=max_abs(u), final_weighted_global=max_abs(weighted),
        final_weighted_target=max_abs(weighted[np.abs(state.x)<=1]),
        outside_query_values=counts.outside, operator_seconds=operator_seconds)
    return weighted, counts, summary


def solve_once(problem, config, grid, expected_T, target, output):
    """Same fixed-n quadrature pipeline as production; only supplied grid differs."""
    began = time.perf_counter()
    # Check T without substituting a rounded number in the configuration.
    from remizov_solver import compute_truncation_T
    T = compute_truncation_T(problem, config.tail_eps) if config.T is None else config.T
    assert T == expected_T
    a, b, c = _evaluate_coefficients(problem, grid)
    forcing = _as_values(problem.g_remizov, grid, 'nonfinite_state')
    state = _State(grid, a, b, c, forcing, problem.lambda_value, config.n,
        config.interp, config.boundary, config.taper_width or .1*(grid[-1]-grid[0]))
    preparation_seconds = time.perf_counter()-began
    s = np.linspace(0, 1, config.num_t+(config.num_t % 2 == 0))
    times = T*(s**2 if config.t_grading == 'quadratic' else s)
    weights = _simpson_weights(times)
    integral = np.zeros_like(grid)
    counts = _Counts()
    summaries = []
    with (output/'time_nodes.jsonl').open('x') as handle:
        for j, (t, w) in enumerate(zip(times, weights)):
            values, work, observation = observed_chain(state, float(t), problem.c_abs_upper_bound, j)
            integral += w*values
            counts.merge(work)
            summaries.append(observation)
            handle.write(json.dumps(clean(observation), sort_keys=True, allow_nan=False)+'\n')
    counts.boundary_max = max(counts.boundary_max, abs(float(integral[0])), abs(float(integral[-1])))
    f = CubicSpline(grid, integral, extrapolate=False)(target)
    counts.splines += 1
    assert np.all(np.isfinite(f))
    # Match production's boundary metric and also retain the final-step edges
    # that production's pre-step-only observation does not include.
    return dict(problem_id=problem.problem_id, scheme='solve', success=True, failure_reason='',
        x=target, f=f, T=T, n=config.n, time_nodes=len(times), actual_dx=.0125,
        work_grid_points=len(grid), work_domain=[float(grid[0]), float(grid[-1])],
        runtime_seconds=time.perf_counter()-began, preparation_seconds=preparation_seconds,
        n_operator_applications=counts.operators, n_spline_builds=counts.splines,
        outside_query_values=counts.outside, boundary_state_max_abs=counts.boundary_max,
        max_intermediate_global=max(z['max_global'] for z in summaries),
        max_intermediate_target=max(z['max_target'] for z in summaries),
        max_intermediate_boundary_including_final_steps=max(z['max_boundary'] for z in summaries),
        max_weighted_intermediate_global=max(z['max_weighted_global'] for z in summaries),
        observed_step_bound_violations=sum(z['violations'] for z in summaries),
        max_observed_step_ratio_to_bound=max(z['max_step_ratio_to_bound'] for z in summaries),
        observed_steps=counts.operators, operator_seconds=sum(z['operator_seconds'] for z in summaries),
        total_error_certified=False, refinement_controls_performed=False,
        runtime_includes_observation=True), times, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    write_json(output/'before_sha256.json', protected_snapshot(output))
    (output/'PROJECT_STATUS.before.md').write_bytes((ROOT/'PROJECT_STATUS.md').read_bytes())
    try:
        baseline, prior_plan, production_plan = read_json(BASE), read_json(PRIOR/'plan.json'), read_json(BATCH/'plan.json')
        artifact_hashes = read_json(PRIOR/'artifacts_sha256.json')
        for path in (BASE, PRIOR/'plan.json', ROOT/'diagnostics/chernov_boundary_diagnosis.py'):
            assert digest(path) == artifact_hashes[str(path.relative_to(ROOT))]
        for name, expected in prior_plan['source_data_sha256'].items():
            assert digest(ROOT/name) == expected == production_plan['protected_sha256'][name]
        versions = {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'sympy')}
        assert versions == prior_plan['environment']['versions']
        assert platform.python_version() == prior_plan['environment']['python'].split()[0]
        equation = load_all_methods_equation(PID, DATA/'all_methods_equations_v1.csv',
                                           DATA/'all_methods_equations_v1_metadata.jsonl')
        production = read_json(BATCH/'solutions'/f'{PID}.json')
        assert production['metadata'] == equation.metadata
        assert digest(BATCH/'solutions'/f'{PID}.json') == baseline['production_result_sha256']
        old_config = ChernovConfig(**baseline['config'])
        new_config = replace(old_config, padding=24.)
        assert [k for k in asdict(old_config) if getattr(old_config,k)!=getattr(new_config,k)] == ['padding']
        assert old_config.n == 128 and old_config.num_t == 513 and old_config.dx == .0125
        assert old_config.interp == 'cubic' and old_config.boundary == 'zero'
        target, exact, old_f = np.asarray(baseline['result']['x']), np.asarray(baseline['y_exact']), np.asarray(baseline['result']['f'])
        np.testing.assert_array_equal(target, np.linspace(-1, 1, 33))
        np.testing.assert_array_equal(exact, equation.y_exact(target))
        assert max_abs(old_f-exact) == baseline['max_abs_error']
        old, times, weights = prepare(equation.remizov_problem, old_config, baseline['result']['T'])
        assert fingerprints(old, times, weights, target) == baseline['fingerprints']
        grid, grid_check = nested_grid(old.x)
        a, b, c = _evaluate_coefficients(equation.remizov_problem, grid)
        g = _as_values(equation.remizov_problem.g_remizov, grid, 'nonfinite_state')
        for new_a, old_a in ((a,old.a),(b,old.b),(c,old.c),(g,old.g)):
            np.testing.assert_array_equal(new_a[960:3041], old_a)
        target_indices = np.searchsorted(grid, target)
        np.testing.assert_array_equal(grid[target_indices], target)
        plan = dict(schema='chernov_window_padding24_v1', created_utc=datetime.now(timezone.utc).isoformat(),
            problem_id=PID, old_config=asdict(old_config), new_config=asdict(new_config),
            changed_config_fields=['padding'], baseline_path=str(BASE.relative_to(ROOT)),
            baseline_sha256=digest(BASE), production_result_sha256=digest(BATCH/'solutions'/f'{PID}.json'),
            source_data_sha256=prior_plan['source_data_sha256'], source_script_sha256=digest(__file__),
            environment=dict(python=sys.version, platform=platform.platform(), versions=versions,
                thread_environment={k:os.environ[k] for k in THREAD_VARS}),
            T=baseline['result']['T'], lambda_value=equation.remizov_problem.lambda_value,
            grid_check=grid_check, target_x=target, time_nodes_sha256=array_digest(times),
            time_weights_sha256=array_digest(weights), shared_coefficients_bitwise_equal=True,
            no_previous_solutions_recomputed=True, planned_full_solves=1,
            observer='all 65536 actual _apply steps, no modifications to state',
            step_bound_tolerance_relative=1e-10,
            resource_policy=dict(compute_processes=1, blas_threads=1, memory_mib=768,
                stage_wall_limit_seconds=baseline['result']['runtime_seconds']*(4001/2081)*6,
                expected_seconds=[25,60]))
        write_json(output/'plan.json', plan)
        with ResourceGuard(output/'resources.jsonl', plan['resource_policy']['stage_wall_limit_seconds']):
            result, new_times, new_weights = solve_once(equation.remizov_problem, new_config,
                grid, baseline['result']['T'], target, output)
        np.testing.assert_array_equal(times, new_times)
        np.testing.assert_array_equal(weights, new_weights)
        assert result['n_operator_applications'] == 65536
        error, delta = max_abs(result['f']-exact), max_abs(result['f']-old_f)
        record = dict(config=asdict(new_config), grid_check=grid_check, result=result,
            y_exact=exact, baseline_f=old_f, baseline_error=baseline['max_abs_error'],
            max_abs_error=error, solution_max_abs_difference=delta,
            error_change=error-baseline['max_abs_error'], accuracy_reached=error<=.01,
            peak_rss_mib=peak_mib(), mathematical_core_modified=False)
        write_json(output/'result.json', record)
        write_json(output/'comparison.json', dict(padding12=baseline,
            padding24=record, no_full_padding12_maximum_available=True,
            padding12_trace_only='four selected time nodes; no baseline recomputation'))
        print(json.dumps(clean(dict(error=error, solution_difference=delta, result=result)),indent=2),flush=True)
    except BaseException:
        write_json(output/'stopped.json', dict(traceback=traceback.format_exc()))
        raise
    finally:
        before = read_json(output/'before_sha256.json')
        changed = [name for name,h in before.items() if not (ROOT/name).exists() or digest(ROOT/name)!=h]
        write_json(output/'run_integrity.json',dict(checked_files=len(before),changed_files=changed,
            protected_files_unchanged=not changed))
        assert not changed


if __name__ == '__main__':
    main()
