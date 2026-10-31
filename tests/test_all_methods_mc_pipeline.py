"""Read-only integration with real small-dataset vector evaluators and CLI."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from all_methods_dataset_loader import load_all_methods_equations
from mc_solver import MCConfig, solve_monte_carlo

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def equations():
    return {e.equation_id: e for e in load_all_methods_equations(
        ROOT/'data/generated/all_methods_equations_v1.csv',
        ROOT/'data/generated/all_methods_equations_v1_metadata.jsonl')}


@pytest.mark.parametrize('pid', ['AMV1_F0000_S00', 'AMV1_F0001_S00', 'AMV1_F0002_S00'])
def test_existing_vector_functions_are_consumed_without_scalar_fallback(equations, pid):
    e = equations[pid]
    x = np.linspace(-1., 1., 33)
    p = e.remizov_problem
    for func in (p.a, p.b, p.c_remizov, p.g_remizov):
        assert np.asarray(func(x)).shape == x.shape
    # Deliberately a small compatibility run, not an assertion of total accuracy.
    cfg = MCConfig(M=24, chunk=8, T=.2, dt=.05, seed=2488)
    r = solve_monte_carlo(p, x, cfg)
    assert r.success, r.message
    np.testing.assert_array_equal(r.x, x)
    assert r.f.shape == r.stderr.shape == (33,)
    assert np.all(np.isfinite(r.f)) and np.all(r.stderr >= 0)
    assert r.coefficient_values['a'] == 33*24*4
    assert r.n_path_steps == 33*24*4
    assert r.tail_error_bound is None and not r.total_error_certified
    np.testing.assert_allclose([e.cauchy_problem.c(t) for t in x], p.c_remizov(x)-p.lambda_value)
    np.testing.assert_array_equal([e.cauchy_problem.g(t) for t in x], -p.g_remizov(x))
    # The new solver is also compatible with the existing JSON-oriented workflow.
    from examples.solve_mc_equation import json_ready
    json.dumps(json_ready(asdict(r)), allow_nan=False)


def test_cli_reports_explicit_uncertified_tail_and_exact_error():
    proc = subprocess.run([sys.executable, '-B', str(ROOT/'examples/solve_mc_equation.py'),
        '--equation-id', 'AMV1_F0000_S00', '--points', '3', '--M', '8',
        '--T', '.2', '--dt', '.1', '--chunk', '4', '--uncertified-tail'], cwd=ROOT, text=True,
        capture_output=True, timeout=30, env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    r = data['result']
    assert r['success'] and r['tail_error_bound'] is None
    assert r['total_error_certified'] is False
    assert data['accuracy_reached'] is None
    assert data['max_abs_error'] == max(abs(a-b) for a,b in zip(r['f'],data['y_exact']))


def test_cli_requires_explicit_T_when_no_forcing_certificate():
    proc = subprocess.run([sys.executable, '-B', str(ROOT/'examples/solve_mc_equation.py'), '--uncertified-tail'],
        cwd=ROOT, text=True, capture_output=True, timeout=30,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
    assert proc.returncode == 2 and 'analytical' in proc.stderr


def test_existing_comparison_worker_and_preflight_accept_mc(tmp_path):
    from diagnostics.compare_chernov import _worker_calculate, preflight
    spec = dict(problem_id='MC_constant', method='monte_carlo',
                config=asdict(MCConfig(M=4, T=.2, dt=.1)))
    estimate = preflight(spec)
    assert estimate['path_steps'] == 33*4*2
    assert estimate['tail_bound_available'] and not estimate['total_error_certified']
    request, output = tmp_path/'request.json', tmp_path/'result.json'
    request.write_text(json.dumps(spec))
    _worker_calculate(request, output)
    data = json.loads(output.read_text())
    assert data['result']['success']
    assert data['result']['n_path_steps'] == estimate['path_steps']
    assert data['result']['tail_error_bound'] is not None
    assert data['max_abs_error'] == max(abs(a-b) for a,b in zip(data['result']['f'], data['y_exact']))


def test_comparison_preflight_does_not_invent_dataset_G():
    from diagnostics.compare_chernov import preflight
    spec = dict(problem_id='AMV1_F0000_S00', method='monte_carlo',
                config=asdict(MCConfig(M=4, T=.2, dt=.1)))
    estimate = preflight(spec)
    assert estimate['tail_bound_available']
    assert estimate['forcing_certificate']['version'] == 'all_methods_forcing_bounds_v1'
    spec['config']['T'] = None
    assert preflight(spec)['T'] > 0
    spec['forcing_bound'] = None  # explicit opt-out retains the old uncertified mode
    with pytest.raises(ValueError, match='global forcing certificate'):
        preflight(spec)
