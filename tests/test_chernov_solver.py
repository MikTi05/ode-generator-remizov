"""Independent operator/oracle checks and compatibility with supplied Chern."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import simpson

from chernov_solver import ChernovConfig, ChernovSolver, _Counts, _State, _apply, _geometry, _simpson_weights
from remizov_benchmark_loader import load_remizov_problem
from remizov_solver import RemizovProblem


def problem(**changes):
    p = RemizovProblem('scalar', lambda x: 1., lambda x: 0., lambda x: 0.,
        lambda x: 1., 1., 1., 1., 0., 0., 'analytic', 'all_real_x', -1., 1.)
    return replace(p, **changes)


def test_simpson_accumulation_matches_scipy_on_both_grids():
    for t in (np.linspace(0, 3, 17), 3*np.linspace(0, 1, 17)**2):
        values = np.column_stack((np.exp(-t), np.sin(t), t**3))
        np.testing.assert_allclose(_simpson_weights(t) @ values,
                                   simpson(values, x=t, axis=0), rtol=2e-15, atol=2e-15)


def test_operator_polynomial_weights_sign_and_no_lambda():
    x = np.linspace(-3, 3, 601)
    state = _State(x, np.full_like(x, 1.2), np.full_like(x, -.3),
                   np.full_like(x, .4), x*x, 1., 1, 'cubic', 'zero', 1.)
    tau = .04
    q = _geometry(state, tau)
    values = _apply(state, x*x, tau, q, _Counts())
    inside = np.abs(x) <= 1
    expected = .25*q[0]**2 + .25*q[1]**2 + .5*q[2]**2 + tau*.4*x*x
    np.testing.assert_allclose(values[inside], expected[inside], rtol=0, atol=2e-14)
    np.testing.assert_array_equal(values, _apply(replace(state, lam=9), x*x, tau, q, _Counts()))


@pytest.mark.parametrize('n,oracle', [(1, 1.48), (2, 95/81)])
def test_finite_n_manual_r1_oracle(n, oracle):
    result = ChernovSolver(load_remizov_problem('R1'), ChernovConfig(
        n=n, num_t=513, dx=.0125, padding=16, tail_eps=1e-10)).solve([0.])
    assert result.success, result.failure_reason
    assert abs(result.f[0]-oracle) < 2e-7
    assert result.n_operator_applications == n*(513-1)


@pytest.mark.parametrize('scheme', ['solve', 'streaming'])
@pytest.mark.parametrize('interp', ['cubic', 'linear'])
def test_matches_supplied_source_on_same_grid(scheme, interp):
    spec = importlib.util.spec_from_file_location('supplied_chern_reference',
        Path(__file__).resolve().parents[1]/'comparison_inputs/Chern.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    # Binary-exact spacing/endpoints avoid differences between arange/linspace.
    p = problem(g_remizov=lambda x: (3-4*x*x)*np.exp(-x*x))
    config = ChernovConfig(n=3, num_t=32, n_steps=20, dx=.125, padding=3, T=2.,
                          interp=interp,boundary='taper' if interp=='cubic' else 'zero')
    legacy = reference.ChernovSolver(p.a,p.b,p.c_remizov,p.g_remizov,1.,-4.,4.,.125,
                                     T=2.,n=3,num_t=32,c_sup=0.,interp=interp)
    adapted = ChernovSolver(p, config)
    result = adapted.solve([-1.,0.,1.]) if scheme == 'solve' else adapted.solve_streaming([-1.,0.,1.])
    exact_discrete = legacy.solve() if scheme == 'solve' else legacy.solve_streaming(n_steps=20)
    assert result.success
    np.testing.assert_allclose(result.f, exact_discrete[[24,32,40]], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('scheme', ['solve', 'streaming'])
def test_scalar_forcing_and_numerical_failures(scheme):
    c = ChernovConfig(n=1, num_t=33, n_steps=32, dx=.1, padding=3)
    for p, expected in ((problem(), ''),
                        (problem(lambda_value=0.), 'invalid_remizov_conditions'),
                        (problem(g_remizov=lambda x: np.nan), 'nonfinite_state'),
                        (problem(a=lambda x: 2.), 'certified_bound_violation')):
        s = ChernovSolver(p,c)
        r = s.solve([0.]) if scheme == 'solve' else s.solve_streaming([0.])
        assert r.failure_reason == expected
        assert r.success == (not expected)
        assert r.f.size == (1 if r.success else 0)
        assert not r.total_error_certified
    r = ChernovSolver(problem(), replace(c,max_work_grid_points=4)).solve([0.])
    assert r.failure_reason == 'work_domain_too_large' and r.f.size == 0


def test_config_and_target_validation_and_zero_T():
    for changes in ({'n':0}, {'n':True}, {'dx':0}, {'T':-1}, {'n_steps':1},
                    {'num_t':2}, {'padding':0}, {'taper_width':0}, {'n_jobs':0}):
        with pytest.raises(ValueError):
            ChernovSolver(problem(), replace(ChernovConfig(), **changes))
    s = ChernovSolver(problem(), ChernovConfig(T=0))
    for target in ([], [0.,0.], [np.nan], [-2.]):
        with pytest.raises(ValueError):
            s.solve(target)
    for r in (s.solve([0.]), s.solve_streaming([0.])):
        assert r.success and r.f[0] == 0 and r.n_operator_applications == 0


def test_streaming_is_a_separate_scheme_and_rounds_steps():
    p = load_remizov_problem('R1')
    c = ChernovConfig(n=2, n_steps=3, num_t=5, T=1., dx=.05, padding=4)
    fixed = ChernovSolver(p,c).solve([0.])
    streaming = ChernovSolver(p,c).solve_streaming([0.])
    assert fixed.success and streaming.success
    assert streaming.n_steps == 4 and streaming.n_operator_applications == 4
    assert abs(fixed.f[0]-streaming.f[0]) > 1e-3


def test_spawn_parallel_accepts_local_callables_and_matches_serial():
    p = problem(g_remizov=lambda x: np.exp(-x*x))
    c = ChernovConfig(n=2,num_t=9,T=1.,dx=.125,padding=3)
    serial = ChernovSolver(p,c).solve([-1.,0.,1.])
    parallel = ChernovSolver(p,replace(c,n_jobs=2)).solve([-1.,0.,1.])
    assert serial.success and parallel.success
    np.testing.assert_array_equal(serial.f,parallel.f)
    assert serial.n_operator_applications == parallel.n_operator_applications


def test_both_schemes_accept_every_existing_dataset_record():
    from all_methods_dataset_loader import load_all_methods_equations
    root=Path(__file__).resolve().parents[1]/'data/generated'
    equations=load_all_methods_equations(root/'all_methods_equations_v1.csv',
                                       root/'all_methods_equations_v1_metadata.jsonl')
    assert len(equations)==20
    # Interface/finite-arithmetic smoke test, not an accuracy claim at this budget.
    c=ChernovConfig(n=2,num_t=9,n_steps=8,T=.25,dx=.1,padding=2)
    for e in equations:
        for result in (ChernovSolver(e.remizov_problem,c).solve([-1.,0.,1.]),
                       ChernovSolver(e.remizov_problem,c).solve_streaming([-1.,0.,1.])):
            assert result.success, (e.equation_id,result.failure_reason)
            assert result.f.shape==(3,) and np.all(np.isfinite(result.f))
