"""Independent finite-time/discrete oracles and failure contracts for MC."""
from dataclasses import replace
import math

import numpy as np
import pytest

from mc_solver import (MCConfig, MCForcingBound, _Moments, _Run,
                       _path_contributions, auto_T, paths_for_tolerance,
                       solve_monte_carlo)
from remizov_solver import RemizovProblem


def problem(**changes):
    p = RemizovProblem('analytic', lambda x: 1., lambda x: 0., lambda x: 0.,
        lambda x: 1., 2., 1., 1., 0., 0., 'analytic', 'all_real_x', -2., 2.)
    return replace(p, **changes)


def bound(G):
    return MCForcingBound(G, 'explicit analytical bound for the test formula')


def test_auto_horizon_uses_G_scaling_zero_and_extreme_logarithms():
    for G in (1., 100., 1e200):
        T = auto_T(2., 0., G, 1e-4)
        assert math.log(G) - math.log(2.) - 2*T <= math.log(1e-4)
        assert T == pytest.approx(math.log(G/(2e-4))/2)
    assert auto_T(2., 0., 0., 1e-4) == 0
    assert auto_T(2., 0., 1., 1.) == 0
    assert math.isfinite(auto_T(1., 0., 1., 1e-320))
    assert auto_T(2., 0., 100., 1e-4) - auto_T(2., 0., 1., 1e-4) == pytest.approx(math.log(100.)/2)


def test_constant_solution_and_independent_trapezoid_oracle():
    r = solve_monte_carlo(problem(), [-1., 0., 1.],
        MCConfig(M=8, dt=.01, tail_eps=1e-6, chunk=3), forcing_bound=bound(1.))
    assert r.success, r.message
    # Integrand is deterministic: an independently summed exponential trapezoid.
    t = np.linspace(0., r.T, r.n_steps+1)
    oracle = np.trapezoid(np.exp(-2*t), x=t)
    np.testing.assert_allclose(r.f, oracle, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(r.stderr, 0.)
    # Infinite trapezoid bias <= lambda*h^2/12; tail here <= 1e-6.
    assert np.max(abs(r.f-.5)) < 2e-5
    assert r.tail_error_bound <= 1e-6
    assert r.tail_tolerance_met
    assert not r.total_error_certified and r.discretization_error_bound is None
    assert r.n_path_steps == r.n_normal_draws == 3*8*r.n_steps
    np.testing.assert_array_equal(r.completed_paths, 8)


def test_nonzero_potential_and_drift_diffusion_step_oracle():
    # Prescribed normals: verify old-point EM and both trapezoids independently.
    normals = np.array([[.5, -.3], [.2, .7], [-.4, .1]])
    class Draws:
        def __init__(self):
            self.i = 0
        def standard_normal(self, n):
            assert n == 2
            z = normals[self.i]; self.i += 1
            return z
    p = problem(a=lambda x: 1.+.1*np.tanh(x), b=lambda x: .2*np.cos(x),
        c_remizov=lambda x: .3*np.tanh(x), g_remizov=np.cos,
        a_lower_bound=.9, a_upper_bound=1.1, b_abs_upper_bound=.2, c_abs_upper_bound=.3)
    h = .2; x = np.full(2, .25); I = np.zeros(2)
    oracle = .5*h*np.cos(x)
    for k, z in enumerate(normals, 1):
        old = x.copy()
        x = old + .2*np.cos(old)*h + np.sqrt(2*(1+.1*np.tanh(old))*h)*z
        I += .5*h*(.3*np.tanh(old)+.3*np.tanh(x))
        oracle += (h/2 if k == 3 else h)*np.exp(-2*k*h+I)*np.cos(x)
    actual = _path_contributions(p, .25, .6, 3, 2, Draws(), bound(1.), _Run())
    np.testing.assert_allclose(actual, oracle, rtol=0, atol=3e-16)


def test_gaussian_against_exact_discrete_expectation_and_exact_solution():
    p = problem(g_remizov=lambda x: (4.-4*x*x)*np.exp(-x*x))
    r = solve_monte_carlo(p, [-1., 0., 1.],
        MCConfig(M=12000, dt=.01, tail_eps=1e-5, seed=2488, chunk=4000),
        forcing_bound=bound(4.))
    assert r.success, r.message
    t = np.linspace(0., r.T, r.n_steps+1)
    A = 1+4*t[:, None]
    x = r.x[None, :]
    heat_y = np.exp(-x*x/A)/np.sqrt(A)
    integrand = np.exp(-2*t[:, None])*(2+2/A-4*x*x/(A*A))*heat_y
    expected_discrete = np.trapezoid(integrand, x=t, axis=0)
    exact = np.exp(-r.x*r.x)
    # Predefined 6-SE diagnostic envelope: tests stochastic mean, not a total CI.
    assert np.all(abs(r.f-expected_discrete) <= 6*r.stderr+2e-14)
    assert np.max(abs(r.f-exact)) < .03
    assert np.all(r.stderr > 0)
    np.testing.assert_allclose(r.ci95, 1.96*r.stderr)
    assert r.interval_kind.endswith('discrete_truncated_mean')


def test_time_bias_is_not_hidden_by_zero_statistical_interval():
    r = solve_monte_carlo(problem(), [0.], MCConfig(M=2, T=1., dt=1.), forcing_bound=bound(1.))
    assert r.success
    assert r.ci95[0] == 0
    assert abs(r.f[0]-.5) > .06
    assert not r.total_error_certified
    assert not r.tail_tolerance_met


def test_centered_moments_keep_small_variance_on_large_mean_and_chunk_merges():
    values = 1e8 + np.linspace(-.3, .7, 1001)
    for chunks in ([values], np.array_split(values, 17), [[v] for v in values]):
        m = _Moments()
        for v in chunks:
            m.add(np.asarray(v))
        assert m.mean == pytest.approx(np.mean(values), abs=2e-6)
        assert m.stderr() == pytest.approx(np.std(values, ddof=1)/np.sqrt(len(values)), rel=2e-5)
    m = _Moments(); m.add(np.full(10, 1e200))
    assert m.mean == 1e200 and m.stderr() == 0


def test_solver_statistics_match_centered_identical_path_samples():
    from mc_solver import _spawn_key
    p = problem(g_remizov=lambda x: 1e8+np.tanh(x))
    cfg = MCConfig(M=1000, T=1., dt=.1, seed=42, chunk=1000)
    r = solve_monte_carlo(p, [0.], cfg)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(42,
        spawn_key=_spawn_key(p.problem_id, 0., 0))))
    values = _path_contributions(p, 0., r.T, r.n_steps, 1000, rng, None, _Run())
    assert r.success
    assert r.stderr[0] > 0
    assert r.stderr[0] == pytest.approx(np.std(values, ddof=1)/math.sqrt(1000), rel=1e-12)


def test_seed_is_local_repeatable_and_point_addressed():
    p = problem(g_remizov=np.cos)
    cfg = MCConfig(M=30, T=.4, dt=.1, seed=42, chunk=7)
    np.random.seed(918); expected_global = np.random.random()
    np.random.seed(918)
    a = solve_monte_carlo(p, [-1., 0., 1.], cfg)
    b = solve_monte_carlo(p, [-1., 0., 1.], cfg)
    single = solve_monte_carlo(p, [0.], cfg)
    assert np.random.random() == expected_global
    np.testing.assert_array_equal(a.f, b.f)
    np.testing.assert_array_equal(a.stderr, b.stderr)
    assert a.f[1] == single.f[0] and a.stderr[1] == single.stderr[0]
    other = solve_monte_carlo(p, [0.], replace(cfg, stream_id=1))
    assert other.f[0] != single.f[0]
    random_seed = solve_monte_carlo(p, [0.], replace(cfg, seed=None))
    replay = solve_monte_carlo(p, [0.], replace(cfg, seed=random_seed.seed_entropy))
    np.testing.assert_array_equal(random_seed.f, replay.f)


def test_explicit_T_without_G_is_uncertified_and_no_probe_is_called():
    def potential(x):
        assert x.size <= 4
        return np.zeros_like(x)
    p = problem(c_remizov=potential)
    r = solve_monte_carlo(p, [0.], MCConfig(M=4, chunk=4, dt=.1, T=.2))
    assert r.success and r.tail_error_bound is None and r.tail_tolerance_met is None
    assert r.forcing_bound_source is None and not r.total_error_certified
    r = solve_monte_carlo(p, [0.], MCConfig(M=4))
    assert not r.success and r.failure_reason == 'missing_forcing_bound'
    assert r.n_path_steps == 0 and not any(r.coefficient_calls.values())


@pytest.mark.parametrize('changes', [dict(M=0), dict(M=1), dict(M=-2), dict(M=True),
    dict(M=2.5), dict(dt=0), dict(dt=-1), dict(dt=np.inf), dict(chunk=0), dict(chunk=-1),
    dict(chunk=True), dict(T=-1), dict(T=np.nan), dict(T=np.inf), dict(tail_eps=0),
    dict(tail_eps=np.nan), dict(seed=-1), dict(seed=True), dict(stream_id=-1),
    dict(max_steps=0), dict(max_path_steps=0), dict(max_batch_paths=0),
    dict(max_runtime_seconds=0)])
def test_invalid_configs_raise_before_sampling(changes):
    with pytest.raises(ValueError):
        solve_monte_carlo(problem(), [0.], replace(MCConfig(T=.2), **changes))


@pytest.mark.parametrize('x', [[], [0., 0.], [np.nan], [np.inf], [1., 0.],
                             [[0., 1.]], [1j], [True], [-3.], 0.])
def test_invalid_target_vectors(x):
    with pytest.raises(ValueError):
        solve_monte_carlo(problem(), x, MCConfig(T=.1, M=2))


@pytest.mark.parametrize('changes,reason', [
    (dict(a=lambda x: -1.), 'nonpositive_diffusion'),
    (dict(a=lambda x: 0.), 'nonpositive_diffusion'),
    (dict(a=lambda x: np.nan), 'nonfinite_coefficient'),
    (dict(b=lambda x: np.inf), 'nonfinite_coefficient'),
    (dict(g_remizov=lambda x: np.inf), 'nonfinite_coefficient'),
    (dict(g_remizov=lambda x: np.ones_like(x, dtype=complex)), 'invalid_coefficient'),
    (dict(g_remizov=lambda x: np.ones((2, 2))), 'coefficient_evaluation_failure'),
    (dict(a=lambda x: 2.), 'certified_bound_violation'),
    (dict(b=lambda x: 1.), 'certified_bound_violation'),
    (dict(c_remizov=lambda x: 3.), 'certified_bound_violation'),
    (dict(g_remizov=lambda x: 2.), 'certified_bound_violation'),
    (dict(lambda_value=0.), 'invalid_remizov_conditions'),
    (dict(bounds_certification='sampled'), 'invalid_remizov_conditions'),
    (dict(bounds_domain='[-20,20]'), 'invalid_remizov_conditions')])
def test_numerical_failures_preserve_empty_arrays_and_diagnostics(changes, reason):
    r = solve_monte_carlo(problem(**changes), [0.], MCConfig(T=.2, M=4, dt=.1), forcing_bound=bound(1.))
    assert not r.success and r.failure_reason == reason, r.message
    for a in (r.f, r.stderr, r.ci95, r.ci_lower, r.ci_upper):
        assert a.size == 0
    np.testing.assert_array_equal(r.x, [0.])
    assert r.message and not r.total_error_certified


def test_callable_exception_and_actual_arithmetic_overflow_are_failures():
    def broken(x):
        raise RuntimeError('callback failed')
    r = solve_monte_carlo(problem(g_remizov=broken), [0.], MCConfig(M=2, T=.1))
    assert r.failure_reason == 'coefficient_evaluation_failure'
    # A genuine nonrepresentable state; no coefficient clipping or silent reset.
    p = problem(b=lambda x: 1e308, b_abs_upper_bound=1e308)
    r = solve_monte_carlo(p, [0.], MCConfig(M=2, T=10., dt=10.))
    assert r.failure_reason == 'numerical_overflow' and r.f.size == 0


@pytest.mark.parametrize('changes', [dict(max_steps=1), dict(max_path_steps=1),
                                   dict(chunk=3, max_batch_paths=2)])
def test_preflight_limits_do_not_sample(changes):
    cfg = replace(MCConfig(M=4, T=1., dt=.1), **changes)
    r = solve_monte_carlo(problem(), [0.], cfg)
    assert r.failure_reason == 'work_limit_exceeded'
    assert r.n_path_steps == r.n_normal_draws == 0
    assert not any(r.coefficient_calls.values())


def test_runtime_limit_and_zero_horizon(monkeypatch):
    import mc_solver
    ticks = iter([0., 0., 1., 1.])
    monkeypatch.setattr(mc_solver.time, 'perf_counter', lambda: next(ticks))
    r = solve_monte_carlo(problem(), [0.], MCConfig(M=2, T=.1, max_runtime_seconds=.5))
    assert r.failure_reason == 'time_limit_exceeded' and r.f.size == 0
    monkeypatch.undo()
    r = solve_monte_carlo(problem(), [0.], MCConfig(M=2, T=0), forcing_bound=bound(1.))
    assert r.success and r.n_steps == 0 and r.dt_actual == 0
    assert r.n_path_steps == r.n_normal_draws == r.n_batches == 0
    assert r.completed_paths[0] == 0 and not any(r.coefficient_calls.values())
    assert r.f[0] == r.stderr[0] == 0 and r.tail_error_bound == .5


def test_actual_step_never_exceeds_requested_and_bad_G_is_not_a_certificate():
    r = solve_monte_carlo(problem(), [0.], MCConfig(M=2, T=1., dt=.3))
    assert r.n_steps == 4 and r.dt_actual == .25
    for cert in (MCForcingBound(-1., 'bad'), MCForcingBound(1., ''),
                 MCForcingBound(1., 'grid', bounds_certification='sampled')):
        with pytest.raises(ValueError):
            solve_monte_carlo(problem(), [0.], MCConfig(M=2, T=.1), forcing_bound=cert)


def test_tiny_positive_horizon_is_not_mistaken_for_zero():
    p = problem(g_remizov=lambda x: 1e308)
    r = solve_monte_carlo(p, [0.], MCConfig(M=2, T=1e-320, dt=1e308))
    assert r.success and r.n_steps == 1 and r.dt_actual > 0
    assert r.f[0] > 0
    assert r.f[0] == pytest.approx(1e-320*1e308, rel=1e-3, abs=0)
    r = solve_monte_carlo(p, [0.], MCConfig(M=2, T=math.nextafter(0., 1.), dt=1.))
    assert not r.success and r.failure_reason == 'numerical_underflow'
    assert r.n_path_steps == 0 and r.f.size == 0


def test_paths_for_tolerance_is_only_a_validated_pilot_estimate():
    assert paths_for_tolerance(.01, 1000, .005) == math.ceil((1.96*.01*math.sqrt(1000)/.005)**2)
    for args in [(0., 100, .01), (-1., 100, .01), (.1, 1, .01),
                 (.1, 0, .01), (.1, 100, 0), (np.nan, 100, .01)]:
        with pytest.raises(ValueError):
            paths_for_tolerance(*args)
