"""Small independent checks for the experimental observer; no production solves."""
from dataclasses import replace

import numpy as np
import pytest

from chernov_solver import ChernovConfig, _Counts, _State, _apply, _geometry
from diagnostics.chernov_boundary_diagnosis import (
    fingerprints, inspect_step, prepare, summarize_trace,
)
from remizov_solver import RemizovProblem


def state(interp='cubic', boundary='taper'):
    x = np.linspace(-3, 3, 97)
    return _State(x, 1+.1*np.sin(x), .3*np.cos(x), .2*np.sin(2*x),
                  np.exp(-x*x), 1., 4, interp, boundary, .6)


@pytest.mark.parametrize('boundary', ['taper', 'zero'])
def test_observer_preserves_production_step_and_weight(boundary):
    s = state(boundary=boundary)
    u = np.sin(4*s.x)+.1*s.x**2
    tau = .075
    q = _geometry(s, tau)
    actual, row = inspect_step(s, u, tau, q, 2, _Counts())
    np.testing.assert_array_equal(actual, _apply(s, u, tau, q, _Counts()))
    expected_weight = np.exp(-s.lam*tau*3)
    assert row['weighted_global_after'] == expected_weight*max(abs(actual))
    if boundary == 'zero':
        assert row['outside_extension_max'] == 0
        assert row['outside_contribution_max'] == 0


def test_outside_contribution_is_exact_boundary_intervention_on_same_state():
    s = state()
    u = np.sin(4*s.x)+.1*s.x**2
    tau = .1
    q = _geometry(s, tau)
    taper, row = inspect_step(s, u, tau, q, 0, _Counts())
    zero = _apply(replace(s, boundary='zero'), u, tau, q, _Counts())
    np.testing.assert_allclose(row['outside_contribution_max'], max(abs(taper-zero)), rtol=2e-15)
    np.testing.assert_allclose(row['zero_extension_same_input_after_max'], max(abs(zero)), rtol=2e-15)
    assert row['outside_contribution_target'] == 0


@pytest.mark.parametrize('tau', [.0001, .025, .25, 2.])
def test_linear_zero_max_norm_bound_without_a_matrix(tau):
    s = state('linear', 'zero')
    rng = np.random.default_rng(8102)
    vectors = [np.ones_like(s.x), (-1.)**np.arange(s.x.size), rng.uniform(-1, 1, s.x.size)]
    q = _geometry(s, tau)
    for u in vectors:
        out = _apply(s, u, tau, q, _Counts())
        assert max(abs(out)) <= (1+tau*max(abs(s.c)))*max(abs(u))*(1+1e-14)


def test_preparation_fingerprints_unchanged_under_boundary_and_interpolation():
    p = RemizovProblem('p', lambda x: 1., lambda x: .3, lambda x: -.2,
        lambda x: np.exp(-x*x), 1., 1., 1., .3, .2,
        'analytic', 'all_real_x', -1., 1.)
    c = ChernovConfig(n=4, num_t=9, dx=.125, padding=2., T=2.)
    s, t, w = prepare(p, c, 2.)
    for variant in (replace(c, boundary='zero'), replace(c, interp='linear', boundary='zero')):
        ss, tt, ww = prepare(p, variant, 2.)
        assert fingerprints(s, t, w, np.linspace(-1, 1, 33)) == fingerprints(
            ss, tt, ww, np.linspace(-1, 1, 33))
    with pytest.raises(ValueError, match='effective T'):
        prepare(p, c, 3.)


def test_onset_uses_documented_thresholds():
    rows = []
    for k, before, after, difference in [(1, 10., 9., 0.), (2, 9., 25., .02)]:
        rows.append(dict(k_after=k, global_before=before, global_after=after,
            target_after=after, outside_extension_max=after,
            weighted_global_after=after/2, weighted_target_after=after/2,
            target_pair_difference=difference, weighted_target_pair_difference=difference/2))
    result = summarize_trace(rows, 10., .5, .1)
    assert result['first_global_gt_twice_initial'] == 2
    assert result['first_step_bound_violation'] == 2
    assert result['first_target_difference_gt_001'] == 2
    assert result['first_weighted_target_difference_gt_001'] is None
