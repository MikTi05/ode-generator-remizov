"""Small observer/grid checks; no production equation is solved."""
import numpy as np
from scipy.interpolate import CubicSpline

from chernov_solver import ChernovConfig, _State, _at_t
from diagnostics.chernov_window_padding24 import nested_grid, observed_chain, solve_once
from remizov_solver import RemizovProblem


def test_nested_grid_retains_old_float_coordinates_and_nominal_step():
    old = np.linspace(-13., 13., 2081)
    grid, check = nested_grid(old)
    assert grid.size == 4001 and check['all_2081_old_nodes_bitwise_retained']
    np.testing.assert_array_equal(grid[960:3041], old)
    target = np.linspace(-1, 1, 33)
    np.testing.assert_array_equal(grid[np.searchsorted(grid, target)], target)
    np.testing.assert_allclose(np.diff(grid), .0125, atol=8e-15, rtol=0)


def test_observed_chain_equals_unmodified_production_chain():
    x = np.linspace(-3, 3, 241)
    state = _State(x, 1+.1*np.sin(x), .3*np.cos(x), .2*np.sin(x),
                   np.exp(-x*x), 1., 7, 'cubic', 'zero', .6)
    for t in (.0, .15, .5):
        actual, counts, summary = observed_chain(state, t, .2, 0)
        expected, original_counts = _at_t(state, t)
        np.testing.assert_array_equal(actual, expected)
        assert counts == original_counts
        assert summary['max_global'] >= summary['initial_global']
        assert summary['max_global'] >= summary['final_global']
        assert summary['outside_query_values'] == counts.outside


def test_single_integration_pipeline_constant_forcing_oracle(tmp_path):
    p = RemizovProblem('constant-oracle',lambda x:1.,lambda x:0.,lambda x:0.,
        lambda x:1.,1.,1.,1.,0.,0.,'analytic','all_real_x',-1.,1.)
    c = ChernovConfig(n=2,num_t=5,dx=.0125,padding=24.,interp='cubic',boundary='zero',T=.2)
    grid = np.linspace(-25,25,4001)
    target = np.linspace(-1,1,33)
    result,t,w = solve_once(p,c,grid,.2,target,tmp_path)
    # At these central nodes, all two-step paths sample constant forcing.
    expected = sum(w*np.exp(-t))
    np.testing.assert_allclose(result['f'],expected,atol=2e-16,rtol=0)
    assert result['n_operator_applications'] == 8
    # Cubic/zero is not generally a contraction. Independently construct its
    # first boundary jump from constant forcing and interpolate that profile:
    # the second step overshoots despite C=0. This checks that the observer
    # detects a real violation instead of assuming the bound must hold.
    shift = 2*np.sqrt(.2/2)
    plus, minus = grid+shift, grid-shift
    u1 = .25*(plus<=25)+.25*(minus>=-25)+.5
    spline = CubicSpline(grid,u1,extrapolate=False)
    vp, vm = spline(plus), spline(minus)
    vp[plus>25] = 0
    vm[minus<-25] = 0
    u2 = .25*vp+.25*vm+.5*u1
    assert max(abs(u2)) > 1+1e-10
    assert result['observed_step_bound_violations'] > 0
    assert result['max_intermediate_global'] >= max(abs(u2))*(1-1e-14)
