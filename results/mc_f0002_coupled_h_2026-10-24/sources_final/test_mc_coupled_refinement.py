"""Physical Brownian coupling, original EM marginal, and pilot selection."""
from dataclasses import replace
import math

import numpy as np
import pytest

from diagnostics.mc_f0002_time_refinement import CoupledNormals, costs, coupled_paths
from diagnostics.report_mc_f0002_time_refinement import centered_stats, order_diagnostic, pilot_decision
from mc_solver import MCConfig, MCForcingBound, _Run, _path_contributions, solve_monte_carlo
from remizov_solver import RemizovProblem


def problem():
    return RemizovProblem('coupling_test',lambda x:1.,lambda x:0.,lambda x:0.,
        lambda x:(4-4*x*x)*np.exp(-x*x),2.,1.,1.,0.,0.,'analytic','all_real_x',-2.,2.)


class Prescribed:
    def __init__(self,draws):
        self.draws=iter(draws)
    def standard_normal(self,n):
        z=next(self.draws).copy()
        assert z.shape==(n,)
        return z


def test_nested_brownian_blocks_are_the_same_physical_increments():
    z=np.array([[.5,-.25],[.25,.75],[-.5,.25],[.75,-.5]])
    fine=CoupledNormals(Prescribed(z),1)
    middle=CoupledNormals(Prescribed(z),2)
    coarse=CoupledNormals(Prescribed(z),4)
    fine_dW=np.array([math.sqrt(1/128)*fine.standard_normal(2) for _ in range(4)])
    middle_dW=np.array([math.sqrt(1/64)*middle.standard_normal(2) for _ in range(2)])
    coarse_dW=math.sqrt(1/32)*coarse.standard_normal(2)
    np.testing.assert_allclose(middle_dW,fine_dW.reshape(2,2,2).sum(axis=1),rtol=0,atol=3e-17)
    np.testing.assert_allclose(coarse_dW,fine_dW.sum(axis=0),rtol=0,atol=3e-17)
    assert fine.normal_values_generated==middle.normal_values_generated==coarse.normal_values_generated==8


@pytest.mark.parametrize('q',[1,2])
def test_coupled_variable_coefficients_match_manual_em_and_both_trapezoids(q):
    z=np.array([[.5,-.3],[.2,.7],[-.4,.1],[.1,-.2]])
    p=replace(problem(),a=lambda x:1+.1*np.tanh(x),b=lambda x:.2*np.cos(x),
        c_remizov=lambda x:.3*np.tanh(x),g_remizov=np.cos,
        a_lower_bound=.9,a_upper_bound=1.1,b_abs_upper_bound=.2,c_abs_upper_bound=.3)
    T=.5; h=q/8; N=4//q
    x=np.full(2,.25); potential=np.zeros(2); oracle=.5*h*np.cos(x)
    for k,block in enumerate(z.reshape(N,q,2),1):
        old=x.copy()
        dW=math.sqrt(1/8)*block.sum(axis=0)
        x=old+.2*np.cos(old)*h+np.sqrt(2*(1+.1*np.tanh(old)))*dW
        potential+=.5*h*(.3*np.tanh(old)+.3*np.tanh(x))
        oracle+=(.5*h if k==N else h)*np.exp(-2*k*h+potential)*np.cos(x)
    actual=_path_contributions(p,.25,T,N,2,CoupledNormals(Prescribed(z),q),
        MCForcingBound(1.,'cos bounded by one'),_Run())
    np.testing.assert_allclose(actual,oracle,rtol=0,atol=8e-16)


def test_finest_level_preserves_production_result_and_repeatability():
    p=problem();bound=MCForcingBound(4.,'Gaussian exact forcing bound')
    base=MCConfig(M=32,chunk=32,T=.5,dt=1/8,seed=23)
    configs=[base,replace(base,dt=1/16),replace(base,dt=1/32)]
    Z,record=coupled_paths(p,0.,configs,bound)
    again,_=coupled_paths(p,0.,configs,bound)
    assert record['success']
    np.testing.assert_array_equal(Z,again)
    r=solve_monte_carlo(p,[0.],configs[-1],forcing_bound=bound)
    mean,cov=centered_stats(Z)
    assert r.success
    assert mean[-1]==pytest.approx(r.f[0],abs=2e-15)
    assert math.sqrt(cov[-1,-1]/32)==pytest.approx(r.stderr[0],rel=1e-14)
    assert record['preflight']['path_steps']==sum(l['kernel_statistics']['n_path_steps'] for l in record['levels'])
    assert record['preflight']['normal_values_generated_with_replay']==3*32*16


def test_failure_and_nonisolated_configs_do_not_produce_valid_samples():
    p=problem();c=MCConfig(M=8,chunk=8,T=.5,dt=1/8,seed=1);b=MCForcingBound(4.,'Gaussian')
    with pytest.raises(ValueError,match='only dt'):
        coupled_paths(p,0.,[c,replace(c,dt=1/16,seed=2)],b)
    with pytest.raises(ValueError,match='full batch'):
        coupled_paths(p,0.,[replace(c,chunk=4)],b)
    Z,r=coupled_paths(replace(p,a=lambda x:-1.),0.,[c],b)
    assert Z is None and not r['success']
    assert r['failure']['reason']=='nonpositive_diffusion'
    assert r['levels'][0]['kernel_statistics']['n_path_steps']==0


@pytest.mark.parametrize('factor',[0,3,-2,True,1.5])
def test_invalid_normal_aggregation_is_rejected(factor):
    with pytest.raises(ValueError):
        CoupledNormals(np.random.default_rng(1),factor)


def test_stable_covariance_does_not_subtract_large_squares():
    x=np.linspace(-.1,.1,101)
    Z=np.column_stack([1e8+x,1e8+2*x])
    mean,cov=centered_stats(Z)
    assert mean==pytest.approx([1e8,1e8],abs=2e-8)
    assert cov[1,1]==pytest.approx(4*cov[0,0],rel=3e-8)
    assert cov[0,0]>0


def test_pilot_gate_stops_if_noise_is_unresolved_and_limits_128(monkeypatch):
    import diagnostics.report_mc_f0002_time_refinement as module
    summary=dict(repeats=4,differences=[dict(mean=.03,ci95_half=.003,
        variance_reduction_vs_independent=20.)],levels=[dict(signed_error=.02,ci95_half=.02)])
    monkeypatch.setattr(module,'summarize_stage',lambda out,stage:(summary,[]))
    plan=dict(tail_error_bound=1e-4,decision_rule=dict(minimum_pilot_repeats=4,
        maximum_relative_ci95_half=.5,minimum_variance_reduction=2.))
    d=pilot_decision(None,plan)
    assert d['proceed'] and d['confirmation_hs']==[1/32,1/64,1/128]
    summary['differences'][0]['ci95_half']=.02
    assert not pilot_decision(None,plan)['proceed']
    summary['differences'][0]['ci95_half']=.003
    summary['levels'][0]=dict(signed_error=.002,ci95_half=.002)
    assert pilot_decision(None,plan)['confirmation_hs']==[1/32,1/64]


def test_joint_bootstrap_preserves_common_noise_in_order():
    shared=np.array([-.6,.2,-.1,.5,.8,-.5,.1,.4])
    means=shared[:,None]+np.array([3.,1.,0.])
    stage=dict(means_by_seed=means,hs=[1/32,1/64,1/128],
        levels=[dict(mean=float(np.mean(means[:,i]))) for i in range(3)])
    d=order_diagnostic(stage,seed=11,B=1000)
    assert d['observed_p']==pytest.approx(1.)
    assert d['bootstrap_p_ci95']==pytest.approx([1.,1.],abs=1e-14)
    assert d['finest_bias_power_law_model']==pytest.approx(1.)


def test_preflight_counts_updates_and_replayed_gaussians_separately():
    c=costs(2048,63.625,[1/32,1/64,1/128])
    assert c['time_steps']==[2036,4072,8144]
    assert c['path_steps']==29_188_096
    assert c['normal_values_generated_with_replay']==50_036_736
    assert c['unique_finest_normal_values']==16_678_912
