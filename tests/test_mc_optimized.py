"""Same-increment regression and independent analytical MC oracles.

Absolute 1e-8 path tolerance was frozen before measurements. No exact-based
parameter selection. Statistical test uses a fixed seed and 5 standard errors
against the analytically known discrete/truncated expectation, not a CI claim
for the continuous solution.
"""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from all_methods_dataset_loader import load_all_methods_equations
from mc_solver import MCConfig,MCForcingBound,solve_monte_carlo,_path_contributions,_Run,_MCFailure
from mc_optimized import (prepare_all_methods_mc,prepare_analytic_mc,compile_backend,
    solve_monte_carlo_optimized,fused_values,flatten_trees,value_tape_numpy,path_contributions)
from src.all_methods_expressions import SafeExpressionNode,evaluate_jet2,ALLOWED_OPS

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def equations():
    d=ROOT/'data/generated'
    return load_all_methods_equations(d/'all_methods_equations_v1.csv',d/'all_methods_equations_v1_metadata.jsonl')


@pytest.fixture(scope='module')
def compiled():
    pytest.importorskip('numba');compile_backend()
    import mc_ast_kernels
    return mc_ast_kernels


def node(op,*children,**params):
    return SafeExpressionNode(op,children,params)


def test_all_opcodes_against_reference_jets(compiled):
    leaves=[node('constant',value=.2),node('sin_affine',omega=1.2,phi=.3),
        node('cos_affine',omega=.8,phi=-.2),node('tanh_affine',omega=.7,phi=.1),
        node('gaussian',alpha=.8,mu=-.4),node('lorentz',alpha=1.3,mu=.2),node('saturating',mu=.1)]
    nodes=leaves+[node(op,leaves[1]) for op in ('sin','cos','tanh','exp_neg_square','rational_saturate')]
    nodes += [node('lincomb',nodes[-1],leaves[2],alpha=.7,beta=-.3),node('mul',nodes[-2],nodes[-3])]
    assert set(n.op for n in nodes)==ALLOWED_OPS
    x=np.linspace(-4,4,101)
    for n in nodes:
        tape,_=flatten_trees((n,));v=np.empty(len(tape));d=v.copy();dd=v.copy()
        ref=evaluate_jet2(n,x)
        np.testing.assert_allclose(value_tape_numpy(tape,x)[-1],ref.value,rtol=0,atol=1e-12)
        for i,point in enumerate(x):
            assert compiled._jet(tape,point,v,d,dd)
            np.testing.assert_allclose([v[-1],d[-1],dd[-1]],
                [ref.value[i],ref.d1[i],ref.d2[i]],rtol=0,atol=1e-11)


@pytest.mark.parametrize('index',range(20))
def test_fused_coefficients_all_small_equations(equations,compiled,index):
    e=equations[index];p=prepare_all_methods_mc(e.metadata);mu,R=p.parameters[6:8]
    x=np.unique(np.r_[np.linspace(-5,5,81),mu-R,mu+R,
        np.nextafter(mu-R,mu),np.nextafter(mu+R,mu)])
    old=e.remizov_problem
    expected=np.array([old.a(x),old.b(x),old.c_remizov(x),old.g_remizov(x)])
    for engine in ('numpy','numba'):
        np.testing.assert_allclose(fused_values(p,x,engine),expected,rtol=0,atol=1e-12)


class Replay:
    def __init__(self,z):self.z=z;self.row=0
    def standard_normal(self,m):
        value=self.z[self.row];self.row+=1;assert value.size==m
        return value.copy()


@pytest.mark.parametrize('index',[0,2,14,17,18])
def test_per_path_replay(equations,compiled,index):
    e=equations[index];p=prepare_all_methods_mc(e.metadata);T=2.;N=256;m=64
    z=np.random.default_rng(20261025).standard_normal((N,m))
    expected=_path_contributions(e.remizov_problem,0.,T,N,m,Replay(z),p.forcing_bound,_Run())
    for engine in ('numpy','numba'):
        for block in (1,37,64):
            got=path_contributions(p,0.,T,N,m,Replay(z),_Run(),engine,block)
            np.testing.assert_allclose(got,expected,rtol=0,atol=1e-8)


@pytest.mark.parametrize('engine',['numpy','numba'])
def test_full_api_rng_moments_and_point_address(equations,compiled,engine):
    e=equations[17];p=prepare_all_methods_mc(e.metadata)
    cfg=MCConfig(M=73,chunk=32,dt=1/64,T=2.,seed=20261025)
    old=solve_monte_carlo(e.remizov_problem,[-.25,0.,.25],cfg,forcing_bound=p.forcing_bound)
    got=solve_monte_carlo_optimized(p,old.x,cfg,engine=engine,block_steps=37)
    assert old.success and got.success
    for field in ('f','stderr','ci95','ci_lower','ci_upper'):
        np.testing.assert_allclose(getattr(got,field),getattr(old,field),rtol=0,atol=1e-8)
    for field in ('n_path_steps','n_normal_draws','coefficient_calls','coefficient_values','point_spawn_keys','n_batches'):
        assert getattr(got,field)==getattr(old,field)
    again=solve_monte_carlo_optimized(p,[0.],cfg,engine=engine,block_steps=1)
    np.testing.assert_array_equal(got.f[1:2],again.f)
    np.testing.assert_array_equal(got.stderr[1:2],again.stderr)
    assert not got.total_error_certified and got.discretization_error_bound is None


@pytest.mark.parametrize('engine',['numpy','numba'])
def test_constant_discrete_analytical_oracle(compiled,engine):
    p=prepare_analytic_mc();T=8.;h=1/128;cfg=MCConfig(M=16,chunk=8,dt=h,T=T,seed=19)
    got=solve_monte_carlo_optimized(p,[0.],cfg,engine=engine)
    oracle=.5*h*(1/math.tanh(h))*(-math.expm1(-2*T))
    assert got.success
    np.testing.assert_allclose(got.f,[oracle],rtol=0,atol=1e-13)
    assert got.stderr[0]==0 # genuinely deterministic path integrals, M>=2
    assert abs(got.f[0]-.5)<2e-5
    assert got.tail_error_bound>=.5*math.exp(-2*T)


@pytest.mark.parametrize('engine',['numpy','numba'])
def test_gaussian_analytical_discrete_expectation(compiled,engine):
    p=prepare_analytic_mc('gaussian');T=4.;h=1/32;N=128
    cfg=MCConfig(M=8192,chunk=1024,dt=h,T=T,seed=2026102511)
    got=solve_monte_carlo_optimized(p,[0.],cfg,engine=engine)
    t=np.arange(N+1)*h
    integrand=np.exp(-2*t)*4*(1+2*t)/(1+4*t)**1.5
    oracle=h*(.5*(integrand[0]+integrand[-1])+sum(integrand[1:-1]))
    assert got.success
    assert abs(got.f[0]-oracle)<=5*got.stderr[0]
    assert abs(got.f[0]-1.)<=5*got.stderr[0]+abs(oracle-1.)


@pytest.mark.parametrize('engine',['numpy','numba'])
def test_failures_work_budget_and_certificates(compiled,engine):
    p=prepare_analytic_mc()
    cfg=MCConfig(M=4,dt=.1,T=1.,seed=13)
    limits=solve_monte_carlo_optimized(p,[0.],replace(cfg,max_path_steps=1),engine=engine)
    assert not limits.success and limits.failure_reason=='work_limit_exceeded' and limits.f.size==0
    invalid=replace(p,problem=replace(p.problem,a_lower_bound=0.))
    got=solve_monte_carlo_optimized(invalid,[0.],cfg,engine=engine)
    assert got.failure_reason=='invalid_remizov_conditions'
    wrong=replace(p,forcing_bound=MCForcingBound(.1,'deliberately invalid test certificate'))
    got=solve_monte_carlo_optimized(wrong,[0.],cfg,engine=engine)
    assert got.failure_reason=='certified_bound_violation' and not got.f.size
    tiny=solve_monte_carlo_optimized(p,[0.],replace(cfg,max_runtime_seconds=1e-12),engine=engine)
    assert tiny.failure_reason=='time_limit_exceeded'
    zero=solve_monte_carlo_optimized(p,[0.],replace(cfg,T=0.),engine=engine)
    assert zero.success and zero.f[0]==0 and zero.n_path_steps==0
    under=solve_monte_carlo_optimized(p,[0.],replace(cfg,T=np.nextafter(0.,1.)),engine=engine)
    assert under.failure_reason=='numerical_underflow'
    for target in ([np.nan],[np.inf],[1j],[True],[0.,0.]):
        with pytest.raises(ValueError):solve_monte_carlo_optimized(p,target,cfg,engine=engine)
    with pytest.raises(ValueError):solve_monte_carlo_optimized(p,[0.],replace(cfg,M=1),engine=engine)


def test_bad_metadata_and_kernel_nonfinite(equations,compiled):
    import copy
    meta=copy.deepcopy(equations[0].metadata);meta['a_lower_cert']=-1.
    with pytest.raises(ValueError):prepare_all_methods_mc(meta)
    p=prepare_analytic_mc()
    with pytest.raises(_MCFailure,match='trajectory|invalid|rejected'):
        path_contributions(p,0.,1.,1,2,Replay(np.array([[np.nan,np.nan]])),_Run(),'numba')


def test_numpy_backend_needs_no_compiler(monkeypatch):
    import mc_optimized
    def forbidden():raise AssertionError('NumPy backend must not import/compile Numba')
    monkeypatch.setattr(mc_optimized,'compile_backend',forbidden)
    p=prepare_analytic_mc();cfg=MCConfig(M=4,dt=.1,T=.2,seed=51)
    got=solve_monte_carlo_optimized(p,[0.],cfg,engine='numpy')
    assert got.success and got.compilation_seconds==0
    for x in ([1j],[True],[np.nan]):
        with pytest.raises(ValueError):fused_values(p,x)


def test_auto_horizon_preserves_certification(equations,compiled):
    e=equations[2];p=prepare_all_methods_mc(e.metadata)
    cfg=MCConfig(M=16,chunk=16,dt=1/32,T=None,seed=20261025)
    old=solve_monte_carlo(e.remizov_problem,[0.],cfg,forcing_bound=p.forcing_bound)
    got=solve_monte_carlo_optimized(p,[0.],cfg)
    assert old.success and got.success and got.tail_tolerance_met
    assert (got.T,got.tail_error_bound,got.n_steps)==(old.T,old.tail_error_bound,old.n_steps)
    np.testing.assert_allclose(got.f,old.f,rtol=0,atol=1e-8)


def test_physical_diffusion_and_declared_bounds_are_checked(equations,compiled):
    p=prepare_all_methods_mc(equations[2].metadata);params=p.parameters.copy();params[0]=-10.
    params.setflags(write=False);cfg=MCConfig(M=4,T=.1,dt=.1,seed=7)
    # Deliberate corruption, to exercise runtime defences behind the safe factory.
    invalid=replace(p,parameters=params)
    for engine in ('numpy','numba'):
        got=solve_monte_carlo_optimized(invalid,[0.],cfg,engine=engine)
        assert got.failure_reason=='nonpositive_diffusion' and not got.f.size
    p=prepare_analytic_mc();invalid=replace(p,problem=replace(p.problem,a_lower_bound=.1,a_upper_bound=.5))
    got=solve_monte_carlo_optimized(invalid,[0.],cfg)
    assert got.failure_reason=='certified_bound_violation' and not got.f.size
