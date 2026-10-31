"""Analytical oracles, independent symbolic derivatives, and v1 certificate wiring."""
from copy import deepcopy
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import sympy as sp

from all_methods_bounds import (BoundCertificationError, DerivativeBounds,
    _add, _divide, _mul, _sqrt_upper, bump_bounds, certify_all_methods_forcing,
    expression_bounds, forcing_bound_from_solution_bounds, solution_bounds)
from all_methods_dataset_loader import load_all_methods_equations
from mc_solver import MCConfig, auto_T, solve_monte_carlo
from src.all_methods_expressions import SafeExpressionNode, normalized_bump_jet2
from test_all_methods_expressions import _representative_roots, _sympy_expression

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def equations():
    p = ROOT/'data/generated'
    return load_all_methods_equations(p/'all_methods_equations_v1.csv', p/'all_methods_equations_v1_metadata.jsonl')


@pytest.mark.parametrize('node', _representative_roots(), ids=lambda n:n.op)
def test_all_whitelisted_jet_envelopes_against_independent_sympy(node):
    x = sp.Symbol('x', real=True)
    expr = _sympy_expression(node, x)
    bounds = expression_bounds(node)
    points = np.linspace(-15., 15., 4001)
    for order, upper in enumerate((bounds.value, bounds.d1, bounds.d2)):
        f = sp.lambdify(x, sp.diff(expr, x, order), modules='numpy')
        assert np.max(np.abs(np.asarray(f(points)))) <= upper + 1e-13


def test_primitive_sharp_or_known_analytical_limits():
    node = SafeExpressionNode('sin_affine', params={'omega':-3.,'phi':.1})
    assert expression_bounds(node) == DerivativeBounds(1.,3.,9.)
    gaussian = expression_bounds(SafeExpressionNode('gaussian',params={'alpha':4.,'mu':-10.}))
    assert gaussian.value == 1 and gaussian.d2 == 8
    assert gaussian.d1 >= float(sp.sqrt(8/sp.E))
    # Exact stationary values for the two less familiar primitives.
    assert 48/(25*sp.sqrt(5)) < 1                  # saturating second derivative
    assert (3+2*sp.sqrt(2))/4 < sp.Rational(3,2)  # rational_saturate second derivative
    assert 4/(3*sp.sqrt(3)) < 1                  # tanh second derivative
    zero = SafeExpressionNode('constant',params={'value':0.})
    assert expression_bounds(SafeExpressionNode('sin',(zero,))) == DerivativeBounds(0.,0.,0.)


def test_bump_algebra_and_rational_global_constant():
    z,u = sp.symbols('z u', real=True)
    phi = sp.exp(1-1/(1-z*z))
    polynomial = sp.cancel(sp.diff(phi,z,2)/phi).subs(z*z,1-1/u)
    assert sp.simplify(polynomial-(4*u**4-12*u**3+6*u**2)) == 0
    # e >= 8/3 and exp(1/2)>=3/2 give a rational majorant of the positive part.
    positive = sp.Rational(8,3)*(256*sp.Rational(3,8)**4+81*sp.Rational(3,8)**3+9*sp.Rational(3,8)**2)
    assert positive == sp.Rational(1809,64)
    assert 24*sp.Rational(3,8) == 9 < positive
    b = bump_bounds(2.)
    assert b == DerivativeBounds(1.,1.5,1809/256)
    points = np.linspace(-2.1,2.1,8001)
    jet = normalized_bump_jet2(points,0.,2.)
    for values, upper in zip((jet.value,jet.d1,jet.d2),(b.value,b.d1,b.d2)):
        assert np.max(abs(values)) <= upper


def test_upward_arithmetic_including_subnormals_and_overflow():
    assert Fraction(_add(.1,.2)) >= Fraction(.1)+Fraction(.2)
    assert Fraction(_mul(.1,.2)) >= Fraction(.1)*Fraction(.2)
    assert Fraction(_divide(1.,3.)) >= Fraction(1,3)
    assert Fraction(_divide(1.,.1,2)) >= Fraction(1)/Fraction(.1)**2
    assert Fraction(_sqrt_upper(.1))**2 >= Fraction(.1)
    assert _mul(math.nextafter(0.,1.), .5) == math.nextafter(0.,1.)
    assert _mul(0.,1e308) == 0
    with pytest.raises(BoundCertificationError, match='float64'):
        _mul(1e308,2.)


def test_constant_and_gaussian_exact_forcing_norms():
    c = forcing_bound_from_solution_bounds(DerivativeBounds(.5,0.,0.),
        a_upper=1.,b_abs=0.,c_abs=0.,lambda_value=2.)
    assert c.abs_upper_bound == 1.  # exact forcing norm for y=1/2
    y = expression_bounds(SafeExpressionNode('gaussian',params={'alpha':1.,'mu':0.}))
    g = forcing_bound_from_solution_bounds(y,a_upper=1.,b_abs=0.,c_abs=0.,lambda_value=2.)
    assert g.abs_upper_bound == 4.  # exact norm of (4-4x^2)exp(-x^2)
    assert auto_T(2.,0.,g.abs_upper_bound,1e-5) == pytest.approx(math.log(2e5)/2)


def test_horizon_and_tail_round_up_against_high_precision_oracle():
    from mc_solver import _gap_lower, _tail_bound
    for lam,C,G,eps in [(1.,2.**-54,100.,1e-4), (2.,0.,4.,1e-320),
                        (1.,0.,1.,1.), (2.,0.,1.,.5)]:
        T = auto_T(lam,C,G,eps)
        gap = _gap_lower(lam,C)
        tail = _tail_bound(G,gap,T)
        with localcontext() as ctx:
            ctx.prec = 200
            l,c,g,e,t = map(Decimal.from_float,(lam,C,G,eps,T))
            exact_gap = l-c
            required = max(Decimal(0),(g/(exact_gap*e)).ln()/exact_gap)
            reference_tail = g/exact_gap * (-exact_gap*t).exp()
            assert Decimal.from_float(T) >= required
            assert Decimal.from_float(tail) >= reference_tail
            assert Decimal.from_float(tail) <= e
    assert 0 < _tail_bound(1.,1e308,1e308) == math.nextafter(0.,1.)


def test_provably_conservative_cancellation_example():
    sine = SafeExpressionNode('sin_affine',params={'omega':1.,'phi':0.})
    eta = 2.**-40
    unreduced = SafeExpressionNode('lincomb',(sine,sine),{'alpha':1.,'beta':-(1-eta)})
    # eta^-1*(sin(x)-(1-eta)sin(x)) == sin(x) identically.
    reduced_y = solution_bounds(expression_bounds(sine),1.,2.)
    expanded_y = solution_bounds(expression_bounds(unreduced),1/eta,2.)
    args = dict(a_upper=1.,b_abs=0.,c_abs=0.,lambda_value=2.)
    tight = forcing_bound_from_solution_bounds(reduced_y,**args).abs_upper_bound
    loose = forcing_bound_from_solution_bounds(expanded_y,**args).abs_upper_bound
    assert loose/tight > 1e12  # certified lower bound on conservatism: true norm <= tight


def test_certification_never_evaluates_or_samples_functions(equations, monkeypatch):
    import all_methods_dataset_loader as loader
    def forbidden(*a,**kw):
        raise AssertionError('sampling/evaluation is not part of the proof')
    for name in ('evaluate_expression','evaluate_solution'):
        monkeypatch.setattr(loader,name,forbidden)
    monkeypatch.setattr(np,'linspace',forbidden)
    e = equations[0]
    c = certify_all_methods_forcing(e.remizov_problem,e.metadata)
    mutated = deepcopy(e.metadata)
    mutated['validation'] = {'sampled_ranges':{'g':{'min':-1e300,'max':1e300}}}
    assert certify_all_methods_forcing(e.remizov_problem,mutated) == c


def test_all_twenty_global_bounds_and_automatic_MC_preflight(equations):
    for e in equations:
        p = e.remizov_problem
        c = certify_all_methods_forcing(p,e.metadata)
        x = np.linspace(c.support[0]-.1,c.support[1]+.1,4097)
        for func,upper in [(e.y_exact,c.solution_bounds.value),
                           (e.dy_exact,c.solution_bounds.d1),
                           (e.d2y_exact,c.solution_bounds.d2),
                           (p.g_remizov,c.forcing_bound.abs_upper_bound)]:
            assert np.max(abs(func(x))) <= upper
        assert c.forcing_bound.bounds_domain == 'all_real_x'
        assert len(c.serialized_ast_sha256) == 64
        json.dumps(asdict(c),allow_nan=False)
        # No trajectories: force refusal after automatic T, before sampling.
        r = solve_monte_carlo(p,[-1.,0.,1.],MCConfig(M=2,max_path_steps=1),forcing_bound=c.forcing_bound)
        assert r.failure_reason == 'work_limit_exceeded'
        assert r.n_normal_draws == r.n_path_steps == 0
        assert r.T == auto_T(p.lambda_value,p.c_abs_upper_bound,c.forcing_bound.abs_upper_bound,1e-4)
        assert r.tail_error_bound <= 1e-4 and not r.total_error_certified


def test_mismatched_or_invalid_metadata_are_rejected(equations):
    e = equations[0]
    for key,value in [('equation_id','different'),('generation_version','unknown')]:
        m = deepcopy(e.metadata);m[key] = value
        with pytest.raises(ValueError):
            certify_all_methods_forcing(e.remizov_problem,m)
    m = deepcopy(e.metadata);m['serialized_ast']['y_shape']['op']='unrestricted_exp'
    with pytest.raises(ValueError):
        certify_all_methods_forcing(e.remizov_problem,m)
    with pytest.raises(BoundCertificationError, match='mismatch'):
        certify_all_methods_forcing(replace(e.remizov_problem,lambda_value=e.remizov_problem.lambda_value+1),e.metadata)
    for radius in (0.,-1.,math.inf):
        with pytest.raises(BoundCertificationError):
            bump_bounds(radius)


def test_cli_automatically_derives_G_before_resource_refusal():
    proc = subprocess.run([sys.executable,'-B',str(ROOT/'examples/solve_mc_equation.py'),
        '--M','2','--points','1','--max-path-steps','1'],cwd=ROOT,
        capture_output=True,text=True,timeout=30)
    assert proc.returncode == 1, proc.stderr
    data = json.loads(proc.stdout)
    assert data['result']['failure_reason'] == 'work_limit_exceeded'
    assert data['result']['n_normal_draws'] == 0
    assert data['forcing_certificate']['forcing_bound'] == data['forcing_bound']
    assert data['result']['tail_tolerance_met']


def test_derived_certificate_is_accepted_by_real_MC_paths(equations):
    e = equations[0]
    cert = certify_all_methods_forcing(e.remizov_problem,e.metadata)
    r = solve_monte_carlo(e.remizov_problem,[0.],MCConfig(M=16,dt=.1,seed=2488),
                          forcing_bound=cert.forcing_bound)
    assert r.success, r.message
    assert r.n_normal_draws == 16*r.n_steps and r.n_steps > 0
    assert r.tail_error_bound <= 1e-4 and r.tail_tolerance_met
    assert r.forcing_abs_upper_bound == cert.forcing_bound.abs_upper_bound
    assert np.all(np.isfinite(r.f)) and not r.total_error_certified
