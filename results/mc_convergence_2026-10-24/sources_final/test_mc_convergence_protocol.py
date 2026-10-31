"""One-factor isolation and independent closed-form study references."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from diagnostics.mc_convergence import configurations, step_cost
from diagnostics.report_mc_convergence import analytic_means, difference, ensemble, holm


def test_factor_series_keep_actual_step_and_other_parameters_fixed():
    for Tstar in (4.9517437763,8.4620836924,57.3241718063,63.5598715637):
        configs=configurations(Tstar)
        assert len({(c['M'],c['dt'],c['T']) for c in configs})==len(configs)
        for factor, changed in [('M','M'),('h','dt'),('T','T')]:
            series=[c for c in configs if any(v['factor']==factor for v in c['views'])]
            assert len(series)>=3
            for fixed in {'M','dt','T'}-{changed}:
                assert len({c[fixed] for c in series})==1
        assert max(c['T'] for c in configs)>=Tstar
        for c in configs:
            cost=step_cost(c['M'],c['dt'],c['T'])
            assert cost['actual_dt']==c['dt']
            assert cost['path_steps']==c['M']*cost['time_steps']<=5_000_000


@pytest.mark.parametrize('x',[0.,.7,1.5])
def test_gaussian_continuous_oracle_by_independent_normal_integration(x):
    # Integrate the normal transition density directly rather than reuse the
    # heat-kernel integrand in the report implementation.
    T=.5
    def heat_g(t):
        if t==0:
            return (4-4*x*x)*math.exp(-x*x)
        def f(z):
            q=x+math.sqrt(2*t)*z
            return (4-4*q*q)*math.exp(-q*q-z*z/2)/math.sqrt(2*math.pi)
        return quad(f,-12,12,epsabs=1e-11)[0]*math.exp(-2*t)
    oracle=analytic_means('MC_gaussian',1/32,T,x)
    assert quad(heat_g,0,T,epsabs=1e-10)[0]==pytest.approx(oracle['continuous_finite_mean'],abs=2e-10)


def test_constant_oracle_and_bias_order_are_deterministic():
    biases=[]
    for h in (1/8,1/16,1/32):
        o=analytic_means('MC_constant',h,8)
        # Independent geometric sum for the exponential trapezoid.
        geometric=(h/2)*(1+math.exp(-2*h))*(-math.expm1(-16))/(-math.expm1(-2*h))
        assert o['discrete_mean']==pytest.approx(geometric,abs=1e-15)
        biases.append(o['time_bias'])
    assert 1.99<math.log2(biases[1]/biases[2])<2.01


def test_ensemble_uses_seed_scatter_and_welch_difference():
    a=ensemble([1.,2.,3.,4.],[.1]*4,2.)
    assert a['mean']==2.5
    assert a['rmse']==pytest.approx(math.sqrt(1.5))
    assert a['replicate_sd']==pytest.approx(np.std([1.,2.,3.,4.],ddof=1))
    # Across-run uncertainty must not be replaced by reported within-run SE.
    assert a['mean_ci95_half']>1
    b=ensemble([11.,12.,13.,14.],[.1]*4,2.)
    d=difference(b,a)
    assert d['difference']==10 and d['excludes_zero'] and d['lower']>0


def test_holm_retains_the_full_comparison_family():
    rows=[dict(pvalue=p) for p in (.01,.03,.04,.8)]
    holm(rows)
    assert [r['holm_pvalue'] for r in rows]==pytest.approx([.04,.09,.09,.8])
    assert [r['holm_significant_005'] for r in rows]==[True,False,False,False]
