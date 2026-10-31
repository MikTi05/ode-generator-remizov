"""Independent statistical oracles and synthetic outcomes; no ODE solver."""
import math

import pytest

from diagnostics.report_joint_benchmark_pilot import cp_bounds,holm,log_time_ratio,compare


def test_cp_extremes_match_exact_binomial_formulas():
    for n in (4,8,59):
        lo,hi=cp_bounds(n,n)
        assert lo==pytest.approx(.05**(1/n)) and hi==1.
        lo,hi=cp_bounds(0,n)
        assert lo==0. and hi==pytest.approx(1-.05**(1/n))
    assert cp_bounds(8,8)[0]<.95<cp_bounds(59,59)[0]


def test_holm_preserves_original_order_and_monotonicity():
    assert holm([.04,.001,.02])==pytest.approx([.04,.003,.04])
    assert holm([1.,0.,.9])==pytest.approx([1.,0.,1.])


def test_log_time_ratio_does_not_claim_more_samples_than_available():
    assert log_time_ratio([1.],[2.,3.]) is None
    assert log_time_ratio([0.,1.],[2.,3.]) is None
    v=log_time_ratio([1.,1.,1.],[2.,2.,2.])
    assert v['ratio']==pytest.approx(.5) and v['lower']==pytest.approx(v['upper'])


def profile(method,**kwargs):
    p=dict(mode='point',problem_id='synthetic',method=method,complexity='L1',regime='constant',
        T=10.,max_error=.001,bitwise_reproducible=True,accuracy_state='qualified_observed',
        median_response_seconds=10.,median_solver_seconds=9.,max_rss_mib=80,
        accuracy_successes=8,requested_repeats=8,median_error=.001,rmse_E=.001,
        max_abs_mean_bias=.001,bias_ci_state='unresolved',median_max_sampling_half=.02,
        median_max_simultaneous_sampling_half=.02,reliability_state='unresolved',
        p_lower=.687,p_upper=1.,below_target_holm_p=1.)
    p.update(kwargs);return p


def test_fast_eight_of_eight_does_not_become_qualified_mc_winner():
    profiles=[profile('solve'),profile('monte_carlo',median_response_seconds=.1)]
    rows=[dict(mode='point',problem_id='synthetic',method=m,status='completed',worker_response_seconds=t)
          for m,t in [('solve',10.),('solve',11.),('monte_carlo',.1),('monte_carlo',.11)]]
    r=compare(profiles,rows)[0]
    assert r['winner']=='unknown' and r['winner_reason']=='accuracy_unresolved'
    assert r['training_eligible'] is False


def test_only_qualified_label_is_accuracy_not_speed_claim():
    profiles=[profile('solve'),profile('monte_carlo',reliability_state='below_target',below_target_holm_p=.001)]
    r=compare(profiles,[])[0]
    assert r['winner']=='remizov' and r['winner_reason']=='only_accuracy_qualified'
    assert r['winner_survives_holm26'] is True
    assert r['descriptive_mc_over_r_time_ratio'] is None


def test_missing_r_profile_does_not_become_neither():
    profiles=[profile('solve',accuracy_state='infrastructure_unresolved'),
              profile('monte_carlo',reliability_state='below_target')]
    assert compare(profiles,[])[0]['winner']=='unknown'
