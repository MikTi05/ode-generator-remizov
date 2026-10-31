"""Statistical-label and feature isolation regressions; no ODE solves."""
import math
import numpy as np
from diagnostics.analyze_mc1000_x0 import seed_summary


def test_seed_aggregation_uses_between_and_within_variance():
    values=[.8,.9,1.1,1.2];se=[.02]*4
    s=seed_summary(values,se,1024,1.,1e-4,.001)
    assert s['mc_mean']==1. and s['mc_aggregate_error']==0.
    assert s['mc_pooled_stderr']>0 and s['mc_seed_t95']>s['mc_pooled_ci95']
    assert s['quality_label']=='uncertain'
    pooled=((1023)*sum(np.array(se)**2*1024)+1024*sum((np.array(values)-1)**2))/(4096-1)
    assert math.isclose(s['mc_pooled_stderr'],math.sqrt(pooled/4096),rel_tol=1e-14)


def test_winner_requires_separated_interval_and_margin():
    s=seed_summary([1.,1.001,.999,1.], [.01]*4,1024,1.,1e-4,.1)
    assert s['quality_label']=='mc_mean_closer_exploratory'
    s=seed_summary([1.1,1.101,1.099,1.1],[.01]*4,1024,1.,1e-4,.001)
    assert s['quality_label']=='remizov_closer_exploratory'
    assert s['mc_bias_t95_low']>0
    s=seed_summary([1.,1.,1.,1.],[.01]*4,1024,1.,1e-4,.001)
    assert s['quality_label']=='uncertain' # .0025 margin prevents tiny-error ordering


def test_missing_remizov_does_not_generate_comparative_label():
    s=seed_summary([1.1]*4,[.01]*4,1024,1.,1e-4,None)
    assert s['quality_label']=='uncertain'
