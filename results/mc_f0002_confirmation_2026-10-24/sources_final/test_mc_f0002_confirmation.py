"""Independent phase policy and prospective grid statistical cost."""
import math

import pytest
from scipy.stats import norm

from diagnostics.report_mc_f0002_confirmation import gate_from_summary, statistical_cost


def summary(mean=.02,half=.01,variance=2.,repeats=8):
    return dict(repeats=repeats,differences=[dict(mean=mean,ci95_half=half,
        variance_reduction_vs_independent=variance)])


def test_gate_keeps_original_inclusive_thresholds_and_eight_new_repeats():
    rule=dict(required_repeats=8,maximum_relative_ci95_half=.5,minimum_variance_reduction=2.)
    assert gate_from_summary(summary(),rule)['proceed']
    assert not gate_from_summary(summary(half=.01001),rule)['proceed']
    assert not gate_from_summary(summary(variance=1.99),rule)['proceed']
    assert not gate_from_summary(summary(repeats=4),rule)['proceed']
    assert not gate_from_summary(summary(mean=0),rule)['proceed']


def test_grid_cost_distinguishes_pointwise_and_simultaneous_coverage():
    one=statistical_cost(.8,1/128,63.625,.005,reference_seconds=4.)
    pointwise=statistical_cost(.8,1/128,63.625,.005,points=33,reference_seconds=4.)
    simultaneous=statistical_cost(.8,1/128,63.625,.005,points=33,simultaneous=True,reference_seconds=4.)
    assert one['M_per_point']==math.ceil((1.96*.8/.005)**2)
    assert pointwise['path_steps']==33*one['path_steps']
    assert simultaneous['M_per_point']>pointwise['M_per_point']
    assert 33*2*norm.sf(simultaneous['z'])==pytest.approx(.05)
    assert one['serial_seconds_at_chunk_1024']==4*math.ceil(one['M_per_point']/1024)
    assert one['time_steps']==8144


def test_halving_statistical_budget_quadruples_required_paths_up_to_rounding():
    a=statistical_cost(.77,1/64,63.625,.01)
    b=statistical_cost(.77,1/64,63.625,.005)
    assert 4*a['M_per_point']-3<=b['M_per_point']<=4*a['M_per_point']
    assert b['path_steps']==4072*b['M_per_point']


@pytest.mark.parametrize('changes',[dict(sigma=0),dict(h=0),dict(T=-1),dict(halfwidth=0),dict(points=0)])
def test_invalid_planning_inputs_are_not_reported_as_costs(changes):
    cfg=dict(sigma=.8,h=1/64,T=63.625,halfwidth=.005);cfg.update(changes)
    with pytest.raises(ValueError):statistical_cost(**cfg)
