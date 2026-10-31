"""Pure cost/summary fixtures; no solver or fitted ML data."""
import pytest
from diagnostics.report_switch5_r_timing import aggregate,times


def test_cost_of_shared_inaccurate_case_is_charged_without_creating_delivery():
    cases=[dict(R128=dict(accurate=True,timing=dict(median=10.)),R256=dict(accurate=True,timing=dict(median=20.))),
           dict(R128=dict(accurate=False,timing=dict(median=12.)),R256=dict(accurate=False,timing=dict(median=24.)))]
    a=aggregate(cases)
    assert a['all5']['R128_median_cost']==22. and a['all5']['accurate_deliveries']==1
    assert a['all5']['method_seconds_per_accurate']==22. and a['both_accurate']['R128_median_cost']==10.
    assert a['all5']['lost_baseline_deliveries']==0


def test_incomplete_global_denominator_never_becomes_measured_percentage():
    c=dict(R128=dict(accurate=True,timing=dict(median=10.)),R256=dict(accurate=True,timing=dict(median=20.)))
    a=aggregate([c]);assert a['global40']['measured_percent_saving'] is None
    assert a['global40']['attributable_switch_cost_delta_seconds']==10.


def test_four_timings_preserve_large_observation_and_MAD_instability():
    t=times([1.,2.,3.,100.])
    assert t['raw']==[1.,2.,3.,100.] and t['max']==100.
    assert t['median']==2.5 and t['MAD_fraction']>.2
