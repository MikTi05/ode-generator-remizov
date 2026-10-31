"""Survey protocol/outcome logic, without running the numerical study."""
import pytest

from diagnostics.plan_chernov_dataset20 import build_plan
from diagnostics.report_chernov_dataset20 import accuracy_status, summarize


def test_plan_covers_every_equation_once_per_method_without_individual_tuning():
    ids=[f'AMV1_F{i:04d}_S00' for i in range(20)]
    plan=build_plan(ids)
    assert len(plan['attempts'])==40
    assert {(s['problem_id'],s['method']) for s in plan['attempts']}=={
        (pid,method) for pid in ids for method in ('solve','streaming')}
    assert all(s['config']==plan['attempts'][0]['config'] for s in plan['attempts'])
    assert plan['resource_policy']['automatic_retries']==0
    with pytest.raises(ValueError):build_plan(ids[:-1])
    with pytest.raises(ValueError):build_plan(ids[:-1]+ids[:1])


@pytest.mark.parametrize('status,error,expected',[
    ('success',.01,'accuracy_pass'),('success',.010001,'accuracy_not_met'),
    ('success',None,'invalid_error_metric'),('success',float('nan'),'invalid_error_metric'),
    ('external_time_limit',None,'external_time_limit'),
    ('nonfinite_state',None,'nonfinite_state')])
def test_outcomes_do_not_confuse_computational_and_accuracy_success(status,error,expected):
    assert accuracy_status(status,error,.01)==expected


def test_statistics_include_failed_attempt_wall_time_but_no_invented_solver_time():
    rows=[dict(method='solve',computation_status='success',status='accuracy_pass',
        solver_seconds=2.,worker_seconds=3.,peak_rss_mib=90.),
        dict(method='solve',computation_status='success',status='accuracy_not_met',
        solver_seconds=4.,worker_seconds=5.,peak_rss_mib=100.),
        dict(method='solve',computation_status='external_time_limit',status='external_time_limit',
        solver_seconds=None,worker_seconds=60.,peak_rss_mib=80.)]
    s=summarize(rows)['solve']
    assert s['numerical_successes']==2 and s['accuracy_passes']==1
    assert s['accuracy_fraction']==1/3
    assert s['completed_solver_seconds']['mean']==3.
    assert s['all_attempt_wall_seconds']['total']==68.
