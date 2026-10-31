"""Frozen R0 scope and warm timing tests; no numerical solver calls."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from diagnostics.remizov_r0 import CONTRACT, measured_r_call, scope_jobs


def frozen():
    contract=json.loads((CONTRACT/'plan.json').read_text())
    jobs=[json.loads(s) for s in (CONTRACT/'pilot_jobs.jsonl').read_text().splitlines()]
    return contract,jobs


def test_frozen_scope_exactly_38_unique_jobs():
    contract,jobs=frozen();selected=scope_jobs(contract,jobs)
    assert len(selected)==38 and len({j['equation_id'] for j in selected})==35
    assert sum(j['profile']=='R128' for j in selected)==3
    assert sum(j['profile']=='R256' for j in selected)==35
    assert all(j['stage']==0 and j['episode']==0 for j in selected)


@pytest.mark.parametrize('change',[
    lambda j:j['config'].update(n=64),
    lambda j:j.update(expected_T=j['expected_T']+1),
    lambda j:j.update(target_x=[0.]),
    lambda j:j.update(split='validation'),
])
def test_frozen_scope_rejects_mutation(change):
    contract,jobs=frozen();stage0=next(j for j in jobs if j['stage']==0)
    change(stage0)
    with pytest.raises(AssertionError):scope_jobs(contract,jobs)


def test_warm_timer_includes_G_T_config_constructor_and_solve():
    calls=[];clock=iter((100,201))
    def tick():calls.append('clock');return next(clock)
    def certify(*_):calls.append('G');return SimpleNamespace(forcing_bound=SimpleNamespace(abs_upper_bound=3.))
    def auto_t(*_):calls.append('T');return 2.
    def cfg(**_):calls.append('config');return object()
    class Solver:
        def __init__(self,*_):calls.append('construct')
        def solve(self,_):calls.append('solve');return 'result'
    e=SimpleNamespace(remizov_problem=SimpleNamespace(lambda_value=1.,c_abs_upper_bound=1.),metadata={})
    job=dict(forcing_G=3.,expected_T=2.,config=dict(T=2.),target_x=[0.])
    _,T,z,wall,start,end=measured_r_call(e,job,certify,auto_t,cfg,Solver,tick)
    assert calls==['clock','G','T','config','construct','solve','clock']
    assert (T,z,start,end,wall)==(2.,'result',100,201,101e-9)


def test_warm_timer_records_exception_interval():
    def fail(*_):raise RuntimeError('failure')
    with pytest.raises(RuntimeError) as caught:
        measured_r_call(SimpleNamespace(remizov_problem=object(),metadata={}),{},fail,None,None,None,iter((2,8)).__next__)
    assert caught.value.r0_timer==(2,8)
