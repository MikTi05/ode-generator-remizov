"""P1 scope, timing isolation and frozen gates; no trajectories or model fit."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from diagnostics.mc_sentinel_p1 import scope_jobs,measured_call
from diagnostics.report_mc_sentinel_p1 import choose_profile

ROOT=Path(__file__).resolve().parents[1]


def inputs():
    p=ROOT/'reports/ml_selector_contract'
    return json.loads((p/'plan.json').read_text()),[json.loads(s) for s in (p/'pilot_jobs.jsonl').read_text().splitlines()]


def test_scope_preserves_exact60_frozen_jobs_and_seeds():
    p,j=inputs();selected=scope_jobs(p,j)
    assert len(selected)==60 and all(x['stage']==1 and x['profile'].startswith('MC') for x in selected)
    assert {x['key'] for x in selected}=={x['key'] for x in j if x['stage']==1}
    assert scope_jobs(p,list(reversed(j)))==selected
    for pid in p['pilot']['sentinels']:
        for e in range(4):assert {x['profile'] for x in selected if x['equation_id']==pid and x['episode']==e}=={'MC0','MC1','MC2'}


@pytest.mark.parametrize('field,value',[('split','test'),('profile','MC3'),('target_x',[0.])])
def test_bad_scope_rejected(field,value):
    p,j=inputs();j=copy.deepcopy(j);item=next(x for x in j if x['stage']==1);item[field]=value
    with pytest.raises((AssertionError,KeyError,IndexError)):scope_jobs(p,j)


def test_wrong_seed_and_M_rejected():
    p,j=inputs()
    for field,value in [('seed',2488),('M',16384)]:
        altered=copy.deepcopy(j);item=next(x for x in altered if x['stage']==1);item['config'][field]=value
        with pytest.raises(AssertionError):scope_jobs(p,altered)


def test_timing_includes_factory_and_config_before_solver():
    events=[];p,j=inputs();job=scope_jobs(p,j)[0]
    def clock():events.append('clock');return 1000000000 if len(events)==1 else 4000000000
    def prepare(m):events.append('AST_G');return 'prepared'
    def config(**kwargs):events.append('config');return 'config'
    def solve(p,x,c,**kwargs):
        events.append('solver');assert p=='prepared' and c=='config' and len(x)==33
        return 'f33'
    prepared,z,wall,t0,t1=measured_call({},job,prepare,config,solve,clock)
    assert events==['clock','AST_G','config','solver','clock']
    assert wall==3. and z=='f33' and t1-t0==3000000000


def test_exception_preserves_measured_interval_without_extra_calls():
    p,j=inputs();job=scope_jobs(p,j)[0];ticks=iter([10,30]);calls=[]
    def failed(*args,**kwargs):calls.append('solve');raise RuntimeError('expected mock failure')
    with pytest.raises(RuntimeError) as error:
        measured_call({},job,lambda m:None,lambda **k:None,failed,lambda:next(ticks))
    assert error.value.p1_timer==(10,30) and calls==['solve']


def profiles():
    return {f'MC{i}':dict(failures=0,accuracy_pass=s,wall=dict(median=t),error=dict(median=e))
            for i,s,t,e in [(0,0,1.,.04),(1,5,4.,.02),(2,6,8.,.019)]}


def test_cost_gate_inclusive8_and_success_ranking():
    p,_=inputs();v=profiles();r=choose_profile(v,p['gates'],True)
    assert r['selected']=='MC2' and r['profiles']['MC2']['eligible']
    v['MC2']['wall']['median']=8.001;r=choose_profile(v,p['gates'],True)
    assert r['selected']=='MC1' and r['profiles']['MC2']['reasons']==['cost_ratio_gt8']


def test_futility_and_allzero_never_choose_nextstage():
    p,_=inputs();v=profiles()
    for r in v.values():r['accuracy_pass']=0
    v['MC1']['error']['median']=.032
    r=choose_profile(v,p['gates'],True)
    assert r['selected'] is None and r['status']=='all_zero_stop_before_stage2'
    assert 'zero_success_and_error_ratio_ge_0.8' in r['profiles']['MC1']['reasons']
    assert choose_profile(profiles(),p['gates'],False)['selected'] is None


def test_tie_priority_and_failures_are_not_redefined():
    p,_=inputs();v=profiles();v['MC1']['accuracy_pass']=v['MC2']['accuracy_pass']=5
    v['MC2']['wall']['median']=4.1
    assert choose_profile(v,p['gates'],True)['selected']=='MC1'
    v['MC1']['failures']=2
    assert choose_profile(v,p['gates'],True)['selected']=='MC2'
