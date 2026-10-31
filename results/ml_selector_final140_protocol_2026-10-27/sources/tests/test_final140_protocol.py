"""Protocol/control tests only. Scratch fixtures are not experimental observations."""
import copy
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from diagnostics import run_final140 as run
from diagnostics import evaluate_final140 as evaluate
from diagnostics.final140_protocol import ROOT, SOURCE, REVIEW, validate_scope, schedule
from ml.revision_evidence import read, sha

OUT=ROOT/'results/ml_selector_final140_protocol_2026-10-27'


@pytest.fixture
def plan():return read(OUT/'plan.json')


def request(j,ph='test'):return dict(job=j,plan_sha256=ph,session='mock_session')


def budget(ph='test'):
    return dict(plan_sha256=ph,approved_cap_seconds=7464,maximum_physical_calls=140,
        started_epoch=100.,deadline_epoch=7564.,started_monotonic=200.,deadline_monotonic=7664.)


def test_fixed_scope_preserves_frozen_inputs_predictions_and_jobs(plan):
    validate_scope(plan)
    old=read(SOURCE/'plan.json');options=read(REVIEW/'options.json')['options']
    assert plan['cases']==old['cases'] and plan['MC_equation_ids']==options[0]['equation_ids']
    assert Counter(j['profile'] for j in plan['jobs'])=={'R128':60,'R256':60,'MC1':20}
    assert all(j in old['jobs'] and j['episode']==0 for j in plan['jobs'])
    assert sha(OUT/'frozen_predictions.json')==sha(SOURCE/'frozen_predictions.json')
    assert sha(OUT/'independence.json')==sha(SOURCE/'independence.json')
    jobs,order=schedule(list(reversed(plan['cases'])),list(reversed(plan['jobs'])),set(plan['MC_equation_ids']))
    assert jobs==plan['jobs'] and order==plan['measurement_order']
    assert sum(r['profiles'].index('R128')<r['profiles'].index('R256') for r in order)==30


@pytest.mark.parametrize('mutation',['profile','seed','equation','cap','duplicate','MCsubset'])
def test_scope_tampering_rejected(plan,mutation):
    if mutation=='profile':plan['jobs'][0]['config']['n']=999
    elif mutation=='seed':next(j for j in plan['jobs'] if j['profile']=='MC1')['config']['seed']+=1
    elif mutation=='equation':plan['cases'][0]['expression_hash']='bad'
    elif mutation=='cap':plan['resources']['series_seconds']+=1
    elif mutation=='duplicate':plan['jobs'][-1]=plan['jobs'][0]
    else:plan['MC_equation_ids'][0]=plan['cases'][1]['equation_id']
    with pytest.raises(AssertionError):validate_scope(plan)


def test_no_approval_no_worker_no_solver(monkeypatch):
    def forbidden(*a,**k):pytest.fail('execution forbidden in protocol preparation')
    monkeypatch.setattr(run,'GuardedProcess',forbidden)
    monkeypatch.setattr(run,'run_authorized',forbidden)
    # Approval guard itself does not need a thermal sensor.
    monkeypatch.setattr(run,'preflight',lambda out: {})
    with pytest.raises(PermissionError,match='Separate start'):run.execute(OUT)
    assert not (OUT/'execution_authorization.json').exists() and not (OUT/'budget.json').exists()
    assert not list((OUT/'records').glob('*.json')) and not list((OUT/'events').glob('*.request.json'))


def test_persistent_resume_never_retries_reserved_slot(plan):
    j=plan['jobs'][0];r=request(j)
    state=run.resume_state(plan,{},[r],budget(),'test',101.,201.)
    assert state['orphan_keys']==[j['key']] and j['key'] not in state['pending_keys']
    assert len(state['pending_keys'])==139 and not state['expired']
    for epoch,mono in [(7564.,201.),(101.,7664.),(101.,199.)]:
        assert run.resume_state(plan,{},[r],budget(),'test',epoch,mono)['expired']
    b=budget();b['deadline_epoch']+=1
    with pytest.raises(ValueError,match='reset or extension'):run.resume_state(plan,{},[r],b,'test',101,201)
    with pytest.raises(ValueError,match='duplicate'):run.resume_state(plan,{},[r,r],budget(),'test',101,201)
    with pytest.raises(ValueError,match='mismatched'):run.resume_state(plan,{},[request(j,'wrong')],budget(),'test',101,201)
    with pytest.raises(ValueError,match='record/request'):
        run.resume_state(plan,{j['key']:dict(r,status='unknown_new_status')},[r],budget(),'test',101,201)
    state=run.resume_state(plan,{j['key']:dict(r,status='invalid_result')},[r],budget(),'test',101,201)
    assert state['critical']


def scratch(tmp_path,plan):
    for n in ('records','events','workers','sessions','reconciliations'):(tmp_path/n).mkdir()
    run.write_json(tmp_path/'plan.json',plan);return sha(tmp_path/'plan.json')


@pytest.mark.parametrize('footprint',['request','worker','session','active'])
def test_missing_original_budget_cannot_be_reset(tmp_path,plan,footprint):
    ph=scratch(tmp_path,plan)
    path={'request':'events/a.request.json','worker':'workers/state.json','session':'sessions/a.json','active':'active.json'}[footprint]
    run.write_json(tmp_path/path,request(plan['jobs'][0],ph) if footprint=='request' else {})
    with pytest.raises(ValueError,match='refusing deadline reset'):run.collect(tmp_path,plan)


def test_orphan_reconciliation_is_idempotent_unknown_not_failure(tmp_path,plan):
    ph=scratch(tmp_path,plan);j=plan['jobs'][0];req=request(j,ph);records={}
    assert run.reconcile_orphans(tmp_path,plan,records,[req])==[j['key']]
    first=(tmp_path/'records'/f"{j['key']}.json").read_bytes()
    assert records[j['key']]['status']=='interrupted_unknown' and records[j['key']]['accuracy_met'] is None
    assert run.reconcile_orphans(tmp_path,plan,records,[req])==[]
    assert (tmp_path/'records'/f"{j['key']}.json").read_bytes()==first
    state=run.resume_state(plan,records,[req],budget(ph),ph,101,201)
    assert not state['orphan_keys'] and j['key'] not in state['pending_keys']


def fake_controller(monkeypatch,tmp_path,plan):
    """No processes, guardian or actual numerical methods. Only dispatch writes."""
    plan['protected_sha256']={}
    ph=scratch(tmp_path,plan)
    monkeypatch.setattr(run,'validate_scope',lambda p:None)
    monkeypatch.setattr(run,'check_authorization',lambda out:{'mock':True})
    @contextmanager
    def locks(out):yield []
    monkeypatch.setattr(run,'controller_locks',locks)
    monkeypatch.setattr(run,'thermal',lambda:{'thermal_state':0})
    def new_control(out,session,lease_seconds):
        control=out/'control'/session;control.mkdir(parents=True)
        return dict(control_dir=str(control))
    monkeypatch.setattr(run,'new_control',new_control)
    monkeypatch.setattr(run,'finish_control',lambda *a:None)
    monkeypatch.setattr(run,'wait',lambda *a:(None,1.))
    dispatched=[]
    class Child:
        stop_reason=None
        def __init__(self,*a):self.stdin=self;self.alive=True
        def write(self,line):
            req=read(Path(json.loads(line)['request_path']));j=req['job'];dispatched.append(j['key'])
            run.write_json(tmp_path/'records'/f"{j['key']}.json",dict(req,status='completed',E_grid=0.,method_wall_seconds=1.,result={'failure_reason':''}),exclusive=True)
        def flush(self):pass
        def poll(self):return None if self.alive else 0
        def terminate(self):self.alive=False
        def wait(self,timeout):return 0
    monkeypatch.setattr(run,'GuardedProcess',Child)
    return ph,dispatched


def test_mock_dispatch_exactly140_and_second_resume_no_worker(tmp_path,plan,monkeypatch):
    ph,dispatched=fake_controller(monkeypatch,tmp_path,plan)
    run.run_authorized(tmp_path)
    assert dispatched==[j['key'] for j in plan['jobs']] and len(set(dispatched))==140
    original_budget=(tmp_path/'budget.json').read_bytes()
    def forbidden(*a,**k):pytest.fail('completed run must not start a worker')
    monkeypatch.setattr(run,'GuardedProcess',forbidden)
    run.run_authorized(tmp_path)
    assert (tmp_path/'budget.json').read_bytes()==original_budget
    assert read(tmp_path/'resume_summary.json')['reason']=='all_slots_accounted'


def test_expired_budget_no_worker_or_reset(tmp_path,plan,monkeypatch):
    ph,dispatched=fake_controller(monkeypatch,tmp_path,plan)
    run.write_json(tmp_path/'budget.json',budget(ph))
    original=(tmp_path/'budget.json').read_bytes()
    run.run_authorized(tmp_path)
    assert dispatched==[] and read(tmp_path/'resume_summary.json')['reason']=='original_budget_exhausted'
    assert (tmp_path/'budget.json').read_bytes()==original


def test_critical_saved_record_blocks_remaining_calls(tmp_path,plan,monkeypatch):
    ph,dispatched=fake_controller(monkeypatch,tmp_path,plan);j=plan['jobs'][0]
    run.write_json(tmp_path/'budget.json',budget(ph))
    req=request(j,ph);run.write_json(tmp_path/'events/mock.request.json',req)
    run.write_json(tmp_path/'records'/f"{j['key']}.json",dict(req,status='invalid_result'))
    run.run_authorized(tmp_path)
    assert not dispatched and read(tmp_path/'resume_summary.json')['reason']=='critical_saved_result_requires_review'


def test_resume_orphan_reserved_no_redispatch(tmp_path,plan,monkeypatch):
    ph,dispatched=fake_controller(monkeypatch,tmp_path,plan);j=plan['jobs'][0]
    now=run.time.time();mono=run.time.monotonic();b=budget(ph)
    b.update(started_epoch=now,deadline_epoch=now+7464,started_monotonic=mono,deadline_monotonic=mono+7464)
    run.write_json(tmp_path/'budget.json',b);run.write_json(tmp_path/'events/mock.request.json',request(j,ph))
    run.run_authorized(tmp_path)
    assert len(dispatched)==139 and j['key'] not in dispatched
    assert read(tmp_path/'records'/f"{j['key']}.json")['status']=='interrupted_unknown'


def observations(plan):
    # Synthetic fixtures exercise accounting; they never enter OUT/records.
    return {j['key']:dict(status='completed',correct=1,delivered=1,violation=0,error=0.,seconds=10. if j['profile']=='R128' else 20.,observed_spent_seconds=10.) for j in plan['jobs']}


def predictions():return {r['equation_id']:r for r in read(OUT/'frozen_predictions.json')['rows']}


def test_metrics_primary60_secondary20_missing_MC40_never_none(plan):
    r=evaluate.build_metrics(plan,predictions(),observations(plan))
    assert r['fixed_N_primary_complete'] and r['primary']['selector']['accurate_deliveries']==60
    assert r['primary']['selector']['total_warm_cost']>21*10+39*20
    assert r['primary']['constant_R128']['total_warm_cost']==600
    assert len(r['paired_comparisons'])==2
    assert r['MC_secondary']['known_outcomes']==20 and r['MC_secondary']['not_measured_families']==40
    assert not r['MC_secondary']['uncertainty']['within_cell_bootstrap_used']
    assert r['probability_scores']['MC1']['planned_families']==20
    unknown=[c for c in r['cases'] if c['case']['equation_id'] not in plan['MC_equation_ids']]
    assert len(unknown)==40 and all(c['full_three_profile_oracle_status']=='unknown_MC_not_measured' for c in unknown)
    assert all(c['profiles']['MC1']['correct'] is None for c in unknown)
    assert not r['MC_secondary']['conditional_MC_choice_quality_confirmed']


def test_missing_primary_not_fabricated_and_inaccurate_time_charged(plan):
    obs=observations(plan);mc=next(j for j in plan['jobs'] if j['profile']=='MC1');obs.pop(mc['key'])
    r=evaluate.build_metrics(plan,predictions(),obs,bootstrap=False)
    assert r['fixed_N_primary_complete'] and not r['MC_secondary']['complete'] and r['MC_secondary']['unknown_outcomes']==1
    obs=observations(plan);j=plan['jobs'][0];obs.pop(j['key'])
    r=evaluate.build_metrics(plan,predictions(),obs)
    assert not r['fixed_N_primary_complete'] and not r['paired_comparisons']
    assert r['primary'][f"constant_{j['profile']}"]['total_warm_cost'] is None
    assert r['primary'][f"constant_{j['profile']}"]['accuracy_bounds']==[59/60,1.]
    obs=observations(plan)
    for value in obs.values():value.update(correct=0,violation=1,error=1.)
    r=evaluate.build_metrics(plan,predictions(),obs,bootstrap=False)
    assert r['primary']['constant_R128']['total_warm_cost']==600 and r['primary']['selector']['finite_violations']==60
    assert r['primary']['selector']['cost_per_accurate'] is None


def test_decode_internal_resource_censor_remains_unknown(plan):
    j=plan['jobs'][0]
    r=dict(job=j,plan_sha256='test',status='solver_failure',result=dict(problem_id=j['equation_id'],T=j['expected_T'],success=False,failure_reason='work_limit_exceeded'),
        method_wall_seconds=1.,timer_start_ns=0,timer_stop_ns=1000000000,peak_rss_mib=1.,seed=j['config'].get('seed'),
        **{n:j[n] for n in ('equation_id','expression_hash','family_id','profile','config','split')})
    d=evaluate.decode(r,j,'test',np.zeros(33))
    assert d['status']=='resource_censored' and d['correct'] is None and d['observed_spent_seconds']==1. and d['seconds'] is None
    r['seed']='wrong'
    with pytest.raises(AssertionError):evaluate.decode(r,j,'test',np.zeros(33))


def test_decode_completed_R_recomputes_accuracy_and_checks_timer(plan):
    j=plan['jobs'][0];f=[0.]*33;f[0]=.01
    z=dict(problem_id=j['equation_id'],T=j['expected_T'],success=True,failure_reason='',x=j['target_x'],f=f,
        n=j['config']['n'],n_operator_applications=j['numerical_work'],n_spline_builds=j['numerical_work']+1,
        work_grid_points=2081,time_nodes=513,actual_dx=.0125,work_domain=[-13.,13.])
    r=dict(job=j,plan_sha256='test',status='completed',result=z,method_wall_seconds=1.,timer_start_ns=0,timer_stop_ns=1000000000,
        peak_rss_mib=1.,seed=j['config'].get('seed'),y_exact=[0.]*33,E_grid=.01,accuracy_met=True,solver_called=True,
        **{n:j[n] for n in ('equation_id','expression_hash','family_id','profile','config','split')})
    assert evaluate.decode(r,j,'test',np.zeros(33))['correct']==1
    r['E_grid']=.001
    with pytest.raises(AssertionError):evaluate.decode(r,j,'test',np.zeros(33))
    r['E_grid']=.01;r['timer_stop_ns']+=1
    with pytest.raises(AssertionError):evaluate.decode(r,j,'test',np.zeros(33))


def test_saved_only_empty_analysis_never_invents_outcomes():
    r=evaluate.analyze(OUT)
    assert r['status']=='not_executed' and not r['final_evaluation_complete'] and r['records']==r['requests']==0
