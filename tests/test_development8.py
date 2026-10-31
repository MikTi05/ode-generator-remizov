"""No solver calls: scope, frozen selection and physical-call limits."""
import copy
import json
from pathlib import Path
import pytest
from diagnostics.prepare_development8 import select_cases
from diagnostics.run_development8 import validate_scope

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ml_development8_2026-10-26'


def plan():return json.loads((OUT/'plan.json').read_text())


def test_registered_scope():
    p=plan();validate_scope(p)
    assert all(j['split']=='train' for j in p['jobs'])
    assert all(c['identity'].split('|')[1]==c['structure_signature'] for c in p['cases'])


@pytest.mark.parametrize('field,value',[('series_seconds',5461),('compute_workers',2),('rss_mib',513)])
def test_reject_unapproved_resources(field,value):
    p=plan();p['resources'][field]=value
    with pytest.raises(AssertionError):validate_scope(p)


def test_reject_profile_and_seed_drift():
    p=plan();j=next(j for j in p['jobs'] if j['profile']=='MC1');j['config']['seed']+=99
    with pytest.raises(AssertionError):validate_scope(p)
    p=plan();p['jobs'][0]['config']['dt']=1/64
    with pytest.raises(AssertionError):validate_scope(p)


def test_selection_is_order_invariant_and_outcome_blind():
    rules=json.loads((ROOT/'results/ml_selector_generalization_plan_2026-10-26/sampling_rules.json').read_text())
    audit=json.loads((OUT/'identity_preflight.json').read_text())
    rows=[json.loads(l) for l in (OUT/'input_pool/all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    expected=[c['equation_id'] for c in plan()['cases']]
    for order in (rows,list(reversed(rows))):
        for r in order:r['E_grid']=-100;r['runtime']=-1
        selected=select_cases(order,rules,audit['dataset_sha256'],set(audit['excluded_signatures']))
        assert [r['equation_id'] for r in selected]==expected
    with pytest.raises(ValueError):select_cases(rows,rules,audit['dataset_sha256'],{r['structure_signature'] for r in rows})


def test_immutable_seeds_and_exclusion():
    p=plan();audit=json.loads((OUT/'identity_preflight.json').read_text());excluded=set(audit['excluded_signatures'])
    assert not excluded & {c['structure_signature'] for c in p['cases']}
    assert not audit['unresolved_train_exposures']
    assert {j['config']['seed'] for j in p['jobs'] if j['profile']=='MC1'}==set(p['MC_seed_entropy'])
    for profile in ('R128','R256','MC1'):
        for case in p['cases']:
            assert {j['episode'] for j in p['jobs'] if j['profile']==profile and j['identity']==case['identity']}==set(range(4))


def test_saved_only_score_and_timing_formulas():
    from diagnostics.report_development8 import timing,proper_scores,interval,cluster_score_uncertainty
    assert timing([1,2,3,4])['median']==2.5
    assert timing([1,2,3,4])['MAD']==1
    assert proper_scores([.5,.5],[0,1])['brier']==.25
    assert interval(0,4)[0]==0 and interval(4,4)[1]==1
    # Four successes are insufficient evidence for new-seed p>=.95.
    assert interval(4,4)[0]<.5
    s=cluster_score_uncertainty([.5,.5],[[0]*4,[1]*4])
    assert s['brier']['mean']==.25 and s['brier']['family_SE']==0


def test_resume_never_redispatches_requested_keys(tmp_path,monkeypatch,capsys):
    from diagnostics import run_development8 as driver
    from diagnostics.prepare_development8 import write_json,digest
    p=plan();p['protected_sha256']={}
    for n in ('records','events'):(tmp_path/n).mkdir()
    write_json(tmp_path/'plan.json',p);ph=digest(tmp_path/'plan.json')
    for j in p['jobs']:write_json(tmp_path/'events'/f"{j['key']}.request.json",dict(job=j,plan_sha256=ph))
    def forbidden(*a,**kw):raise AssertionError('resume must not acquire compute lock or spawn')
    monkeypatch.setattr(driver,'controller_locks',forbidden)
    monkeypatch.setattr(driver,'GuardedProcess',forbidden)
    driver.run(tmp_path)
    result=json.loads(capsys.readouterr().out)
    assert result['new_calls']==0 and result['worker_started'] is False
    assert result['reason']=='all_missing_keys_already_requested'


def test_completed_resume_starts_no_worker(tmp_path,monkeypatch,capsys):
    from diagnostics import run_development8 as driver
    from diagnostics.prepare_development8 import write_json,digest
    p=plan();p['protected_sha256']={};(tmp_path/'records').mkdir()
    write_json(tmp_path/'plan.json',p);ph=digest(tmp_path/'plan.json')
    for j in p['jobs']:write_json(tmp_path/'records'/f"{j['key']}.json",dict(job=j,plan_sha256=ph,status='completed'))
    monkeypatch.setattr(driver,'GuardedProcess',lambda *a,**kw:pytest.fail('no worker allowed'))
    driver.run(tmp_path);result=json.loads(capsys.readouterr().out)
    assert result==dict(new_calls=0,worker_started=False,reason='already_complete')
