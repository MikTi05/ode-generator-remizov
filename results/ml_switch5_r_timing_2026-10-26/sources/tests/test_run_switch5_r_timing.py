"""Execution-controller scope/resume guards; all computation forbidden in tests."""
import json,time
import pytest
import diagnostics.run_switch5_r_timing as d


def setup_plan(tmp_path,done=0):
    jobs=d.read_json(d.SOURCE/'plan.json')['jobs']
    plan=dict(execution_authorized=True,jobs=jobs,resources=dict(series_seconds=1766,rss_mib=512,compute_workers=1),protected_sha256={})
    (tmp_path/'plan.json').write_text(json.dumps(plan));(tmp_path/'records').mkdir();(tmp_path/'events').mkdir()
    ph=d.digest(tmp_path/'plan.json')
    for j in jobs[:done]:(tmp_path/'records'/f"{j['key']}.json").write_text(json.dumps(dict(job=j,plan_sha256=ph,status="completed")))
    return plan,ph


def test_approved_scope_exact_frozen_jobs_and_caps(tmp_path):
    p,_=setup_plan(tmp_path);d.validate_scope(p)
    p['jobs'][0]['config']['n']=64
    with pytest.raises(AssertionError):d.validate_scope(p)


@pytest.mark.parametrize('limit,value',[('series_seconds',1767),('rss_mib',513),('compute_workers',2)])
def test_cannot_expand_approved_limits(tmp_path,limit,value):
    p,_=setup_plan(tmp_path);p['resources'][limit]=value
    with pytest.raises(AssertionError):d.validate_scope(p)


def test_completed_resume_changes_nothing_and_never_starts_worker(tmp_path,monkeypatch,capsys):
    setup_plan(tmp_path,40)
    def forbidden(*a,**kw):raise AssertionError('compute forbidden')
    monkeypatch.setattr(d,'GuardedProcess',forbidden)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()};d.run(tmp_path)
    assert json.loads(capsys.readouterr().out)['reason']=='already_complete'
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_interrupted_request_not_retried_under40_ceiling(tmp_path,monkeypatch,capsys):
    p,ph=setup_plan(tmp_path,39)
    for j in p['jobs']:(tmp_path/'events'/f"{j['key']}.request.json").write_text(json.dumps(dict(job=j,plan_sha256=ph,status="completed")))
    def forbidden(*a,**kw):raise AssertionError('compute forbidden')
    monkeypatch.setattr(d,'GuardedProcess',forbidden);d.run(tmp_path)
    result=json.loads(capsys.readouterr().out);assert not result['worker_started'] and result['pending']==1


def test_expired_original_deadline_not_reset(tmp_path,monkeypatch,capsys):
    setup_plan(tmp_path)
    path=tmp_path/'budget.json';path.write_text(json.dumps(dict(deadline_epoch=time.time()-1,deadline_monotonic=time.monotonic()-1)))
    before=path.read_bytes()
    def forbidden(*a,**kw):raise AssertionError('compute forbidden')
    monkeypatch.setattr(d,'GuardedProcess',forbidden);d.run(tmp_path)
    assert json.loads(capsys.readouterr().out)['reason']=='original_budget_exhausted'
    assert path.read_bytes()==before


def test_R_only_worker_never_invokes_MC_JIT_or_ML():
    import ast,inspect
    tree=ast.parse(inspect.getsource(d.worker))
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    assert not calls&{'solve_monte_carlo_optimized','solve_monte_carlo','compile_backend','fit','fit_head'}


def test_known_inaccurate_F0172_remains_expected_not_a_retry_trigger():
    jobs=d.read_json(d.SOURCE/'plan.json')['jobs']
    own=[j for j in jobs if j['equation_id']=='AMV1_F0172_S03']
    assert len(own)==8 and all(not j['expected_accurate'] for j in own)
