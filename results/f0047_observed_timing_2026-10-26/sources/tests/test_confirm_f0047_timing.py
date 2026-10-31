"""Confirmation scope/resume tests; never start a numerical worker."""
import copy,json,hashlib
from collections import Counter
import pytest
import diagnostics.confirm_f0047_timing as driver


def inputs():
    return driver.read_json(driver.R0/'NEXT_STEP.json'),[json.loads(s) for s in (driver.CONTRACT/'pilot_jobs.jsonl').read_text().splitlines()]


def test_exact_eight_same_task_parameters_semantic_seeds_and_stable_order():
    p,old=inputs();jobs=driver.make_jobs(p,old)
    assert len(jobs)==len({j['key'] for j in jobs})==8
    assert {j['equation_id'] for j in jobs}=={driver.PID}
    assert Counter(j['profile'] for j in jobs)=={'R256':4,'MC1':4}
    assert [j['config']['seed'] for j in jobs if j['profile']=='MC1']==[248800010100+i for i in range(4)]
    for i in range(4):
        pair=jobs[2*i:2*i+2]
        shift=int(hashlib.sha256(f'{driver.PID}|{i}'.encode()).hexdigest(),16)%2
        assert [j['profile'] for j in pair]==(['MC1','R256'] if shift else ['R256','MC1'])
        assert next(j for j in pair if j['profile']=='R256')['config']==p['R_config']
        assert next(j for j in pair if j['profile']=='MC1')['config']==p['MC_original_jobs'][i]['config']


@pytest.mark.parametrize('field,value',[('calls',9),('equation_id','other'),('R_profile','R128')])
def test_reject_expanded_or_different_scope(field,value):
    p,old=inputs();p[field]=value
    with pytest.raises(AssertionError):driver.make_jobs(p,old)


def test_completed_resume_no_worker_no_file_mutations(tmp_path,monkeypatch,capsys):
    p,old=inputs();jobs=driver.make_jobs(p,old)
    plan=dict(execution_authorized=True,jobs=jobs,resources={},protected_sha256={})
    (tmp_path/'plan.json').write_text(json.dumps(plan));(tmp_path/'records').mkdir()
    ph=driver.digest(tmp_path/'plan.json')
    for j in jobs:(tmp_path/'records'/f"{j['key']}.json").write_text(json.dumps(dict(job=j,plan_sha256=ph)))
    def forbidden(*a,**kw):raise AssertionError('numerical worker forbidden')
    monkeypatch.setattr(driver,'GuardedProcess',forbidden)
    before={str(f):f.read_bytes() for f in tmp_path.rglob('*') if f.is_file()}
    driver.run(tmp_path)
    assert json.loads(capsys.readouterr().out)['worker_started'] is False
    assert before=={str(f):f.read_bytes() for f in tmp_path.rglob('*') if f.is_file()}
