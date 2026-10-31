"""Twelve-call scope, fresh seeds and completed-resume tests; no solver calls."""
import json,hashlib
from collections import Counter
import pytest
import diagnostics.f0047_three_profile as d


def inputs():
    return d.read_json(d.CONTRACT/'plan.json'),[json.loads(s) for s in (d.CONTRACT/'pilot_jobs.jsonl').read_text().splitlines()]


def test_three_profiles_preserve_originals_and_fixed_new_seed_schedule():
    c,old=inputs();before=json.dumps(old,sort_keys=True);jobs=d.make_jobs(c,old)
    assert len(jobs)==len({j['key'] for j in jobs})==12
    assert Counter(j['profile'] for j in jobs)=={'R128':4,'R256':4,'MC1':4}
    assert {j['equation_id'] for j in jobs}=={d.PID}
    assert [j['config']['seed'] for j in jobs if j['profile']=='MC1']==list(d.SEEDS)
    assert not set(d.SEEDS)&{j['config'].get('seed') for j in old}
    for j in jobs:
        ref=d.read_json(d.ROOT/j['reference_path'])['job']['config'].copy()
        if j['profile']=='MC1':ref['seed']=d.SEEDS[j['episode']]
        assert j['config']==ref
    assert json.dumps(old,sort_keys=True)==before
    for i in range(4):
        shift=int(hashlib.sha256(f'{d.PID}|{i}'.encode()).hexdigest(),16)%3
        order=['R128','R256','MC1'];assert [j['profile'] for j in jobs[3*i:3*i+3]]==order[shift:]+order[:shift]


def test_seed_reuse_rejected(monkeypatch):
    c,old=inputs();monkeypatch.setattr(d,'SEEDS',(248800010100,248800040101,248800040102,248800040103))
    with pytest.raises(AssertionError):d.make_jobs(c,old)


def test_completed_resume_no_worker_or_writes(tmp_path,monkeypatch,capsys):
    c,old=inputs();jobs=d.make_jobs(c,old)
    (tmp_path/'plan.json').write_text(json.dumps(dict(execution_authorized=True,jobs=jobs,resources={},protected_sha256={})))
    ph=d.digest(tmp_path/'plan.json');(tmp_path/'records').mkdir()
    for j in jobs:(tmp_path/'records'/f"{j['key']}.json").write_text(json.dumps(dict(job=j,plan_sha256=ph)))
    def forbidden(*a,**kw):raise AssertionError('worker forbidden')
    monkeypatch.setattr(d,'GuardedProcess',forbidden)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    d.run(tmp_path)
    assert json.loads(capsys.readouterr().out)['worker_started'] is False
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
