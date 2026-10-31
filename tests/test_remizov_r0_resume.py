"""Resume checks with files only; starting any compute worker is forbidden."""
import json
from pathlib import Path

import pytest

import diagnostics.resume_remizov_r0 as resume


def prepare(tmp_path,complete):
    jobs=[dict(key=f'job{i}',stage=0,profile='R128') for i in range(38)]
    plan=dict(execution_authorized=True,jobs=jobs,resources={},protected_sha256={})
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    (tmp_path/'records').mkdir();(tmp_path/'events').mkdir()
    ph=resume.digest(tmp_path/'plan.json')
    for job in jobs[:complete]:
        (tmp_path/'records'/f"{job['key']}.json").write_text(json.dumps(dict(job=job,plan_sha256=ph)))
    return jobs,ph


def test_completed_resume_starts_no_worker_and_changes_no_files(tmp_path,monkeypatch,capsys):
    prepare(tmp_path,38)
    def forbidden(*a,**kw):raise AssertionError('compute must not start')
    monkeypatch.setattr(resume,'GuardedProcess',forbidden)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    resume.run(tmp_path)
    assert json.loads(capsys.readouterr().out)['worker_started'] is False
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_resume_rejects_unapproved_interrupted_key_before_worker(tmp_path,monkeypatch):
    jobs,ph=prepare(tmp_path,1)
    (tmp_path/'events'/'old.request.json').write_text(json.dumps(dict(job=jobs[1])))
    (tmp_path/'resume_amendment.json').write_text(json.dumps(dict(original_plan_sha256=ph,maximum_physical_attempts=39,permitted_retry_keys=['other'])))
    def forbidden(*a,**kw):raise RuntimeError('compute must not start')
    monkeypatch.setattr(resume,'GuardedProcess',forbidden)
    with pytest.raises(AssertionError):resume.run(tmp_path)
