"""Persistence/identity tests; historical numerical results are independent oracles."""
import copy
import json
from pathlib import Path

import pytest

from diagnostics import mc33_parallel as p


@pytest.fixture
def historical():
    path = p.ROOT/'reports/ml_selector_contract/pilot_p2_2026-10-26/records/AMV1_F0004_S04_MC1_stage2_episode0.json'
    record = p.read_json(path)
    return copy.deepcopy(record['job']), copy.deepcopy(record)


def test_historical_record_and_identity_mutations(historical):
    job, r = historical
    p.validate_success(job, r)
    for field, value in [('seed_entropy', 1), ('M', 1024), ('n_path_steps', 0),
                         ('dt_actual', .01), ('x', [0.]), ('f', [float('nan')]*33)]:
        bad = copy.deepcopy(r); bad['result'][field] = value
        with pytest.raises(ValueError):
            p.validate_success(job, bad)
    bad = copy.deepcopy(r); bad['E_grid'] /= 2
    with pytest.raises(ValueError, match='error metric'):
        p.validate_success(job, bad)


def test_atomic_no_clobber(tmp_path):
    f = tmp_path/'result.json'
    p.publish(f, {'a':1}); before = f.read_bytes()
    with pytest.raises(FileExistsError):
        p.publish(f, {'a':2})
    assert f.read_bytes() == before


def checkpoint(tmp_path, job, r):
    (tmp_path/'records').mkdir(); (tmp_path/'attempts').mkdir()
    p.publish(tmp_path/'plan.json', {'jobs':[job]})
    r = dict(r, plan_sha256=p.digest(tmp_path/'plan.json'), job=job)
    return {'record':r, 'sha256':p.sha_object(r)}


def test_recovery_after_attempt_before_record_and_pending_request(tmp_path, historical):
    job, r = historical; w = checkpoint(tmp_path, job, r)
    # A request without an answer is still pending, preserving its seed.
    p.publish(tmp_path/'attempts'/'old.request.json', {'job':job})
    assert p.scan(tmp_path, {'jobs':[job]}) == {}
    p.publish(tmp_path/'attempts'/'new.result.json', w)
    assert p.scan(tmp_path, {'jobs':[job]})[job['key']] == w['record']
    f = tmp_path/'records'/(job['key']+'.json'); before = f.read_bytes()
    assert len(p.scan(tmp_path, {'jobs':[job]})) == 1
    assert f.read_bytes() == before


def test_corrupt_checkpoint_fails_closed(tmp_path, historical):
    job, r = historical; w = checkpoint(tmp_path, job, r)
    w['record']['result']['f'][0] += 1
    f = tmp_path/'records'/(job['key']+'.json'); p.publish(f, w)
    with pytest.raises(ValueError, match='corrupt checkpoint'):
        p.scan(tmp_path, {'jobs':[job]})


def test_completed_resume_starts_no_worker(tmp_path, historical, monkeypatch):
    job, r = historical; w = checkpoint(tmp_path, job, r)
    p.publish(tmp_path/'records'/(job['key']+'.json'), w)
    monkeypatch.setattr(p, 'load_plan', lambda _: {'jobs':[job]})
    def forbidden(*args, **kwargs):
        pytest.fail('completed resume started a worker')
    monkeypatch.setattr(p, 'GuardedProcess', forbidden)
    result = p.run(tmp_path, 8, resume=True)
    assert result['worker_startups'] == result['new_calls'] == result['pending'] == 0


def test_rng_address_is_task_order_independent():
    import numpy as np
    from mc_solver import _spawn_key
    ids = ['AMV1_F0004_S04', 'AMV1_F0067_S01']
    def draws(order):
        return {pid:np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            248800020100, spawn_key=_spawn_key(pid, .25, 0)))).standard_normal(100)
                for pid in order}
    first, second = draws(ids), draws(list(reversed(ids)))
    assert all(np.array_equal(first[pid], second[pid]) for pid in ids)
    assert not np.array_equal(first[ids[0]], first[ids[1]])
