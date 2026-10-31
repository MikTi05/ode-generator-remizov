"""Executor contracts, using synthetic kernels; no benchmark computations."""
from collections import Counter
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from diagnostics import joint_benchmark_pilot as pilot


def test_262_fixed_attempts_use_authorized_profiles_and_seed_lists():
    proposal=pilot.read(pilot.PROPOSAL/'proposal.json')
    tasks={r['problem_id']:dict(T=float(r['T']),identity={}) for r in
           pilot.read(pilot.ROOT/'results/forcing_bounds_2026-10-24/survey/certificates.json')}
    attempts=pilot.build_attempts(proposal,tasks)
    assert len(attempts)==len({a['attempt_id'] for a in attempts})==262
    assert Counter(a['method'] for a in attempts)=={'solve':78,'monte_carlo':184}
    assert Counter(a['mode'] for a in attempts)=={'point':220,'grid33':42}
    for phase in proposal['phases']:
        for pid in phase['ids']:
            rr=[a for a in attempts if a['mode']==phase['name'] and a['problem_id']==pid]
            mc=[a for a in rr if a['method']=='monte_carlo']
            assert [a['config']['seed'] for a in mc]==phase['root_seeds']
            assert all(a['config']['M']==phase['M'] and a['config']['dt']==1/128
                       and a['config']['max_runtime_seconds'] is None for a in mc)
            assert len([a for a in rr if a['method']=='solve'])==3


LIMITS=dict(rss_limit_mib=512,telemetry_stale_seconds=30,cpu_idle_seconds=120,progress_stall_seconds=600)


def test_active_worker_is_not_killed_for_long_total_duration():
    assert pilot.safeguards(dict(peak_rss_mib=180),elapsed=1000000,
                            idle_cpu=1,idle_progress=4,stale=1,limits=LIMITS) is None


@pytest.mark.parametrize('field,value,reason',[
    ('stale',31,'worker_telemetry_stale'),
    ('idle_cpu',121,'worker_cpu_stalled'),
    ('idle_progress',601,'worker_progress_stalled')])
def test_liveness_watchdogs_are_separate_from_duration(field,value,reason):
    args=dict(sample={'peak_rss_mib':100},elapsed=10000,idle_cpu=0,idle_progress=0,stale=0,limits=LIMITS)
    args[field]=value
    assert pilot.safeguards(**args)==reason


def test_memory_guard_is_external_failure():
    assert pilot.safeguards({'peak_rss_mib':513},1,0,0,0,LIMITS)=='external_memory_limit'


def test_mc_observer_returns_original_object_and_restores_kernel(monkeypatch):
    import mc_solver
    output=object();run=SimpleNamespace(n_path_steps=123,n_batches=2)
    args=(object(),object(),run);calls=[]
    def fake(*a,**kw):
        calls.append((a,kw));return output
    monkeypatch.setattr(mc_solver,'_path_contributions',fake)
    state={}
    with pilot.observe('monte_carlo',state):
        assert mc_solver._path_contributions(*args) is output
    assert mc_solver._path_contributions is fake
    assert calls==[(args,{})]
    assert state==dict(completed_units=123,completed_batches=2)


def test_chernov_observer_does_not_change_result_or_arguments(monkeypatch):
    import chernov_solver
    original_result=object();calls=[]
    def fake(*a,**kw):calls.append((a,kw));return original_result
    monkeypatch.setattr(chernov_solver,'_at_t',fake)
    state={'completed_units':0}
    with pilot.observe('solve',state):
        assert chernov_solver._at_t('state',.125) is original_result
    assert calls==[(('state',.125),{})] and state['completed_units']==1
    assert chernov_solver._at_t is fake


def test_observer_restores_kernel_after_exception(monkeypatch):
    import chernov_solver
    def bad(*a):raise ArithmeticError('synthetic')
    monkeypatch.setattr(chernov_solver,'_at_t',bad)
    with pytest.raises(ArithmeticError):
        with pilot.observe('solve',{'completed_units':0}):chernov_solver._at_t(None,0)
    assert chernov_solver._at_t is bad


def test_expired_external_lease_is_never_revived(tmp_path):
    control=tmp_path/'control';control.mkdir()
    pilot.write(tmp_path/'active.json',dict(control_dir=str(control),token='t',finished=False,owner_lease_seconds=120))
    pilot.write(control/'lease.json',dict(token='t',renewed_monotonic=time.monotonic()-121))
    pilot.write(tmp_path/'controller.heartbeat.json',dict(monotonic=time.monotonic()))
    before=(control/'lease.json').read_bytes()
    with pytest.raises(RuntimeError,match='expired'):pilot.renew_checked(tmp_path)
    assert (control/'lease.json').read_bytes()==before
