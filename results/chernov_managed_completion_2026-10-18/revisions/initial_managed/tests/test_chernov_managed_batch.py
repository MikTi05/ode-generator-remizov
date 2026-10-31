"""Managed series, checked lease renewal and short sleeping processes; no ODEs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import diagnostics.chernov_monitored_batch as monitored
from diagnostics.chernov_batch_control import GuardedProcess, atomic, new_control, lock_is_held
from diagnostics.run_chernov_batch import plan_hash


def renewal_case(tmp_path,monkeypatch):
    control=tmp_path/'control';control.mkdir()
    active=dict(token='t',controller_pid=12,control_dir=str(control),finished=False,
                session_id='s',owner_lease_seconds=120)
    atomic(tmp_path/'active.json',active)
    atomic(control/'lease.json',dict(token='t',renewed_monotonic=990.))
    sample=dict(monotonic=1000.,thermal_state=0,safety_stop_reason=None,
                compute_workers=[dict(pid=14,cpu_percent=100.,rss_mib=100.)],
                guardians=[dict(guardian_pid=13,controller_pid=12,worker_pid=14,finished=False)])
    calls=[]
    monkeypatch.setattr(monitored.time,'monotonic',lambda:1000.)
    monkeypatch.setattr(monitored,'renew_owner',lambda out:calls.append(out) or dict(renewed=True,session='s'))
    return active,sample,SimpleNamespace(pid=12,poll=lambda:None),calls


@pytest.mark.parametrize('fault',[
    'dead_controller','changed_session','finished_session','stale_sample',
    'expired_lease','wrong_lease_token','no_worker','extra_worker',
    'thermal','finished_guardian','unowned_worker'])
def test_unhealthy_or_expired_owner_never_renews(tmp_path,monkeypatch,fault):
    active,sample,child,calls=renewal_case(tmp_path,monkeypatch)
    if fault=='dead_controller':child.poll=lambda:130
    elif fault=='changed_session':atomic(tmp_path/'active.json',dict(active,token='other'))
    elif fault=='finished_session':atomic(tmp_path/'active.json',dict(active,finished=True))
    elif fault=='stale_sample':sample['monotonic']=993.
    elif fault=='expired_lease':atomic(Path(active['control_dir'])/'lease.json',dict(token='t',renewed_monotonic=880.))
    elif fault=='wrong_lease_token':atomic(Path(active['control_dir'])/'lease.json',dict(token='wrong',renewed_monotonic=999.))
    elif fault=='no_worker':sample['compute_workers']=[]
    elif fault=='extra_worker':sample['safety_stop_reason']='multiple_compute_workers'
    elif fault=='thermal':sample['safety_stop_reason']='thermal_state_not_nominal';sample['thermal_state']=1
    elif fault=='finished_guardian':sample['guardians'][0]['finished']=True
    elif fault=='unowned_worker':sample['compute_workers'][0]['pid']=99
    with pytest.raises(monitored.SafetyStop):monitored.renew_checked(tmp_path,child,active,sample)
    assert calls==[]


def test_fresh_owned_worker_can_renew(tmp_path,monkeypatch):
    active,sample,child,calls=renewal_case(tmp_path,monkeypatch)
    assert monitored.renew_checked(tmp_path,child,active,sample)['renewed']
    assert calls==[tmp_path]


def test_completed_batch_accepts_ps_observation_just_before_guardian_reap(tmp_path,monkeypatch):
    active,sample,child,calls=renewal_case(tmp_path,monkeypatch)
    sample['guardians'][0].update(finished=True,cleanup_success=True)
    assert monitored.process_reason(child,active,sample,all_saved=True) is None
    with pytest.raises(monitored.SafetyStop):monitored.renew_checked(tmp_path,child,active,sample)
    assert not calls


def test_managed_batches_resume_and_hold_lock_between_batches(tmp_path,monkeypatch):
    monkeypatch.setattr(monitored,'ROOT',tmp_path)
    out=tmp_path/'batch';out.mkdir();(out/'solutions').mkdir();(out/'attempts').mkdir()
    plan=dict(config={'n_jobs':1},equation_ids=['old_answer','old_error']+[f'E{i}' for i in range(43)],
              implementation='chernov_solver.ChernovSolver.solve',protected_sha256={})
    atomic(out/'plan.json',plan);atomic(out/'safety_policy_v2.json',{})
    def save(pid,status='accuracy_pass'):
        record=dict(problem_id=pid,config=plan['config'],plan_sha256=plan_hash(plan),
                    implementation=plan['implementation'],status=status)
        with (out/'solutions'/f'{pid}.json').open('x') as f:json.dump(record,f)
    save('old_answer','accuracy_not_met');save('old_error','worker_exception')
    preserved={p.name:p.read_bytes() for p in (out/'solutions').glob('*.json')}
    requests=[];pauses=[]
    def cool(*args):
        assert lock_is_held(tmp_path/'results/.chernov_small_batch.lock')
        with pytest.raises(RuntimeError):
            with monitored.controller_locks(tmp_path/'other',tmp_path/'results/.chernov_small_batch.lock'):pass
        pauses.append(args[2])
    def fake_run(out,folder,count,*args,**kwargs):
        assert kwargs['managed'] and kwargs['lock_held']
        pending=[p for p in plan['equation_ids'] if not (out/'solutions'/f'{p}.json').exists()]
        requests.append(count)
        # Simulate one committed answer then an operator stop during first batch.
        ids=pending[:1] if len(requests)==1 else pending[:count]
        for pid in ids:save(pid)
        return dict(new_ids=ids,returncode=130 if len(requests)==1 else 0,
                    automatic_lease_renewals=2,cleanup_confirmed=True)
    monkeypatch.setattr(monitored,'check_protected',lambda *a:None)
    monkeypatch.setattr(monitored,'cooldown_remaining',lambda *a:60.)
    monkeypatch.setattr(monitored,'wait_cooldown',cool)
    monkeypatch.setattr(monitored,'run',fake_run)
    first=tmp_path/'first';first.mkdir();r=monitored.run_all(out,first,20,True)
    assert r['phase']=='stopped' and r['pending']==42 and len(requests)==1
    second=tmp_path/'second';second.mkdir();r=monitored.run_all(out,second,20,True)
    assert r['phase']=='completed' and r['pending']==0
    assert requests==[20,20,20,2] and pauses==[60.]*4
    assert all((out/'solutions'/name).read_bytes()==value for name,value in preserved.items())


def test_stop_and_thermal_failure_during_cooldown(tmp_path,monkeypatch):
    with pytest.raises(monitored.SafetyStop,match='operator_stop'):
        monitored.wait_cooldown(tmp_path,tmp_path,60,lambda:True,os.getppid())
    monkeypatch.setattr(monitored,'snapshot',lambda *a:dict(safety_stop_reason='thermal_state_not_nominal'))
    with pytest.raises(monitored.SafetyStop,match='thermal_state_not_nominal'):
        monitored.wait_cooldown(tmp_path,tmp_path,60,lambda:False,os.getppid())


def test_series_stop_command_writes_token_during_cooldown(tmp_path,monkeypatch):
    monkeypatch.setattr(monitored,'ROOT',tmp_path)
    out=tmp_path/'batch';out.mkdir();monitor=tmp_path/'monitor';monitor.mkdir()
    atomic(tmp_path/'results/chernov_monitored_active.json',dict(token='owner',output_dir=str(out),
           monitor_dir=str(monitor),finished=False))
    atomic(out/'active.json',dict(finished=True))
    assert monitored.stop_monitor(out)['series_stop_requested']
    assert json.loads((monitor/'stop.json').read_text())['token']=='owner'


def test_lost_external_renewer_expires_lease_with_live_controller(tmp_path):
    """A real guardian kills worker AND descendant after external monitor SIGKILL.

    Only test timings are shortened. Production still uses its fixed 120 s.
    The test process keeps the controller pipe/heartbeats alive throughout.
    """
    root=Path(__file__).resolve().parents[1]
    active=new_control(tmp_path,'short',lease_seconds=.6)
    state=tmp_path/'control/short/test.guardian.json'
    worker='''
import json,os,signal,subprocess,sys,time
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(12)'])
Path(sys.argv[1]).write_text(json.dumps(dict(worker=os.getpid(),descendant=p.pid)))
def stop(s,f):p.terminate();p.wait()
signal.signal(signal.SIGTERM,stop)
while True:time.sleep(.02)
'''
    renewer='''
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
from diagnostics.chernov_monitored_batch import renew_checked
out=Path(sys.argv[2]);active=json.loads((out/'active.json').read_text())
child=SimpleNamespace(pid=active['controller_pid'],poll=lambda:None)
for i in range(100):
 g=json.loads((out/'control/short/test.guardian.json').read_text())
 sample=dict(monotonic=time.monotonic(),thermal_state=0,safety_stop_reason=None,
   compute_workers=[dict(pid=g['worker_pid'])],guardians=[g])
 renew_checked(out,child,active,sample)
 (out/'renewed.json').write_text(json.dumps(dict(count=i+1)))
 time.sleep(.05)
'''
    observer=None
    with (tmp_path/'guardian.log').open('w') as log:
        child=GuardedProcess([sys.executable,'-B','-c',worker,str(tmp_path/'pids.json')],
            log,state,active,compute_lock_path=tmp_path/'compute.lock',heartbeat_timeout=1)
        try:
            deadline=time.monotonic()+4
            while not (tmp_path/'pids.json').exists():
                child.poll();assert time.monotonic()<deadline;time.sleep(.02)
            observer=subprocess.Popen([sys.executable,'-B','-c',renewer,str(root),str(tmp_path)],
                                      stdout=log,stderr=log)
            until=time.monotonic()+1.1
            while time.monotonic()<until:
                assert child.poll() is None;time.sleep(.02)
            assert json.loads((tmp_path/'renewed.json').read_text())['count']>=5
            observer.kill();observer.wait(timeout=3)
            deadline=time.monotonic()+4
            while child.poll() is None:
                assert time.monotonic()<deadline;time.sleep(.02)
            child.wait(timeout=3)
            final=json.loads(state.read_text())
            assert final['reason']=='owner_lease_expired' and final['cleanup_success']
            for pid in json.loads((tmp_path/'pids.json').read_text()).values():
                with pytest.raises(ProcessLookupError):os.kill(pid,0)
        finally:
            if observer is not None and observer.poll() is None:observer.terminate();observer.wait(timeout=3)
            if child.poll() is None:child.terminate();child.wait(timeout=4)
