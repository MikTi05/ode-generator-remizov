"""Short sleeping processes only: independent cleanup, lease, locks, resume."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from diagnostics.chernov_batch_control import (GuardedProcess, atomic,
    controller_locks, new_control, request_stop, renew_owner, lock_is_held)

ROOT=Path(__file__).resolve().parents[1]


def wait_until(predicate,seconds=5):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.03)
    raise AssertionError('short-process condition timed out')


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


@pytest.fixture
def sleeping_controller(tmp_path):
    """A dummy controller, guardian and worker; no solver imports/calls."""
    script=tmp_path/'controller.py'
    script.write_text('''
import json,os,signal,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from diagnostics.chernov_batch_control import *
out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
lease=float(sys.argv[3]); heartbeat=float(sys.argv[4])
worker_code="""
import json,os,signal,subprocess,sys,time
from pathlib import Path
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])
Path(sys.argv[1]).write_text(json.dumps(dict(pid=os.getpid(),descendant=p.pid)))
def stop(s,f):
    p.terminate();p.wait()  # reap the test descendant before forced worker kill
signal.signal(signal.SIGTERM,stop)
while True:time.sleep(.05)
"""
stop=False
def on_signal(s,f):
    global stop
    stop=True
for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(s,on_signal)
with controller_locks(out,out/'global.controller.lock') as fds:
    active=new_control(out,'session',lease)
    with (out/'guardian.log').open('w') as log:
        child=GuardedProcess([sys.executable,'-c',worker_code,str(out/'pids.json')],
            log,out/'control/session/test.guardian.json',active,fds,
            compute_lock_path=out/'global.compute.lock',heartbeat_timeout=heartbeat)
        try:
            while not stop and child.poll() is None:time.sleep(.02)
        finally:
            child.terminate();child.wait(timeout=4)
''')
    processes=[]
    def launch(lease=10,heartbeat=1):
        out=tmp_path/f'run{len(processes)}';out.mkdir()
        log=(out/'controller.log').open('w')
        p=subprocess.Popen([sys.executable,'-B',str(script),str(ROOT),str(out),
                            str(lease),str(heartbeat)],stdout=log,stderr=subprocess.STDOUT)
        processes.append((p,log,out))
        wait_until(lambda:(out/'pids.json').exists())
        return p,out,json.loads((out/'pids.json').read_text())
    yield launch
    for p,log,out in processes:
        if p.poll() is None:
            # Resume SIGSTOP tests before requesting normal cleanup.
            p.send_signal(signal.SIGCONT);p.terminate()
        p.wait(timeout=5);log.close()


def assert_clean(out,pids):
    state=out/'control/session/test.guardian.json'
    wait_until(lambda:json.loads(state.read_text()).get('finished'))
    wait_until(lambda:not alive(pids['pid']) and not alive(pids['descendant']))
    return json.loads(state.read_text())


def test_sigkill_controller_cleans_worker_and_descendant(sleeping_controller):
    p,out,pids=sleeping_controller()
    p.kill();p.wait()
    assert assert_clean(out,pids)['reason']=='controller_pipe_closed'


def test_stalled_living_controller_is_stopped_independently(sleeping_controller):
    p,out,pids=sleeping_controller(heartbeat=.4)
    p.send_signal(signal.SIGSTOP)
    try:
        assert assert_clean(out,pids)['reason']=='controller_heartbeat_expired'
    finally:p.send_signal(signal.SIGCONT)


def test_external_owner_lease_expires_even_when_controller_keeps_running(sleeping_controller):
    p,out,pids=sleeping_controller(lease=.5)
    assert assert_clean(out,pids)['reason']=='owner_lease_expired'
    p.wait(timeout=3)


@pytest.mark.parametrize('sig',[signal.SIGINT,signal.SIGTERM,signal.SIGHUP])
def test_signals_clean_worker_group(sleeping_controller,sig):
    p,out,pids=sleeping_controller()
    p.send_signal(sig);p.wait(timeout=4)
    assert_clean(out,pids)


def test_stop_command_cleans_group_without_pid_guessing(sleeping_controller):
    p,out,pids=sleeping_controller()
    assert request_stop(out)['guardians_stopped']
    assert assert_clean(out,pids)['reason']=='stop_requested'
    p.wait(timeout=3)


def test_owner_can_renew_but_cannot_restart_a_stopped_session(sleeping_controller):
    p,out,pids=sleeping_controller(lease=.6)
    deadline=time.monotonic()+.9
    while time.monotonic()<deadline:
        assert renew_owner(out)['renewed']
        time.sleep(.12)
    assert alive(pids['pid'])
    assert request_stop(out)['guardians_stopped']
    assert_clean(out,pids)
    with pytest.raises(ValueError):renew_owner(out)
    p.wait(timeout=3)


def test_run_locks_remain_held_until_orphan_cleanup_finishes(sleeping_controller):
    p,out,pids=sleeping_controller()
    p.kill();p.wait()
    with pytest.raises(RuntimeError,match='running or cleaning'):
        with controller_locks(out,out/'global.controller.lock'):pass
    assert_clean(out,pids)
    wait_until(lambda:not alive(pids['pid']))
    # The guardian publishes its final state just before exiting; inherited
    # run-lock FDs deliberately stay locked through that final unwind.
    wait_until(lambda:not lock_is_held(out/'global.controller.lock'))
    with controller_locks(out,out/'global.controller.lock'):pass


def test_global_lock_blocks_different_output_directory(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    common=tmp_path/'global.lock'
    assert not lock_is_held(common)
    with controller_locks(a,common):
        assert lock_is_held(common)
        with pytest.raises(RuntimeError):
            with controller_locks(b,common):pass
    assert not lock_is_held(common)


def test_direct_compute_worker_entry_is_rejected(tmp_path):
    p=subprocess.run([sys.executable,'-B',str(ROOT/'diagnostics/run_chernov_batch.py'),
        '--worker-plan',str(tmp_path/'unused.json')],capture_output=True,text=True)
    assert p.returncode==2 and 'independent guardian' in p.stderr


def test_nested_solver_pool_is_rejected():
    from diagnostics.run_chernov_batch import check_protected
    with pytest.raises(ValueError,match='n_jobs=1'):
        check_protected(dict(protected_sha256={},config={'n_jobs':2}))


def test_denied_signal_zero_probe_does_not_abort_kill_and_reap(monkeypatch):
    import diagnostics.chernov_batch_control as control
    sent=[]
    class Child:
        pid=123
        def poll(self):return None
        def wait(self,timeout):sent.append('reaped')
    def killpg(pid,sig):
        if sig==0:raise PermissionError('sandbox probe denied')
        sent.append(sig)
    monkeypatch.setattr(control.os,'killpg',killpg)
    control.kill_group(Child(),grace=.005)
    assert sent==[signal.SIGTERM,signal.SIGKILL,'reaped']


def test_compute_slot_blocks_second_guardian(sleeping_controller):
    p,out,pids=sleeping_controller()
    other=out/'control/session/second.guardian.json'
    active=json.loads((out/'active.json').read_text())
    with (out/'second.log').open('w') as log:
        q=GuardedProcess([sys.executable,'-c','raise AssertionError("must not launch")'],
            log,other,active,compute_lock_path=out/'global.compute.lock')
        q.wait(timeout=3)
    assert json.loads(other.read_text())['reason']=='compute_slot_busy'
    assert json.loads(other.read_text())['worker_pid'] is None
    assert alive(pids['pid'])


def test_real_scheduler_ignores_legacy_eight_workers_and_resumes(tmp_path,monkeypatch):
    import diagnostics.run_chernov_batch as batch
    import diagnostics.chernov_batch_control as control
    out=tmp_path/'batch';out.mkdir()
    for name in ('attempts','solutions','workers','sessions'):(out/name).mkdir()
    plan=dict(config={'n_jobs':1},equation_ids=['A','B'],pilot_ids=['A'],
        protected_sha256={},workers=8,method='solve',
        implementation='chernov_solver.ChernovSolver.solve',target_x=[0.],
        threshold=.01,startup_wall_seconds=3,worker_rss_limit_mib=512,
        poll_seconds=.02,task_wall_seconds=3)
    atomic(out/'plan.json',plan);atomic(out/'pilot_check.json',{})
    dummy=tmp_path/'worker.py'
    dummy.write_text('''
import argparse,hashlib,json,sys,time,os
from pathlib import Path
p=argparse.ArgumentParser()
for x in ('worker-plan','worker-state','parent-pid','guard-token','compute-lock-fd'):p.add_argument('--'+x)
a=p.parse_args();plan=json.loads(Path(a.worker_plan).read_text())
Path(a.worker_state).write_text(json.dumps(dict(phase='ready',peak_rss_mib=1)))
for line in sys.stdin:
    request=Path(json.loads(line)['request_path']);r=json.loads(request.read_text())
    value=dict(problem_id=r['problem_id'],attempt_id=r['attempt_id'],
      plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
      config=plan['config'],implementation=plan['implementation'],
      status='accuracy_pass',computation_status='success',max_abs_error=0.,solver_seconds=.05)
    time.sleep(float(os.environ.get('FAKE_BATCH_DELAY','.05')))
    target=request.parent/'result.json';tmp=request.parent/'dummy.tmp'
    tmp.write_text(json.dumps(value));os.replace(tmp,target)
''')
    original=batch.GuardedProcess
    def launch(command,*args,**kwargs):
        return original([sys.executable,'-B',str(dummy),*command[3:]],*args,
                        compute_lock_path=tmp_path/'compute.lock',**kwargs)
    monkeypatch.setattr(batch,'GuardedProcess',launch)
    monkeypatch.setattr(control,'ROOT',tmp_path)
    batch.run_batch(out)
    sessions=[json.loads(p.read_text()) for p in (out/'sessions').glob('*.json')]
    assert len(sessions[0]['worker_startups'])==1
    assert sessions[0]['completed_new']==2
    before={p.name:p.read_bytes() for p in (out/'solutions').glob('*.json')}
    batch.run_batch(out)
    assert before=={p.name:p.read_bytes() for p in (out/'solutions').glob('*.json')}
    assert len(list((out/'attempts').glob('*/request.json')))==2
    # A control interruption is pending work, not a terminal worker_exception.
    fresh=tmp_path/'interrupted';fresh.mkdir()
    for name in ('attempts','solutions','workers','sessions'):(fresh/name).mkdir()
    atomic(fresh/'plan.json',plan);atomic(fresh/'pilot_check.json',{})
    original_control=batch.new_control
    monkeypatch.setattr(batch,'new_control',lambda out,session:control.new_control(out,session,.3))
    monkeypatch.setenv('FAKE_BATCH_DELAY','.7')
    with pytest.raises(SystemExit) as stopped:
        batch.run_batch(fresh)
    assert stopped.value.code==130
    assert not list((fresh/'solutions').glob('*.json'))
    abandoned=list((fresh/'attempts').glob('*/request.json'))
    assert len(abandoned)==1
    saved=abandoned[0].read_bytes()
    assert (abandoned[0].parent/'interrupted.json').exists()
    monkeypatch.setattr(batch,'new_control',original_control)
    monkeypatch.setenv('FAKE_BATCH_DELAY','.05')
    batch.run_batch(fresh)
    assert abandoned[0].read_bytes()==saved
    assert len(list((fresh/'solutions').glob('*.json')))==2
    assert len(list((fresh/'attempts').glob('*/request.json')))==3
