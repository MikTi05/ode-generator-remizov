"""Independent POSIX guardian for one batch compute process group.

No numerical imports. The controller must keep its pipe/heartbeat alive AND
an external owner must renew the finite lease. Closing an IDE is not itself
an OS process-lifetime event. The guardian runs in a separate session, keeps
the run locks, and kills/reaps the worker group without needing worker Python
threads to run. The worker inherits the compute-slot lock as a second fence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
OWNER_LEASE_SECONDS=120.
CONTROLLER_TIMEOUT=5.
CONTROL_STOPS={'controller_pipe_closed','controller_heartbeat_expired',
    'owner_lease_expired','stop_requested','controller_stop','guardian_signal'}


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.control_',
                                     suffix='.tmp',delete=False) as f:
        tmp=Path(f.name);json.dump(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def controller_locks(out, global_path=None):
    """Hold both locks through guardian cleanup; pass these FDs to guardian."""
    handles=[]
    try:
        for path in (global_path or ROOT/'results/.chernov_batch_controller.lock',out/'batch.lock'):
            path.parent.mkdir(parents=True,exist_ok=True)
            f=path.open('a');handles.append(f)
            try:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'batch already running or cleaning up: {path}') from exc
        yield [f.fileno() for f in handles]
    finally:
        # Closing, rather than LOCK_UN, preserves the guardian's inherited lock.
        for f in reversed(handles):
            f.close()


def new_control(out,session,lease_seconds=OWNER_LEASE_SECONDS):
    control=out/'control'/session;control.mkdir(parents=True,exist_ok=False)
    token=uuid.uuid4().hex
    active=dict(session_id=session,token=token,controller_pid=os.getpid(),
        control_dir=str(control.resolve()),owner_lease_seconds=lease_seconds,
        finished=False)
    atomic(control/'lease.json',dict(token=token,renewed_monotonic=time.monotonic()))
    atomic(control/'controller.json',active)
    atomic(out/'active.json',active)
    return active


def renew_owner(out):
    active=read(out/'active.json')
    if active.get('finished'):
        raise ValueError('no active batch')
    control=Path(active['control_dir'])
    guardians=[read(p) for p in control.glob('*.guardian.json')]
    if guardians and all(g.get('finished') for g in guardians):
        raise ValueError('guardians have finished; continue with a new session')
    if (control/'stop.json').exists():
        raise ValueError('batch has been stopped; start a new session to continue')
    atomic(control/'lease.json',dict(token=active['token'],renewed_monotonic=time.monotonic()))
    return dict(renewed=True,lease_seconds=active['owner_lease_seconds'],session=active['session_id'])


def request_stop(out):
    active_path=out/'active.json'
    if not active_path.exists():
        return dict(stop_requested=False,reason='no managed session')
    active=read(active_path);control=Path(active['control_dir'])
    if not (control/'stop.json').exists():
        atomic(control/'stop.json',dict(token=active['token'],requested_monotonic=time.monotonic()))
    # No PID-based kill: an independent guardian watches this token-scoped file.
    began=time.monotonic()
    while time.monotonic()-began<8:
        guardians=[read(p) for p in control.glob('*.guardian.json')]
        if guardians and all(g.get('finished') and g.get('cleanup_success',True) for g in guardians):
            return dict(stop_requested=True,guardians_stopped=True,session=active['session_id'])
        if read(control/'controller.json').get('finished') and not guardians:
            return dict(stop_requested=True,guardians_stopped=True,session=active['session_id'])
        time.sleep(.1)
    return dict(stop_requested=True,guardians_stopped=False,
                note='inspect --status; stop request remains active',session=active['session_id'])


def lock_is_held(path):
    if not path.exists():return False
    with path.open('a') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def finish_control(out,active):
    value=dict(active,finished=True,finished_monotonic=time.monotonic())
    atomic(Path(active['control_dir'])/'controller.json',value)
    atomic(out/'active.json',value)


def group_exists(pgid):
    try:
        os.killpg(pgid,0);return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS sandbox can deny the signal-0 probe during group teardown.
        # Treat the group as potentially alive; still send TERM/KILL and reap.
        return True


def kill_group(child,grace=.7):
    """Kill the owned session, even if the group leader has already exited."""
    pgid=child.pid
    for sig,wait in ((signal.SIGTERM,grace),(signal.SIGKILL,grace)):
        try:
            os.killpg(pgid,sig)
        except ProcessLookupError:
            break
        deadline=time.monotonic()+wait
        while time.monotonic()<deadline:
            child.poll()
            if not group_exists(pgid):
                break
            time.sleep(.02)
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        child.kill();child.wait()


def guardian(command,state_path,control_dir,token,lease_seconds,heartbeat_timeout,
             compute_lock_path):
    """Independent event loop: parent EOF, heartbeats, owner lease and stop file."""
    child=None;compute=None;reason='guardian_exception';signalled=False
    state=dict(guardian_pid=os.getpid(),controller_pid=os.getppid(),token=token,
        finished=False,worker_pid=None,worker_pgid=None)
    atomic(state_path,state)
    def on_signal(signum,frame):
        nonlocal signalled
        signalled=True
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(sig,on_signal)
    last_heartbeat=time.monotonic();buffer=b''
    def stop_reason():
        if signalled:return 'guardian_signal'
        if (control_dir/'stop.json').exists():return 'stop_requested'
        lease=read(control_dir/'lease.json')
        if lease['token']!=token:return 'owner_lease_expired'
        if time.monotonic()-lease['renewed_monotonic']>lease_seconds:
            return 'owner_lease_expired'
        if time.monotonic()-last_heartbeat>heartbeat_timeout:
            return 'controller_heartbeat_expired'
        return None
    try:
        compute_lock_path.parent.mkdir(parents=True,exist_ok=True)
        compute=compute_lock_path.open('a')
        try:
            fcntl.flock(compute,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            reason='compute_slot_busy';return
        early=stop_reason()
        if early:
            reason=early;return
        env=os.environ.copy();env['CHERNOV_GUARD_TOKEN']=token
        command=[part.replace('{GUARD_PID}',str(os.getpid()))
            .replace('{GUARD_TOKEN}',token).replace('{COMPUTE_FD}',str(compute.fileno()))
            for part in command]
        child=subprocess.Popen(command,stdin=subprocess.PIPE,text=True,env=env,
            start_new_session=True,close_fds=True,pass_fds=(compute.fileno(),))
        state.update(worker_pid=child.pid,worker_pgid=child.pid)
        atomic(state_path,state)
        while True:
            current=stop_reason()
            if current:
                reason=current;break
            if child.poll() is not None:
                reason='worker_exit';break
            ready,_,_=select.select([sys.stdin.fileno()],[],[],.1)
            if not ready:continue
            chunk=os.read(sys.stdin.fileno(),65536)
            if not chunk:
                reason='controller_pipe_closed';break
            buffer+=chunk
            while b'\n' in buffer:
                line,buffer=buffer.split(b'\n',1)
                message=json.loads(line)
                if message.get('_control')=='heartbeat':
                    last_heartbeat=time.monotonic()
                elif message.get('_control')=='stop':
                    reason='controller_stop';return
                else:
                    child.stdin.write(line.decode()+'\n');child.stdin.flush()
    finally:
        if child is not None:
            try:
                kill_group(child)
                state['cleanup_success']=True
            except Exception as exc:
                # Always publish failed cleanup; never claim the group stopped.
                # A surviving worker keeps the inherited compute-slot lock.
                state['cleanup_success']=False
                state['cleanup_error']=f'{type(exc).__name__}: {exc}'
            state['worker_returncode']=child.returncode
        state.update(finished=True,reason=reason,finished_monotonic=time.monotonic())
        atomic(state_path,state)
        if compute is not None:compute.close()


class GuardedProcess:
    """Small Popen-compatible facade used by the existing resource watchdog."""
    def __init__(self,command,log,guardian_path,active,lock_fds=(),
                 compute_lock_path=None,heartbeat_timeout=CONTROLLER_TIMEOUT):
        self.guardian_path=guardian_path
        args=[sys.executable,'-B',str(Path(__file__).resolve()),
            '--command-json',json.dumps(command),'--state',str(guardian_path),
            '--control-dir',active['control_dir'],'--token',active['token'],
            '--lease-seconds',str(active['owner_lease_seconds']),
            '--heartbeat-timeout',str(heartbeat_timeout),'--compute-lock',
            str(compute_lock_path or ROOT/'results/.chernov_batch_compute.lock')]
        self.process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=log,
            stderr=subprocess.STDOUT,text=True,start_new_session=True,
            close_fds=True,pass_fds=tuple(lock_fds))
        self.stdin=self.process.stdin;self.pid=self.process.pid
        self._last_pulse=0.

    @property
    def returncode(self):return self.process.returncode

    @property
    def stop_reason(self):
        return read(self.guardian_path).get('reason') if self.guardian_path.exists() else None

    def poll(self):
        code=self.process.poll()
        if code is None and time.monotonic()-self._last_pulse>.4:
            try:
                self.stdin.write('{"_control":"heartbeat"}\n');self.stdin.flush()
                self._last_pulse=time.monotonic()
            except (BrokenPipeError,ValueError):pass
        return code

    def terminate(self):
        try:
            self.stdin.write('{"_control":"stop"}\n');self.stdin.flush()
        except (BrokenPipeError,ValueError):pass
        if self.process.poll() is None:self.process.terminate()

    def kill(self):
        # Give the independent guardian a signal it can handle, never SIGKILL
        # it while its compute process could remain alive.
        self.process.terminate()

    def wait(self,timeout=None):return self.process.wait(timeout=timeout)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command-json',required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--control-dir',type=Path,required=True)
    parser.add_argument('--token',required=True)
    parser.add_argument('--lease-seconds',type=float,required=True)
    parser.add_argument('--heartbeat-timeout',type=float,required=True)
    parser.add_argument('--compute-lock',type=Path,required=True)
    args=parser.parse_args()
    guardian(json.loads(args.command_json),args.state,args.control_dir,args.token,
        args.lease_seconds,args.heartbeat_timeout,args.compute_lock)


if __name__=='__main__':main()
