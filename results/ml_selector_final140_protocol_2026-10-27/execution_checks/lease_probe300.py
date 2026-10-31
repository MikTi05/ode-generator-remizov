"""Finite sleeping-process lease probe. Never import numerical code."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from diagnostics.chernov_batch_control import controller_locks,new_control,GuardedProcess,atomic,read,finish_control,lock_is_held
out=Path(__file__).resolve().parent/'lease_probe300';out.mkdir(exist_ok=False)
began=time.monotonic()
with controller_locks(out,out/'probe_controller.lock') as fds:
 active=new_control(out,'external_renewal_probe',lease_seconds=300.)
 state=Path(active['control_dir'])/'probe.guardian.json'
 with (out/'guardian.log').open('x') as log:
  child=GuardedProcess([sys.executable,'-B','-c','import time; time.sleep(60)'],log,state,active,fds,compute_lock_path=out/'probe_compute.lock')
  atomic(out/'probe_started.json',dict(started_monotonic=began,initial_lease_seconds=300,controller_pid=active['controller_pid'],numerical_calls=0))
  survived_initial=False
  try:
   while child.poll() is None and time.monotonic()-began<90:
    if time.monotonic()-began>1.:
     lease=read(Path(active['control_dir'])/'lease.json')
     if lease['renewed_monotonic']>began+1:survived_initial=True
    time.sleep(.05)
   if child.poll() is None:child.terminate()
   child.wait(timeout=5)
  finally:
   if child.poll() is None:child.terminate();child.wait(timeout=5)
   finish_control(out,active)
 guardian=read(state)
 result=dict(passed=survived_initial and guardian.get('reason')=='stop_requested' and guardian.get('finished') and guardian.get('cleanup_success'),
  external_cli_renewal_observed_while_worker_alive=survived_initial,external_stop_cleanup_confirmed=guardian.get('reason')=='stop_requested',
  guardian=guardian,elapsed_seconds=time.monotonic()-began,numerical_calls=0)
 atomic(out/'result.json',result);print(json.dumps(result),flush=True)
 if not result['passed']:raise SystemExit(1)
