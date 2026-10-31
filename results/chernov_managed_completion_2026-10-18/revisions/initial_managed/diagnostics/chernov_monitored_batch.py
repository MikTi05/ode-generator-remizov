"""Foreground batches with independent activity/thermal monitoring.

Small batches keep the original non-renewing lease. Explicit --all-pending
authorizes sequential batches and checked renewal by this external monitor,
never by the backend controller. Loss/stall of this monitor expires the lease.
Backend mathematical sources and its frozen safety policy stay intact.
macOS thermalState is NOT a CPU temperature in degrees (unknown can report 0).
The operator explicitly acknowledges this limitation for a state-only run.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.chernov_batch_control import atomic, controller_locks, lock_is_held, request_stop, renew_owner
from diagnostics.run_chernov_batch import check_protected, digest, read_json, validate_identity

THERMAL_SCRIPT='ObjC.import("Foundation"); JSON.stringify({thermalState:Number($.NSProcessInfo.processInfo.thermalState)})'
COOLDOWN_SECONDS=60.
RENEW_SECONDS=20.
SAMPLE_SECONDS=2.


class SafetyStop(RuntimeError):
    """A condition that ends the entire series; never automatically retried."""


def thermal_sample():
    r=subprocess.run(['/usr/bin/osascript','-l','JavaScript','-e',THERMAL_SCRIPT],
                     capture_output=True,text=True,timeout=4)
    if r.returncode:
        raise RuntimeError(f'thermal telemetry unavailable: {r.stderr.strip()}')
    value=json.loads(r.stdout)['thermalState']
    if type(value) is not int or value not in (0,1,2,3):
        raise RuntimeError('invalid thermal state')
    return dict(thermal_state=value,thermal_name=('nominal','fair','serious','critical')[value],
        temperature_celsius=None,measurement='NSProcessInfo.thermalState',
        limitation='not degrees; unsupported/unknown systems can also return nominal')


def parse_activity(text):
    rows=[]
    script=str(ROOT/'diagnostics/run_chernov_batch.py')
    for line in text.splitlines():
        fields=line.split(None,5)
        if len(fields)!=6:continue
        try:
            pid,ppid,pgid=int(fields[0]),int(fields[1]),int(fields[2])
            cpu,rss=float(fields[3]),int(fields[4])
        except ValueError:continue
        command=fields[5]
        if script in command and ' --worker-plan ' in command:
            rows.append(dict(pid=pid,ppid=ppid,pgid=pgid,cpu_percent=cpu,rss_mib=rss/1024))
    return rows


def activity_sample():
    r=subprocess.run(['/bin/ps','-axo','pid=,ppid=,pgid=,%cpu=,rss=,command='],
                     capture_output=True,text=True,timeout=4)
    if r.returncode:raise RuntimeError(f'process telemetry unavailable: {r.stderr.strip()}')
    return parse_activity(r.stdout)


def safety_reason(thermal,workers):
    if thermal['thermal_state']!=0:return 'thermal_state_not_nominal'
    if len(workers)>1:return 'multiple_compute_workers'
    if any(w['cpu_percent']>125 for w in workers):return 'worker_cpu_above_single_core_budget'
    if any(w['rss_mib']>512 for w in workers):return 'worker_memory_limit'
    return None


def cooldown_remaining(history,now=None):
    now=time.time() if now is None else now
    ends=[]
    for p in history.glob('*.json'):
        r=read_json(p)
        # Unfinished external monitor: conservatively include the backend lease.
        ends.append(r.get('finished_epoch',r.get('last_sample_epoch',r['started_epoch'])+120))
    return max(0.,max(ends,default=now-COOLDOWN_SECONDS)+COOLDOWN_SECONDS-now)


def snapshot(out,active=None):
    thermal=thermal_sample();workers=activity_sample()
    solving=[];guardians=[]
    if active:
        for p in Path(active['control_dir']).glob('*.guardian.json'):
            g=read_json(p);guardians.append(g)
            token=p.name.removesuffix('.guardian.json')
            state=out/'workers'/f'{token}.json'
            if state.exists():
                w=read_json(state)
                if w.get('phase')=='solving':solving.append(w.get('problem_id'))
    return dict(utc=datetime.now(timezone.utc).isoformat(),epoch=time.time(),monotonic=time.monotonic(),
        **thermal,compute_workers=workers,compute_worker_count=len(workers),
        solving_ids=solving,guardians=guardians,safety_stop_reason=safety_reason(thermal,workers))


def stop_owned(out,child):
    active=read_json(out/'active.json') if (out/'active.json').exists() else None
    if active and active.get('controller_pid')==child.pid:
        return request_stop(out)
    return dict(stop_requested=False,reason='our controller has no active session')


def backend_busy():
    return any(lock_is_held(ROOT/'results'/name) for name in
               ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'))


def terminal_ids(out,plan):
    """Include committed but not yet adopted answers without mutating them."""
    ids=set()
    for path in list((out/'solutions').glob('*.json'))+list((out/'attempts').glob('*/result.json')):
        record=read_json(path);validate_identity(record,plan);ids.add(record['problem_id'])
    return ids


def process_reason(child,active,sample,all_saved=False):
    if sample['safety_stop_reason']:return sample['safety_stop_reason']
    if not active:return None
    if active['controller_pid']!=child.pid:return 'controller_identity_changed'
    guards=sample['guardians'];live=[g for g in guards if not g.get('finished')]
    if len(live)>1:return 'multiple_live_guardians'
    if any(g.get('cleanup_success') is False for g in guards):return 'guardian_cleanup_failed'
    if any(g.get('finished') for g in guards) and not all_saved:return 'guardian_ended_before_batch_completed'
    for worker in sample['compute_workers']:
        # ps and guardian JSON are consecutive observations, not an atomic
        # snapshot. A just-reaped worker may still appear in the earlier ps.
        owners=guards if all_saved else live
        if not any(g.get('worker_pid')==worker['pid'] and g.get('controller_pid')==child.pid for g in owners):
            return 'worker_not_owned_by_live_guardian'
    return None


def renew_checked(out,child,active,sample):
    """Renew only a live, owned session with fresh healthy telemetry.

    An expired lease is never revived, even if a stalled guardian has not yet
    observed expiry. This function has no scheduler/restart side effects.
    """
    now=time.monotonic()
    if child.poll() is not None:raise SafetyStop('controller_exited')
    current=read_json(out/'active.json')
    if (current.get('finished') or current['token']!=active['token'] or
        current['controller_pid']!=child.pid):raise SafetyStop('active_session_changed')
    if not 0<=now-sample['monotonic']<=6:raise SafetyStop('stale_telemetry')
    reason=process_reason(child,current,sample)
    if reason:raise SafetyStop(reason)
    if len(sample['compute_workers'])!=1 or len([g for g in sample['guardians'] if not g.get('finished')])!=1:
        raise SafetyStop('no_single_live_worker_for_renewal')
    lease=read_json(Path(current['control_dir'])/'lease.json')
    if lease['token']!=current['token'] or now-lease['renewed_monotonic']>=current['owner_lease_seconds']:
        raise SafetyStop('owner_lease_already_expired')
    result=renew_owner(out)
    if result['session']!=current['session_id']:raise SafetyStop('renewed_session_changed')
    return dict(**result,utc=datetime.now(timezone.utc).isoformat(),monotonic=now,
                worker_pid=sample['compute_workers'][0]['pid'],thermal_state=sample['thermal_state'])


def persist_run(monitor_dir,history,record):
    atomic(monitor_dir/'run.json',record)
    atomic(history/f"{record['run_id']}.json",{k:record[k] for k in
        ('run_id','started_epoch','last_sample_epoch','finished_epoch','monitor_dir') if k in record})


def run(out,monitor_dir,count,state_only,test_stop=False,*,managed=False,
        lock_held=False,external_stop=lambda:False):
    if count not in (range(1,21) if managed else (1,2,3)):
        raise ValueError('batch size must be 1..20 in managed mode; small batches must be 1..3')
    if not state_only:
        raise ValueError('Celsius unavailable: explicitly select --thermal-state-only or do not start')
    if test_stop and count!=2:raise ValueError('stop/resume test requires --max-new 2')
    # A monitor lock, distinct from backend locks. No second thermal monitor run.
    with (nullcontext() if lock_held else controller_locks(monitor_dir,ROOT/'results/.chernov_small_batch.lock')):
        plan=read_json(out/'plan.json');check_protected(plan,out)
        if backend_busy():
            raise RuntimeError('a backend is already running/cleaning up')
        history=ROOT/'results/chernov_small_batch_history';history.mkdir(exist_ok=True)
        wait=cooldown_remaining(history)
        if wait>0:raise RuntimeError(f'cooling interval: wait {math.ceil(wait)} seconds; no worker started')
        before={p.stem:digest(p) for p in (out/'solutions').glob('*.json')}
        completed=terminal_ids(out,plan)
        selected=[pid for pid in plan['equation_ids'] if pid not in completed][:count]
        if not selected:raise ValueError('no pending equations')
        initial=snapshot(out)
        if initial['safety_stop_reason'] or initial['compute_worker_count']:
            raise RuntimeError(f'preflight failed: {initial}')
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
        command=[sys.executable,'-B',str(ROOT/'diagnostics/run_chernov_batch.py'),
                 '--output-dir',str(out),'--max-new',str(count)]
        record=dict(run_id=run_id,started_epoch=time.time(),selected_ids=selected,
            max_new=count,backend_command=command,config=plan['config'],
            plan_sha256=digest(out/'plan.json'),safety_policy_sha256=digest(out/'safety_policy_v2.json'),
            source_sha256=digest(Path(__file__)),monitor_dir=str(monitor_dir),
            owner_lease_seconds=120,automatic_lease_renewals=0,inter_batch_cooldown_seconds=60,
            sampling_seconds=2,thermal_mode='macos_state_only_acknowledged',test_stop=test_stop,
            managed_renewal_authorized=managed,before_solutions_sha256=before,initial=initial)
        persist_run(monitor_dir,history,record)
        stop=False;child=None;active=None;triggered=False;next_solving_since=None
        parent_pid=os.getppid();last_renew=time.monotonic();failure=None
        printed_ids=None;last_print=0.
        def on_signal(signum,frame):
            nonlocal stop
            stop=True
        old={s:signal.signal(s,on_signal) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        try:
            with (monitor_dir/'controller.log').open('x') as log,(monitor_dir/'samples.jsonl').open('x') as samples,(monitor_dir/'lease_renewals.jsonl').open('x') as renewals:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
                record['controller_pid']=child.pid
                persist_run(monitor_dir,history,record)
                while True:
                    if (out/'active.json').exists():
                        candidate=read_json(out/'active.json')
                        if candidate.get('controller_pid')==child.pid:active=candidate
                        elif active is not None and child.poll() is None:raise SafetyStop('active_session_replaced')
                    sample=snapshot(out,active)
                    sample['saved_new_ids']=[pid for pid in selected if (out/'solutions'/f'{pid}.json').exists()]
                    samples.write(json.dumps(sample)+'\n');samples.flush();os.fsync(samples.fileno())
                    record['last_sample_epoch']=sample['epoch']
                    reason=process_reason(child,active,sample,len(sample['saved_new_ids'])==len(selected))
                    if stop or external_stop():reason=reason or 'operator_stop'
                    if os.getppid()!=parent_pid:reason=reason or 'monitor_parent_lost'
                    if managed:
                        failed=[pid for pid in sample['saved_new_ids'] if
                                read_json(out/'solutions'/f'{pid}.json')['computation_status']!='success']
                        if failed:reason=reason or 'new_terminal_computation_failure';record['failed_ids']=failed
                    if sample['saved_new_ids']!=printed_ids or time.monotonic()-last_print>=30:
                        print(json.dumps(dict(batch=str(monitor_dir.name),saved_new=len(sample['saved_new_ids']),
                            workers=sample['compute_worker_count'],thermal=sample['thermal_name'],
                            worker_cpu=[w['cpu_percent'] for w in sample['compute_workers']],
                            automatic_lease_renewals=record['automatic_lease_renewals'])),flush=True)
                        printed_ids=sample['saved_new_ids'];last_print=time.monotonic()
                    if reason:
                        record['stop_reason']=reason
                        record['stop_response']=stop_owned(out,child);triggered=True
                    if test_stop and not triggered and selected[0] in sample['saved_new_ids']:
                        if selected[1] in sample['solving_ids'] and selected[1] not in sample['saved_new_ids']:
                            next_solving_since=next_solving_since or time.monotonic()
                            if time.monotonic()-next_solving_since>=2:
                                record['stop_reason']='planned_stop_during_second_real_equation'
                                record['stop_response']=stop_owned(out,child);triggered=True
                    if child.poll() is not None:break
                    if triggered:
                        child.wait(timeout=10);break
                    if (managed and active and len(sample['saved_new_ids'])<len(selected) and
                        sample['compute_worker_count']==1 and time.monotonic()-last_renew>=RENEW_SECONDS):
                        renewed=renew_checked(out,child,active,sample)
                        renewals.write(json.dumps(renewed)+'\n');renewals.flush();os.fsync(renewals.fileno())
                        record['automatic_lease_renewals']+=1;last_renew=time.monotonic()
                    persist_run(monitor_dir,history,record)
                    time.sleep(SAMPLE_SECONDS)
                child.wait(timeout=10)
        except BaseException as exc:
            failure=exc;record['monitor_error']=f'{type(exc).__name__}: {exc}'
        finally:
            if child is not None and child.poll() is None:
                record['final_stop_response']=stop_owned(out,child)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=8)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=3)
            for sig,handler in old.items():signal.signal(sig,handler)
            record['finished_epoch']=time.time()
            record['returncode']=child.returncode if child else None
            record['new_ids']=[pid for pid in selected if (out/'solutions'/f'{pid}.json').exists()]
            record['old_solutions_unchanged']=all(digest(out/'solutions'/f'{pid}.json')==h for pid,h in before.items())
            try:record['final_sample']=snapshot(out,active)
            except Exception as exc:record['final_sample']=dict(telemetry_error=f'{type(exc).__name__}: {exc}')
            final=record['final_sample']
            record['cleanup_confirmed']=final.get('compute_worker_count')==0 and all(
                g.get('finished') and g.get('cleanup_success',True) for g in final.get('guardians',[])) and not backend_busy()
            persist_run(monitor_dir,history,record)
        if not record['old_solutions_unchanged'] or not record['cleanup_confirmed']:
            raise RuntimeError('preservation/cleanup validation failed; inspect run.json')
        print(json.dumps(dict(run_id=run_id,new_ids=record['new_ids'],
            returncode=record['returncode'],cleanup_confirmed=True,
            automatic_lease_renewals=record['automatic_lease_renewals'])))
        if failure:raise SafetyStop(record['monitor_error']) from failure
        if record['returncode'] not in (0,130):raise RuntimeError('backend failed; inspect controller.log')
        return record


def wait_cooldown(out,monitor_dir,seconds,should_stop,parent_pid):
    """Cool without a worker, checking thermal state and stop requests throughout."""
    deadline=time.monotonic()+seconds
    print(json.dumps(dict(phase='cooling',seconds=round(seconds,1))),flush=True)
    with (monitor_dir/'cooling_samples.jsonl').open('a') as log:
        while time.monotonic()<deadline:
            if should_stop():raise SafetyStop('operator_stop_during_cooldown')
            if os.getppid()!=parent_pid:raise SafetyStop('monitor_parent_lost')
            sample=snapshot(out)
            log.write(json.dumps(sample)+'\n');log.flush();os.fsync(log.fileno())
            if sample['safety_stop_reason']:raise SafetyStop(sample['safety_stop_reason'])
            if sample['compute_worker_count'] or backend_busy():raise SafetyStop('unexpected_backend_during_cooldown')
            time.sleep(min(SAMPLE_SECONDS,max(0,deadline-time.monotonic())))


def stop_monitor(out):
    """Token-scoped series stop, including pauses; never kill a guessed PID."""
    pointer=ROOT/'results/chernov_monitored_active.json'
    active=read_json(pointer) if pointer.exists() else None
    if active and not active.get('finished') and Path(active['output_dir'])==out:
        atomic(Path(active['monitor_dir'])/'stop.json',dict(token=active['token'],requested_epoch=time.time()))
        backend=read_json(out/'active.json') if (out/'active.json').exists() else None
        response=request_stop(out) if backend and not backend.get('finished') else None
        return dict(series_stop_requested=True,monitor_dir=active['monitor_dir'],backend=response)
    return dict(series_stop_requested=False,backend=request_stop(out))


def run_all(out,monitor_dir,batch_size,state_only):
    if not 10<=batch_size<=20:raise ValueError('managed batch size must be 10..20')
    if not state_only:raise ValueError('Celsius unavailable: select --thermal-state-only')
    with controller_locks(monitor_dir,ROOT/'results/.chernov_small_batch.lock'):
        plan=read_json(out/'plan.json');check_protected(plan,out)
        if backend_busy():raise SafetyStop('a backend is already running/cleaning up')
        completed=terminal_ids(out,plan)
        pending=[pid for pid in plan['equation_ids'] if pid not in completed]
        before={p.stem:digest(p) for p in (out/'solutions').glob('*.json')}
        token=uuid.uuid4().hex;parent_pid=os.getppid();stop=False
        pointer=dict(token=token,monitor_pid=os.getpid(),parent_pid=parent_pid,
                     output_dir=str(out),monitor_dir=str(monitor_dir),finished=False)
        record=dict(**pointer,phase='starting',started_epoch=time.time(),batch_size=batch_size,
                    pending_ids_at_start=pending,previous_terminal_records=len(completed),
                    config=plan['config'],plan_sha256=digest(out/'plan.json'),
                    safety_policy_sha256=digest(out/'safety_policy_v2.json'),source_sha256=digest(Path(__file__)),
                    owner_lease_seconds=120,renew_interval_seconds=RENEW_SECONDS,
                    checked_automatic_renewal_authorized=True,inter_batch_cooldown_seconds=60,
                    sampling_seconds=SAMPLE_SECONDS,thermal_mode='macos_state_only_acknowledged',
                    before_solutions_sha256=before,batches=[])
        atomic(ROOT/'results/chernov_monitored_active.json',pointer);atomic(monitor_dir/'series.json',record)
        def on_signal(signum,frame):
            nonlocal stop
            stop=True
        def should_stop():return stop or (monitor_dir/'stop.json').exists()
        old={s:signal.signal(s,on_signal) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        try:
            while True:
                if should_stop():raise SafetyStop('operator_stop')
                if os.getppid()!=parent_pid:raise SafetyStop('monitor_parent_lost')
                remaining=[pid for pid in pending if pid not in terminal_ids(out,plan)]
                if not remaining:record['phase']='completed';break
                wait=cooldown_remaining(ROOT/'results/chernov_small_batch_history')
                if wait>0:
                    record['phase']='cooling';atomic(monitor_dir/'series.json',record)
                    wait_cooldown(out,monitor_dir,wait,should_stop,parent_pid)
                if should_stop():raise SafetyStop('operator_stop')
                if digest(Path(__file__))!=record['source_sha256']:raise SafetyStop('monitor_source_changed')
                folder=monitor_dir/f"batch_{len(record['batches'])+1:04d}";folder.mkdir(exist_ok=False)
                record['phase']='running';record['current_batch']=str(folder)
                atomic(monitor_dir/'series.json',record)
                result=run(out,folder,min(batch_size,len(remaining)),True,managed=True,
                           lock_held=True,external_stop=should_stop)
                record['batches'].append(dict(directory=str(folder),new_ids=result['new_ids'],
                    returncode=result['returncode'],automatic_lease_renewals=result['automatic_lease_renewals'],
                    cleanup_confirmed=result['cleanup_confirmed']))
                atomic(monitor_dir/'series.json',record)
                if result.get('stop_reason') or result['returncode']!=0:
                    raise SafetyStop(result.get('stop_reason','backend_stopped'))
                if len(result['new_ids'])!=min(batch_size,len(remaining)):
                    raise SafetyStop('batch_did_not_commit_expected_answers')
        except BaseException as exc:
            record['phase']='stopped';record['stop_reason']=f'{type(exc).__name__}: {exc}'
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
            record['finished']=True;record['finished_epoch']=time.time()
            record['terminal_records']=len(terminal_ids(out,plan))
            record['pending']=len(plan['equation_ids'])-record['terminal_records']
            record['old_solutions_unchanged']=all(digest(out/'solutions'/f'{pid}.json')==h for pid,h in before.items())
            record['backend_locks_free']=not backend_busy()
            if not record['old_solutions_unchanged'] or not record['backend_locks_free']:
                record['phase']='stopped';record['stop_reason']='preservation_or_cleanup_failed'
            atomic(monitor_dir/'series.json',record)
            atomic(ROOT/'results/chernov_monitored_active.json',dict(pointer,finished=True,phase=record['phase']))
        print(json.dumps({k:record[k] for k in ('phase','terminal_records','pending','old_solutions_unchanged','backend_locks_free')}),flush=True)
        return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--max-new',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--run',action='store_true',help='start computation in the foreground')
    parser.add_argument('--all-pending',action='store_true',help='explicitly authorize sequential managed batches with checked lease renewal')
    parser.add_argument('--batch-size',type=int,choices=range(10,21),default=20)
    parser.add_argument('--stop',action='store_true',help='stop the entire managed series, including cooldown')
    parser.add_argument('--thermal-state-only',action='store_true',help='acknowledge macOS thermal-state monitoring, without degrees C')
    parser.add_argument('--monitor-dir',type=Path)
    parser.add_argument('--test-stop-after-first',action='store_true')
    args=parser.parse_args();out=args.output_dir.resolve()
    if args.stop:
        if args.run:parser.error('--stop cannot be combined with --run')
        print(json.dumps(stop_monitor(out)));return
    if args.all_pending and (args.test_stop_after_first or args.max_new!=3):
        parser.error('--all-pending uses --batch-size, not --max-new or the pilot stop')
    if not args.run:
        print(json.dumps(snapshot(out)));return
    monitor=args.monitor_dir.resolve() if args.monitor_dir else ROOT/'results'/(
        'chernov_small_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6])
    if not monitor.is_relative_to(ROOT/'results'):parser.error('monitor-dir must be under results')
    monitor.mkdir(parents=True,exist_ok=False)
    if args.all_pending:
        result=run_all(out,monitor,args.batch_size,args.thermal_state_only)
        if result['phase']!='completed':raise SystemExit(130)
    else:run(out,monitor,args.max_new,args.thermal_state_only,args.test_stop_after_first)


if __name__=='__main__':main()
