"""Foreground, finite batches with external activity/thermal monitoring.

Never renews the backend's 120-second owner lease. No loop over all pending
tasks. Backend mathematical sources and its frozen safety policy stay intact.
macOS thermalState is NOT a CPU temperature in degrees (unknown can report 0).
The operator explicitly acknowledges this limitation for a state-only run.
"""
from __future__ import annotations

import argparse
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
from diagnostics.chernov_batch_control import atomic, controller_locks, lock_is_held, request_stop
from diagnostics.run_chernov_batch import check_protected, digest, read_json

THERMAL_SCRIPT='ObjC.import("Foundation"); JSON.stringify({thermalState:Number($.NSProcessInfo.processInfo.thermalState)})'
COOLDOWN_SECONDS=60.


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
    return dict(utc=datetime.now(timezone.utc).isoformat(),epoch=time.time(),
        **thermal,compute_workers=workers,compute_worker_count=len(workers),
        solving_ids=solving,guardians=guardians,safety_stop_reason=safety_reason(thermal,workers))


def stop_owned(out,child):
    active=read_json(out/'active.json') if (out/'active.json').exists() else None
    if active and active.get('controller_pid')==child.pid:
        return request_stop(out)
    return dict(stop_requested=False,reason='our controller has no active session')


def run(out,monitor_dir,count,state_only,test_stop=False):
    if count not in (1,2,3):raise ValueError('batch size must be 1..3')
    if not state_only:
        raise ValueError('Celsius unavailable: explicitly select --thermal-state-only or do not start')
    if test_stop and count!=2:raise ValueError('stop/resume test requires --max-new 2')
    # A monitor lock, distinct from backend locks. No second thermal monitor run.
    with controller_locks(monitor_dir,ROOT/'results/.chernov_small_batch.lock'):
        plan=read_json(out/'plan.json');check_protected(plan,out)
        if lock_is_held(ROOT/'results/.chernov_batch_controller.lock') or lock_is_held(ROOT/'results/.chernov_batch_compute.lock'):
            raise RuntimeError('a backend is already running/cleaning up')
        history=ROOT/'results/chernov_small_batch_history';history.mkdir(exist_ok=True)
        wait=cooldown_remaining(history)
        if wait>0:raise RuntimeError(f'cooling interval: wait {math.ceil(wait)} seconds; no worker started')
        before={p.stem:digest(p) for p in (out/'solutions').glob('*.json')}
        selected=[pid for pid in plan['equation_ids'] if pid not in before][:count]
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
            before_solutions_sha256=before,initial=initial)
        atomic(monitor_dir/'run.json',record);atomic(history/f'{run_id}.json',record)
        stop=False;child=None;active=None;triggered=False;next_solving_since=None
        def on_signal(signum,frame):
            nonlocal stop
            stop=True
        old={s:signal.signal(s,on_signal) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        try:
            with (monitor_dir/'controller.log').open('x') as log,(monitor_dir/'samples.jsonl').open('x') as samples:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                record['controller_pid']=child.pid
                while True:
                    if (out/'active.json').exists():
                        candidate=read_json(out/'active.json')
                        if candidate.get('controller_pid')==child.pid:active=candidate
                    sample=snapshot(out,active)
                    sample['saved_new_ids']=[pid for pid in selected if (out/'solutions'/f'{pid}.json').exists()]
                    samples.write(json.dumps(sample)+'\n');samples.flush();os.fsync(samples.fileno())
                    record['last_sample_epoch']=sample['epoch']
                    atomic(history/f'{run_id}.json',record)
                    print(json.dumps(dict(saved_new=len(sample['saved_new_ids']),
                        workers=sample['compute_worker_count'],thermal=sample['thermal_name'],
                        worker_cpu=[w['cpu_percent'] for w in sample['compute_workers']])),flush=True)
                    if sample['safety_stop_reason'] or stop:
                        record['stop_reason']=sample['safety_stop_reason'] or 'operator_signal'
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
                    time.sleep(2)
                child.wait(timeout=10)
        finally:
            if child is not None and child.poll() is None:
                record['final_stop_response']=stop_owned(out,child)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.terminate();child.wait(timeout=8)
            for sig,handler in old.items():signal.signal(sig,handler)
            record['finished_epoch']=time.time()
            record['returncode']=child.returncode if child else None
            record['new_ids']=[pid for pid in selected if (out/'solutions'/f'{pid}.json').exists()]
            record['old_solutions_unchanged']=all(digest(out/'solutions'/f'{pid}.json')==h for pid,h in before.items())
            record['final_sample']=snapshot(out,active)
            record['cleanup_confirmed']=record['final_sample']['compute_worker_count']==0 and all(
                g.get('finished') and g.get('cleanup_success',True) for g in record['final_sample']['guardians']) and not (
                    lock_is_held(ROOT/'results/.chernov_batch_controller.lock') or
                    lock_is_held(ROOT/'results/.chernov_batch_compute.lock'))
            atomic(monitor_dir/'run.json',record);atomic(history/f'{run_id}.json',record)
        if not record['old_solutions_unchanged'] or not record['cleanup_confirmed']:
            raise RuntimeError('preservation/cleanup validation failed; inspect run.json')
        print(json.dumps(dict(run_id=run_id,new_ids=record['new_ids'],
            returncode=record['returncode'],cleanup_confirmed=True,automatic_lease_renewals=0)))
        if record['returncode'] not in (0,130):raise RuntimeError('backend failed; inspect controller.log')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--max-new',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--run',action='store_true',help='explicitly start ONE finite batch')
    parser.add_argument('--thermal-state-only',action='store_true',help='acknowledge macOS thermal-state monitoring, without degrees C')
    parser.add_argument('--monitor-dir',type=Path)
    parser.add_argument('--test-stop-after-first',action='store_true')
    args=parser.parse_args();out=args.output_dir.resolve()
    if not args.run:
        print(json.dumps(snapshot(out)));return
    monitor=args.monitor_dir.resolve() if args.monitor_dir else ROOT/'results'/(
        'chernov_small_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6])
    if not monitor.is_relative_to(ROOT/'results'):parser.error('monitor-dir must be under results')
    monitor.mkdir(parents=True,exist_ok=False)
    run(out,monitor,args.max_new,args.thermal_state_only,args.test_stop_after_first)


if __name__=='__main__':main()
