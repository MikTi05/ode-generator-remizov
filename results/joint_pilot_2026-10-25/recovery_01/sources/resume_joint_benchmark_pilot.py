"""Freeze exact unfinished logical trials after the recorded operator interruption.

The original plan, stopped summary and partial trial are immutable. A continuation
uses the same production driver, configurations and seeds in a separate directory.
Only the external operator grace period changes; no numerical parameter changes.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')


def prepare(out):
    original=read(out/'plan.json');stopped=read(out/'summary.json')
    if stopped['status']!='stopped_safely':raise ValueError('expected recorded safe stop')
    if stopped['sessions'][-1]['stop_reason']!='owner_lease_expired':raise ValueError('other interruption requires audit')
    pairs=list(zip(original['attempts'],stopped['runs']))
    interrupted=[a for a,r in pairs if r['status']=='safety_stopped']
    pending=[a for a,r in pairs if r['status']=='not_run']
    if len(interrupted)!=1 or len(pending)!=28:raise ValueError('unexpected recovery scope')
    trial=out/'trials'/interrupted[0]['attempt_id']
    if (trial/'prediction.json').exists() or (trial/'score.json').exists():
        raise ValueError('do not silently retry an already scored answer')
    guard=read(out/next(r['guardian'] for a,r in pairs if r['status']=='safety_stopped'))
    if not guard['finished'] or not guard['cleanup_success']:raise ValueError('cleanup incomplete')
    cont=out/'recovery_01';cont.mkdir(exist_ok=False);(cont/'trials').mkdir();(cont/'sources').mkdir()
    plan=copy.deepcopy(original);plan['attempts']=interrupted+pending
    plan['resources']['external_owner_lease_seconds']=300
    plan['parent_plan_sha256']=sha(out/'plan.json')
    plan['continuation_reason']='Operator missed external renewal; guardian stopped an unscored trial. Same seed/config restarted, remaining fixed queue follows.'
    plan['external_lease_change']='300s operator grace accommodates tool/code/report communication; does not limit worker duration. Parent pipe/heartbeat cleanup still immediate/15s; CPU/RSS/progress guards unchanged.'
    plan['maximum_work']=dict(attempts=len(plan['attempts']),
        mc_path_steps=sum(a['work'].get('path_steps',0) for a in plan['attempts']),
        remizov_operator_applications=sum(a['work'].get('operator_applications',0) for a in plan['attempts']))
    plan['source_sha256']['diagnostics/resume_joint_benchmark_pilot.py']=sha(Path(__file__))
    write(cont/'plan.json',plan)
    write(cont/'summary.json',dict(plan_sha256=sha(cont/'plan.json'),status='prepared',sessions=[],
        runs=[dict(attempt_id=a['attempt_id'],status='not_run',reason='not_started') for a in plan['attempts']]))
    shutil.copy2(__file__,cont/'sources'/Path(__file__).name)
    last=read(trial/'telemetry.json')
    write(out/'interruption.json',dict(reason='operator_owner_lease_expired',mathematical_failure=False,
        original_trial=str(trial.relative_to(out)),retry_directory='recovery_01',retry_identical_seed_config=True,
        no_partial_prediction_or_exact_score=True,partial_telemetry=last,
        sampled_cpu_seconds_lower_bound=last['cpu_seconds'],
        completed_path_steps_lower_bound=last['completed_units'],
        actual_partial_path_steps_upper_bound=interrupted[0]['work']['path_steps'],
        partial_worker_wall_seconds=next(r['worker_process_seconds'] for a,r in pairs if r['status']=='safety_stopped'),
        cleared_guardian=guard,uncontrolled_workers_after_cleanup=False,
        original_plan_sha256=sha(out/'plan.json'),original_summary_sha256=sha(out/'summary.json'),
        continuation_plan_sha256=sha(cont/'plan.json')))
    print(json.dumps(dict(continuation=str(cont),logical_attempts=len(plan['attempts']),
                         same_parameters_and_seeds=True,external_owner_lease_seconds=300)))


def combine(out):
    original=read(out/'plan.json');first=read(out/'summary.json');cont=out/'recovery_01'
    second=read(cont/'summary.json');cp=read(cont/'plan.json')
    if second['status']!='completed':raise ValueError('continuation unfinished')
    parent={a['attempt_id']:a for a in original['attempts']}
    for a in cp['attempts']:
        if a!=parent[a['attempt_id']]:raise ValueError('logical parameters changed')
    resolved={r['attempt_id']:r for r in first['runs'] if r['status']=='completed'}
    directories={r['attempt_id']:'trials/'+r['attempt_id'] for r in first['runs'] if r['status']=='completed'}
    for r in second['runs']:
        if r['status']!='completed' or r['attempt_id'] in resolved:raise ValueError('duplicate/incomplete logical answer')
        rr=copy.deepcopy(r);rr['guardian']='recovery_01/'+r['guardian']
        resolved[r['attempt_id']]=rr;directories[r['attempt_id']]='recovery_01/trials/'+r['attempt_id']
    if len(resolved)!=262:raise ValueError('262 logical answers required')
    sessions=[dict(s,segment='initial') for s in first['sessions']]+[dict(s,segment='recovery_01') for s in second['sessions']]
    merged=dict(status='completed',plan_sha256=sha(out/'plan.json'),
        runs=[resolved[a['attempt_id']] for a in original['attempts']],sessions=sessions,
        trial_dirs=directories,outcomes={'completed':262},physical_attempts=263,
        physical_outcomes={'completed':262,'operator_safety_stop':1},
        interruption=read(out/'interruption.json'),
        segment_active_files=['active.json','recovery_01/active.json'])
    write(out/'combined_summary.json',merged)
    print(json.dumps(dict(logical_answers=262,physical_attempts=263,operator_stops=1)))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('prepare','combine'))
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.action=='prepare':prepare(args.out.resolve())
    else:combine(args.out.resolve())


if __name__=='__main__':main()
