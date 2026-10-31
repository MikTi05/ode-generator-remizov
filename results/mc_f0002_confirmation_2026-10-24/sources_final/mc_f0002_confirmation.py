"""Bounded independent F0002 control, then separately frozen three-level study.

Reuse the existing diagnostic Brownian adapter and unmodified production kernel.
No path-count adaptation, retries, pooling with prior pilots, or h<1/128.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.compare_chernov import THREAD_VARS, load_task, mc_forcing_certificate, write_json
from diagnostics.mc_convergence import digest, now, protected_hashes
from diagnostics.mc_f0002_time_refinement import config_dict, costs, coupled_paths

for name in THREAD_VARS:
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
SOURCES=('diagnostics/mc_f0002_confirmation.py','diagnostics/report_mc_f0002_confirmation.py',
         'tests/test_mc_f0002_confirmation.py')
PHASES=('control','refinement128')


def old_seeds():
    paths=[ROOT/'results/mc_convergence_2026-10-24/plan.json',
           ROOT/'results/mc_f0002_coupled_h_2026-10-24/plan.json']
    return {a.get('seed',a.get('config',{}).get('seed'))
            for p in paths for a in json.loads(p.read_text())['attempts']}


def protected(out):
    result=protected_hashes(out)
    # Protect existing diagnostic and test sources as well as the numerical core.
    for folder in ('diagnostics','tests','docs','.claude'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and str(p.relative_to(ROOT)) not in SOURCES and p.suffix!='.pyc':
                result[str(p.relative_to(ROOT))]=digest(p)
    return result


def prepare(out):
    import numpy as np
    from mc_solver import _gap_lower, _tail_bound, auto_T
    out.mkdir(parents=True,exist_ok=False)
    (out/'before').mkdir();shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'before/PROJECT_STATUS.md')
    write_json(out/'protected_before.json',protected(out))
    p,_,_,meta=load_task('AMV1_F0002_S00')
    cert,proof=mc_forcing_certificate({},p,meta)
    task=dict(problem_id=p.problem_id,x=0.,T=63.625,tail_eps=1e-4,
        certified_T=auto_T(p.lambda_value,p.c_abs_upper_bound,cert.abs_upper_bound,1e-4),
        forcing_bound=asdict(cert),forcing_proof=proof,
        tail_error_bound=_tail_bound(cert.abs_upper_bound,_gap_lower(p.lambda_value,p.c_abs_upper_bound),63.625))
    assert task['tail_error_bound']<=task['tail_eps']
    attempts=[]
    for phase_index,(phase,hs) in enumerate([('control',[1/32,1/64]),
                                             ('refinement128',[1/32,1/64,1/128])]):
        for rep in range(8):
            seed=int(np.random.SeedSequence(2026102410,spawn_key=(phase_index,rep)).generate_state(1,dtype=np.uint64)[0])
            attempts.append(dict(attempt_id=f'{phase}_{rep:02d}',stage=phase,repeat=rep,
                M=1024,seed=seed,configs=[config_dict(1024,h,seed) for h in hs],
                maximum_cost=costs(1024,63.625,hs)))
    seeds={a['seed'] for a in attempts}
    assert len(seeds)==16 and not seeds&old_seeds()
    total=sum(a['maximum_cost']['path_steps'] for a in attempts)
    assert total<=170_000_000
    policy=dict(version='F0002_independent_completion_v1',frozen_utc=now(),**task,
        master_seed=2026102410,attempts=attempts,
        gate=dict(required_repeats=8,maximum_relative_ci95_half=.5,minimum_variance_reduction=2.,
            condition='All 8 new control repeats succeed; t95 half-width/abs(paired mean)<=0.5 '
                'and estimated variance reduction>=2. No prior pilot data in the calculation.'),
        next_phase=dict(only_if_gate_passes=True,freeze_separate_plan_before_worker=True,
            levels=[1/32,1/64,1/128],repeats=8,M_per_seed=1024,
            independent_from_control=True,stop='Fixed sample size. No automatic increase if .01 '
                'remains unconfirmed, contrasts unresolved, or costs excessive. No h<1/128.'),
        analysis=dict(intervals='t95 across independent run means; Bonferroni simultaneous intervals '
                'within each phase for level bias and adjacent contrasts',
            bootstrap_seed=2026102411,bootstrap_replicates=10000,
            order='Three-level contrast ratio is only a model diagnostic, not an asymptotic-order claim; '
                'do not fit a power if adjacent simultaneous contrast intervals include zero.',
            cost='sigma measured only at x=0. Estimate normal pointwise .005/.01 statistical half-width '
                'and 33-point simultaneous Bonferroni cost, assuming the same sigma at all points. '
                'No full-error certificate, grid sampling, or benchmark winner claim.',
            timing='Extrapolate serial chunk=1024 full batches from observed finest-level runtime. '
                'Not a guaranteed wall-time prediction.'),
        resources=dict(workers=1,blas_threads=1,rss_limit_mib=384.,total_wall_seconds=240.,
            phase_wall_seconds=180.,attempt_cooperative_seconds=20.,max_path_steps=170_000_000,
            planned_maximum_path_steps=total,
            planned_maximum_normal_values_generated=sum(a['maximum_cost']['normal_values_generated_with_replay'] for a in attempts),
            prior_seconds_estimate=[65.,100.],
            stop='Stop on failure or external resource limit, preserve partial counters and not_run; no retries'))
    write_json(out/'policy.json',policy)
    write_json(out/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','sympy','matplotlib','pytest')},
        thread_environment={k:os.environ[k] for k in THREAD_VARS}))
    (out/'sources').mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT/name,out/'sources'/Path(name).name)
    print(json.dumps(dict(attempts=16,max_path_steps=total,
        phase_costs={phase:sum(a['maximum_cost']['path_steps'] for a in attempts if a['stage']==phase) for phase in PHASES},
        tail_bound=task['tail_error_bound'])),flush=True)


def freeze_phase(out,phase,policy):
    phase_out=out/phase;phase_out.mkdir(exist_ok=False);(phase_out/'records').mkdir()
    plan={k:policy[k] for k in ('problem_id','x','T','tail_eps','tail_error_bound','forcing_bound','analysis')}
    plan.update(phase=phase,frozen_utc=now(),parent_policy_sha256=digest(out/'policy.json'),
        attempts=[a for a in policy['attempts'] if a['stage']==phase],resources=policy['resources'])
    if phase=='refinement128':
        plan['gate_decision_sha256']=digest(out/'gate.json')
    write_json(phase_out/'plan.json',plan)
    return phase_out


def worker(phase_out):
    import numpy as np
    from mc_solver import MCConfig, MCForcingBound
    plan=json.loads((phase_out/'plan.json').read_text())
    began=time.perf_counter();p,exact,_,_=load_task(plan['problem_id'])
    bound=MCForcingBound(**plan['forcing_bound']);stop=threading.Event()
    def monitor():
        with (phase_out/'rss.jsonl').open('x') as f:
            while not stop.is_set():
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
                f.write(json.dumps(dict(seconds=time.perf_counter()-began,peak_rss_mib=rss))+'\n');f.flush();stop.wait(.1)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        for spec in plan['attempts']:
            aid=spec['attempt_id'];write_json(phase_out/'records'/f'{aid}.started.json',dict(started_utc=now(),spec=spec))
            configs=[MCConfig(**cfg) for cfg in spec['configs']]
            Z,result=coupled_paths(p,plan['x'],configs,bound,
                runtime_seconds=plan['resources']['attempt_cooperative_seconds'])
            hs=[cfg.dt for cfg in configs]
            y=float(exact(np.array([plan['x']]))[0])
            if Z is not None:
                np.savez_compressed(phase_out/'records'/f'{aid}.samples.npz',Z=Z,hs=hs)
            write_json(phase_out/'records'/f'{aid}.result.json',dict(spec=spec,hs=hs,result=result,
                y_exact=y,process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                (1024**2 if sys.platform=='darwin' else 1024)))
            print(json.dumps(dict(attempt=aid,success=result['success'],seconds=result['runtime_seconds'])),flush=True)
            if not result['success']:
                break
    finally:
        stop.set();thread.join()


def bounded_phase(phase_out,overall_began,limits):
    start=time.perf_counter();reason=None;peak=0.;frozen=digest(phase_out/'plan.json')
    with (phase_out/'worker.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker',
            '--out',str(phase_out)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            sample=phase_out/'rss.jsonl'
            if sample.exists():
                lines=sample.read_text().splitlines()
                try:
                    if lines:peak=max(peak,json.loads(lines[-1])['peak_rss_mib'])
                except json.JSONDecodeError:pass
            if time.perf_counter()-overall_began>limits['total_wall_seconds']:reason='external_total_time_limit'
            elif time.perf_counter()-start>limits['phase_wall_seconds']:reason='external_phase_time_limit'
            elif peak>limits['rss_limit_mib']:reason='external_memory_limit'
            if reason:
                child.terminate()
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                break
            time.sleep(.1)
    if child.returncode and not reason:reason='worker_exception'
    files=list((phase_out/'records').glob('*.result.json'))
    if any(not json.loads(p.read_text())['result']['success'] for p in files):reason='solver_failure'
    assert frozen==digest(phase_out/'plan.json')
    return dict(phase=phase_out.name,wall_seconds=time.perf_counter()-start,reason=reason,
        completed=len(files),returncode=child.returncode,sampled_peak_rss_mib=peak,plan_sha256=frozen)


def run(out):
    from diagnostics.report_mc_f0002_confirmation import gate_from_summary
    from diagnostics.report_mc_f0002_time_refinement import summarize_stage
    policy=json.loads((out/'policy.json').read_text());frozen=digest(out/'policy.json')
    with (out/'run.lock').open('x') as f:f.write(now())
    began=time.perf_counter();events=[]
    control=freeze_phase(out,'control',policy)
    events.append(bounded_phase(control,began,policy['resources']))
    if events[-1]['reason']:
        gate=dict(proceed=False,reason=events[-1]['reason'])
    else:
        summary,_=summarize_stage(control,'control')
        gate=gate_from_summary(summary,policy['gate'])
    gate['decided_utc']=now()
    write_json(out/'gate.json',gate)
    print(json.dumps(dict(control_event=events[-1],gate=gate)),flush=True)
    if gate['proceed']:
        # This separate plan is persisted only after the independent gate,
        # before the refinement process is even created.
        refinement=freeze_phase(out,'refinement128',policy)
        events.append(bounded_phase(refinement,began,policy['resources']))
        print(json.dumps(events[-1]),flush=True)
    attempts=[]
    for a in policy['attempts']:
        path=out/a['stage']/'records'/f"{a['attempt_id']}.result.json"
        status='not_run'
        if path.exists():
            r=json.loads(path.read_text())['result'];status='success' if r['success'] else r['failure']['reason']
        elif path.with_name(path.name.replace('.result.json','.started.json')).exists():
            status=events[-1]['reason'] or 'incomplete'
        attempts.append(dict(attempt_id=a['attempt_id'],phase=a['stage'],status=status))
    write_json(out/'run_summary.json',dict(finished_utc=now(),wall_seconds=time.perf_counter()-began,
        events=events,attempts=attempts))
    before=json.loads((out/'protected_before.json').read_text());after=protected(out)
    changed=[p for p,h in before.items() if after.get(p)!=h]
    write_json(out/'verification.json',dict(protected_count=len(before),changed_or_removed=changed,
        after_sha256=after,policy_sha256_before=frozen,policy_sha256_after=digest(out/'policy.json')))
    assert not changed and frozen==digest(out/'policy.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','worker'));parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    if not out.is_relative_to(ROOT/'results') or out==ROOT/'results':parser.error('use a new directory under results/')
    (worker if args.action=='worker' else prepare if args.action=='prepare' else run)(out)


if __name__=='__main__':main()
