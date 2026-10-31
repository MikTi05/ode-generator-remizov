"""Two independent bounded n=4 runs. Instrumentation is diagnostic only.

First run F0005 uninstrumented; second run F0000 with cProfile and stage logs.
The original solver functions receive exactly the original arguments once.
No caching, interpolation substitutions, tolerance changes, or retries.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict
from datetime import datetime, timezone
import functools
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import pstats
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.remizov_n_study import clean, snapshot, write_json, THREAD_VARS

WALL_LIMIT = 45.0
RSS_LIMIT_MIB = 512
POLL_SECONDS = .25
EXPECTED_CONFIG = dict(n=4, tail_eps=1e-8, nt_initial=8, dx_initial=.05,
    quadrature_rtol=1e-7, spatial_rtol=1e-7, quadrature_atol=1e-9,
    spatial_atol=1e-9, max_quadrature_refinements=8,
    max_spatial_refinements=5, max_work_grid_points=100000)


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    write_json(temporary, data)
    temporary.replace(path)


class SolverObserver:
    """Observe original function boundaries; retain state across interruption."""
    def __init__(self, folder):
        self.folder = folder
        self.profiler = cProfile.Profile()
        self.stack = []
        self.completed = {}
        self.diag = None
        self.events = (folder/'stage_events.jsonl').open('x', buffering=1)
        self.started = time.perf_counter()
        self.last_flush = self.started
        self.observer_seconds = 0.0
        self.patched = []

    def state(self):
        now = time.perf_counter()
        return {
            'elapsed_solver_observed_seconds': now-self.started,
            'completed_inclusive_timings': self.completed,
            'active_calls': [{'function': x['name'], 'context': x['context'],
                             'elapsed_seconds': now-x['start']} for x in self.stack],
            'partial_solver_diagnostics': asdict(self.diag) if self.diag is not None else None,
            'observer_io_seconds_completed': self.observer_seconds,
            'inclusive_timings_overlap_do_not_sum': True,
            'instrumentation_changes_timing_not_arithmetic': True,
        }

    def flush(self, event, force=False):
        if not force and time.perf_counter()-self.last_flush < 1.0:
            return
        began = time.perf_counter()
        self.profiler.disable()
        state = self.state()
        self.events.write(json.dumps(clean({'event': event, **state}), allow_nan=False)+'\n')
        atomic_json(self.folder/'progress.json', state)
        self.profiler.dump_stats(str(self.folder/'partial_profile.pstats'))
        self.observer_seconds += time.perf_counter()-began
        self.last_flush = time.perf_counter()
        self.profiler.enable()

    def wrap(self, owner, name, context):
        original = getattr(owner, name)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            frame = {'name': name, 'context': context(args, kwargs),
                     'start': time.perf_counter()}
            if name == '_converge_spatial':
                self.diag = args[4]
            self.stack.append(frame)
            if name in ('_converge_spatial', '_converge_quadrature', '_integrate_fixed_nt'):
                self.flush('enter:'+name, force=True)
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter()-frame['start']
                stats = self.completed.setdefault(name, {'calls_completed': 0,
                                                         'inclusive_seconds': 0.0})
                stats['calls_completed'] += 1
                stats['inclusive_seconds'] += elapsed
                self.stack.pop()
                self.flush('return:'+name, force=name in
                           ('_converge_spatial', '_converge_quadrature', '_integrate_fixed_nt'))
        setattr(owner, name, observed)
        self.patched.append((owner, name, original))

    def install(self):
        import numpy as np
        import remizov_solver as solver
        self.wrap(solver, '_converge_spatial', lambda a,k: {})
        self.wrap(solver, '_converge_quadrature', lambda a,k: {'dx_limit': a[4]})
        self.wrap(solver, '_integrate_fixed_nt', lambda a,k: {'dx_limit': a[4], 'nt': a[5]})
        self.wrap(solver, '_compute_chernoff_at_t', lambda a,k: {'t': a[3], 'dx_limit': a[5]})
        self.wrap(solver, '_apply_operator_on_grid', lambda a,k:
                  {'source_points': len(a[1]), 'destination_points': len(a[3]), 'tau': a[4]})
        self.wrap(np.polynomial.legendre, 'leggauss', lambda a,k: {'nt': a[0]})
        # No replacement of CubicSpline or its results: cProfile observes SciPy calls.
        self.previous_sigterm_handler = signal.signal(signal.SIGTERM, self.interrupted)
        self.profiler.enable()

    def interrupted(self, signum, frame):
        self.flush('external_SIGTERM', force=True)
        self.profiler.disable()
        self.events.close()
        os._exit(128+signum)

    def close(self):
        self.flush('solver_return', force=True)
        self.profiler.disable()
        self.events.close()
        signal.signal(signal.SIGTERM, self.previous_sigterm_handler)
        for owner, name, original in reversed(self.patched):
            setattr(owner, name, original)


def load_equation(equation_id):
    from all_methods_dataset_loader import load_all_methods_equation
    return load_all_methods_equation(equation_id,
        csv_path=ROOT/'data/generated/all_methods_equations_v1.csv',
        metadata_path=ROOT/'data/generated/all_methods_equations_v1_metadata.jsonl')


def worker(equation_id, profile, folder):
    import numpy as np
    from remizov_solver import solve_remizov
    from verify_remizov_solver import make_verification_config
    e = load_equation(equation_id)
    config = make_verification_config(4)
    assert asdict(config) == EXPECTED_CONFIG
    x = np.linspace(-1, 1, 33)
    observer = SolverObserver(folder) if profile else None
    if observer:
        observer.install()
    result = solve_remizov(e.remizov_problem, x, config)
    if observer:
        observer.close()
    np.testing.assert_array_equal(x, result.x)
    exact = e.y_exact(x)
    write_json(folder/'solver_return.json', {
        'equation_id': equation_id, 'n': 4, 'config': asdict(config),
        'status': 'success' if result.success else result.failure_reason,
        'failure_reason': result.failure_reason, 'solver_result': asdict(result),
        'y_exact': exact,
        'max_abs_error': float(np.max(np.abs(result.f-exact))) if result.success else None,
        'pointwise_error': result.f-exact if result.success else None,
        'runtime_seconds': result.runtime_seconds,
        'instrumented': profile, 'total_error_certified': False,
    })


def profile_summary(folder):
    path = folder/'partial_profile.pstats'
    if not path.exists():
        return None
    stats = pstats.Stats(str(path))
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, callers) in stats.stats.items():
        rows.append({'filename': filename, 'line': line, 'function': function,
            'primitive_calls': primitive, 'calls': calls,
            'self_seconds': own, 'cumulative_seconds': cumulative})
    data = {'self_seconds_total': stats.total_tt,
        'inclusive_cumulative_times_overlap_do_not_sum': True,
        'profile_is_partial_if_no_solver_return': not (folder/'solver_return.json').exists(),
        'active_function_times_may_be_incomplete': True,
        'rows_by_self_seconds': sorted(rows,key=lambda row: row['self_seconds'], reverse=True)}
    write_json(folder/'profile_summary.json', data)
    with (folder/'profile_top.txt').open('w') as handle:
        stats.stream=handle
        stats.sort_stats('tottime').print_stats(45)
        stats.sort_stats('cumtime').print_stats(45)
    return data


def bounded_run(equation_id, profile, folder):
    folder.mkdir()
    started = time.perf_counter()
    peak_rss = 0
    reason = ''
    termination_at = None
    with (folder/'worker.log').open('x') as log, (folder/'resource_samples.jsonl').open('x') as resources:
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
            '--worker-id', equation_id, '--worker-dir', str(folder)] +
            (['--profile'] if profile else []), cwd=ROOT, env=os.environ.copy(),
            stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            elapsed = time.perf_counter()-started
            probe=subprocess.run(['ps','-o','rss=','-p',str(child.pid)],
                                 capture_output=True,text=True)
            rss_text=probe.stdout.strip()
            rss=int(rss_text) if rss_text else None
            if rss is not None:
                peak_rss=max(peak_rss,rss)
            resources.write(json.dumps({'wall_seconds': elapsed,'rss_kib': rss})+'\n')
            resources.flush()
            if elapsed >= WALL_LIMIT:
                reason='external_time_limit'
            elif peak_rss > RSS_LIMIT_MIB*1024:
                reason='external_memory_limit'
            if reason:
                termination_at=time.perf_counter()-started
                child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                break
            time.sleep(min(POLL_SECONDS,max(0.0,WALL_LIMIT-(time.perf_counter()-started))))
    elapsed=time.perf_counter()-started
    solver_path=folder/'solver_return.json'
    if reason or child.returncode:
        record={'equation_id':equation_id,'n':4,'status':reason or 'worker_exception',
            'failure_reason':reason or 'worker_exception','max_abs_error':None,
            'runtime_seconds':None,'solver_result':None,'config':EXPECTED_CONFIG,
            'instrumented':profile}
    else:
        record=json.loads(solver_path.read_text())
    record.update(worker_wall_seconds=elapsed,worker_peak_rss_mib=peak_rss/1024,
        termination_requested_wall_seconds=termination_at,child_returncode=child.returncode)
    if profile:
        progress=folder/'progress.json'
        record['partial_diagnostics']=json.loads(progress.read_text()) if progress.exists() else None
        profile_summary(folder)
    write_json(folder/'run.json',record)
    print(json.dumps({k:record[k] for k in ('equation_id','status','runtime_seconds',
          'worker_wall_seconds','worker_peak_rss_mib','max_abs_error')}) ,flush=True)
    return record


def tree_hashes(folder):
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--worker-id',choices=['AMV1_F0005_S00','AMV1_F0000_S00','AMV1_F0006_S00','AMV1_F0007_S00'])
    parser.add_argument('--worker-dir',type=Path)
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    if args.worker_id:
        worker(args.worker_id,args.profile,args.worker_dir)
        return
    if args.output_dir is None:
        parser.error('--output-dir required')
    out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'results'):
        parser.error('new output directory must be under project/results')
    out.mkdir(parents=True,exist_ok=False)
    before=snapshot(ROOT)
    original=ROOT.parent/'generator_DE'
    original_before=snapshot(original)
    previous=ROOT/'results/remizov_n_study_2026-10-11'
    prior=json.loads((previous/'summary.json').read_text())
    previous_before=tree_hashes(previous)
    summary={'started_utc':datetime.now(timezone.utc).isoformat(),
        'config':EXPECTED_CONFIG,'x':prior['x'],'runs':[],
        'environment':{'python':sys.version,'executable':sys.executable,
           'platform':platform.platform(),'versions':{name:importlib.metadata.version(name)
              for name in ('numpy','scipy','sympy','matplotlib')},
           'thread_environment':{key:os.environ[key] for key in THREAD_VARS}},
        'resource_policy':{'wall_seconds_per_worker':WALL_LIMIT,
           'observed_rss_limit_mib':RSS_LIMIT_MIB,'poll_seconds':POLL_SECONDS,
           'termination_flush_grace_seconds':2,'serial':True,'retries':0,
           'independent_runs_continue_after_first_external_time_limit':True},
        'cost_estimate':{'F0005_n1_solver_seconds':8.891003625001758,
           'F0005_n2_solver_seconds':21.14037262497004,
           'n4_finest_source_points':99707,'n2_processed_points':64624136,
           'n4_max_operator_applications':98112,
           'naive_n_proportional_seconds':2*21.14037262497004,
           'estimate_is_not_a_runtime_bound':True,
           'n4_three_intermediate_grids_vs_one_at_n2':True,
           'completion_within_45_seconds_uncertain':True},
        'previous_dataset_and_code_sha256':before,
        'previous_results_sha256':previous_before,
        'original_sha256':original_before}
    write_json(out/'summary.json',summary)
    for equation_id,profile,label in [('AMV1_F0005_S00',False,'F0005_n4'),
                                     ('AMV1_F0000_S00',True,'F0000_n4_profile')]:
        print(f'START {equation_id} n=4 instrumented={profile}',flush=True)
        record=bounded_run(equation_id,profile,out/label)
        summary['runs'].append(record)
        write_json(out/'summary.json',summary)
        if record['status'] in ('external_memory_limit','worker_exception'):
            break
    after=snapshot(ROOT); original_after=snapshot(original)
    summary['integrity']={'project_changed_or_removed':[p for p,h in before.items() if after.get(p)!=h],
        'original_changed_or_removed':[p for p,h in original_before.items() if original_after.get(p)!=h],
        'original_added':sorted(set(original_after)-set(original_before)),
        'previous_results_identical':tree_hashes(previous)==previous_before,
        'excluded_from_source_snapshots':['.venv','.git','.pytest_cache','__pycache__','results','.DS_Store']}
    summary['finished_utc']=datetime.now(timezone.utc).isoformat()
    write_json(out/'summary.json',summary)
    print(f'FINISHED {out}',flush=True)


if __name__=='__main__':
    main()
