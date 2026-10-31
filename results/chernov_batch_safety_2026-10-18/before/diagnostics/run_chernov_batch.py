"""Resumable, resource-bounded production batch: only ChernovSolver.solve.

--init freezes the protocol. --pilot runs five predeclared equations; a
saved-only pilot check gates the full run. Persistent isolated workers load
the validated dataset once, then solve independent tasks serially. Terminal
attempts (including failures) are never automatically retried. Unfinished
requests remain on disk and may be resumed in a new attempt.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import queue
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
for name in THREADS:
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
PILOT_IDS = ['AMV1_F0000_S00', 'AMV1_F0060_S00', 'AMV1_F0120_S00',
             'AMV1_F0180_S00', 'AMV1_F0199_S04']


def utc():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value, *, exclusive=False, durable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(clean(value), sort_keys=True, allow_nan=False)+'\n'
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.batch_',
                                     suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        if durable:
            os.fsync(handle.fileno())
    try:
        if exclusive:
            os.link(temporary, path)  # atomic publication, never replace old results
        else:
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def accuracy_status(computation_status, error, threshold):
    if computation_status != 'success':
        return computation_status
    if error is None or not math.isfinite(error):
        return 'invalid_error_metric'
    return 'accuracy_pass' if error <= threshold else 'accuracy_not_met'


def limit_reason(elapsed, peak_rss, wall_limit, rss_limit):
    if elapsed > wall_limit:
        return 'external_time_limit'
    if peak_rss > rss_limit:
        return 'external_memory_limit'
    return None


def initialize(out, data):
    import numpy as np
    from all_methods_dataset_loader import load_all_methods_equations
    from chernov_solver import ChernovConfig
    from remizov_solver import compute_truncation_T
    if not out.is_relative_to(ROOT/'results'):
        raise ValueError('output must be under results')
    if out.exists():
        raise FileExistsError('refusing to overwrite existing batch')
    # Use the full previous successful configuration, including unused streaming K.
    previous = read_json(ROOT/'results/chernov_dataset20_2026-10-18/plan.json')
    config = previous['attempts'][0]['config']
    ChernovConfig(**config).validate()
    equations = load_all_methods_equations(data/'all_methods_equations_v1.csv',
        data/'all_methods_equations_v1_metadata.jsonl')
    if len(equations) != 1000:
        raise ValueError('expected validated production dataset of 1000 tasks')
    expected = read_json(ROOT/'results/dataset1000_2026-10-18/artifacts_sha256.json')
    data_hashes = {str((data/name).relative_to(ROOT)): digest(data/name) for name in expected}
    if list(data_hashes.values()) != [expected[name] for name in expected]:
        raise ValueError('production artifacts changed since validation')
    source_names = ['chernov_solver.py', 'remizov_solver.py',
        'all_methods_dataset_loader.py', 'src/all_methods_expressions.py',
        'src/all_methods_dataset_v1.py', 'diagnostics/run_chernov_batch.py',
        'diagnostics/report_chernov_batch.py']
    protected = {name: digest(ROOT/name) for name in source_names}
    protected.update(data_hashes)
    preflight = [dict(problem_id=e.equation_id,
        complexity=e.metadata['complexity_level'], regime=e.metadata['regime'],
        split=e.metadata['split'], family_id=e.metadata['family_id'],
        gap=e.remizov_problem.lambda_value-e.remizov_problem.c_abs_upper_bound,
        T=compute_truncation_T(e.remizov_problem, config['tail_eps']),
        work_grid_points=2081, operator_applications=128*512) for e in equations]
    plan = dict(schema='chernov_batch_v1', created_utc=utc(),
        implementation='chernov_solver.ChernovSolver.solve', method='solve',
        config=config, target_x=np.linspace(-1., 1., 33).tolist(), threshold=.01,
        data_dir=str(data.relative_to(ROOT)), equation_ids=[e.equation_id for e in equations],
        pilot_ids=PILOT_IDS, workers=8, task_wall_seconds=120., startup_wall_seconds=120.,
        worker_rss_limit_mib=512., total_worker_rss_budget_mib=4096.,
        poll_seconds=.25, heartbeat_seconds=1., blas_threads=1,
        parameter_policy='identical prior successful config for every equation; T only from certified gap; exact evaluated after solve only',
        retry_policy='no retries of terminal outcomes; interrupted requests retained and resumed as new attempts',
        expected_cost=dict(serial_solver_seconds=15494.,
            estimated_parallel_wall_seconds=[2400,5400],
            estimate_basis='previous mean 15.494 s/task; concurrency changes wall time',
            max_task_wall_seconds_sum=120000., grid_points=2081,
            operator_applications_per_task=65536), protected_sha256=protected,
        environment=dict(python=sys.version, executable=sys.executable,
            platform=platform.platform(), cpu_count=os.cpu_count(),
            physical_memory_bytes=os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'),
            versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','sympy','pytest')},
            thread_environment={n:os.environ[n] for n in THREADS}))
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('attempts','solutions','sessions','workers'):
        (out/folder).mkdir()
    before = {}
    for directory, subdirs, files in os.walk(ROOT):
        parent = Path(directory)
        subdirs[:] = [s for s in subdirs if s not in ('.venv','.git','__pycache__','.pytest_cache') and parent/s != out]
        for name in files:
            if name != '.DS_Store':
                p=parent/name
                before[str(p.relative_to(ROOT))]=digest(p)
    write_json(out/'before_sha256.json', before, exclusive=True)
    (out/'PROJECT_STATUS.before.md').write_bytes((ROOT/'PROJECT_STATUS.md').read_bytes())
    write_json(out/'plan.json', plan, exclusive=True)
    write_json(out/'preflight.json', preflight, exclusive=True)
    print(json.dumps(dict(initialized=str(out), tasks=len(equations), config=config,
        T_range=[min(r['T'] for r in preflight),max(r['T'] for r in preflight)])))


def validate_identity(record, plan):
    if record['plan_sha256'] != plan_hash(plan) or record['config'] != plan['config']:
        raise ValueError('saved result belongs to another plan/configuration')
    if record['problem_id'] not in plan['equation_ids']:
        raise ValueError('unknown equation ID')
    if record['implementation'] != 'chernov_solver.ChernovSolver.solve':
        raise ValueError('wrong numerical implementation')


def collect_completed(out, plan):
    """Adopt durable attempt results even after a crash before checkpointing."""
    completed = {}
    for p in sorted((out/'solutions').glob('*.json')):
        record=read_json(p);validate_identity(record,plan)
        if p.stem != record['problem_id']:
            raise ValueError('result filename/ID mismatch')
        completed[record['problem_id']]=record
    for p in sorted((out/'attempts').glob('*/result.json')):
        record=read_json(p);validate_identity(record,plan)
        pid=record['problem_id']
        if pid not in completed:
            write_json(out/'solutions'/f'{pid}.json',record,exclusive=True)
            completed[pid]=record
        elif completed[pid]['attempt_id'] != record['attempt_id']:
            raise ValueError('multiple terminal attempts for the same equation')
    return completed


def check_protected(plan):
    changed=[name for name,h in plan['protected_sha256'].items()
             if not (ROOT/name).is_file() or digest(ROOT/name)!=h]
    if changed:
        raise ValueError(f'protected code/data changed: {changed}')


def worker(plan_path, state_path, parent_pid):
    """Persistent process; exact evaluator is used only after solve returns."""
    plan=read_json(plan_path);stop=threading.Event()
    state=dict(pid=os.getpid(), phase='loading', problem_id=None)
    def monitor():
        with state_path.with_suffix('.rss.jsonl').open('x') as log:
            while not stop.is_set():
                if os.getppid()!=parent_pid:
                    os._exit(75)  # orphan must not compete with a resumed session
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
                sample=dict(state,utc=utc(),peak_rss_mib=rss)
                write_json(state_path,sample,durable=False)
                log.write(json.dumps(sample)+'\n');log.flush()
                stop.wait(plan['heartbeat_seconds'])
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        import numpy as np
        from all_methods_dataset_loader import load_all_methods_equations
        from chernov_solver import ChernovConfig, ChernovSolver
        began=time.perf_counter();data=ROOT/plan['data_dir']
        equations={e.equation_id:e for e in load_all_methods_equations(
            data/'all_methods_equations_v1.csv', data/'all_methods_equations_v1_metadata.jsonl')}
        load_seconds=time.perf_counter()-began
        state.update(phase='ready', load_seconds=load_seconds)
        write_json(state_path,dict(state,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)),durable=False)
        for line in sys.stdin:
            request_path=Path(json.loads(line)['request_path'])
            request=read_json(request_path);pid=request['problem_id']
            state.update(phase='solving',problem_id=pid)
            began=time.perf_counter()
            base=dict(problem_id=pid,attempt_id=request['attempt_id'],
                plan_sha256=plan_hash(plan),implementation=plan['implementation'],
                config=plan['config'],target_x=plan['target_x'],worker_pid=os.getpid(),
                dataset_load_seconds_once=load_seconds,total_error_certified=False)
            try:
                equation=equations[pid];config=ChernovConfig(**plan['config'])
                result=ChernovSolver(equation.remizov_problem,config).solve(plan['target_x'])
                # Evaluation happens strictly after the numerical method.
                exact=equation.y_exact(np.asarray(plan['target_x']))
                error=float(np.max(np.abs(result.f-exact))) if result.success else None
                computation='success' if result.success else result.failure_reason
                amplitude=float(np.max(np.abs(exact)))
                base.update(result=asdict(result),y_exact=exact,
                    computation_status=computation,max_abs_error=error,
                    relative_error=error/amplitude if error is not None else None,
                    accuracy_reached=bool(error is not None and error<=plan['threshold']),
                    status=accuracy_status(computation,error,plan['threshold']),
                    solver_seconds=result.runtime_seconds,metadata=dict(equation.metadata))
            except Exception:
                base.update(result=None,y_exact=None,computation_status='worker_exception',
                    max_abs_error=None,relative_error=None,accuracy_reached=False,
                    status='worker_exception',solver_seconds=None,traceback=traceback.format_exc())
            base.update(finished_utc=utc(),task_wall_seconds=time.perf_counter()-began,
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024))
            write_json(request_path.parent/'result.json',base,exclusive=True)
            state.update(phase='ready',problem_id=None)
    finally:
        stop.set();thread.join()


def terminate(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill();child.wait()


def wait_for_result(child, result_path, state_path, wall_limit, rss_limit, poll, stop):
    began=time.perf_counter();peak=0.
    while True:
        if result_path is not None and result_path.exists():
            return None,time.perf_counter()-began,peak
        state=read_json(state_path) if state_path.exists() else {}
        peak=max(peak,state.get('peak_rss_mib',0.))
        if result_path is None and state.get('phase')=='ready':
            return None,time.perf_counter()-began,peak
        reason=limit_reason(time.perf_counter()-began,peak,wall_limit,rss_limit)
        if stop.is_set():
            reason='interrupted'
        if child.poll() is not None:
            reason='worker_exception'
        if reason:
            terminate(child)
            return reason,time.perf_counter()-began,peak
        time.sleep(poll)


def run_batch(out, pilot=False, max_new=None):
    lock=(out/'batch.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read_json(out/'plan.json');check_protected(plan)
    completed=collect_completed(out,plan)
    if not pilot and not (out/'pilot_check.json').exists():
        raise ValueError('run and verify pilot before the full batch')
    chosen=plan['pilot_ids'] if pilot else plan['equation_ids']
    pending=[pid for pid in chosen if pid not in completed]
    if max_new is not None:
        pending=pending[:max_new]
    session=utc().replace(':','').replace('.','')+'_'+uuid.uuid4().hex[:6]
    session_path=out/'sessions'/f'{session}.json'
    request_queue=queue.Queue()
    for pid in pending:
        request_queue.put(pid)
    stop=threading.Event();guard=threading.Lock();failures=[]
    session_info=dict(session_id=session,started_utc=utc(),pilot=pilot,
        previously_completed=len(completed),planned_new=len(pending),completed_new=0,
        worker_startups=[],abandoned_requests=[str(p.relative_to(out))
            for p in (out/'attempts').glob('*/request.json') if not (p.parent/'result.json').exists()])
    write_json(session_path,session_info,exclusive=True)
    def checkpoint(record=None):
        with guard:
            if record is not None:
                validate_identity(record,plan)
                write_json(out/'solutions'/f"{record['problem_id']}.json",record,exclusive=True)
                completed[record['problem_id']]=record
                session_info['completed_new']+=1
            write_json(session_path,session_info)
            write_json(out/'progress.json',dict(updated_utc=utc(),total=len(plan['equation_ids']),
                completed=len(completed),pending=len(plan['equation_ids'])-len(completed),
                outcomes=dict(Counter(r['status'] for r in completed.values())),session=session))
            if record is not None:
                print(json.dumps(dict(completed=len(completed),id=record['problem_id'],
                    status=record['status'],error=record['max_abs_error'],
                    seconds=record['solver_seconds'])),flush=True)
    def work(slot):
        child=None;log=None
        try:
            while not stop.is_set():
                try:
                    pid=request_queue.get_nowait()
                except queue.Empty:
                    break
                if child is None or child.poll() is not None:
                    if log is not None:
                        log.close()
                    token=f'{session}_w{slot}_{uuid.uuid4().hex[:6]}'
                    state_path=out/'workers'/f'{token}.json'
                    log=(out/'workers'/f'{token}.log').open('x')
                    child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),
                        '--worker-plan',str(out/'plan.json'),'--worker-state',str(state_path),
                        '--parent-pid',str(os.getpid())],stdin=subprocess.PIPE,
                        stdout=log,stderr=subprocess.STDOUT,text=True,cwd=ROOT)
                    reason,elapsed,peak=wait_for_result(child,None,state_path,
                        plan['startup_wall_seconds'],plan['worker_rss_limit_mib'],plan['poll_seconds'],stop)
                    with guard:
                        session_info['worker_startups'].append(dict(slot=slot,pid=child.pid,
                            elapsed_seconds=elapsed,peak_rss_mib=peak,status=reason or 'ready',
                            state_file=str(state_path.relative_to(out))))
                    if reason:
                        raise RuntimeError(f'worker startup failed: {reason}; {state_path}')
                attempt=f'{session}_{pid}'
                folder=out/'attempts'/attempt
                folder.mkdir(exist_ok=False)
                request=folder/'request.json'
                write_json(request,dict(problem_id=pid,attempt_id=attempt,
                    plan_sha256=plan_hash(plan),requested_utc=utc()),exclusive=True)
                child.stdin.write(json.dumps(dict(request_path=str(request)))+'\n');child.stdin.flush()
                reason,elapsed,peak=wait_for_result(child,folder/'result.json',state_path,
                    plan['task_wall_seconds'],plan['worker_rss_limit_mib'],plan['poll_seconds'],stop)
                # A complete durable result may have appeared between the final
                # watchdog observation and process termination; preserve it.
                if reason and (folder/'result.json').exists():
                    reason=None
                if reason=='interrupted':
                    write_json(folder/'interrupted.json',dict(status=reason,elapsed=elapsed,utc=utc()),exclusive=True)
                    break
                if reason:
                    record=dict(problem_id=pid,attempt_id=attempt,plan_sha256=plan_hash(plan),
                        implementation=plan['implementation'],config=plan['config'],
                        target_x=plan['target_x'],computation_status=reason,status=reason,
                        max_abs_error=None,relative_error=None,accuracy_reached=False,
                        solver_seconds=None,result=None,y_exact=None,
                        task_wall_seconds=elapsed,process_peak_rss_mib=peak,
                        worker_pid=child.pid,finished_utc=utc(),total_error_certified=False)
                    write_json(folder/'result.json',record,exclusive=True)
                else:
                    record=read_json(folder/'result.json')
                checkpoint(record)
                if child.poll() is not None:
                    log.close();log=None;child=None
        except BaseException:
            failures.append(traceback.format_exc());stop.set()
        finally:
            if child is not None:
                # No task is in flight here: normal shutdown uses EOF.
                if child.poll() is None:
                    child.stdin.close()
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        terminate(child)
            if log is not None:
                log.close()
    def handle_signal(signum,frame):
        stop.set()
    old_handlers={s:signal.signal(s,handle_signal) for s in (signal.SIGINT,signal.SIGTERM)}
    began=time.perf_counter();checkpoint()
    threads=[threading.Thread(target=work,args=(slot,)) for slot in range(min(plan['workers'],len(pending)))]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        for s,handler in old_handlers.items():
            signal.signal(s,handler)
        session_info.update(finished_utc=utc(),wall_seconds=time.perf_counter()-began,
            stopped=stop.is_set(),infrastructure_errors=failures)
        checkpoint();lock.close()
    if failures:
        raise RuntimeError('\n'.join(failures))
    if stop.is_set():
        raise SystemExit(130)
    return completed


def verify_pilot(out):
    import numpy as np
    plan=read_json(out/'plan.json');check_protected(plan)
    complete=collect_completed(out,plan)
    rows=[complete[pid] for pid in plan['pilot_ids']]
    for r in rows:
        validate_identity(r,plan)
        if r['computation_status']!='success' or r['result']['scheme']!='solve':
            raise ValueError('pilot did not complete numerically; inspect before full run')
        np.testing.assert_array_equal(r['result']['x'],plan['target_x'])
        error=float(np.max(np.abs(np.array(r['result']['f'])-r['y_exact'])))
        if error!=r['max_abs_error'] or r['result']['n_operator_applications']!=65536:
            raise ValueError('pilot validation failed')
    value=dict(checked_utc=utc(),plan_sha256=plan_hash(plan),ids=plan['pilot_ids'],
        infrastructure_pass=True,config_unchanged=True,no_individual_tuning=True,
        accuracy_passes=sum(r['accuracy_reached'] for r in rows),
        resume_skip_verified=len([pid for pid in plan['pilot_ids'] if pid not in complete])==0)
    write_json(out/'pilot_check.json',value,exclusive=True)
    print(json.dumps(value))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/all_methods_v1_1000_2026-10-18')
    parser.add_argument('--init',action='store_true')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--verify-pilot',action='store_true')
    parser.add_argument('--max-new',type=int)
    parser.add_argument('--worker-plan',type=Path)
    parser.add_argument('--worker-state',type=Path)
    parser.add_argument('--parent-pid',type=int)
    args=parser.parse_args()
    if args.worker_plan:
        worker(args.worker_plan,args.worker_state,args.parent_pid);return
    if args.output_dir is None:
        parser.error('--output-dir is required')
    if args.max_new is not None and args.max_new<=0:
        parser.error('--max-new must be positive')
    out=args.output_dir.resolve()
    if args.init:
        initialize(out,args.data_dir.resolve())
    elif args.verify_pilot:
        verify_pilot(out)
    else:
        run_batch(out,args.pilot,args.max_new)


if __name__=='__main__':
    main()
