"""Execute an explicit comparison plan; one isolated process per attempt.

No automatic retries or accuracy-based selection. Follow-up plans are separate
files, preserving all pilots/failures. Relative paths are rooted at this repo.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
THREAD_VARS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREAD_VARS:
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [clean(v) for v in value]
    if isinstance(value,np.ndarray):
        return clean(value.tolist())
    if isinstance(value,(float,np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value,np.integer):
        return int(value)
    return value


def write_json(path, data):
    path.write_text(json.dumps(clean(data),indent=2,sort_keys=True,allow_nan=False)+'\n')


def hashes():
    paths = [ROOT/'remizov_solver.py', ROOT/'all_methods_dataset_loader.py',
             ROOT/'src/all_methods_expressions.py',ROOT/'src/all_methods_dataset_v1.py',
             ROOT/'chernov_solver.py',ROOT/'mc_solver.py',Path(__file__).resolve()]
    paths += list((ROOT/'data').rglob('*')) + list((ROOT/'comparison_inputs').rglob('*'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file() and p.name != '.DS_Store'}


def load_task(problem_id):
    import numpy as np
    if problem_id in ('MC_constant', 'MC_gaussian'):
        from mc_solver import MCForcingBound
        from remizov_solver import RemizovProblem
        gaussian = problem_id == 'MC_gaussian'
        g = (lambda x: (4-4*x*x)*np.exp(-x*x)) if gaussian else (lambda x: 1.)
        exact = (lambda x: np.exp(-x*x)) if gaussian else (lambda x: np.full_like(x, .5))
        p = RemizovProblem(problem_id, lambda x: 1., lambda x: 0., lambda x: 0.,
            g, 2., 1., 1., 0., 0., 'analytic', 'all_real_x', -2., 2.)
        cert = MCForcingBound(4. if gaussian else 1.,
            '|(1-u)exp(-u)|<=1 for u=x^2>=0' if gaussian else 'g_R=1 identically')
        return p, exact, np.linspace(-2., 2., 33), {'forcing_bound': asdict(cert)}
    if problem_id.startswith('AMV1_'):
        from all_methods_dataset_loader import load_all_methods_equation
        e=load_all_methods_equation(problem_id,
            ROOT/'data/generated/all_methods_equations_v1.csv',
            ROOT/'data/generated/all_methods_equations_v1_metadata.jsonl')
        p=e.remizov_problem
        return p,e.y_exact,np.linspace(-1,1,33),dict(e.metadata)
    from remizov_benchmark_loader import load_remizov_problem
    p=load_remizov_problem(problem_id,ROOT/'data/remizov_benchmark.csv')
    return p,lambda x:np.exp(-x*x),np.linspace(-2,2,257),{}


def worker(request, output):
    stop=threading.Event()
    def sample():
        began=time.perf_counter()
        with output.with_suffix('.rss.jsonl').open('x') as handle:
            while not stop.is_set():
                rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # macOS reports bytes, Linux reports KiB.
                mib=rss/(1024**2 if sys.platform=='darwin' else 1024)
                handle.write(json.dumps({'worker_wall_seconds':time.perf_counter()-began,
                    'peak_rss_mib':mib})+'\n'); handle.flush()
                stop.wait(.1)
    monitor=threading.Thread(target=sample,daemon=True)
    monitor.start()
    try:
        _worker_calculate(request,output)
    finally:
        stop.set(); monitor.join()


def _worker_calculate(request, output):
    import numpy as np
    from remizov_solver import RemizovConfig,solve_remizov
    from chernov_solver import ChernovConfig,ChernovSolver
    spec=json.loads(request.read_text())
    started=time.perf_counter()
    p,exact,x,metadata=load_task(spec['problem_id'])
    load_seconds=time.perf_counter()-started
    if spec['method']=='remizov':
        config=RemizovConfig(**spec['config'])
        result=solve_remizov(p,x,config)
    elif spec['method']=='monte_carlo':
        from mc_solver import MCConfig, MCForcingBound, solve_monte_carlo
        config = MCConfig(**spec['config'])
        raw_bound = spec.get('forcing_bound', metadata.get('forcing_bound'))
        cert = None if raw_bound is None else MCForcingBound(**raw_bound)
        result = solve_monte_carlo(p, x, config, forcing_bound=cert)
    else:
        config=ChernovConfig(**spec['config'])
        solver=ChernovSolver(p,config)
        result=solver.solve(x) if spec['method']=='solve' else solver.solve_streaming(x)
    y=exact(x)
    record={'spec':spec,'config':asdict(config),'result':asdict(result),
        'load_seconds':load_seconds,'y_exact':y,'metadata':metadata,
        'max_abs_error':float(np.max(np.abs(result.f-y))) if result.success else None,
        'exact_amplitude':float(np.max(np.abs(y))),
        'total_error_certified':False}
    record['relative_error']=record['max_abs_error']/record['exact_amplitude'] if result.success else None
    record['process_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    write_json(output,record)


def preflight(spec):
    from remizov_solver import compute_truncation_T
    p,_,x,metadata=load_task(spec['problem_id'])
    c=spec['config']
    if spec['method']=='monte_carlo':
        from mc_solver import MCConfig, MCForcingBound, auto_T
        config = MCConfig(**c); config.validate()
        raw_bound = spec.get('forcing_bound', metadata.get('forcing_bound'))
        cert = None if raw_bound is None else MCForcingBound(**raw_bound)
        if cert is not None:
            cert.validate()
        if config.T is None and cert is None:
            raise ValueError('MC preflight requires explicit T or global forcing certificate')
        T = config.T if config.T is not None else auto_T(p.lambda_value,
            p.c_abs_upper_bound, cert.abs_upper_bound, config.tail_eps)
        N = math.ceil(T/config.dt) if T else 0
        if N and T/N > config.dt:
            N += 1
        return {'T': T, 'target_points':len(x), 'time_steps':N,
            'actual_dt':T/N if N else 0., 'path_steps':int(config.M)*N*len(x),
            'peak_batch_paths':min(config.chunk,config.M),
            'tail_bound_available':cert is not None,
            'total_error_certified':False}
    T=c.get('T')
    if T is None:
        T=compute_truncation_T(p,c.get('tail_eps',1e-8))
    if spec['method']=='remizov':
        n=c['n']; q=c.get('max_quadrature_refinements',4); s=c.get('max_spatial_refinements',4)
        shift=max(2*math.sqrt(p.a_upper_bound*T/n),2*p.b_abs_upper_bound*T/n)
        width=p.x_target_max-p.x_target_min+2*n*shift
        dx=c.get('dx_initial',.05)/2**s
        nodes=c.get('nt_initial',8)*(2**(q+1)-1)*(s+1)
        return {'T':T,'target_points':len(x),'source_width_bound':width,
            'finest_source_points_bound':max(3,math.ceil(width/dx))+1,
            'max_node_evaluations':nodes,'max_operator_applications':n*nodes}
    width=p.x_target_max-p.x_target_min+2*c.get('padding',8)
    nodes=c.get('num_t',257); nodes += nodes%2==0
    steps=c.get('n_steps',2000); steps+=steps%2
    ops=c.get('n',100)*(nodes-1) if spec['method']=='solve' else steps
    return {'T':T,'target_points':len(x),'source_width_bound':width,
        'work_points':math.ceil(width/c.get('dx',.025))+1,'operator_applications':ops}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--worker-request',type=Path)
    parser.add_argument('--worker-output',type=Path)
    args=parser.parse_args()
    if args.worker_request:
        worker(args.worker_request,args.worker_output)
        return
    if args.plan is None or args.output_dir is None:
        parser.error('--plan and --output-dir required')
    plan=json.loads(args.plan.read_text())
    out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'results'):
        parser.error('output must be a new directory under results')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'plan.json',plan)
    before=hashes()
    # Preflight is read-only and does not solve any equation.
    summary={'started_utc':datetime.now(timezone.utc).isoformat(),'plan':plan,
        'environment':{'python':sys.version,'executable':sys.executable,
            'platform':platform.platform(),'versions':{n:importlib.metadata.version(n)
            for n in ('numpy','scipy','sympy','pytest')},
            'thread_environment':{key:os.environ[key] for key in THREAD_VARS}},
        'before_sha256':before,'preflight':[preflight(s) for s in plan['attempts']],
        'runs':[]}
    write_json(out/'summary.json',summary)
    for index,spec in enumerate(plan['attempts']):
        label=f'{index:03d}_{spec["problem_id"]}_{spec["method"]}'
        request=out/f'{label}.request.json'; result_path=out/f'{label}.result.json'
        write_json(request,spec)
        began=time.perf_counter(); reason=''; peak=0
        wall=spec.get('wall_seconds',plan.get('wall_seconds',120))
        rss_limit=plan.get('rss_limit_mib',1024)
        with (out/f'{label}.log').open('x') as log, (out/f'{label}.resources.jsonl').open('x') as resource:
            child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),
                '--worker-request',str(request),'--worker-output',str(result_path)],
                cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                elapsed=time.perf_counter()-began
                rss=0
                sample_path=result_path.with_suffix('.rss.jsonl')
                if sample_path.exists():
                    lines=sample_path.read_text().splitlines()
                    try:
                        rss=int(json.loads(lines[-1])['peak_rss_mib']*1024) if lines else 0
                    except json.JSONDecodeError:
                        pass  # worker may be appending its last line
                peak=max(peak,rss)
                resource.write(json.dumps({'wall_seconds':elapsed,'rss_kib':rss})+'\n')
                if elapsed>wall:
                    reason='external_time_limit'
                elif rss>rss_limit*1024:
                    reason='external_memory_limit'
                if reason:
                    child.terminate()
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                    break
                time.sleep(.1)
        record={'spec':spec,'worker_wall_seconds':time.perf_counter()-began,
            'observed_peak_rss_mib':peak/1024,'returncode':child.returncode,
            'result_file':result_path.name if result_path.exists() else None}
        if reason or child.returncode or not result_path.exists():
            record.update(status=reason or 'worker_exception',max_abs_error=None,runtime_seconds=None)
        else:
            result=json.loads(result_path.read_text()); r=result['result']
            record.update(status='success' if r['success'] else r['failure_reason'],
                max_abs_error=result['max_abs_error'],relative_error=result['relative_error'],
                runtime_seconds=r['runtime_seconds'],
                observed_peak_rss_mib=result['process_peak_rss_mib'])
        summary['runs'].append(record)
        write_json(out/'summary.json',summary)
        print(json.dumps({'attempt':label,**{k:record[k] for k in
            ('status','max_abs_error','runtime_seconds','worker_wall_seconds')}}),flush=True)
    after=hashes()
    summary.update(finished_utc=datetime.now(timezone.utc).isoformat(),after_sha256=after,
                   changed_or_removed=[p for p,h in before.items() if after.get(p)!=h])
    write_json(out/'summary.json',summary)


if __name__=='__main__':
    main()
