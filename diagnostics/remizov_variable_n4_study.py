"""One unprofiled n=4 attempt each for F0006 and F0007, independently.

Uses the existing worker, serializer and watchdog. No numerical changes.
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
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.remizov_n_study import snapshot, write_json, THREAD_VARS
from diagnostics.remizov_n4_followup import (
    bounded_run,tree_hashes,EXPECTED_CONFIG,WALL_LIMIT,RSS_LIMIT_MIB,POLL_SECONDS,
)

TASKS=(('AMV1_F0006_S00','slow_varying'),('AMV1_F0007_S00','moderate_varying'))
TASK_CONFIG=dict(n=4,tail_eps=1e-8,nt_initial=8,dx_initial=.05,
    quadrature_rtol=1e-7,quadrature_atol=1e-9,spatial_rtol=1e-7,spatial_atol=1e-9,
    max_quadrature_refinements=8,max_spatial_refinements=5,max_work_grid_points=100000)
PREVIOUS=('remizov_n_study_2026-10-11','remizov_n4_followup_2026-10-12',
          'remizov_f0000_n4_control_2026-10-12')


def preflight(equation_id):
    import numpy as np
    from diagnostics.remizov_n4_followup import load_equation
    from remizov_solver import compute_truncation_T, _problem_conditions_valid
    e=load_equation(equation_id)
    p=e.remizov_problem
    assert _problem_conditions_valid(p)
    x=np.linspace(-1,1,33)
    T=compute_truncation_T(p,TASK_CONFIG['tail_eps'])
    shift=max(2*math.sqrt(p.a_upper_bound*T/4),2*p.b_abs_upper_bound*T/4)
    left=p.x_target_min-4*shift; right=p.x_target_max+4*shift
    width=right-left
    global_points=np.linspace(left,right,1025)
    certificate={'a_lower':p.a_lower_bound,'a_upper':p.a_upper_bound,
        'b_abs':p.b_abs_upper_bound,'c_R_abs':p.c_abs_upper_bound,
        'lambda':p.lambda_value,'gap':p.lambda_value-p.c_abs_upper_bound,
        'certification':p.bounds_certification,'domain':p.bounds_domain}
    values={name:np.asarray(function(global_points),dtype=float)
            for name,function in [('a',p.a),('b',p.b),('c_R',p.c_remizov),('g_R',p.g_remizov)]}
    assert all(np.all(np.isfinite(v)) for v in values.values())
    assert np.min(values['a'])>=p.a_lower_bound-1e-12
    assert np.max(values['a'])<=p.a_upper_bound+1e-12
    assert np.max(np.abs(values['b']))<=p.b_abs_upper_bound+1e-12
    assert np.max(np.abs(values['c_R']))<=p.c_abs_upper_bound+1e-12
    cauchy=e.cauchy_problem
    ordinary_g=np.array([cauchy.g(float(t)) for t in x])
    ordinary_c=np.array([cauchy.c(float(t)) for t in x])
    y=e.y_exact(x); dy=e.dy_exact(x); d2y=e.d2y_exact(x)
    residual=p.a(x)*d2y+p.b(x)*dy+(p.c_remizov(x)-p.lambda_value)*y+p.g_remizov(x)
    return {
        'equation_id':equation_id,'metadata':dict(e.metadata),
        'certificate':certificate,'T':T,'target_x':x,'y_exact_target':y,
        'exact_amplitude_on_target_33':float(np.max(np.abs(y))),
        'support':[e.metadata['support_left'],e.metadata['support_right']],
        'work_domain_bound':[left,right],'work_width_bound':width,
        'work_shift_per_operator_bound':shift,
        'source_points_initial_bound':max(3,math.ceil(width/.05))+1,
        'source_points_finest_bound':max(3,math.ceil(width/(.05/32)))+1,
        'max_quadrature_node_evaluations':6*4088,
        'max_operator_applications':4*6*4088,
        'working_domain_sample_ranges':{name:{'min':float(np.min(v)),
            'max':float(np.max(v)),'max_abs':float(np.max(np.abs(v)))} for name,v in values.items()},
        'working_domain_sample_points':1025,
        'sampled_checks_do_not_replace_global_certificates':True,
        'sign_checks':{'ordinary_g_plus_g_R':float(np.max(np.abs(ordinary_g+p.g_remizov(x)))),
            'ordinary_c_minus_c_R_plus_lambda':float(np.max(np.abs(ordinary_c-(p.c_remizov(x)-p.lambda_value)))),
            'manufactured_residual_on_33':float(np.max(np.abs(residual))),
            'y0_difference':float(e.y_exact(-1)-cauchy.y0),
            'v0_difference':float(e.dy_exact(-1)-cauchy.v0)},
        'cost_bound_is_not_runtime_or_success_guarantee':True,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'results'):
        parser.error('new results directory must be under project/results')
    from verify_remizov_solver import make_verification_config
    assert TASK_CONFIG==EXPECTED_CONFIG==asdict(make_verification_config(4))
    assert WALL_LIMIT==45.0 and RSS_LIMIT_MIB==512 and POLL_SECONDS==.25
    out.mkdir(parents=True,exist_ok=False)
    before=snapshot(ROOT)
    original=ROOT.parent/'generator_DE'
    original_before=snapshot(original)
    old_hashes={name:tree_hashes(ROOT/'results'/name) for name in PREVIOUS}
    manifest=json.loads((ROOT/'MIGRATION_MANIFEST.json').read_text())
    core={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest['source_sha256'][name]
          for name in manifest['unchanged_mathematical_core']}
    assert all(core.values())
    checks={equation_id:preflight(equation_id) for equation_id in
            ['AMV1_F0005_S00',*[task[0] for task in TASKS]]}
    # F0005 is reconstructed only for comparison, never solved in this study.
    for equation_id,regime in TASKS:
        assert checks[equation_id]['metadata']['regime']==regime
    baseline=json.loads(args.baseline.read_text()) if args.baseline else None
    summary={'client_study_date':'2026-10-12','started_utc':datetime.now(timezone.utc).isoformat(),
        'config':TASK_CONFIG,'tasks':[{'equation_id':eid,'regime':regime} for eid,regime in TASKS],
        'environment':{'python':sys.version,'executable':sys.executable,
            'platform':platform.platform(),'cpu_count':os.cpu_count(),
            'versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','sympy')},
            'thread_environment':{key:os.environ[key] for key in THREAD_VARS}},
        'resource_policy':{'wall_seconds_per_worker':45.0,'observed_rss_limit_mib':512,
            'poll_seconds':.25,'max_parallel_solver_processes':1,'profiling_enabled':False,
            'instrumentation_enabled':False,'retries':0,'attempts_per_task':1,
            'second_task_independent_of_first_status':True,
            'termination_wait_seconds':2,'termination_wait_is_not_extra_solver_budget':True},
        'preflight':checks,'project_before_execution_sha256':before,
        'original_before_sha256':original_before,'previous_results_before_sha256':old_hashes,
        'baseline_before_infrastructure_changes':baseline,
        'core_matches_migration_before':core,'runs':[]}
    write_json(out/'summary.json',summary)
    write_json(out/'preflight.json',checks)
    for equation_id,regime in TASKS:
        print(f'START {equation_id} {regime} n=4 unprofiled',flush=True)
        record=bounded_run(equation_id,False,out/f'{equation_id}_n4')
        record['regime']=regime
        record['exact_amplitude_on_target_33']=checks[equation_id]['exact_amplitude_on_target_33']
        if record['solver_result'] is not None:
            import numpy as np
            np.testing.assert_array_equal(record['solver_result']['x'],np.linspace(-1,1,33))
        write_json(out/f'{equation_id}_n4/run.json',record)
        summary['runs'].append(record)
        write_json(out/'summary.json',summary)
        # Deliberately no break or automatic retry for any returned status.
    after=snapshot(ROOT); original_after=snapshot(original)
    summary['integrity']={'project_changed_during_execution':[p for p,h in before.items() if after.get(p)!=h],
        'original_changed_or_removed':[p for p,h in original_before.items() if original_after.get(p)!=h],
        'original_added':sorted(set(original_after)-set(original_before)),
        'previous_results_identical':{name:tree_hashes(ROOT/'results'/name)==hashes for name,hashes in old_hashes.items()},
        'excluded_from_source_snapshots':['.venv','.git','.pytest_cache','__pycache__','results','.DS_Store']}
    summary['finished_utc']=datetime.now(timezone.utc).isoformat()
    assert len(summary['runs'])==2
    assert [r['equation_id'] for r in summary['runs']]==[task[0] for task in TASKS]
    write_json(out/'summary.json',summary)
    print(f'FINISHED after exactly two attempts: {out}',flush=True)


if __name__=='__main__':
    main()
