"""One num_t=1025 control against saved F0180/S04 cubic/zero, padding=12.

Only num_t changes. Reuse the tested observation wrapper and unmodified
production operators; no baseline rerun or automatic refinement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.chernov_boundary_diagnosis import (
    BATCH, DATA, ResourceGuard, THREAD_VARS, array_digest, clean, digest,
    fingerprints, max_abs, peak_mib, prepare, protected_snapshot, read_json, write_json,
)
from diagnostics.chernov_window_padding24 import BASE, PRIOR, PID, solve_once
import numpy as np
from all_methods_dataset_loader import load_all_methods_equation
from chernov_solver import ChernovConfig, _simpson_weights


def quadrature_check(T):
    arrays, checks = {}, {}
    for nodes in (513, 1025):
        t = T*np.linspace(0, 1, nodes)**2
        w = _simpson_weights(t)
        assert np.all(np.diff(t)>0) and t[0]==0 and t[-1]==T
        assert np.flatnonzero(w<0).tolist()==[0]
        assert abs(float(w.sum())-T)<=4*np.finfo(float).eps*T
        expected_first = -2*T/(3*(nodes-1)**2)
        np.testing.assert_allclose(w[0],expected_first,rtol=2e-15,atol=0)
        moments = [dict(degree=k, actual=math.fsum(float(a)*float(b) for a,b in zip(w,t**k)),
            expected=T**(k+1)/(k+1), absolute_error=
            math.fsum(float(a)*float(b) for a,b in zip(w,t**k))-T**(k+1)/(k+1))
            for k in range(3)]
        checks[str(nodes)] = dict(time_nodes_sha256=array_digest(t),
            weights_sha256=array_digest(w), sum_weights=float(w.sum()),
            sum_weights_minus_T=float(w.sum()-T), negative_weight_indices=[0],
            first_weight=float(w[0]), first_weight_analytic=expected_first,
            moments=moments, min_spacing=float(np.min(np.diff(t))),
            max_spacing=float(np.max(np.diff(t))))
        arrays[str(nodes)] = dict(t=t,weights=w)
    np.testing.assert_array_equal(arrays['513']['t'],arrays['1025']['t'][::2])
    checks.update(old_nodes_bitwise_equal_new_even_nodes=True,
                  same_method='integrated quadratic interpolant on every nonuniform three-node panel',
                  nodes_formula='T*(j/(num_t-1))**2', quadrature_bug_detected=False)
    return arrays, checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    write_json(output/'before_sha256.json',protected_snapshot(output))
    (output/'PROJECT_STATUS.before.md').write_bytes((ROOT/'PROJECT_STATUS.md').read_bytes())
    try:
        prior=read_json(PRIOR/'plan.json');baseline=read_json(BASE)
        hashes=read_json(PRIOR/'artifacts_sha256.json')
        for path in (BASE,PRIOR/'plan.json',ROOT/'diagnostics/chernov_boundary_diagnosis.py'):
            assert digest(path)==hashes[str(path.relative_to(ROOT))]
        window=ROOT/'results/chernov_window_padding24_2026-10-24'
        whashes=read_json(window/'artifacts_sha256.json')
        wrapper=ROOT/'diagnostics/chernov_window_padding24.py'
        assert digest(wrapper)==whashes[str(wrapper.relative_to(ROOT))]
        production_plan=read_json(BATCH/'plan.json')
        for name,h in prior['source_data_sha256'].items():
            assert digest(ROOT/name)==h==production_plan['protected_sha256'][name]
        versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','sympy')}
        assert versions==prior['environment']['versions']
        assert platform.python_version()==prior['environment']['python'].split()[0]
        equation=load_all_methods_equation(PID,DATA/'all_methods_equations_v1.csv',
            DATA/'all_methods_equations_v1_metadata.jsonl')
        production_path=BATCH/'solutions'/f'{PID}.json'
        production=read_json(production_path)
        assert production['metadata']==equation.metadata
        assert digest(production_path)==baseline['production_result_sha256']
        old_config=ChernovConfig(**baseline['config'])
        new_config=replace(old_config,num_t=1025)
        assert [k for k in asdict(old_config) if getattr(old_config,k)!=getattr(new_config,k)]==['num_t']
        required=dict(n=128,num_t=513,t_grading='quadratic',dx=.0125,padding=12.,
            interp='cubic',boundary='zero',tail_eps=1e-8,T=None,n_jobs=1)
        assert all(getattr(old_config,k)==v for k,v in required.items())
        T=baseline['result']['T']
        assert T==18.692237908953764
        target=np.asarray(baseline['result']['x']);exact=np.asarray(baseline['y_exact'])
        old_f=np.asarray(baseline['result']['f'])
        np.testing.assert_array_equal(target,np.linspace(-1,1,33))
        np.testing.assert_array_equal(exact,equation.y_exact(target))
        assert max_abs(old_f-exact)==baseline['max_abs_error']
        old,old_t,old_w=prepare(equation.remizov_problem,old_config,T)
        assert fingerprints(old,old_t,old_w,target)==baseline['fingerprints']
        new,new_t,new_w=prepare(equation.remizov_problem,new_config,T)
        for key in ('x','a','b','c','g'):
            np.testing.assert_array_equal(getattr(old,key),getattr(new,key))
        assert old.lam==new.lam==production['metadata']['remizov_lambda']
        assert old.taper_width==new.taper_width==2.6
        assert (new.x[0],new.x[-1],len(new.x))==(-13.,13.,2081)
        qarrays,qcheck=quadrature_check(T)
        np.testing.assert_array_equal(qarrays['513']['t'],old_t)
        np.testing.assert_array_equal(qarrays['513']['weights'],old_w)
        np.testing.assert_array_equal(qarrays['1025']['t'],new_t)
        np.testing.assert_array_equal(qarrays['1025']['weights'],new_w)
        plan=dict(schema='chernov_time_quadrature1025_v1',created_utc=datetime.now(timezone.utc).isoformat(),
            problem_id=PID,baseline_path=str(BASE.relative_to(ROOT)),baseline_sha256=digest(BASE),
            old_config=asdict(old_config),new_config=asdict(new_config),changed_config_fields=['num_t'],
            T=T,lambda_value=new.lam,baseline_error=baseline['max_abs_error'],
            unchanged_spatial_array_hashes={k:v for k,v in baseline['fingerprints'].items() if k not in ('t','weights')},
            new_fingerprints=fingerprints(new,new_t,new_w,target),quadrature_check=qcheck,
            source_data_sha256=prior['source_data_sha256'],source_script_sha256=digest(__file__),
            observer_source_sha256=digest(wrapper),environment=dict(python=sys.version,
                platform=platform.platform(),versions=versions,thread_environment={k:os.environ[k] for k in THREAD_VARS}),
            resource_policy=dict(compute_processes=1,blas_threads=1,memory_mib=768,
                expected_seconds=[25,50],stage_wall_limit_seconds=baseline['result']['runtime_seconds']*2*6),
            planned_full_solves=1,baseline_recomputed=False)
        write_json(output/'plan.json',plan)
        write_json(output/'quadrature.json',dict(checks=qcheck,arrays=qarrays))
        write_json(output/'source_metadata.json',dict(equation.metadata))
        with ResourceGuard(output/'resources.jsonl',plan['resource_policy']['stage_wall_limit_seconds']):
            result,t,w=solve_once(equation.remizov_problem,new_config,new.x,T,target,output)
        np.testing.assert_array_equal(t,new_t)
        np.testing.assert_array_equal(w,new_w)
        assert result['n_operator_applications']==128*1024
        error=max_abs(result['f']-exact);difference=max_abs(result['f']-old_f)
        record=dict(config=asdict(new_config),result=result,y_exact=exact,baseline_f=old_f,
            baseline_error=baseline['max_abs_error'],max_abs_error=error,D_Q=difference,
            delta_E=error-baseline['max_abs_error'],D_Q_over_baseline_error=difference/baseline['max_abs_error'],
            accuracy_reached=error<=.01,peak_rss_mib=peak_mib(),
            baseline_sha256=digest(BASE),source_data_sha256=prior['source_data_sha256'])
        write_json(output/'result.json',record)
        import json
        print(json.dumps(clean(dict(E_513=baseline['max_abs_error'],E_1025=error,D_Q=difference,
            delta_E=record['delta_E'],D_Q_relative=record['D_Q_over_baseline_error'],
            runtime_seconds=result['runtime_seconds'],boundary_max=result['boundary_state_max_abs'],
            observed_norm_violations=result['observed_step_bound_violations'])),indent=2),flush=True)
    except BaseException:
        write_json(output/'stopped.json',dict(traceback=traceback.format_exc()))
        raise
    finally:
        before=read_json(output/'before_sha256.json')
        changed=[p for p,h in before.items() if not (ROOT/p).exists() or digest(ROOT/p)!=h]
        write_json(output/'run_integrity.json',dict(checked_files=len(before),changed_files=changed,
            protected_files_unchanged=not changed))
        assert not changed


if __name__=='__main__':
    main()
