"""Exactly one uninstrumented F0000 n=4 call using the existing watchdog.

No solver-function wrappers, cProfile sessions, retries, or parameter changes.
The supervisor uses bounded_run(..., profile=False) from the prior experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.remizov_n_study import snapshot, write_json, THREAD_VARS
from diagnostics.remizov_n4_followup import (
    bounded_run, tree_hashes, EXPECTED_CONFIG, WALL_LIMIT, RSS_LIMIT_MIB, POLL_SECONDS,
)

TASK_CONFIG=dict(n=4,tail_eps=1e-8,nt_initial=8,dx_initial=.05,
    quadrature_rtol=1e-7,quadrature_atol=1e-9,spatial_rtol=1e-7,spatial_atol=1e-9,
    max_quadrature_refinements=8,max_spatial_refinements=5,max_work_grid_points=100000)
PREVIOUS=('remizov_n_study_2026-10-11','remizov_n4_followup_2026-10-12')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'results'):
        parser.error('output must be a new directory under project/results')
    from verify_remizov_solver import make_verification_config
    assert EXPECTED_CONFIG==TASK_CONFIG==asdict(make_verification_config(4))
    assert WALL_LIMIT==45.0 and RSS_LIMIT_MIB==512 and POLL_SECONDS==.25
    out.mkdir(parents=True,exist_ok=False)
    before=snapshot(ROOT)
    original=ROOT.parent/'generator_DE'
    original_before=snapshot(original)
    old_hashes={name:tree_hashes(ROOT/'results'/name) for name in PREVIOUS}
    prior={name:json.loads((ROOT/'results'/name/'summary.json').read_text()) for name in PREVIOUS}
    manifest=json.loads((ROOT/'MIGRATION_MANIFEST.json').read_text())
    core={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==manifest['source_sha256'][name]
          for name in manifest['unchanged_mathematical_core']}
    assert all(core.values())
    summary={
        'client_study_date':'2026-10-12','started_utc':datetime.now(timezone.utc).isoformat(),
        'equation_id':'AMV1_F0000_S00','n':4,'config':TASK_CONFIG,
        'x':prior[PREVIOUS[0]]['x'],
        'environment':{'python':sys.version,'executable':sys.executable,
            'platform':platform.platform(),'cpu_count':os.cpu_count(),
            'versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','sympy')},
            'thread_environment':{key:os.environ[key] for key in THREAD_VARS}},
        'resource_policy':{'wall_seconds_per_solver_process':45.0,'observed_rss_limit_mib':512,
            'poll_seconds':.25,'solver_processes':1,'parallel_solver_processes':0,'retries':0,
            'profiling_enabled':False,'function_instrumentation_enabled':False,
            'watchdog_reused_without_changes':True,
            'termination_wait_seconds':2,'termination_wait_is_not_extra_solver_budget':True},
        'wrapper':{'file':'diagnostics/remizov_n4_followup.py','call':
            "bounded_run('AMV1_F0000_S00', False, new_results_dir / 'F0000_n4_unprofiled')",
            'module_import_of_cProfile_does_not_enable_profiling':True},
        'preflight':{'previous_unprofiled_external_stop_seconds':45.262741292011924,
            'previous_profiled_external_stop_seconds':45.01400120800827,
            'previous_profiled_last_dx':.003125,'previous_profiled_last_nt':2048,
            'n4_finest_grid_points_bound':26331,'n4_max_operator_applications':98112,
            'completion_within_limit_not_guaranteed':True},
        'project_before_sha256':before,'original_before_sha256':original_before,
        'previous_results_before_sha256':old_hashes,'core_matches_migration_before':core,
        'runs':[],
    }
    write_json(out/'summary.json',summary)
    print('START exactly one AMV1_F0000_S00 n=4; no instrumentation',flush=True)
    result=bounded_run('AMV1_F0000_S00',False,out/'F0000_n4_unprofiled')
    summary['runs'].append(result)
    summary['finished_utc']=datetime.now(timezone.utc).isoformat()
    after=snapshot(ROOT); original_after=snapshot(original)
    summary['integrity']={
        'project_changed_or_removed':[p for p,h in before.items() if after.get(p)!=h],
        'original_changed_or_removed':[p for p,h in original_before.items() if original_after.get(p)!=h],
        'original_added':sorted(set(original_after)-set(original_before)),
        'previous_results_identical':{name:tree_hashes(ROOT/'results'/name)==hashes
                                    for name,hashes in old_hashes.items()},
        'excluded_from_source_snapshots':['.venv','.git','.pytest_cache','__pycache__','results','.DS_Store'],
    }
    write_json(out/'summary.json',summary)
    assert len(summary['runs'])==1
    print(f'FINISHED {out}',flush=True)


if __name__=='__main__':
    main()
