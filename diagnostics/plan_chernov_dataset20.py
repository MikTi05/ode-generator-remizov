"""Freeze one global configuration before the complete 20-equation survey.

Reads IDs/labels/certificates, never evaluates exact solutions or runs a solver.
The numerical executor is the existing diagnostics/compare_chernov.py.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chernov_solver import ChernovConfig

COMMON_CONFIG = ChernovConfig(
    n=128, num_t=513, n_steps=32768, dx=.0125, padding=12., tail_eps=1e-8,
    T=None, t_grading='quadratic', interp='cubic', boundary='taper',
    taper_width=None, max_work_grid_points=100000, n_jobs=1, start_method='spawn',
)


def build_plan(equation_ids: list[str]) -> dict:
    if len(equation_ids) != 20 or len(set(equation_ids)) != 20:
        raise ValueError('the survey requires exactly 20 unique equation IDs')
    COMMON_CONFIG.validate()
    return {
        'purpose': 'One-pass coverage of both Chern schemes on all 20 existing equations; no per-equation tuning or retries',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'parameter_policy': 'identical complete ChernovConfig for every equation and both schemes; only T derives from the certified gap by the unchanged formula',
        'selection_rationale': 'n=128 is twice the largest selected fixed n in the preceding four-task study; 513 time nodes double its panels; K=32768 is four times its largest selected streaming K; dx is halved, padding increased from 8 to 12. No current exact errors are inspected before freezing this plan.',
        'target_grid': {'x_min': -1., 'x_max': 1., 'points': 33},
        'accuracy_threshold': .01,
        'wall_seconds': 60., 'rss_limit_mib': 512,
        'resource_policy': {'concurrent_solver_processes': 1, 'blas_threads': 1,
                            'poll_seconds': .1, 'automatic_retries': 0,
                            'continue_after_failure': True,
                            'max_attempts': 40, 'worst_case_worker_wall_seconds': 2400},
        'attempts': [dict(problem_id=pid, method=method, config=asdict(COMMON_CONFIG))
                     for pid in equation_ids for method in ('solve', 'streaming')],
    }


def snapshot_old_files(new_root: Path) -> dict[str, str]:
    """Include all earlier result files, data and mathematical sources."""
    paths = [ROOT/name for name in ('remizov_solver.py','chernov_solver.py',
        'all_methods_dataset_loader.py','src/all_methods_expressions.py',
        'src/all_methods_dataset_v1.py','MIGRATION_MANIFEST.json',
        'diagnostics/compare_chernov.py','diagnostics/plan_chernov_dataset20.py',
        'diagnostics/report_chernov_dataset20.py')]
    for folder in ('data', 'results', 'comparison_inputs'):
        paths.extend(p for p in (ROOT/folder).rglob('*')
                     if p.is_file() and not p.is_relative_to(new_root))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file() and p.name != '.DS_Store'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'results'):
        parser.error('new output directory must be under results')
    data = ROOT/'data/generated'
    with (data/'all_methods_equations_v1.csv').open(newline='') as f:
        ids = [row['equation_id'] for row in csv.DictReader(f)]
    metadata = [json.loads(line) for line in
                (data/'all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
    by_id = {m['equation_id']: m for m in metadata}
    if len(by_id) != 20 or set(ids) != set(by_id):
        raise ValueError('CSV/metadata must have identical 20 unique IDs')
    plan = build_plan(ids)
    # Geometry and operation counts only; no functions evaluated.
    preflight = []
    for pid in ids:
        m = by_id[pid]
        gap = m['remizov_lambda'] - m['remizov_c_abs_cert']
        T = max(0., (math.log(2)-math.log(gap)-math.log(COMMON_CONFIG.tail_eps))/gap)
        preflight.append(dict(problem_id=pid, complexity=m['complexity_level'],
            regime=m['regime'], gap=gap, T=T, work_domain=[-13.,13.],
            work_grid_points=2081, fixed_n_operator_applications=128*512,
            streaming_operator_applications=32768, streaming_tau=T/32768,
            estimated_solver_seconds_from_prior_study={'solve': [10,25], 'streaming': [5,15]},
            runtime_estimate_not_a_bound=True))
    out.mkdir(parents=True, exist_ok=False)
    for name, value in (('plan.json',plan), ('preflight.json',preflight),
                        ('before_sha256.json',snapshot_old_files(out))):
        (out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'attempts':len(plan['attempts']), 'config':asdict(COMMON_CONFIG),
                      'T_range':[min(r['T'] for r in preflight),max(r['T'] for r in preflight)],
                      'preflight_saved':str(out/'preflight.json')},indent=2))


if __name__ == '__main__':
    main()
