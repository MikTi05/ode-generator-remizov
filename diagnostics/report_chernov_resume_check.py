"""Audit the saved three-equation lifecycle pilot; never import/run a solver.

Run once from the project root. Outputs are created exclusively, preserving any
existing report. This is a historical pilot audit, not the full-series report.
"""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'results/chernov_resume_check_2026-10-18'
BATCH = ROOT / 'results/chernov_solve1000_2026-10-18'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = read(PILOT / 'plan.json')
    baseline = read(PILOT / 'before_sha256.json')
    stages = {s: read(PILOT / s / 'run.json') for s in ('stop_check', 'resume_two')}
    selected = plan['selected_ids']
    old_solutions = stages['stop_check']['before_solutions_sha256']
    current = {p.stem: read(p) for p in (BATCH / 'solutions').glob('*.json')}
    assert len(old_solutions) == 694
    assert set(current) - old_solutions.keys() == set(selected)
    assert stages['stop_check']['new_ids'] == selected[:1]
    assert stages['resume_two']['new_ids'] == selected[1:]
    assert stages['stop_check']['returncode'] == 130
    assert stages['resume_two']['returncode'] == 0
    assert stages['stop_check']['stop_reason'] == 'planned_stop_during_second_real_equation'
    cooldown = stages['resume_two']['started_epoch'] - stages['stop_check']['finished_epoch']
    assert cooldown >= plan['cooldown_seconds']

    changed = []
    for name, value in baseline.items():
        path = ROOT / name
        assert path.exists(), f'missing baseline file: {name}'
        if sha(path) != value:
            changed.append(name)
    assert set(changed) <= {'PROJECT_STATUS.md', 'docs/CHERNOV_BATCH_EXECUTION.md',
                           'results/chernov_solve1000_2026-10-18/progress.json'}
    for pid, value in old_solutions.items():
        assert sha(BATCH / 'solutions' / f'{pid}.json') == value
    assert sha(BATCH / 'plan.json') == plan['base_plan_sha256']
    assert sha(BATCH / 'safety_policy_v2.json') == plan['base_safety_sha256']
    assert sha(ROOT / 'diagnostics/chernov_monitored_batch.py') == plan['environment']['monitor_source_sha256']

    attempts = []
    for path in sorted((BATCH / 'attempts').glob('*/request.json')):
        if str(path.relative_to(ROOT)) in baseline:
            continue
        request = read(path)
        pid = request['problem_id']
        assert pid in selected
        raw = path.parent / 'result.json'
        interrupted = path.parent / 'interrupted.json'
        assert raw.exists() != interrupted.exists()
        attempts.append(dict(problem_id=pid, attempt_id=request['attempt_id'],
                             status='completed' if raw.exists() else 'interrupted',
                             interrupted=read(interrupted) if interrupted.exists() else None))
    assert Counter(a['problem_id'] for a in attempts) == Counter({selected[0]: 1, selected[1]: 2, selected[2]: 1})
    assert sum(a['status'] == 'interrupted' for a in attempts) == 1

    rows = []
    for pid in selected:
        record = current[pid]
        result = record['result']
        assert record['implementation'] == 'chernov_solver.ChernovSolver.solve'
        assert record['config'] == plan['config']
        assert record['target_x'] == result['x'] == plan['target_x']
        assert len(result['f']) == len(record['y_exact']) == 33
        assert result['success'] and result['scheme'] == 'solve'
        error = max(abs(f-y) for f, y in zip(result['f'], record['y_exact']))
        assert math.isfinite(error) and error == record['max_abs_error']
        assert record['accuracy_reached'] == (error <= plan['threshold'])
        assert record['runtime_compute_workers'] == 1
        assert sha(BATCH / 'solutions' / f'{pid}.json') == sha(BATCH / 'attempts' / record['attempt_id'] / 'result.json')
        rows.append(dict(problem_id=pid, complexity_level=record['metadata']['complexity_level'],
                         regime=record['metadata']['regime'], max_abs_error=error,
                         relative_error=record['relative_error'], solver_seconds=record['solver_seconds'],
                         task_wall_seconds=record['task_wall_seconds'], T=result['T'], actual_dx=result['actual_dx'],
                         process_peak_rss_mib=record['process_peak_rss_mib'],
                         operator_applications=result['n_operator_applications'],
                         spline_builds=result['n_spline_builds'], work_grid_points=result['work_grid_points'],
                         accuracy_reached=record['accuracy_reached'], status=record['status']))

    telemetry = {}
    for name, stage in stages.items():
        samples = [json.loads(line) for line in (PILOT / name / 'samples.jsonl').read_text().splitlines()]
        workers = [w for sample in samples for w in sample['compute_workers']]
        assert max(s['compute_worker_count'] for s in samples) == 1
        assert all(s['thermal_state'] == 0 and not s['safety_stop_reason'] for s in samples)
        assert stage['automatic_lease_renewals'] == 0 and stage['owner_lease_seconds'] == 120
        assert stage['cleanup_confirmed'] and stage['old_solutions_unchanged']
        assert not stage['final_sample']['compute_workers']
        assert all(g['finished'] and g['cleanup_success'] for g in stage['final_sample']['guardians'])
        exit_check = read(PILOT / ('stage_A_process_exit.json' if name == 'stop_check' else 'stage_B_process_exit.json'))
        assert exit_check['all_three_processes_absent'] and exit_check['stdout'] == '' and exit_check['returncode'] == 1
        telemetry[name] = dict(samples=len(samples), max_compute_workers=1,
                               max_cpu_percent=max(w['cpu_percent'] for w in workers),
                               max_sample_rss_mib=max(w['rss_mib'] for w in workers),
                               thermal_states=[0], temperature_celsius=None,
                               wall_seconds=stage['finished_epoch']-stage['started_epoch'],
                               automatic_lease_renewals=0, cleanup_confirmed=True)
    outcomes = dict(Counter(r['status'] for r in current.values()))
    exceptions = sorted(pid for pid, r in current.items() if r['status'] == 'worker_exception')
    assert exceptions == ['AMV1_F0134_S03', 'AMV1_F0136_S01', 'AMV1_F0139_S00']
    summary = dict(pilot_pass=True, saved_files_only=True, baseline_file_count=len(baseline),
                   baseline_changed_files=changed, baseline_missing_files=[], old_694_unchanged=True,
                   selected_ids=selected, results=rows, attempts=attempts, telemetry=telemetry,
                   cooldown_observed_seconds=cooldown, terminal_records=len(current),
                   finite_results=sum(bool((r.get('result') or {}).get('f')) for r in current.values()),
                   pending=1000-len(current), outcomes=outcomes, unchanged_worker_exceptions=exceptions,
                   owner_lease_seconds=120, automatic_lease_renewals=0,
                   full_remaining_series_started=False,
                   thermal_mode_user_accepted='macOS thermalState + operator observation; no Celsius',
                   tests=read(PILOT / 'tests.json'), report_source_sha256=sha(Path(__file__)))
    with (PILOT / 'summary.json').open('x') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write('\n')
    with (PILOT / 'results.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: summary[k] for k in ('pilot_pass', 'terminal_records', 'finite_results', 'pending', 'outcomes')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
