"""Saved-only audit/report for managed completion (also supports a safety stop).

No numerical solver calls, writes exclusively into a new report directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.run_chernov_batch import (accuracy_status,check_protected,digest,
    read_json,validate_identity)
from diagnostics.report_chernov_batch import distribution


def build(batch,experiment,dest):
    protocol=read_json(experiment/'plan_v2.json');plan=read_json(batch/'plan.json')
    baseline=read_json(experiment/'baseline.json')
    before=read_json(experiment/'before_sha256.json')
    check_protected(plan,batch)
    assert digest(batch/'plan.json')==protocol['original_plan_sha256']
    assert digest(batch/'safety_policy_v2.json')==protocol['safety_policy_v2_sha256']
    for name,value in protocol['management_sha256'].items():assert digest(ROOT/name)==value,name
    changed=[]
    for name,value in before.items():
        path=ROOT/name
        assert path.is_file(),f'missing preserved file: {name}'
        if digest(path)!=value:changed.append(name)
    allowed={'PROJECT_STATUS.md','docs/CHERNOV_BATCH_EXECUTION.md',
             'diagnostics/chernov_monitored_batch.py',
             str((batch/'progress.json').relative_to(ROOT)),str((batch/'active.json').relative_to(ROOT))}
    assert set(changed)<=allowed,changed
    for pid,value in baseline['solution_sha256'].items():
        assert digest(batch/'solutions'/f'{pid}.json')==value,pid

    rows=[];by_id={}
    for pid in plan['equation_ids']:
        path=batch/'solutions'/f'{pid}.json'
        if not path.exists():continue
        record=read_json(path);validate_identity(record,plan);result=record['result']
        if record['computation_status']=='success':
            assert result['success'] and result['scheme']=='solve'
            assert result['x']==record['target_x']==plan['target_x']
            assert len(result['f'])==len(record['y_exact'])==33
            assert all(math.isfinite(f) for f in result['f'])
            error=max(abs(f-y) for f,y in zip(result['f'],record['y_exact']))
            assert error==record['max_abs_error']
            assert result['n']==128 and result['time_nodes']==513 and result['actual_dx']==.0125
            assert result['work_grid_points']==2081 and result['n_operator_applications']==65536
        assert record['status']==accuracy_status(record['computation_status'],record['max_abs_error'],.01)
        assert record['accuracy_reached']==(record['status']=='accuracy_pass')
        assert digest(path)==digest(batch/'attempts'/record['attempt_id']/'result.json')
        by_id[pid]=record
        metadata=record.get('metadata',{})
        rows.append(dict(problem_id=pid,status=record['status'],computation_status=record['computation_status'],
            accuracy_reached=record['accuracy_reached'],max_abs_error=record['max_abs_error'],
            relative_error=record['relative_error'],solver_seconds=record['solver_seconds'],
            task_wall_seconds=record['task_wall_seconds'],peak_rss_mib=record['process_peak_rss_mib'],
            complexity=metadata.get('complexity_level'),regime=metadata.get('regime'),split=metadata.get('split'),
            new_in_this_task=pid not in baseline['solution_sha256'],attempt_id=record['attempt_id']))
    assert len(rows)==len(list((batch/'solutions').glob('*.json')))
    assert set(by_id)-baseline['solution_sha256'].keys()<=set(protocol['pending_ids'])
    raw_ids={read_json(p)['problem_id'] for p in (batch/'attempts').glob('*/result.json')}
    assert raw_ids==set(by_id),'committed answer not adopted'

    series=read_json(experiment/'managed_run_v2/series.json')
    assert series['finished'] and series['backend_locks_free'] and series['old_solutions_unchanged']
    telemetry=[];batch_rows=[];renewals=[]
    for folder in sorted(experiment.glob('managed_run*')):
        for path in folder.glob('**/samples.jsonl'):
            telemetry.extend(json.loads(line) for line in path.read_text().splitlines())
        cooling=folder/'cooling_samples.jsonl'
        if cooling.exists():telemetry.extend(json.loads(line) for line in cooling.read_text().splitlines())
        for path in sorted(folder.glob('batch_*/run.json')):
            r=read_json(path)
            assert r['cleanup_confirmed'] and r['old_solutions_unchanged']
            logs=[json.loads(line) for line in (path.parent/'lease_renewals.jsonl').read_text().splitlines()]
            assert len(logs)==r['automatic_lease_renewals']
            assert all(log['thermal_state']==0 and log['lease_seconds']==120 for log in logs)
            renewals.extend(logs)
            batch_rows.append(dict(directory=str(path.parent.relative_to(experiment)),
                selected_ids=r['selected_ids'],new_ids=r['new_ids'],started_epoch=r['started_epoch'],
                finished_epoch=r['finished_epoch'],wall_seconds=r['finished_epoch']-r['started_epoch'],
                returncode=r['returncode'],automatic_lease_renewals=len(logs),cleanup_confirmed=r['cleanup_confirmed'],
                stop_reason=r.get('stop_reason'),monitor_error=r.get('monitor_error')))
    workers=[worker for sample in telemetry for worker in sample['compute_workers']]
    assert max((s['compute_worker_count'] for s in telemetry),default=0)<=1
    completed_batches=[r for r in batch_rows if r['new_ids'] and r['returncode']==0]
    pauses=[b['started_epoch']-a['finished_epoch'] for a,b in zip(completed_batches,completed_batches[1:])]
    assert all(p>=60 for p in pauses),pauses
    outcomes=dict(Counter(r['status'] for r in rows))
    finite=[r for r in rows if r['computation_status']=='success']
    errors=[r['max_abs_error'] for r in finite]
    new=[r for r in rows if r['new_in_this_task']]
    stats=dict(total=1000,terminal_records=len(rows),finite_results=len(finite),pending=1000-len(rows),
        accuracy_pass=outcomes.get('accuracy_pass',0),accuracy_not_met=outcomes.get('accuracy_not_met',0),
        errors=len(rows)-len(finite),outcomes=outcomes,new_terminal_records=len(new),
        new_outcomes=dict(Counter(r['status'] for r in new)),absolute_error=distribution(errors),
        solver_seconds=distribution([r['solver_seconds'] for r in finite]),
        new_solver_seconds=distribution([r['solver_seconds'] for r in new if r['solver_seconds'] is not None]),
        batch_execution_seconds=sum(r['wall_seconds'] for r in batch_rows),
        managed_series_wall_seconds=series['finished_epoch']-series['started_epoch'],
        finished_batches=len(completed_batches),pauses_seconds=pauses,
        thermal_states=dict(Counter(str(s['thermal_state']) for s in telemetry)),samples=len(telemetry),
        max_compute_workers=max((s['compute_worker_count'] for s in telemetry),default=0),
        max_worker_cpu_percent=max((w['cpu_percent'] for w in workers),default=None),
        max_sample_rss_mib=max((w['rss_mib'] for w in workers),default=None),
        temperature_celsius=None,checked_lease_renewals=len(renewals),owner_lease_seconds=120,
        series_phase=series['phase'],series_stop_reason=series.get('stop_reason'),
        computation_error_ids=[r['problem_id'] for r in rows if r['computation_status']!='success'],
        error_bins={label:sum(low<e<=high for e in errors) for label,low,high in
          [('<=1e-3',-1,1e-3),('(1e-3,1e-2]',1e-3,1e-2),('(1e-2,1e-1]',1e-2,.1),('>1e-1',.1,float('inf'))]})
    verification=dict(saved_files_only=True,numerical_solver_calls=0,baseline_files_checked=len(before),
        baseline_changed_files=changed,baseline_missing_files=[],old_700_unchanged=True,
        core_and_dataset_unchanged=True,source_protection_pass=True,errors_recomputed=True,
        canonical_matches_raw_attempts=True,series_finished=True,backend_locks_free=True,
        solution_sha256={pid:digest(batch/'solutions'/f'{pid}.json') for pid in by_id},
        report_source_sha256=digest(Path(__file__)))
    dest.mkdir(parents=True,exist_ok=False)
    for name,value in [('statistics.json',stats),('verification.json',verification),('batches.json',batch_rows)]:
        with (dest/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    with (dest/'all_equations.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Управляемое завершение ChernovSolver.solve на 1000 уравнениях','',
        f"Терминальных записей: **{len(rows)}/1000**; конечных ответов: **{len(finite)}**; "
        f"E∞≤0.01: **{stats['accuracy_pass']}**; выше порога: **{stats['accuracy_not_met']}**; "
        f"вычислительных ошибок: **{stats['errors']}**; pending: **{stats['pending']}**.",'',
        '## Протокол и сохранность','',
        'До запуска сохранено 700 терминальных записей (633 accuracy_pass, 64 accuracy_not_met, '
        '3 worker_exception). Все прежние ответы, включая отказы, пропущены и побайтово '
        'сохранены. Остаток 300 зафиксирован до запуска, без выбора по exact. '
        'Только chernov_solver.ChernovSolver.solve: n=128, 513 quadratic Simpson узлов, '
        'dx=0.0125, padding=12, cubic+taper, tail_eps=1e-8, n_jobs=1, BLAS=1. '
        'T определяется сертификатами; рабочая сетка 2081 точка [-13,13]. '
        'Exact вычисляется после solve; ошибка проверяется на 33 точках [-1,1]. '
        'Численная схема, исходный план и safety_policy_v2 не менялись.','',
        'Внешний монитор выполняет партии по 20 (доступно 10–20), пауза минимум 60 с, '
        'телеметрия каждые 2 с. Разрешение guardian остаётся 120 с; внешний монитор '
        'продлевает его раз в 20 с только после проверки свежей здоровой телеметрии '
        'и владения живым worker/guardian/session. Просроченное разрешение не '
        'восстанавливается, backend его не продлевает. Потеря/зависание монитора '
        'прекращает продления; потеря backend обнаруживается guardian по EOF/heartbeat. '
        'Ctrl+C или --stop монитора останавливает серию и во время паузы. '
        'При тепловой/процессной проблеме или новом вычислительном отказе вся серия '
        'останавливается без автоматического восстановления.','',
        'Первая управляющая попытка managed_run остановлена с 0 новыми ответами: '
        'обнаружено повторное сканирование всех ответов для каждого pending ID. '
        'После исправления на один снимок перед партией добавлена регрессия. '
        'Обе версии/планы/журналы сохранены; основной запуск — managed_run_v2. '
        '45 коротких тестов прошли за 14.84 с, без численных решателей.','',
        '## Итог и ресурсы','',
        f"Новых итогов: {len(new)}, статусы: {stats['new_outcomes']}. "
        f"Фаза монитора: {series['phase']}; причина остановки: {series.get('stop_reason') or 'нет'}.",'',
        f"Завершённых партий: {len(completed_batches)}; checked lease renewals: {len(renewals)}; "
        f"сэмплов: {len(telemetry)}; максимум worker: {stats['max_compute_workers']}; "
        f"CPU worker до {stats['max_worker_cpu_percent']}% одного ядра; "
        f"RSS по сэмплам до {stats['max_sample_rss_mib']} МиБ. "
        f"thermalState: {stats['thermal_states']}.",'',
        'CPU/RSS приведены по наблюдённым сэмплам. Температура в °C не измерялась: '
        'NSProcessInfo.thermalState — системное состояние, неподдерживаемая система '
        'также может вернуть nominal. Пользователь согласовал этот режим вместе '
        'со своим наблюдением за нагревом.','',
        '| Метрика | min | median | mean | p90 | p95 | p99 | max |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,dist in [('E∞',stats['absolute_error']),('Время solve, с (вся серия)',stats['solver_seconds']),
                       ('Время solve, с (новые задачи)',stats['new_solver_seconds'])]:
        if dist['count']:lines.append('| '+label+' | '+' | '.join(f'{dist[k]:.6g}' for k in ('min','median','mean','p90','p95','p99','max'))+' |')
    lines+=['',f"Wall основной управляемой серии: {stats['managed_series_wall_seconds']:.3f} с; "
             f"паузы: {sum(pauses):.3f} с. Ошибки: {stats['error_bins']}.",'',
        'Идентификаторы вычислительных отказов: '+', '.join(stats['computation_error_ids'])+'.','',
        'Прежние три worker_exception не повторялись; точные сигналы старый драйвер '
        'не записывал. accuracy_not_met — конечный ответ выше порога, а не '
        'авария. Источники численной ошибки этой серией не разделялись. '
        'Success не сертифицирует полную ошибку; порог проверен на 33 точках, '
        'не на непрерывной области. Результаты старой параллельной и новой '
        'последовательной частей не являются контролируемым сравнением скорости.','',
        '## Артефакты','',
        'statistics.json — сводка; all_equations.csv — все сохранённые итоги; '
        'batches.json — партии; verification.json — проверки/хеши. '
        'Исходные численные массивы/параметры/время/exact сохранены в '
        'results/chernov_solve1000_2026-10-18/solutions и attempts. '
        'Протоколы plan.json/plan_v2.json, исходные резервные копии before/, '
        'revisions/ и журналы managed_run*/ находятся в родительском каталоге отчёта. '
        'ML, streaming и другие методы не запускались.','']
    with (dest/'REPORT.md').open('x') as f:f.write('\n'.join(lines))
    print(json.dumps({k:stats[k] for k in ('terminal_records','finite_results','accuracy_pass','accuracy_not_met','errors','pending')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,default=ROOT/'results/chernov_solve1000_2026-10-18')
    parser.add_argument('--experiment',type=Path,default=ROOT/'results/chernov_managed_completion_2026-10-18')
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args();build(args.batch.resolve(),args.experiment.resolve(),args.report_dir.resolve())


if __name__=='__main__':main()
