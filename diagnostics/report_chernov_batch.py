"""Read-only numerical analysis of saved batch solutions; never invoke solvers."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.run_chernov_batch import (accuracy_status, digest, read_json,
    validate_identity, write_json, safety_overrides, check_protected)


def distribution(values):
    if not values:
        return dict(count=0)
    a=np.asarray(values,dtype=float)
    return dict(count=len(values),min=float(a.min()),p05=float(np.percentile(a,5)),
        median=float(np.median(a)),mean=float(a.mean()),p90=float(np.percentile(a,90)),
        p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),
        max=float(a.max()),sum=float(a.sum()))


def analyze(out):
    plan=read_json(out/'plan.json')
    labels={r['problem_id']:r for r in read_json(out/'preflight.json')}
    rows=[]
    for pid in plan['equation_ids']:
        path=out/'solutions'/f'{pid}.json'
        if not path.exists():
            raise ValueError(f'incomplete batch: {pid}')
        r=read_json(path);validate_identity(r,plan)
        error=r['max_abs_error'];result=r['result']
        if r['computation_status']=='success':
            if result['scheme']!='solve' or not result['success']:
                raise ValueError('wrong scheme/success')
            np.testing.assert_array_equal(result['x'],plan['target_x'])
            f=np.asarray(result['f']);exact=np.asarray(r['y_exact'])
            if f.shape!=(33,) or exact.shape!=(33,) or not np.all(np.isfinite(f)):
                raise ValueError('invalid numerical values')
            actual=float(np.max(np.abs(f-exact)))
            if error!=actual or result['n']!=128 or result['time_nodes']!=513:
                raise ValueError('inconsistent saved result')
            if result['n_operator_applications']!=65536 or result['work_grid_points']!=2081:
                raise ValueError('unexpected numerical work/configuration')
        if r['status']!=accuracy_status(r['computation_status'],error,plan['threshold']):
            raise ValueError('incorrect accuracy classification')
        if r['accuracy_reached']!=(r['status']=='accuracy_pass'):
            raise ValueError('incorrect accuracy flag')
        label=labels[pid]
        rows.append(dict(problem_id=pid,family_id=label['family_id'],split=label['split'],
            complexity=label['complexity'],regime=label['regime'],status=r['status'],
            computation_status=r['computation_status'],accuracy_reached=r['accuracy_reached'],
            max_abs_error=error,relative_error=r['relative_error'],
            solver_seconds=r['solver_seconds'],task_wall_seconds=r['task_wall_seconds'],
            peak_rss_mib=r['process_peak_rss_mib'],T=result['T'] if result else label['T'],
            actual_dx=result['actual_dx'] if result else None,
            n_operator_applications=result['n_operator_applications'] if result else None,
            result_file=str(path.relative_to(out)),attempt_id=r['attempt_id']))
    if len(list((out/'solutions').glob('*.json')))!=1000 or len(rows)!=1000:
        raise ValueError('expected exactly 1000 unique saved solutions')
    errors=[r['max_abs_error'] for r in rows if r['max_abs_error'] is not None]
    sessions=[read_json(p) for p in sorted((out/'sessions').glob('*.json'))]
    stats=dict(total=1000,computational_successes=sum(r['computation_status']=='success' for r in rows),
        accuracy_passes=sum(r['accuracy_reached'] for r in rows),
        accuracy_fraction=sum(r['accuracy_reached'] for r in rows)/1000,
        outcomes=dict(Counter(r['status'] for r in rows)),
        absolute_error=distribution(errors),relative_error=distribution([r['relative_error'] for r in rows if r['relative_error'] is not None]),
        solver_seconds=distribution([r['solver_seconds'] for r in rows if r['solver_seconds'] is not None]),
        task_wall_seconds=distribution([r['task_wall_seconds'] for r in rows]),
        max_worker_rss_mib=max(r['peak_rss_mib'] for r in rows),
        execution_session_wall_seconds=sum(s.get('wall_seconds',0) for s in sessions),
        session_count=len(sessions),interrupted_attempts=len(list((out/'attempts').glob('*/interrupted.json'))),
        requests_without_result=[str(p.relative_to(out)) for p in (out/'attempts').glob('*/request.json') if not (p.parent/'result.json').exists()],
        error_bins={label:sum(low<e<=high for e in errors) for label,low,high in
            [('<=1e-3',-1,1e-3),('(1e-3,1e-2]',1e-3,1e-2),('(1e-2,1e-1]',1e-2,.1),('>1e-1',.1,float('inf'))]},
        grouped={field:{key:dict(total=len(group),accuracy_passes=sum(r['accuracy_reached'] for r in group),
            max_abs_error=max((r['max_abs_error'] for r in group if r['max_abs_error'] is not None),default=None))
            for key in sorted({r[field] for r in rows})
            for group in [[r for r in rows if r[field]==key]]}
            for field in ('complexity','regime','split')})
    return plan,rows,stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir',type=Path)
    args=parser.parse_args();out=args.output_dir.resolve()
    if (out/'REPORT.md').exists():
        raise FileExistsError('refusing to overwrite report')
    plan,rows,stats=analyze(out)
    before=read_json(out/'before_sha256.json')
    changed=[name for name,h in before.items() if not (ROOT/name).is_file() or digest(ROOT/name)!=h]
    approved=safety_overrides(out,plan);check_protected(plan,out)
    unexpected=[name for name in changed if name!='PROJECT_STATUS.md' and name not in approved]
    if unexpected:
        raise ValueError(f'old code/data/results changed: {unexpected}')
    with (out/'all_equations.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'statistics.json',stats,exclusive=True)
    write_json(out/'verification.json',dict(unique_equations=1000,uniform_config=True,
        implementation=plan['implementation'],errors_recomputed=True,
        previous_files_checked=len(before),changed_files=changed,unexpected_changes=unexpected,
        artifact_sha256={str(p.relative_to(out)):digest(p) for p in (out/'solutions').glob('*.json')}),exclusive=True)
    s=stats;t=s['solver_seconds'];e=s['absolute_error']
    lines=['# ChernovSolver.solve на 1000 уравнениях','',
        f"Вычислительно завершились {s['computational_successes']}/1000; "
        f"E∞≤0.01 достигли **{s['accuracy_passes']}/1000 ({100*s['accuracy_fraction']:.1f}%)**.",'',
        '## Протокол','',
        'Использован только chernov_solver.ChernovSolver.solve. Схема, defaults '
        'и исходный remizov_solver не менялись. Единая конфигурация из успешной '
        'серии на 20 задачах: n=128, 513 quadratic Simpson узлов, dx=0.0125, '
        'padding=12, cubic+taper, tail_eps=1e-8, n_jobs=1; полный asdict в plan.json. '
        '2081 рабочая точка [-13,13]; T по сертификатам. Целевая сетка: 33 точки [-1,1]. '
        'Exact вычислен только после solve для проверки и не участвует в выборе '
        'параметров, вычислении интеграла или остановке.','',
        'До полной серии проверены пять заранее выбранных задач разных сложностей '
        'и режимов; pilot_check.json. При продолжении эти задачи пропущены. '
        'Результаты публикуются атомарно после каждого вызова. Повторяются лишь '
        'незавершённые запросы после прерывания с сохранением старых попыток; '
        'терминальные ошибки и accuracy_not_met не пересчитываются.','',
        'Исходная прерванная сессия использовала до 8 worker. После исправления '
        'надзора действует жёсткий предел 1 вычислительный worker, глобальные '
        'блокировки, независимый guardian и внешнее разрешение на 120 с '
        'с отдельным продлением; история сохранена в safety_policy_v2.json '
        'и sessions/*/runtime_policy. BLAS=1, n_jobs=1. Лимит задачи 120 с, '
        'startup 120 с, RSS 512 МиБ/worker. RSS — getrusage peak за жизнь worker, '
        'не отдельная память последней задачи. Параллельные времена нельзя напрямую '
        'сравнивать с прежним последовательным замером.','',
        '## Статистика','',
        '| Величина | min | median | mean | p90 | p95 | p99 | max |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,d in (('E∞',e),('Время решателя, с',t),('Wall задачи, с',s['task_wall_seconds'])):
        lines.append('| '+name+' | '+' | '.join(f'{d[k]:.6g}' for k in ('min','median','mean','p90','p95','p99','max'))+' |')
    lines += ['',f"Сумма времени решателей: {t.get('sum',0):.3f} с; сумма wall сессий: "
        f"{s['execution_session_wall_seconds']:.3f} с; максимальный RSS worker: {s['max_worker_rss_mib']:.2f} МиБ.",'',
        'Распределение ошибок: '+', '.join(f'{k}: {v}' for k,v in s['error_bins'].items())+'.','',
        'Статусы: '+', '.join(f'{k}: {v}' for k,v in s['outcomes'].items())+'.','',
        'accuracy_not_met — конечный результат выше порога. Разделение источников '
        'ошибки (конечное n, квадратура, интерполяция, окно, хвост) этой серией не '
        'выполнялось. external_* — остановки оболочки; worker_exception — исключение '
        'процесса/вычисления. Прерванным вычислениям не приписываются значения '
        'ошибки или времени решателя.','',
        '## Покрытие групп','',
        '| Группа | Уравнений | E∞≤0.01 | max E∞ |','|---|---:|---:|---:|']
    for field,groups in s['grouped'].items():
        for key,g in groups.items():
            lines.append(f"| {field}: {key} | {g['total']} | {g['accuracy_passes']} | {g['max_abs_error']} |")
    lines += ['', '## Артефакты и ограничения','',
        'all_equations.csv — все 1000 строк; solutions/<ID>.json — f, x, exact, '
        'фактический ChernovResult, конфигурация, параметры, время, E∞/относительная '
        'ошибка, флаг порога и статусы; attempts/ — все запросы и ответы; '
        'workers/ — логи/RSS; sessions/ — время и история продолжений; '
        'statistics.json — распределения; verification.json — контроль источников '
        'и результатов. К каждой записи привязан SHA-256 плана с хешами конкретного '
        'production-набора. Старый набор на 20 задач имеет другую конфигурацию.','',
        'Порог проверен на 33 точках, не на всей непрерывной области. success '
        'означает конечное вычисление, не сертифицированную полную ошибку. '
        'tail_eps не является универсальной абсолютной оценкой. Параметры не '
        'подбирались отдельно по exact, не выполнялись дополнительные уточнения. '
        'Streaming, старый решатель, другие методы и обучение ML не запускались.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()
