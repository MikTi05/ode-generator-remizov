"""Classify saved results of the frozen dataset20 survey; no solver calls."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def accuracy_status(computation_status: str, error: float | None, threshold: float) -> str:
    if computation_status != 'success':
        return computation_status
    if error is None or not math.isfinite(error):
        return 'invalid_error_metric'
    return 'accuracy_pass' if error <= threshold else 'accuracy_not_met'


def time_stats(values: list[float]) -> dict:
    if not values:
        return dict(count=0,total=0.,mean=None,median=None,min=None,max=None,p95=None)
    return dict(count=len(values),total=sum(values),mean=statistics.mean(values),
                median=statistics.median(values),min=min(values),max=max(values),
                p95=float(np.percentile(values,95)))


def summarize(rows: list[dict]) -> dict:
    result = {}
    for method in ('solve','streaming'):
        group = [r for r in rows if r['method']==method]
        result[method] = dict(attempts=len(group),
            numerical_successes=sum(r['computation_status']=='success' for r in group),
            accuracy_passes=sum(r['status']=='accuracy_pass' for r in group),
            accuracy_fraction=sum(r['status']=='accuracy_pass' for r in group)/len(group) if group else 0.,
            outcomes=dict(Counter(r['status'] for r in group)),
            completed_solver_seconds=time_stats([r['solver_seconds'] for r in group
                if r['solver_seconds'] is not None]),
            all_attempt_wall_seconds=time_stats([r['worker_seconds'] for r in group]),
            observed_peak_rss_mib=max((r['peak_rss_mib'] for r in group),default=0.))
    return result


def load_rows(root: Path) -> tuple[dict,dict,list[dict]]:
    frozen = json.loads((root/'plan.json').read_text())
    execution = json.loads((root/'runs/summary.json').read_text())
    if 'finished_utc' not in execution or len(execution['runs']) != 40:
        raise ValueError('all 40 attempts must have been recorded before reporting')
    if frozen != execution['plan']:
        raise ValueError('executed plan differs from the frozen plan')
    preflight = {r['problem_id']:r for r in json.loads((root/'preflight.json').read_text())}
    seen = set(); rows=[]
    for spec,run in zip(frozen['attempts'],execution['runs']):
        if run['spec'] != spec:
            raise ValueError('attempt order/config differs from the frozen plan')
        key=(spec['problem_id'],spec['method'])
        if key in seen:
            raise ValueError('duplicate equation/method')
        seen.add(key)
        r=dict(problem_id=key[0],method=key[1],complexity=preflight[key[0]]['complexity'],
            regime=preflight[key[0]]['regime'],computation_status=run['status'],
            max_abs_error=run['max_abs_error'],relative_error=run.get('relative_error'),
            solver_seconds=run['runtime_seconds'],worker_seconds=run['worker_wall_seconds'],
            peak_rss_mib=run['observed_peak_rss_mib'],T=preflight[key[0]]['T'],
            tau=preflight[key[0]]['streaming_tau'] if key[1]=='streaming' else None,
            n_operator_applications=None,n_spline_builds=None,load_seconds=None,
            outside_query_values=None,boundary_state_max_abs=None,
            result_file=run['result_file'],failure_reason=None)
        if run['status']=='success':
            raw=json.loads((root/'runs'/run['result_file']).read_text())
            numerical=raw['result']
            if raw['config']!=spec['config']:
                raise ValueError('effective config differs from requested full config')
            expected_x=np.linspace(-1,1,33)
            np.testing.assert_array_equal(numerical['x'],expected_x)
            f=np.asarray(numerical['f']);exact=np.asarray(raw['y_exact'])
            if f.shape != (33,) or exact.shape != (33,) or not np.all(np.isfinite(f)):
                raise ValueError('invalid successful solution shape/values')
            error=float(np.max(np.abs(f-exact)))
            if error != raw['max_abs_error'] or error != run['max_abs_error']:
                raise ValueError('saved error is inconsistent')
            for field in ('n_operator_applications','n_spline_builds',
                          'outside_query_values','boundary_state_max_abs'):
                r[field]=numerical[field]
            r['load_seconds']=raw['load_seconds']
        elif run['result_file'] and run['status'] not in (
            'external_time_limit','external_memory_limit','worker_exception'):
            raw=json.loads((root/'runs'/run['result_file']).read_text())
            if raw['result']['f']:
                raise ValueError('solver failure must not expose final values')
            r['failure_reason']=raw['result']['failure_reason']
        r['status']=accuracy_status(r['computation_status'],r['max_abs_error'],frozen['accuracy_threshold'])
        rows.append(r)
    if len(seen)!=40:
        raise ValueError('incomplete coverage')
    return frozen,execution,rows


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir',type=Path)
    args=parser.parse_args();root=args.results_dir
    if (root/'REPORT.md').exists():
        raise SystemExit('Refusing to overwrite report')
    plan,execution,rows=load_rows(root)
    stats=summarize(rows)
    before=json.loads((root/'before_sha256.json').read_text())
    changed=[name for name,h in before.items()
             if not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h]
    verification=dict(complete_coverage=True,attempts=40,uniform_configs=len({
        json.dumps(s['config'],sort_keys=True) for s in plan['attempts']})==1,
        no_retries=True,threshold=.01,errors_recomputed_from_saved_values=True,
        previous_files_checked=len(before),changed_previous_files=changed,
        executor_changed_files=execution.get('changed_or_removed',[]))
    if changed or execution.get('changed_or_removed'):
        raise ValueError(f'source/data/previous result integrity changed: {changed}')
    with (root/'attempts.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name,value in (('attempts.json',rows),('statistics.json',stats),('verification.json',verification)):
        (root/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    def cell(r):
        if r['max_abs_error'] is None:
            return f"— / — / {r['status']}"
        mark='✓' if r['status']=='accuracy_pass' else '✗'
        return f"{r['max_abs_error']:.6g} / {r['solver_seconds']:.3f} / {mark}"
    lines=['# Chern на всех 20 уравнениях: единые параметры','',
        'Дата: 2026-10-18. Цель — оценить покрытие E∞≤0.01 и стоимость '
        'без индивидуального подбора по точному решению. Ровно 40 попыток: '
        '20 уравнений × 2 схемы, без повторов и подстроек.','',
        '## Заранее зафиксированный протокол','',
        'Полная конфигурация и обоснование выбора сохранены до запуска в plan.json. '
        'solve: n=128, 513 квадратичных временных узлов. streaming: K=32768, '
        'равномерные узлы; n и num_t в этой схеме не определяют расчёт. '
        'Общие: dx=0.0125, padding=12 (окно [-13,13], 2081 точка), cubic+taper, '
        'tail_eps=1e-8, глобальные сертификаты из загрузчика. '
        'T вычисляется одинаковым правилом из λ−C; точное решение в выборе '
        'параметров и критерии остановки расчёта не участвует.','',
        'Целевая сетка одинакова: linspace(-1,1,33). '
        'E∞=max abs(f−y_exact), относительная ошибка также сохранена. '
        'Успех численного вызова и достижение порога — разные статусы. '
        'Времена — одиночные замеры, не медианы повторов; включают численную '
        'подготовку и проекцию, исключают загрузку датасета и старт Python. '
        'Полное wall time процесса записано отдельно.','',
        'Ресурсы: один worker за раз, n_jobs=1, BLAS=1, 60 с и 512 МиБ на '
        'попытку. Пик RSS self-reported через getrusage; watchdog опрашивает '
        'его каждые 0.1 с, SIGTERM→SIGKILL после 2 с. Это пределы серии, '
        'не критерии математической применимости. После отказа серия продолжается. '
        'Оценка стоимости до запуска: 65536 применений S для solve, 32768 '
        'для streaming на одной задаче; ожидаемые времена 10–25 и 5–15 с.','',
        '## Все уравнения','',
        'В ячейках: E∞ / секунды решателя / достижение порога. '
        'Для внешней остановки численный ответ и ошибка неизвестны.','',
        '| ID | Сложность / режим | solve | streaming |',
        '|---|---|---:|---:|']
    ids=[s['problem_id'] for s in plan['attempts'][::2]]
    for pid in ids:
        a,b=[r for r in rows if r['problem_id']==pid]
        lines.append(f"| {pid} | {a['complexity']} / {a['regime']} | {cell(a)} | {cell(b)} |")
    lines+=['','## Покрытие и время','',
        '| Схема | Конечных результатов | E∞≤0.01 | Среднее / медиана, с | min…max / p95, с | Сумма solve time, с | Сумма wall time, с | max RSS, МиБ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,s in stats.items():
        t=s['completed_solver_seconds'];w=s['all_attempt_wall_seconds']
        fmt=lambda v:'—' if v is None else f'{v:.3f}'
        lines.append(f"| {method} | {s['numerical_successes']}/20 | {s['accuracy_passes']}/20 ({100*s['accuracy_fraction']:.0f}%) | {fmt(t['mean'])} / {fmt(t['median'])} | {fmt(t['min'])}…{fmt(t['max'])} / {fmt(t['p95'])} | {t['total']:.3f} | {w['total']:.3f} | {s['observed_peak_rss_mib']:.2f} |")
    lines+=['','Время решателя для прерванных worker неизвестно и не включается '
        'в его статистику; wall time учитывает все попытки. При малой выборке '
        'p95 — описательная статистика, не гарантия верхней границы.','',
        '## Причины недостижения цели','']
    for method,s in stats.items():
        lines.append(f"- {method}: "+', '.join(f'{k}={v}' for k,v in s['outcomes'].items())+'.')
    lines+=['','accuracy_not_met означает конечный ответ с фактически слишком '
        'большой ошибкой. Эти ошибки нельзя автоматически приписать n, K, '
        'интерполяции, квадратуре, окну или хвосту: их разделение потребовало бы '
        'отдельных уточнений, которые в этот ограниченный обзор не входят. '
        'external_* — статусы оболочки, не failure_reason решателя.','',
        '## Готовность к 1000 уравнений','']
    if all(s['accuracy_passes']==20 for s in stats.values()):
        lines.append('На данном малом наборе обе схемы прошли общий порог. '
            'Это основание для ограниченного следующего этапа, но не подтверждение '
            'точности на 1000 задачах: отсутствует автоматический контроль полной '
            'ошибки, а проверено только 20 структурных семейств и 33 целевые точки.')
    else:
        lines.append('Единый выбранный бюджет не обеспечивает требуемую точность '
            'на всём малом наборе. Новые реализации пока не готовы к массовому '
            'получению гарантированно точных решений на 1000 уравнениях. '
            'Инфраструктурная возможность завершать/учитывать попытки отдельно '
            'от точности видна из таблицы; success=True нельзя использовать '
            'как метку достижения E∞≤0.01.')
    lines+=['','Ограничения: это один фиксированный протокол, не поиск оптимальных '
        'параметров и не доказательство невозможности достичь точности. '
        'Прошлая серия использовала отдельный подбор на нескольких задачах; '
        'её 100% успех нельзя переносить на этот обзор. Фиксированное n и '
        'streaming — разные схемы. Массовый набор 1000 и ML не запускались.','',
        '## Артефакты и сохранность','',
        'plan.json, preflight.json — до вычислений; runs/ — сырые результаты, '
        'exact, полные конфигурации, окружение, логи, RSS и статусы всех попыток; '
        'attempts.csv/json и statistics.json — таблицы; verification.json — '
        'проверка покрытия и неизменности прежнего кода, данных и результатов. '
        'Отчётный скрипт не запускает решатель.','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(stats,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
