"""Read saved comparison attempts and produce tables; never call a solver."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir',type=Path)
    args=parser.parse_args()
    root=args.results_dir
    if (root/'REPORT.md').exists():
        raise SystemExit('Refusing to overwrite report')
    summaries={p.parent.name:json.loads(p.read_text()) for p in sorted(root.glob('*/summary.json'))}
    all_attempts=[]
    for phase,s in summaries.items():
        for index,run in enumerate(s['runs']):
            all_attempts.append(dict(phase=phase,index=index,problem_id=run['spec']['problem_id'],
                method=run['spec']['method'],label=run['spec'].get('label',''),
                status=run['status'],max_abs_error=run['max_abs_error'],
                runtime_seconds=run['runtime_seconds'],worker_wall_seconds=run['worker_wall_seconds'],
                observed_peak_rss_mib=run['observed_peak_rss_mib'],
                config=json.dumps(run['spec']['config'],sort_keys=True),
                result_file=f"{phase}/{run['result_file']}" if run['result_file'] else ''))
    infrastructure=root/'pilot/infrastructure_failure.json'
    if infrastructure.exists():
        request=json.loads((root/'pilot/000_R1_remizov.request.json').read_text())
        all_attempts.insert(0,dict(phase='pilot',index=0,problem_id=request['problem_id'],
            method=request['method'],label='infrastructure_failure',status='supervisor_exception',
            max_abs_error=None,runtime_seconds=None,worker_wall_seconds=None,
            observed_peak_rss_mib=None,config=json.dumps(request['config'],sort_keys=True),result_file=''))
    with (root/'all_attempts.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_attempts[0])); writer.writeheader();writer.writerows(all_attempts)
    final=summaries['final']; records=[]
    for run in final['runs']:
        assert run['status']=='success'
        record=json.loads((root/'final'/run['result_file']).read_text())
        assert record['max_abs_error']<=.01
        records.append((run,record))
    assert len(records)==36
    tables=[]
    pids=['R1','AMV1_F0005_S00','AMV1_F0006_S00','AMV1_F0007_S00']
    for pid in pids:
        for method in ['remizov','solve','streaming']:
            group=[(r,d) for r,d in records if r['spec']['problem_id']==pid and r['spec']['method']==method]
            assert len(group)==3
            r,d=group[0]; result=d['result']; times=[g[1]['result']['runtime_seconds'] for g in group]
            assert all(g[1]['result']['f']==result['f'] for g in group)
            tables.append(dict(problem_id=pid,method=method,
                max_abs_error=d['max_abs_error'],relative_error=d['relative_error'],
                median_solver_seconds=statistics.median(times),min_solver_seconds=min(times),max_solver_seconds=max(times),
                median_load_seconds=statistics.median(g[1]['load_seconds'] for g in group),
                median_worker_seconds=statistics.median(g[0]['worker_wall_seconds'] for g in group),
                peak_rss_mib=max(g[1]['process_peak_rss_mib'] for g in group),
                n=result['n'],n_steps=result.get('n_steps',0),
                time_nodes=result.get('time_nodes',result.get('final_nt')),
                dx=result.get('actual_dx',result.get('final_dx')),
                n_operator_applications=result['n_operator_applications'],
                grid_points=result.get('work_grid_points',result.get('spatial_grid_points_peak'))))
    with (root/'comparison.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(tables[0]));writer.writeheader();writer.writerows(tables)
    (root/'comparison.json').write_text(json.dumps(tables,indent=2)+'\n')
    controls=[json.loads(p.read_text()) for p in sorted((root/'controls').glob('*.result.json'))]
    control_rows=[]
    for pid in pids:
        for method in ['solve','streaming']:
            selected=[r for r in controls if r['spec']['problem_id']==pid and r['spec']['method']==method]
            base=next(r for r in selected if r['spec']['label']=='base')
            for r in selected:
                if r['spec']['label']=='base':continue
                delta=float(np.max(np.abs(np.array(r['result']['f'])-base['result']['f'])))
                if r['spec']['label'] in ('dx_half','padding_double') or method=='solve':
                    assert delta<.001
                control_rows.append(dict(problem_id=pid,method=method,control=r['spec']['label'],
                                         change=delta,max_abs_error=r['max_abs_error']))
            if method=='streaming':
                tail=next(json.loads(p.read_text()) for p in (root/'tail_fixed_tau').glob('*.result.json')
                    if json.loads(p.read_text())['spec']['problem_id']==pid)
                delta=float(np.max(np.abs(np.array(tail['result']['f'])-base['result']['f'])))
                assert delta<.001
                control_rows.append(dict(problem_id=pid,method=method,control='tail_fixed_tau',
                                         change=delta,max_abs_error=tail['max_abs_error']))
    (root/'controls.json').write_text(json.dumps(control_rows,indent=2)+'\n')
    # Unchanged mathematical core and persisted dataset are checked against the
    # snapshot made before any numerical experiment.
    project=Path(__file__).resolve().parents[1]
    original=json.loads((project/'comparison_inputs/provenance.json').read_text())['before_sha256']
    preserved={p:hashlib.sha256((project/p).read_bytes()).hexdigest()==h for p,h in original.items()}
    assert all(preserved.values())
    assert all(not s.get('changed_or_removed') for name,s in summaries.items() if name!='pilot')
    verification={'preserved_original_inputs_and_core':preserved,
        'recorded_attempts':len(all_attempts),'completed_attempts':len(all_attempts)-1,'final_successes':36,
        'all_final_errors_at_most':.01,'repeated_f_identical':True,
        'infrastructure_failure_attempts':1,
        'full_dataset_or_ml_run':False}
    (root/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    lines=['# Адаптация Chern: сравнение трёх схем', '',
        'Дата: 2026-10-18. Исходный remizov_solver.py и данные сохранены побайтово.', '',
        '## Постановка и критерий', '',
        'Проверяется возможность получить E∞≤0.01 существенно дешевле исходного решателя. '
        'R1: 257 точек [-2,2]; F0005/F0006/F0007: одинаковые 33 точки [-1,1]. '
        'Точное решение R1 — exp(-x²), остальных — equation.y_exact из AST. '
        'Это эмпирическая ошибка на фиксированной сетке, не сертификат всей прямой.', '',
        'F0005: L2 constant; F0006: L2 slow_varying; F0007: L2 moderate_varying. '
        'Наблюдения не обобщаются на все режимы или 1000 уравнений.', '',
        '## Дизайн и подбор', '',
        'Планы всех фаз сохранены до запуска. pilot_v2: 12 попыток с n=8 и K=512; '
        'accuracy: 12 попыток, n=32/64 или K=2048/8192; baseline_accuracy: 3 старых расчёта. '
        'controls: 40 запусков с изменением одного параметра; tail_fixed_tau: 4 дополнительных контроля. '
        'final: 36 попыток (3 повторения × 4 задачи × 3 метода). '
        'Все попытки перечислены в all_attempts.csv; подбор не включён в итоговые времена.', '',
        'Первая инфраструктурная попытка pilot остановилась из-за запрета ps в sandbox. '
        'Результат решателя не получен; это не его штатный отказ. Все её файлы сохранены. '
        'Наблюдатель заменён собственным resource.getrusage в worker; случай не скрыт.', '',
        'Для исходного решателя выбран более мягкий бюджет, чем verification: '
        'rtol квадратуры/пространства=1e-3, atol=1e-6, dx_initial=0.05, '
        'nt_initial=8, пределы уточнений 7/4, max_grid=200000. '
        'Это помогает исходному методу относительно строгих библиотечных проверок; '
        'полная фактическая ошибка проверяется независимо по exact.', '',
        'Новый solve: cubic+taper, dx=0.025, padding=8, quadratic, 257 узлов. '
        'Streaming: те же пространство и продолжение, равномерная сетка K+1. '
        'Везде tail_eps=1e-8, T из одинаковой формулы и глобального C. '
        'n: R1=8, F0005=32, F0006=64, F0007=32. '
        'K: R1=512, F0005=8192, F0006=4096, F0007=2048. '
        'Полные конфигурации находятся в каждом result.json.', '',
        '## Измерения', '',
        'Отдельные последовательные процессы, n_jobs=1, BLAS-потоки=1. '
        'Время — медиана трёх повторов; включает подготовку численных массивов '
        'и проекцию ответа, исключает загрузку датасета и старт Python. '
        'Они учитываются отдельно в comparison.csv и result.json. '
        'Монитор RSS одинаков для всех методов; профилирование не включалось. '
        'Повторения дали идентичные массивы f. Три замера не являются статистической '
        'гарантией времени на другой машине.', '',
        '| Задача | Старый: E∞ / с | solve: E∞ / с | streaming: E∞ / с | Ускорение solve / streaming |',
        '|---|---:|---:|---:|---:|']
    for pid in pids:
        a,b,c=[r for r in tables if r['problem_id']==pid]
        lines.append(f"| {pid} | {a['max_abs_error']:.6g} / {a['median_solver_seconds']:.4f} | {b['max_abs_error']:.6g} / {b['median_solver_seconds']:.4f} | {c['max_abs_error']:.6g} / {c['median_solver_seconds']:.4f} | {a['median_solver_seconds']/b['median_solver_seconds']:.2f}× / {a['median_solver_seconds']/c['median_solver_seconds']:.2f}× |")
    lines+=['','Все 36 итоговых попыток успешны и удовлетворяют общему порогу E∞≤0.01.', '',
        '| Задача | Метод | Время min…max, с | wall медиана, с | RSS max, МиБ | S |',
        '|---|---|---:|---:|---:|---:|']
    for r in tables:
        lines.append(f"| {r['problem_id']} | {r['method']} | {r['min_solver_seconds']:.4f}…{r['max_solver_seconds']:.4f} | {r['median_worker_seconds']:.4f} | {r['peak_rss_mib']:.2f} | {r['n_operator_applications']} |")
    lines+=['','## Контроль численных параметров', '',
        '| Задача | Метод | Изменение | max abs(f_control−f_base) | E∞ control |',
        '|---|---|---|---:|---:|']
    for r in control_rows:
        lines.append(f"| {r['problem_id']} | {r['method']} | {r['control']} | {r['change']:.6g} | {r['max_abs_error']:.6g} |")
    lines+=['',
        'Уменьшение dx, увеличение окна и уточнение квадратуры фиксированного n '
        'изменили ответ меньше чем на 1e-3. Это индикаторы, не строгие границы ошибки.', '',
        'В streaming изменение tail_eps при фиксированном K одновременно меняет τ=T/K; '
        'в частности на F0007 это заметно изменило ответ. Такие строки сохранены как '
        'неизолированные относительно шага времени. Дополнительный tail_fixed_tau '
        'увеличивает K пропорционально T (с округлением до чётного), сохраняя τ '
        'приближённо: изменения во всех задачах меньше 1e-3. '
        'Удвоение K проверяет другую временную схему и уменьшает её ошибку; '
        'это не отдельная оценка ошибки квадратуры.', '',
        '## Вывод', '',
        'На этих четырёх задачах альтернативные схемы сократили время при общем '
        'фактическом пороге точности. Выигрыш неодинаков: streaming особенно '
        'выгоден на F0006/F0007, на F0005 сопоставим с новым solve. '
        'Фиксированный solve удобнее для сравнения прежней конечной аппроксимации. '
        'Это совокупный эффект сеток, кэширования и бюджетов; вклад отдельных '
        'изменений не измерялся. Порядок сходимости не заявляется.', '',
        'Для дальнейшей работы перспективны обе схемы с отдельным контролем окна, '
        'пространства и времени. Streaming нельзя считать эквивалентной заменой '
        'фиксированного n. Массовые 1000 уравнений и ML не запускались.', '',
        '## Воспроизводимость', '',
        'Окружение: final/summary.json. Полные результаты, exact, конфигурации, '
        'ресурсные ряды и логи — в подпапках фаз. Хеши до/после — в summary.json, '
        'проверка исходного ядра/датасета/предоставленных файлов — verification.json. '
        'Численный драйвер compare_chernov.py; этот отчётный скрипт решатели не вызывает.', '',
        'Проверки кода и CLI сохранены отдельно: pytest.json, pytest.txt, cli_check.json. '
        'В comparison.csv поле time_nodes у исходного решателя означает final_nt; '
        'полная история всех квадратур находится в result.json. У новых схем это '
        'число узлов единственной квадратуры. Счётчики S включают всю работу.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(tables,indent=2))


if __name__=='__main__':
    main()
