"""Build the MC33 throughput report strictly from saved artifacts (no solver)."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.mc33_parallel import read_json, publish, require, digest


def build(out):
    audit = read_json(out/'audit.json'); remaining = read_json(out/'remaining.json')
    machine = read_json(out/'machine_before.json')
    protocol = read_json(out/'benchmark_protocol.json'); rows = []; samples = {}
    for workers in protocol['workers']:
        folder = out/f'benchmark_w{workers}'
        require(read_json(folder/'equivalence.json')['passed'], 'equivalence not passed')
        sessions = [read_json(f) for f in (folder/'sessions').glob('*.json')]
        good = [s for s in sessions if s['new_terminal'] == protocol['calls_each'] and not s['stop_reason']]
        require(len(good) == 1, 'need exactly one complete timed session')
        s = good[0]; records = [read_json(p)['record'] for p in (folder/'records').glob('*.json')]
        samples[workers] = {r['job']['key']:r for r in records}
        require(len(records) == protocol['calls_each'], 'incomplete sample')
        rows.append(dict(workers=workers, wall_seconds=s['wall_seconds'],
            compute_seconds=s['compute_seconds'], startup_seconds=s['startup_seconds'],
            method_seconds_sum=sum(r['method_wall_seconds'] for r in records),
            speedup=0., warm_speedup=0.,
            memory_highwater_sum_mib=s['peak_controller_plus_worker_highwater_mib'],
            worker_memory_highwater_sum_mib=s['peak_worker_highwater_sum_mib']))
    for r in rows:
        r['speedup'] = rows[0]['wall_seconds']/r['wall_seconds']
        r['warm_speedup'] = rows[0]['compute_seconds']/r['compute_seconds']
    best = min(rows, key=lambda r:r['wall_seconds'])
    calibration = rows[0]['method_seconds_sum']/protocol['reference_seconds']
    serial = remaining['historical_x0_scaled_serial_seconds']*calibration
    forecast = serial/best['warm_speedup'] + best['startup_seconds']
    # Second estimate separates per-worker slowdown from finite-queue imbalance.
    contention = best['method_seconds_sum']/rows[0]['method_seconds_sum']
    sustained = serial*contention/best['workers'] + best['startup_seconds']
    x0_plan = read_json(ROOT/'results/mc1000_x0_2026-10-25/plan.json')
    jobs = read_json(out/'production/plan.json')['jobs']
    x0 = {}
    for path in (ROOT/'results/mc1000_x0_2026-10-25/attempts').glob('*.json'):
        d = read_json(path); z = d['result']
        x0.setdefault(z['problem_id'], []).append(z['runtime_seconds'])
    # Empirical spatial-proxy check on all 40 historical P1/P2 equations.
    historical = list(csv.DictReader((out/'historical_mc33.csv').open()))
    ratios = []
    for r in historical:
        if r['M'] == '4096' and ('pilot_p2_' in r['group'] or 'sentinel_p1_' in r['group']):
            proxy = statistics.mean(x0[r['equation_id']])*33*4
            ratios.append(float(r['seconds'])/proxy)
    quantiles = statistics.quantiles(ratios, n=10)
    summary = dict(benchmark=rows, selected_workers=best['workers'], selection='minimum measured end-to-end wall among tested counts',
        forecast=dict(calibration_current_vs_historical=calibration,
            serial_seconds=serial, measured_small_queue_seconds=forecast,
            sustained_throughput_seconds=sustained,
            historical_spatial_proxy_ratio_median=statistics.median(ratios),
            historical_spatial_proxy_ratio_p10=quantiles[0], historical_spatial_proxy_ratio_p90=quantiles[8],
            interpretation='scenario, not CI or guaranteed runtime; full run not executed'),
        calls_benchmarked=protocol['calls_each']*len(rows), numerical_equivalence='bitwise arrays; identical non-timing fields/error',
        production_records=len(list((out/'production/records').glob('*.json'))))
    require(summary['production_records'] == 0, 'production unexpectedly started')
    publish(out/'summary.json', summary)
    with (out/'benchmark.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    vm = machine['vm_stat']
    page_size = int(re.search(r'page size of (\d+)', vm)[1])
    pages = {name:int(re.search(r'Pages '+name+r':\s+(\d+)', vm)[1])
             for name in ('free','inactive','speculative')}
    available = page_size*sum(pages.values())/1024**3
    table = '\n'.join(f"| {r['workers']} | {r['wall_seconds']:.2f} | {r['compute_seconds']:.2f} | {r['speedup']:.2f}× | {r['memory_highwater_sum_mib']/1024:.3f} |"
                      for r in rows)
    command = ('PYTHONDONTWRITEBYTECODE=1 caffeinate -i .venv/bin/python -B '
               'diagnostics/mc33_parallel.py run \\\n  --out results/mc33_parallel_2026-11-05/production \\\n  --workers '+str(best['workers'])+' --resume')
    report = f'''# MC33: аудит и параллельный запуск

Дата: 2026-11-05, проект №2488. Полный MC33 **не запускался**. ML после NO-GO
не переобучалась. Датасет, остальные методы, LaTeX и математическое ядро неизменны.

## Объём и исходный протокол

Пользователь выбрал **MC1: M=4096 на узел, четыре полных повтора**. Исходный
контракт — `reports/ml_selector_contract/plan.json`, численный профиль MC1,
финальные seeds **248800030100…248800030103**, episode0 основной. В каждой
реплике `E_grid=max_j abs(f_j-y_exact(x_j))`, 33 узла `x_j=-1+j/16`, порог .01.
Агрегации четырёх ошибок/ответов в основном MC1 target нет. Четыре реплики —
16384 траектории на узел в сумме, но это четыре отдельных MC4096-ответа.

Подтверждено по сырым массивам и SHA:

| Покрытие | Уравнений из 1000 | Смысл |
|---|---:|---|
| MCx0, M1024 × 4 | 1000 | Только x=0; не MC33 |
| MC1024 на 33 узлах | {audit['MC1024_grid33_equations']} | joint_grid100 и P1, другие M/seeds |
| MC1 на 33 узлах, ≥4 исторических seed | {audit['MC1_at_least_four_unique_seeds']} | Численно валидны, разные исторические seed cohorts |
| Любой из этих MC33-профилей | {audit['any_MC33_equations']} | Объединение, не сумма строк |
| MC1 с точными четырьмя финальными seeds | {audit['final_complete_equations']} | Именно задания подготовленного manifest |

Проверены {audit['validated_records']} исторических записей: нет невалидных.
MC1 содержит {audit['MC1_unique_equation_seed_calls']} уникальных пар «уравнение/seed»;
четыре timing-replay F0047 не считаются новыми независимыми репликами. 40 задач
пришли из P1/P2, ещё 8 — из probability validation; другие серии пересекаются.
Наборы новых development/final семейств не смешиваются с основными 1000:
сверяется expression_hash, а не только похожий ID.

У **952** уравнений вовсе нет MC1-grid33 с четырьмя историческими seeds.
Однако исходный финальный seed schedule не совпадает ни с одной старой
записью. Поэтому точный остаток по замороженному финальному manifest:
**{remaining['calls']} вызовов / {remaining['node_seed_estimates']} узловых оценок /
{remaining['paths']:,} траекторий / {remaining['path_updates']:,} обновлений пути**.
Это не пересчёт существующих seed-заданий. Подмена финальных seeds пилотными
означала бы изменение исходного протокола; она не сделана. Старые валидные
файлы сохранены, а benchmark выполняет только отдельно разрешённые replay.

MCx0 (`diagnostics/mc_production_x0.py`) имеет seeds 202610250101…104,
M1024/h1/128, четыре повтора. Его поздняя агрегированная ошибка — ошибка
**среднего четырёх f**, не среднее ошибок. joint_grid100 использует seeds
202610250330…333, один основной M1024-ответ на всех 100 и три дополнительных
на 12 выбранных задачах; это неоднородный исторический протокол.
MC1 P1/P2 (`diagnostics/mc_sentinel_p1.py`, `diagnostics/mc_pilot_p2.py`)
имеют seeds 248800010100…103 и 248800020100…103 соответственно.

## Математика и воспроизводимость

Solver прежний: `mc_optimized.solve_monte_carlo_optimized`, Numba,
Euler–Maruyama, трапеции для потенциала и интеграла, h_requested=1/128,
T=auto_T(λ,C,G,1e-4), chunk=max_batch_paths=1024, block_steps=64,
stream_id=0, fastmath=False. Каждый вызов целиком остаётся последовательным.
Параллелятся независимые equation/seed-задания, не редукции или траектории.

PCG64/SeedSequence/point_address_v1 адресует поток по исходному seed и
SHA256(ID, binary64 координата, stream_id). Нет общего RNG, worker ID,
Python hash или последовательного spawn в очереди. Перестановка задач не
меняет поток; chunk и порядок редукций не изменены. Первая однопроцессная
серия должна пройти сравнение со старыми ответами до остальных серий.

При подготовке заново аналитически вычислены G/T/N для всех 1000 без solver.
Проверены исходные SHA датасета и численных модулей. Полный manifest хранит
точные конфигурации каждого вызова. `max_path_steps=33*M*N` разворачивается
по прежнему правилу `plan_ml_selector_contract.py:job`; это ресурсный счётчик
требуемой работы, не увеличение M или уменьшение h. Для финального профиля
max_runtime_seconds=None, как в исходном контракте; benchmark сохраняет
исходные watchdog/caps каждого пилотного задания.

## Машина и ограничения

Apple M4 Pro, **12 физических = 12 логических ядер (8 performance + 4 efficiency)**,
RAM **24 GiB**. Перед запуском free+inactive+speculative ≈ **{available:.2f} GiB**
(воспроизводимая оценка из vm_stat, не обещание выделения всей памяти).
Swap used 265.69 MiB уже до опыта; thermal/performance warnings отсутствовали,
power mode=0. MC workers до начала — 0. Python 3.10.11, NumPy 2.2.6,
Numba 0.65.1, llvmlite 0.47.0, SciPy 1.15.3; окружение сохранено.

Старые драйверы имеют один compute worker и глобальный exclusive compute lock.
Узкое место — последовательное исполнение MC-заданий; JIT ядра сами serial.
Новый диспетчер сохраняет глобальную блокировку контроллера против других
управляемых серий и выдаёт каждому guardian отдельный compute slot.
BLAS/OpenMP/Numba/VECLIB/NumExpr/BLIS ограничены одним потоком до импорта;
threadpoolctl дополнительно применяет limits=1. В warmup сохранены фактические
numba_threads=1 и threadpool_info (на этой сборке список BLAS/OpenMP пуст).

## Benchmark

16 одинаковых полных заданий, 8 уравнений × 2 исходных P2 seed. Все L1–L4 и
пять coefficient regimes; исходные времена ~10–64 с. ID и все конфигурации
сохранены до запусков в benchmark_protocol и plan каждого опыта. Выбор по
геометрии/стоимости, не по точности. Редкие исторические 100–315-секундные
хвосты этим малым benchmark не представлены; это ограничивает прогноз.

| Workers | Полный wall, с | Очередь после warmup, с | Ускорение wall | RSS, GiB* |
|---:|---:|---:|---:|---:|
{table}

*Консервативная сумма наблюдаемых lifetime RSS maxima контроллера и workers;
это не точный simultaneous peak. Малые guardian-процессы исключены.
Импорт, загрузка и JIT включены в полный wall; отдельно записаны startup и
время очереди. Один опыт на workers, поэтому это измерение, не доверительный
интервал. Параллельные времена не заменяют прежние изолированные cost labels.

Выбрано **{best['workers']} workers**, минимум wall среди проверенных 1/2/4/8.
Все {summary['calls_benchmarked']} возвратов: побитовое совпадение f, stderr,
CI и x со старыми массивами, равенство всех нетайминговых полей, exact и E_grid.
Перестановка завершений не создала численных расхождений.

## Прогноз и resume

Прежний сценарий по 4000 измерениям x0 × 33 узла × масштаб M:
{remaining['historical_x0_scaled_serial_seconds']/3600:.2f} ч последовательной работы.
После калибровки текущим одним worker: **{serial/3600:.2f} ч**.
При {best['workers']} workers прогноз **{forecast/3600:.2f} ч** по измеренному
ускорению малой очереди; модель длительной загрузки без её конечного дисбаланса
даёт **{sustained/3600:.2f} ч**. Это сценарии, не гарантия и не CI: скорость
разных узлов, хвосты стоимости, системная нагрузка, тепловой режим и паузы
могут менять длительность. Историческая проверка x0-прокси на 160 P1/P2
возвратах: median ratio {statistics.median(ratios):.3f}, p10–p90
{quantiles[0]:.3f}–{quantiles[8]:.3f}; это тоже не интервал для времени всей серии.

Старый MCx0/joint_grid100 сохраняет атомарные ответы и возобновляет pending,
но работает последовательно и зависит от внешней owner lease. P2 дополнительно
исключает уже отправленные requests даже без результата (исторический cap140),
поэтому не является универсальным resume для полного MC33.

Новый runner использует immutable plan+SHA, durable request с UUID попытки,
атомарные fsync/no-clobber файлы результата с checksum и identity validation.
Готовый результат — checkpoint. После сбоя между записью attempt-result и
публикацией record он восстанавливается без solver. После прерывания внутри
вызова повторяется только незавершённый equation/seed, с тем же seed и новым
attempt ID; потерянная работа ограничена текущими вызовами workers, внутренние
частичные счётчики не выдумываются. Failures сохраняются и не retry автоматически;
невалидный checkpoint останавливает resume. Контроллер/guardian fencing не
допускает конкурирующего запуска. SIGINT/TERM/HUP или потеря контроллера
останавливает группы workers; lease обновляет сам живой CLI, без участия чата.

Команда после отдельного разрешения пользователя, из корня проекта:

```bash
{command}
```

`caffeinate -i` удерживает Mac от idle sleep только на время команды. После
прерывания применяется та же команда. Сменить workers можно без смены seeds.
Статус без solver:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/mc33_parallel.py status \\\n  --out results/mc33_parallel_2026-11-05/production
```

См. tests/test_mc33_parallel.py, checks.json, lifecycle/ и integrity.json.
Изменения ограничены тремя новыми diagnostics-скриптами, одним тестовым файлом,
каталогом этой серии и PROJECT_STATUS.md. Исторические драйверы не переписаны.
'''
    (out/'REPORT.md').write_text(report)
    (out/'launch_resume.sh').write_text('#!/bin/sh\nset -eu\n'+command+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    build(p.parse_args().out.resolve())
