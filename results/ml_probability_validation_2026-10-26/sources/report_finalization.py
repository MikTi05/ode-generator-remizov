"""Finalize saved validation96 artifacts; no solver, fitting or dataset-test access."""
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path('/Users/mikti/Documents/vsc projects/ode-generator-remizov')
sys.path.insert(0,str(ROOT))
from diagnostics.mc_pilot_p2 import read_json,write_json,digest,snapshot
from diagnostics.chernov_batch_control import lock_is_held
OUT=ROOT/'results/ml_probability_validation_2026-10-26'
SOURCE=ROOT/'results/ml_probability_validation_preflight_2026-10-26'
MODEL=ROOT/'results/ml_probability_selector_2026-10-26'
a=read_json(OUT/'analysis.json');cases=read_json(OUT/'cases.json');plan=read_json(OUT/'plan.json')
budget=read_json(OUT/'budget.json');pre=read_json(OUT/'preflight.json')
assert a['status']=='complete' and a['records']==a['requests']==96 and a['pending']==0

def cmd(args):
    r=subprocess.run(args,capture_output=True,text=True,check=False)
    return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)

before=read_json(OUT/'hardware_power_before.json')
after=dict(captured_utc=dt.datetime.now(dt.timezone.utc).isoformat(),after_last_call=True,
           settings=cmd(['pmset','-g','custom']),source=cmd(['pmset','-g','batt']))
after['settings_unchanged']=before['settings']==after['settings']
after['power_source_unchanged']=before['source']['stdout'].splitlines()[0]==after['source']['stdout'].splitlines()[0]
write_json(OUT/'hardware_power_after.json',after,exclusive=True)

records=[read_json(p) for p in (OUT/'records').glob('*.json')]
sessions=[read_json(p) for p in (OUT/'sessions').glob('*.json')]
rss=[json.loads(s) for p in (OUT/'workers').glob('*.rss.jsonl') for s in p.read_text().splitlines()]
thermals=[json.loads(s) for s in (OUT/'thermal.jsonl').read_text().splitlines()]
guardians=[read_json(p) for p in (OUT/'control').rglob('*.guardian.json')]
active=read_json(OUT/'active.json')
work_R=sum(r['result']['n_operator_applications'] for r in records if r['profile']!='MC1')
work_MC=sum(r['result']['n_path_steps'] for r in records if r['profile']=='MC1')
assert work_R==6291456 and work_MC==10264117248
assert len(sessions)==len({r['worker_pid'] for r in records})==1
assert all(r['status']=='completed' for r in records)
assert active['finished_monotonic']<budget['deadline_monotonic']
finish_epoch=max(dt.datetime.fromisoformat(s['finished_utc']).timestamp() for s in sessions)
assert finish_epoch<budget['deadline_epoch']
assert max(s['peak_rss_mib'] for s in rss)<=512
assert max(s['thermal_state'] for s in thermals)==0
assert all(g['cleanup_success'] and g['finished'] for g in guardians)
assert all(not lock_is_held(ROOT/'results'/n) for n in ('.chernov_batch_controller.lock','.chernov_batch_compute.lock'))
rv=dict(passed=True,physical_calls=96,unique_requests=96,worker_processes=1,thread_environment=read_json(OUT/'environment.json')['thread_environment'],
        numba_threads=1,approved_wall_cap_seconds=5691,controller_wall_seconds=a['controller_wall_sum_seconds'],
        method_wall_seconds=a['method_wall_sum_seconds'],original_deadline_epoch=budget['deadline_epoch'],original_deadline_monotonic=budget['deadline_monotonic'],
        final_controller_monotonic=active['finished_monotonic'],final_epoch=finish_epoch,deadline_reset=False,
        peak_rss_mib=max(s['peak_rss_mib'] for s in rss),rss_samples=len(rss),thermal_samples=len(thermals),maximum_thermal_state=0,
        thermal_measurement='NSProcessInfo.thermalState; nominal=0; Celsius not available',remizov_operator_applications=work_R,MC_path_updates=work_MC,
        guardian_cleanup_success=True,compute_controller_locks_released=True,execution_failures=0,retries=0,pending=0,
        setup=a['resources']['warmups'],power_settings_unchanged=after['settings_unchanged'],power_source_unchanged=after['power_source_unchanged'])
write_json(OUT/'resource_verification.json',rv,exclusive=True)
tests=dict(passed=26,failed=0,duration_seconds=.22,
    command='PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_run_probability_validation.py tests/test_probability_validation_plan.py tests/test_remizov_r0.py tests/test_mc_pilot_p2.py tests/test_probability_validation_report.py',
    scope='unit/mock and saved-only checks, no new numerical experiment, no ML fit, no held-out dataset test',
    stdout='26 passed in 0.22s',recorded_from_completed_tool_execution=True)
write_json(OUT/'tests.json',tests,exclusive=True)
write_json(OUT/'resume_check.json',dict(new_calls=0,worker_started=False,reason='already_complete',
    command='PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/run_probability_validation.py run --out results/ml_probability_validation_2026-10-26',
    recorded_from_completed_tool_execution=True),exclusive=True)

def f(n,k=6):return f'{n:.{k}f}'
b=a['baselines']['R256'];p95=a['policies']['0.95']['point'];p50=a['policies']['0.5']['point']
saving=100*(1-p95['warm_policy_seconds']/b['warm_policy_seconds'])
lines=['# Независимая validation-диагностика замороженного вероятностного селектора', '',
'Дата: 2026-10-26. Статус: **96/96 вызовов завершены**, все исходные ответы и неудачи по точности сохранены.', '',
'**Преимущество над основным baseline constant_R256 не подтверждено.** '
'Точечная политика при каждом заранее заданном q дала 6 точных ответов из 8 запросов, '
'1 неточный ответ и 1 отказ; constant_R256 — 7 точных и 1 неточный без отказов. '
f'При q=0.95 расчётная тёплая стоимость ниже на {saving:.2f}%, но потеряна одна точная доставка. '
'Production API сохраняет research_only/abstain на всех 8 задачах.', '',
'## Контракт, независимость и заморозка', '',
'Исполнен [ранее подготовленный план](../ml_probability_validation_preflight_2026-10-26/plan.json) '
'с явным утверждением бюджета пользователем: **5691 с, не более 96 вызовов, 1 worker/поток, RSS ≤512 MiB**. '
'Исходный approval_request с approved=false оставлен историческим; новое разрешение сохранено в '
'[AUTHORIZATION.txt](AUTHORIZATION.txt) и [исполняемом плане](plan.json). '
'Сценарная оценка 2695.354 с; утверждённый cap = ceil(2 × сценарий + 300) = 5691 с.', '',
'Перед первым вызовом проверены манифест, SHA-256 входов/AST и numerical core, '
'аналитические G/T, конфигурации, лимиты работы, сетка, seeds, отсутствие конфликтов, '
'сохранённые результаты, блокировки, питание и thermal state. '
'Все 8 семейств принадлежат original validation, пересечение с fit/scaler семействами пусто '
'([preflight.json](preflight.json)). Это диагностическая выборка, не случайная окончательная оценка на всём распределении.', '',
f'- Исходный план: `{plan["source_plan_sha256"]}`.',
f'- Registry: `{a["registry_sha256"]}`.',
f'- Прогнозы до outcomes: `{a["frozen_prediction_sha256"]}`.',
'- Все 101 артефакт прежнего ML-пакета и 19 артефактов preflight проверены по прежним манифестам; веса, scalers, профильный контекст, политики и q=0.5/0.8/0.95 неизменны.',
'- Новые MC seed entropy: 248800060100…248800060103, episode 0…3. PCG64/SeedSequence/point_address_v1 включает ID уравнения и координату точки; одинаковая базовая entropy у разных ID не означает одинаковый поток. Старые seeds/результаты не подменялись.',
'- Primary точность и выбор оцениваются по заранее фиксированному episode0; все 4 MC-запуска используются для описания вероятности/разброса, все 4 timing для медианы. Повторы R не считаются независимыми семействами.',
'- Test-уравнения не решались и их outcomes не использовались; benchmark1000 и ML-fit не запускались. Подстройки q, калибратора, признаков или политики по этим результатам нет.', '',
'## Численные параметры и измерение', '',
'Порог строго E_grid = max|f − y_exact| ≤0.01 на 33 узлах [-1,1]. '
'Remizov: ChernovSolver.solve, R128/R256, 513 quadratic time nodes, cubic spline, zero boundary, '
'dx=0.0125, padding=12, рабочая сетка 2081 узел [-13,13]. '
'MC1: M=4096, h=1/128, Numba, chunk=1024, block_steps=64, fastmath=false, tail_eps=1e-4; '
'одинаковый аналитический сертификат forcing G и правило auto-T. Полные asdict(config) сохранены для каждого вызова.', '',
'Wall-clock измерен perf_counter_ns: R включает подготовку сертификата G/auto-T, config, constructor и solve; '
'MC включает AST/G-подготовку, config, solve/auto-T/RNG. Общие импорты, загрузка и JIT вынесены в отдельно сохранённую подготовку; '
'exact, подсчёт E_grid и сериализация выполняются после таймера. Численных/RNG warmup-вызовов нет. '
'Все профили исполнены одним worker в одном окружении; времена MC не заменяют прежние измерения.', '',
'## Точность всех сохранённых вызовов', '',
'| Профиль | Вызовы | E_grid≤0.01 | Нарушения | Точных primary задач |',
'|---|---:|---:|---:|---:|']
for profile in ('R128','R256','MC1'):
    z=a['profile_outcomes'][profile]
    lines.append(f'| {profile} | {z["completed"]}/32 | {z["accurate_calls"]}/32 | {z["finite_accuracy_violations"]}/32 | {z["primary_accurate_equations"]}/8 |')
lines += ['', 'Все 96 возвратов конечны и success=true; это не гарантирует требуемой точности. '
'Всего 40/96 вызовов нарушили порог (8 R128 +4 R256 +28 MC1), execution failures=0. '
'Детерминированные ответы каждого R-профиля побитово совпали во всех четырёх повторах.', '',
'| Задача | E R128 | E R256 | E MC episode0 | MC успехи | Выбор point q=.95 |',
'|---|---:|---:|---:|---:|---|']
frozen={r['equation_id']:r for r in read_json(OUT/'frozen_predictions.json')['rows']}
for c in cases:
    own=c['profiles'];pid=c['equation_id']
    lines.append(f'| {pid.removeprefix("AMV1_")} | {f(own["R128"]["primary_error"],9)} | {f(own["R256"]["primary_error"],9)} | {f(own["MC1"]["primary_error"],9)} | {own["MC1"]["accurate"]}/4 | {frozen[pid]["diagnostic_choices"]["0.95"]["point"]["action"]} |')
lines += ['', '**F0189/R128: 0.01026625331390879 >0.01**, округление не превращает ответ в точный.', '',
'| Задача | R128 median [min,max], с | R256 median [min,max], с | MC1 median [min,max], с |',
'|---|---:|---:|---:|']
for c in cases:
    cells=[]
    for profile in ('R128','R256','MC1'):
        t=c['profiles'][profile]['timing'];cells.append(f'{t["median"]:.3f} [{t["min"]:.3f},{t["max"]:.3f}]')
    lines.append(f'| {c["equation_id"].removeprefix("AMV1_")} | '+ ' | '.join(cells)+' |')
lines += ['', 'Все raw wall times, IQR и MAD сохранены в [cases.json](cases.json); '
'все 96 попыток, ошибки и ссылки на raw JSON — в [attempts.csv](attempts.csv). '
'Offline oracle по наблюдаемому episode0/медианам: R128 на 6 задачах, R256 на F0189, '
'ни один профиль на F0041. MC-winner отсутствует, в том числе с предусмотренным правилом ничьей 10%. '
'Oracle описывает эти реализации и seeds; он не служит новой training-меткой или вероятностной гарантией.', '',
'## Замороженные решения и основной baseline', '',
'| Политика | Точных /8 | Нарушений | Отказов | Coverage | Тёплая стоимость, с |',
'|---|---:|---:|---:|---:|---:|']
rows=[('constant_R256 (основной)',a['baselines']['R256']),('constant_R128 (вторичный)',a['baselines']['R128']),
      ('point q=.5 и .8',a['policies']['0.5']['point']),('point q=.95',p95),
      ('model_spread (все q)',a['policies']['0.95']['model_spread']),('production research guard',a['policies']['production_research_guard'])]
for name,z in rows:
    lines.append(f'| {name} | {z["accurate_primary"]} | {z["finite_accuracy_violations"]} | {z["abstain"]} | {z["coverage"]:.1%} | {z["warm_policy_seconds"]:.3f} |')
lines += ['', 'У point-политик **1 нарушение из 7 выданных ответов (14.29%)**, или 1/8 всех запросов (12.5%); '
'точность условно на выданный ответ 6/7=85.71%, доля точных доставок 6/8=75%. '
'У constant_R256 1/8 нарушений и 7/8=87.5% точных доставок. '
'Все point q исследованы согласно замороженному плану, лучший q по этим outcomes не выбирался.', '',
'Наблюдаемые ошибки: F0041/S02 — point q=.95 выбирает R256 с прогнозом p=0.992167, '
'но E=0.0172385414; q=.5/.8 выбирают также неточный R128 (p=0.920211). '
'F0189/S01 — отказ при всех q, хотя R256 точен (E=0.0051778312), прогноз p_R256=0.189161. '
'При q=.95 R256 на F0032/F0133 увеличивает стоимость без выигрыша в точности относительно точного R128.', '',
f'Point q=.95: {p95["warm_policy_seconds"]:.3f} против {b["warm_policy_seconds"]:.3f} с (−{saving:.2f}%). '
f'Point q=.5/.8: {p50["warm_policy_seconds"]:.3f} с (−{100*(1-p50["warm_policy_seconds"]/b["warm_policy_seconds"]):.2f}%). '
'Это offline сумма медиан выбранных профилей плюс измеренная feature/inference-стоимость всех 8 запросов, включая отказы; '
'она не является фактическим online solver-dispatch и не смешивается со стоимостью 96 исследовательских запусков. '
'Загрузка моделей 0.382659 с отдельно; тёплый feature+inference overhead всех 8 запросов 0.089570 с. '
'Отказ не считается точной доставкой или успешным решением ограничения.', '',
'Парное сравнение только на общих 6 точных задачах даёт экономию 48.377 с (q=.95) или 72.170 с (q=.5/.8). '
'Accuracy-constrained regret определён лишь для этих 6/8 запросов; он не скрывает отказ F0189 и неточный F0041. '
'Model_spread и production guard отказываются на всех задачах: нулевое число нарушений при нулевой доставке '
'не является превосходством над Remizov.', '',
'## Прогнозы вероятности и времени', '',
'| Профиль | Brier модели / train constant | Logloss модели / constant | Median APE времени / constant |',
'|---|---:|---:|---:|']
for profile in ('R128','R256','MC1'):
    z=a['prediction_scores'][profile];m=z['probability']['model'];base=z['probability']['baseline']
    lines.append(f'| {profile} | {m["brier"]:.6f} / {base["brier"]:.6f} | {m["logloss"]:.6f} / {base["logloss"]:.6f} | {z["cost_model"]["median_APE"]:.2%} / {z["constant_cost_baseline"]["median_APE"]:.2%} |')
lines += ['', 'Constant-baselines определены только прежним train: частота успеха для p, геометрическое среднее времени для cost. '
'Для R точность учитывается один раз на уравнение, для MC — все 4 seed-outcomes с равным весом каждого уравнения. '
'R256 probability хуже constant-baseline; ошибки F0041/F0189 показывают необходимость калибровки и корректной неопределённости. '
'Для MC средний прогноз p=0.01724 при наблюдаемой частоте 0.125; улучшение Brier/logloss на восьми семействах '
'не подтверждает калибровку. Median APE MC 14.31%, но отдельные ошибки стоимости достигают ≈43%; '
'на F0133 предсказано 93.337 с против 141.690 с (недооценка 34.13%).', '',
'## Неопределённость Monte Carlo', '',
'Точны только 4/32 MC-вызовов, все на одном семействе F0143/S04. '
'Ошибки там 0.003982393, 0.004098731, 0.003572522, 0.003778285; '
'4/4 — наблюдаемый успех, не гарантия точности нового seed. '
'Индивидуальный двухсторонний 95% Clopper–Pearson интервал при 4/4: [0.397635,1]; '
'при 0/4 на остальных задачах: [0,0.602365]. Это не одновременные интервалы по 8 задачам. '
'Bootstrap-разброс замороженных моделей также не является откалиброванной доверительной границей p.', '',
'На F0143 точный R128 (12.399 с) примерно вдвое быстрее точного MC1 (25.309 с); '
'MC и R256 (24.594 с) находятся в пределах 10% ничьей с предпочтением R. '
'Замороженные политики не выбрали MC ни разу. Следовательно, пакет диагностирует p/time и ошибки выбора, '
'но не проверяет способность селектора выигрывать за счёт MC на новых семействах.', '',
'## Бюджет, ресурсы, сохранение и проверки', '',
f'- Один worker, thread env всех библиотек=1, Numba threads=1; 96 вызовов без повторных запросов, таймаутов, исключений или незавершённых попыток.',
f'- Wall контроллера {a["controller_wall_sum_seconds"]:.3f} с ({a["controller_wall_sum_seconds"]/60:.2f} мин) <5691 с; сумма method wall {a["method_wall_sum_seconds"]:.3f} с. Исходные epoch/monotonic deadlines не сбрасывались, завершение до обоих deadlines подтверждено.',
f'- Пик RSS {rv["peak_rss_mib"]:.3f} MiB ≤512; {rv["rss_samples"]} RSS-сэмплов. Thermal samples {rv["thermal_samples"]}, все state=0 (nominal); это индикатор давления по температуре, не показания °C.',
f'- Работа: {work_R:,} применений S и {work_MC:,} MC path-step updates. Guardian/lease действовали; worker штатно остановлен после серии, cleanup_success=true, compute/controller locks освобождены.',
f'- Настройки питания неизменны: {after["settings_unchanged"]}; источник питания неизменен: {after["power_source_unchanged"]}. Полные снимки питания сохранены до/после.',
'- Все 96 E_grid независимо пересчитаны по каноническим exact; независимо проверены конфигурации, seeds, таймеры, повторы R, все policy-counts и затраты. [independent_verification.json](independent_verification.json), [resource_verification.json](resource_verification.json).',
'- 26 unit/mock/saved-only тестов passed (0 failed). Проверены запреты расширения бюджета/числа workers/RSS, отсутствие повторного вызова после завершения/прерывания и отсутствие сброса deadline; учёт неточного быстрого MC, отказа и overhead, неизвестного oracle, 10% ничьей. Это не запуск held-out dataset test.',
'- Проверка completed-resume: new_calls=0, worker_started=false. [resume_check.json](resume_check.json), [tests.json](tests.json).',
'- Все 18328 существовавших перед init файлов проверены; изменён только PROJECT_STATUS.md с сохранением before/running/after. Входы, ядра, прежние результаты и frozen manifests сохранены. Финальная проверка и манифест SHA-256: [verification.json](verification.json), [artifact_sha256.json](artifact_sha256.json).', '',
'Новый код: diagnostics/run_probability_validation.py, report_probability_validation.py, verify_probability_validation.py; '
'тесты: tests/test_run_probability_validation.py, test_probability_validation_report.py. '
'Исходники скопированы в sources/, численные алгоритмы и файлы ML-моделей не изменены.', '',
'## Что необходимо дальше', '',
'Текущий селектор не готов к заявлению о выборе самого быстрого точного метода. '
'Цель Remizov/MC сохраняется; превращать систему в классификатор Remizov/отказ по этому результату оснований нет.', '',
'1. Для следующей версии выделить отдельные calibration-семейства, особенно около порога R128 и ошибки R256; '
'проверить калибровку p и покрытие неопределённости. Эти 8 уже раскрытых validation-семейств нельзя повторно называть независимой проверкой новой версии.',
'2. Для MC нужны дополнительные независимые семейства с реальными success/failure и повторными seeds, '
'включая режимы, где наблюдается потенциальное преимущество по времени. Старый gate 20+5 MC-winner не является условием обучения вероятностных голов; '
'одного нового MC-success семейства и отсутствия MC-выборов недостаточно для доказательства двухметодного выбора. Не создавать искусственные победы и не экстраполировать 4/4 в p≥0.95.',
'3. Проверить улучшение cost-head на известных до решения G/T/N и более широких train/calibration данных; '
'показать затраты вместе с точностью и отказами. Выбрать допустимый риск и политику до следующей оценки, не подбирать q на текущих 8 outcomes.',
'4. После новой заморозки выполнить заранее спланированную оценку на нетронутых семействах с достаточным числом MC-выборов и заявленной статистической мощностью. '
'Прежде проверить историческую нераскрытость выбранного test-пакета; текущая диагностика восьми семейств не заменяет такой тест. '
'Новые вычисления, калибровка и переобучение в этой задаче не выполнялись.', '']
with (OUT/'REPORT.md').open('x') as h:h.write('\n'.join(lines))

# Keep the pre-approval history verbatim, replacing only this task's transient header.
status=ROOT/'PROJECT_STATUS.md';old=status.read_text()
with (OUT/'PROJECT_STATUS.running.md').open('x') as h:h.write(old)
marker='## Исторический статус до утверждения validation-бюджета\n\n'
assert marker in old
history=old[old.index(marker):]
header='''# Состояние проекта ode-generator-remizov

Обновлено **2026-10-26**: независимая validation-диагностика вероятностного
ML-селектора **завершена — 96/96 вызовов**: 8 исходных validation-семейств,
R128/R256/MC1 ×4. Утверждённый бюджет 5691 с, 96 вызовов максимум,
1 worker/поток, RSS≤512 MiB соблюдён. Фактический wall **2732.966 с (45.55 мин)**,
пик RSS **246.891 MiB**; 10668 RSS-сэмплов, 264 thermal-сэмпла, все state=0.
Guardian/lease, исходные deadlines и безопасное сохранение действовали;
0 retries, execution failures и pending. Completed-resume вернул 0 новых вызовов.

[Отчёт](results/ml_probability_validation_2026-10-26/REPORT.md),
[результаты и метрики](results/ml_probability_validation_2026-10-26/analysis.json),
[все 96 попыток](results/ml_probability_validation_2026-10-26/attempts.csv),
[независимая проверка](results/ml_probability_validation_2026-10-26/independent_verification.json),
[лимиты](results/ml_probability_validation_2026-10-26/resource_verification.json),
[итоговая сохранность](results/ml_probability_validation_2026-10-26/verification.json).

**Основной constant_R256 не превзойдён:** point-политики всех заранее заданных
q=.5/.8/.95 дали **6 точных доставок/8**, **1 неточный ответ/7 выбранных (14.29%)**
и 1 отказ. Constant_R256: **7 точных/8**, 1 неточный, без отказов.
При q=.95 тёплая offline стоимость 119.924 с против 195.127 с (**−38.54%**),
но потеряна одна точная доставка. q=.5/.8: 84.440 с при тех же исходах;
лучший q по outcomes не выбирался. Стоимость включает inference для всех
запросов, включая отказ; загрузка моделей отдельно. Это offline выбор по
замороженным прогнозам и сохранённым медианам, не online dispatch.
Model-spread политика и production research_only guard отказываются на всех
8 задачах: нулевая доставка не подтверждает преимущество по точности.

Подтверждённые ошибки: **F0041/S02** выбран неточный R256 при q=.95
(E=0.01723854, прогноз p=.99217; R128 также неточен), **F0189/S01** отказ
при всех q, хотя R256 точен (E=.00517783, прогноз p=.18916).
F0189/R128 E=.01026625 строго выше .01. R128 точен на 6/8 семействах,
R256 на 7/8; четыре детерминированных повтора не считаются независимыми данными.

**MC1 точен в 4/32 вызовов**, все на F0143/S04; остальные 28 нарушают порог.
4/4 — наблюдение: индивидуальный CP95=[.397635,1], не гарантия точности
нового seed. На F0143 точный R128 12.399 с быстрее MC1 25.309 с;
MC-winner на этих 8 задачах отсутствует, MC ни одной замороженной политикой
не выбран. Поэтому способность получать преимущество именно за счёт MC
остаётся непроверенной. R256 p-head хуже train-constant по Brier/logloss;
median APE времени R128/R256/MC1 =1.37%/2.08%/14.31% (только диагностика 8 семей).

Модели, scalers, q, профильный контекст и прогнозы сохранены по SHA-256;
fit/scaler family overlap пуст. **Нет ML-fit, benchmark1000 или dataset-test**,
численные схемы не менялись. Все 96 ошибок и стоимости политик независимо
пересчитаны; **26 unit/mock/saved-only тестов passed**. Прежние 18328 файлов
проверены, только этот статус изменён с резервными копиями.
Новый код: diagnostics/run_probability_validation.py,
report_probability_validation.py, verify_probability_validation.py;
tests/test_run_probability_validation.py, test_probability_validation_report.py.
Копии исходников, конфигурации, сырые результаты, питание, ресурсы и проверки
сохранены в results/ml_probability_validation_2026-10-26/.

**Следующий шаг:** для новой версии заранее выделить отдельные calibration-
семейства для p_R256/пограничного R128 и неопределённости, добавить реальные
MC-success/failure семейства с независимыми seeds и потенциальной выгодой по
времени. Старый gate20+5 MC-winner не является условием fit вероятностных голов;
эти 8 раскрытых семейств нельзя считать независимыми для следующей версии.
После калибровки и новой заморозки — заранее спланированный нетронутый семейный
тест с достаточным числом MC-выборов. Риск/q не подбирать по текущим outcomes.
Новые расчёты и обучение требуют отдельной следующей задачи.

'''
status.write_text(header+history)
with (OUT/'PROJECT_STATUS.after.md').open('x') as h:h.write(status.read_text())

additional={}
for name in ['diagnostics/run_probability_validation.py','diagnostics/report_probability_validation.py',
             'diagnostics/verify_probability_validation.py','tests/test_run_probability_validation.py',
             'tests/test_probability_validation_report.py']:
    src=ROOT/name;dest=OUT/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert digest(dest)==digest(src)
    else:shutil.copyfile(src,dest)
    additional[name]=digest(src)
shutil.copyfile(__file__,OUT/'sources'/'report_finalization.py')
write_json(OUT/'additional_sources_sha256.json',additional,exclusive=True)

# Independent final preservation pass after the status/report writes.
oldfiles=read_json(OUT/'protected_before.json');current=snapshot(OUT)
changed=[n for n,h in oldfiles.items() if current.get(n)!=h]
assert changed==['PROJECT_STATUS.md'],changed
for name,h in plan['protected_sha256'].items():assert digest(ROOT/name)==h,name
frozen_counts={}
for directory in (MODEL,SOURCE):
    manifest=read_json(directory/'artifact_sha256.json')
    for name,h in manifest.items():assert digest(directory/name)==h,(directory.name,name)
    frozen_counts[directory.name]=len(manifest)
assert digest(SOURCE/'plan.json')==plan['source_plan_sha256']
assert digest(OUT/'frozen_predictions.json')==digest(SOURCE/'frozen_predictions.json')
assert a['ML_fits']==a['benchmark1000_calls']==0 and not a['test_used']
assert p95['accurate_primary']==6 and b['accurate_primary']==7
verify=dict(passed=True,records=96,requests=96,independent_canonical_error_checks=96,
    original_files_checked=len(oldfiles),original_files_changed=changed,old_results_and_numerical_core_preserved=True,
    frozen_artifacts_checked=frozen_counts,source_plan_and_predictions_unchanged=True,models_scalers_policy_unchanged=True,
    approved_budget_and_resources_passed=True,safe_resume_new_calls=0,unit_mock_saved_only_tests_passed=26,
    ML_fits=0,benchmark1000_calls=0,dataset_test_used=False,main_baseline_superiority_demonstrated=False,
    q95_point=dict(accurate_deliveries=6,requests=8,selected=7,violations=1,abstentions=1),
    constant_R256=dict(accurate_deliveries=7,requests=8,violations=1),MC_accurate_calls=4,MC_total_calls=32,
    MC_reliability_guarantee=False,MC_selected_by_any_frozen_policy=False)
write_json(OUT/'verification.json',verify,exclusive=True)
manifest={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_sha256.json'}
write_json(OUT/'artifact_sha256.json',manifest,exclusive=True)
assert all(digest(OUT/name)==h for name,h in manifest.items())
print(json.dumps(dict(passed=True,artifacts=len(manifest),old_paths_checked=len(oldfiles),old_paths_changed=changed,
    power_settings_unchanged=after['settings_unchanged'],power_source_unchanged=after['power_source_unchanged'],
    report=str(OUT/'REPORT.md')),ensure_ascii=False))
