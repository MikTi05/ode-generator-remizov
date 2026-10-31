from pathlib import Path
import json,hashlib,datetime,re
root=Path('/Users/mikti/Documents/vsc projects/ode-generator-remizov');out=root/'reports/ml_selector_contract/pilot_p2_2026-10-26'
s=json.loads((out/'analysis/summary.json').read_text());g=s['groups'];p=g['P2'];c=g['combined'];r=s['R_comparison'];a=s['groups']['P1'];cov=g['coverage'];stress=g['stress'];fc=next(f for f in s['full1000_cost'] if f['scenario']=='central_coverage_median' and f['repeats_per_equation']==4 and f['R_profile']=='R128');f256=next(f for f in s['full1000_cost'] if f['scenario']=='central_coverage_median' and f['repeats_per_equation']==4 and f['R_profile']=='R256');rng=[f['total_hours'] for f in s['full1000_cost'] if f['repeats_per_equation']==4 and f['R_profile']=='R128']
text=f'''# Проект №2488: P2 завершён, объединённый MC1-пилот

## Проверенный объём

P2: **140/140 новых MC-вызовов, 35 разных train-семейств**, по4 seed;
все завершены, операционных отказов, пропущенных результатов и повторов0.
Только MC1: M4096,h1/128,chunk1024,block64,tail_eps1e-4,общий аналитическийT;
seeds248800020100..103. P1 MC1:20 вызовов на5 других семействах,
seeds248800010100..103. Вместе **40 задач,160 вызовов**.
Старые60 записейP1, все данные и численные методы сохранены без изменений.
R0,benchmark1000,обучениеML не запускались.

До старта сверены исходный условный манифест,40 train-ID/семейств,35 оставшихся
идентичностей expression_hash,параметры/G/T/N,независимые seed,сохранностьP1,
выборMC1 по frozen gates,locks,thermal и доступный диск. Явный новый запрос
пользователя разрешил только выбранные140 строк условного плана. Budget18599с
зафиксирован до старта:ceil(2×9149.4729+300),по измеренномуP1 сценарию2.54ч.
При критическом несоответствии запуск/контроллер останавливается.
[План](plan.json),[preflight](preflight.json),[авторизация](AUTHORIZATION.txt).

## Наблюдаемая точность

E_grid=max|f-y_exact| на фиксированных33 узлах[-1,1],порог.01.
Числа ниже — описательные результаты выбранного train-пилота.

| Группа | Задачи | Вызовы | Попадания ≤.01 | Episode0 | Хотя бы1/4 | Все4/4 |
|---|---:|---:|---:|---:|---:|---:|
'''
for label in ['P1','P2','combined','coverage','stress']:
 v=g[label];text+=f"| {label} | {v['equations']} | {v['calls']} | {v['accurate_calls']}/{v['calls']} | {v['episode0_pass']}/{v['equations']} | {v['any_pass']}/{v['equations']} | {v['all4_pass']}/{v['equations']} |\n"
text+=f'''
P2 медиана E_grid={p['error']['median']:.8g},диапазон
[{p['error']['min']:.8g},{p['error']['max']:.8g}]. Объединённая медиана
{c['error']['median']:.8g}. Все160 массивов и ошибки проверены по сохранённым
ответам и каноническому exact. Для каждой задачи сохранены все4 ошибки и
времени,median/MAD/IQR,номинальный индивидуальный двусторонний CP95 интервал для фиксированного
профиля;поправок на выборMC1 вP1 и множественные сравнения40 нет.
Четыре повтора не доказывают p(success)≥.95:даже4/4 дают нижнюю границу.397635.
5280 адресов PRNG(40×4×33) независимо пересчитаны по SHA256(ID,узел) и seeds,
повторяющихся адресов потоков нет. CI дискретного среднего не является
сертификатом полной ошибки. Mean4 даёт {s['mean4_diagnostic_accurate']}/40
попаданий,но это отдельный диагностический оцениватель с суммой стоимости4
вызовов и одним агрегированным повтором;он не подменяет single-call результаты.

P1 использовался для выбораMC1,поэтому его результаты зависимы от выбора
профиля. Coverage20 содержит по одному train-представителю ячейки
complexity×regime;stress20 намеренно включает трудные случаи. Ни pooled160,
ни coverage20 не дают оценки генеральной точности/надёжности на1000 уравнениях.
[Все160 вызовов](analysis/attempts.csv),[40 задач](analysis/equations.csv),
[mean4 отдельно](analysis/mean4_diagnostic.json).

## Сравнение с сохранённым Remizov

Проверены совпаденияID/expression_hash,exact,33 узлов,T и профиля.
R128/R256:cubic/zero,Q513,dx.0125,padding12,tail1e-4,quadratic Simpson,
соответственноn128/256. Исторический taper не смешивался с этими профилями.

| Профиль | Сохранено | Точных R | Точных MC1 episode0 на тех же задачах | Оба | Только R | Только MC | Ни один |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
for label,v in r.items():
 text+=f"| {label} | {v['available']}/40 | {v['R_accurate']}/{v['available']} | {v['MC_episode0_accurate_same_tasks']}/{v['available']} | {v['both']} | {v['R_only']} | {v['MC_only']} | {v['neither']} |\n"
text+=f'''
Для3 задач нет совместимогоR128,для35 —R256;они отмечены missing_not_run,
а не failure/неточный ответ. Эти38 R0-вызовов не входили в разрешениеP2.
R2565/5 — выбранные sentinel,не результат на40/1000. Исторические R timings
не включают новый полный scope AST/G+config+solve,поэтому fastest/winner
метки и относительное преимущество по полной стоимости не объявляются.
[Построчное сравнение](analysis/remizov_comparison.csv).

## Ресурсы и сохранность

Один compute-worker/поток. P2 сумма warm end-to-end времен
**{p['method_seconds_sum']:.3f}с/{p['method_seconds_sum']/3600:.3f}ч**;
wall контроллера **{s['controller_wall_seconds']:.3f}с/{s['controller_wall_seconds']/3600:.3f}ч**.
Основной таймер perf_counter_ns включает prepare_all_methods_mc+MCConfig+solve;
общая загрузка,прогревJIT,exact/scoring/serialization исключены и сохранены
отдельно. Во всех вызовах import/compile0. Выполнено
**{s['P2_path_updates']:,}** path updates/normal draws.
PeakRSS **{s['peak_rss_mib']:.3f}MiB** при cap512;
thermal-сэмплов{s['thermal_samples']},состояния{s['thermal_states']}.
Максимальный внутригрупповой MAD/median времени(включаяP1)
{s['max_within_equation_wall_MAD_fraction']:.3%}.

Использованы штатные locks,независимый guardian,heartbeat,внешняя owner lease300с,
мониторингRSS/.25с и thermal/10с. Deadline единственный,возобновление его не
продлевает. Каждый request и результат сохраняется эксклюзивно;отправленный
ключ автоматически не повторяется. После серии guardian завершён,locks
свободны. Проверка возобновления:0 новых вызовов и0 worker-startups,
хеши140 результатов прежние. Power settings/source зафиксированы до запуска;
подробные аппаратные sysctl-параметры не объявляются независимо доказанными.
[Environment](environment.json),[power до старта](hardware_power_before.json),
[сессии](sessions/),[resume](resume_check.json),
[независимые проверки](independent_verification.json).

## Сценарии стоимости1000

Старый MC-сценарий использует прямые grid33 времена100 задач и для остальных
x0 mean4×33,с масштабированием M×4. Перенос x0→33 на остальных900 заново
не измерялся. Этот operation-scaled прогноз MC1 откалиброван отношениями измеренной
медианы4 вызовов к старому прогнозу на каждой из40 задач. Центральный factor —
медиана coverage20={fc['calibration_factor']:.6g};разброс отношений по40 даёт
только чувствительность модели,не CI/прогнозный интервал/гарантию.
Для свежего1000-уравнённого benchmark с4 вызовами каждого профиля на задачу:

- MC1: **{fc['MC_seconds']/3600:.2f} compute-ч**.
- R128:старый сценарий **{fc['R_seconds_legacy_scenario']/3600:.2f}ч**;
  параMC1+R128 **{fc['total_hours']:.2f}ч**,чувствительность
  **{min(rng):.2f}–{max(rng):.2f}ч**.
- ПараMC1+R256: **{f256['total_hours']:.2f}ч**.

Есть также сценарии по1 вызову/задачу. Старый R-прогноз не откалиброван новым
таймером;будущие подготовка R,прогрев,сохранение,паузы,сбои и повторы не
включены. Это вычислительные сценарии одного worker,не измеренный полный
benchmark и не утверждённый новый бюджет. НеполныеR0 и проверка пригодности
MC1 остаются отдельной работой. Уже потраченноеP1/P2 в эти свежие1000 не входит.
[Калибровка40](analysis/cost_calibration.csv),
[сценарии1/4 и R128/R256](analysis/benchmark1000_cost_scenarios.csv).

## Решение и следующий шаг

P2 выполнен в пределах разрешённого объёма. На этих train-задачах конечность
MC-ответа часто не означает достижение.01;четыре seeds не сертифицируют
надёжность. До решения о benchmark1000 рекомендуется закрыть недостающийR0
(3R128+35R256) и разобрать причины ошибокMC1 на train,затем отдельно
зафиксировать профили и бюджет. К обучению ML пока не переходить:полных
сопоставимых измерений и подтверждённых fastest/reliable меток ещё нет.

Новый код: [driver](../../../diagnostics/mc_pilot_p2.py),
[saved-only report](../../../diagnostics/report_mc_pilot_p2.py),
[independent verifier](../../../diagnostics/verify_mc_pilot_p2.py),
[тесты без траекторий](../../../tests/test_mc_pilot_p2.py).
[Воспроизведение](REPRODUCE.md). Численные схемы не изменялись.
'''
def readable(value):
 value=re.sub(r'([А-Яа-яЁё])(?=[0-9A-Z])',r'\1 ',value)
 value=re.sub(r'([0-9A-Za-z])(?=[А-Яа-яЁё])',r'\1 ',value)
 return re.sub(r'([,:;])(?=[А-Яа-яЁёA-Za-z])',r'\1 ',value)
with (out/'REPORT.md').open('x') as f:f.write(readable(text))
old=(out/'PROJECT_STATUS.before.md').read_text()
intro=f'''# Состояние проекта ode-generator-remizov

Обновлено **2026-10-26**: **P2 завершён —140/140 новых MC1-вызовов на35 train-уравнениях,
все конечные,0 отказов/pending/повторов. Вместе сP1 MC1:40 задач,160 вызовов.**
[ОтчётP2](reports/ml_selector_contract/pilot_p2_2026-10-26/REPORT.md),
[сводка](reports/ml_selector_contract/pilot_p2_2026-10-26/analysis/summary.json),
[проверки](reports/ml_selector_contract/pilot_p2_2026-10-26/independent_verification.json).

Профиль и seeds сохранены:MC1 M4096,h1/128,4seed248800020100..103 из исходных
условныхP2 jobs;P1 MC1 —20 вызовов5 других задач. Перед запуском проверены
идентичности/hash,train/семейства,configs/seeds/G/T/N,все артефактыP1 и budget.
Операционный P2 cap18599с основан на сценарии9149.47с;потолок140 соблюдён.
Численные методы не менялись;новыхR0/benchmark1000/ML-вызовов0.

Попадания E_grid≤.01: **P2 {p['accurate_calls']}/140;P1 {a['accurate_calls']}/20;
вместе {c['accurate_calls']}/160**. Episode0:{c['episode0_pass']}/40,хотя бы1/4:
{c['any_pass']}/40,все4/4:{c['all4_pass']}/40. Coverage20:{cov['accurate_calls']}/80,
stress20:{stress['accurate_calls']}/80. Медиана ошибкиP2 {p['error']['median']:.8g},
всех160 {c['error']['median']:.8g}. Mean4 отдельно:{s['mean4_diagnostic_accurate']}/40,
с суммой стоимости4 вызовов;не подменяет single-call. Это выбранный train-пилот,
P1 участвовал в выбореMC1;точность на1000 и надёжность≥.95 не установлены.

Сохранённый совместимый Remizov R128 accurate **{r['R128']['R_accurate']}/37**,
MC1 episode0 на тех же37:**{r['R128']['MC_episode0_accurate_same_tasks']}/37**.
R256 accurate5/5 наsentinel;для3 R128/35 R256 ответов нет. Missing не считается
численной ошибкой. Taper/старые timings не подменяют новый профиль/полныйтаймер;
fastest/reliable метки пока не подтверждены.

1worker/поток;суммаMC1-P2 warm AST/G+config+solve **{p['method_seconds_sum']:.3f}с**,
wall **{s['controller_wall_seconds']:.3f}с/{s['controller_wall_seconds']/3600:.3f}ч**;
peakRSS{s['peak_rss_mib']:.3f}MiB<512;thermal{s['thermal_states']};
{s['P2_path_updates']:,} updates. JIT отдельно,внутри каждого solve compilation0.
Guardian завершён,locks свободны;resume:0 новых calls/startups и неизменные
хеши140 записей. Power записана до старта;hardware sysctl не объявляется
доказанным. Все прежние файлы,кроме намеренного обновления статуса,сохранены;
старый статус и исходники восстановимы в каталогеP2.13 mock-тестов пройдены,
160 ошибок сверены с exact;независимо140 новых ошибок и5280 RNG-адресов.

Свежий1000×4 сценарий(не запуск/не CI/не гарантия):MC1≈{fc['MC_seconds']/3600:.2f}ч;
сR128≈{fc['total_hours']:.2f}ч(чувствительность{min(rng):.2f}–{max(rng):.2f}ч),
сR256≈{f256['total_hours']:.2f}ч;новые R-preparation и будущие overhead не измерены.
[Расчёт](reports/ml_selector_contract/pilot_p2_2026-10-26/analysis/benchmark1000_cost_scenarios.csv).
Следующий рекомендуемый шаг:закрыть38 недостающихR0 и исследовать причины
ошибокMC1 наtrain,после этого отдельно решить о профилях и бюджете benchmark.
Полныйbenchmark иML не разрешались и не выполнялись.

## Исторический статус после P1 (до текущего P2)

'''
(root/'PROJECT_STATUS.md').write_text(readable(intro)+old.removeprefix('# Состояние проекта ode-generator-remizov\n\n'))
print(json.dumps({'report_written':True,'P2_accurate':p['accurate_calls'],'combined_accurate':c['accurate_calls'],'1000_pair_hours':fc['total_hours']},ensure_ascii=False))
