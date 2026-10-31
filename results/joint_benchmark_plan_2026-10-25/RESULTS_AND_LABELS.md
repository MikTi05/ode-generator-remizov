# Контракт будущих результатов и ML-меток, предложение v1

Эти структуры описывают **будущие** артефакты. В настоящем каталоге нет
новых строк trial/предсказаний/ML-меток. Точечный и сеточный режимы имеют
разные policy_id и разные строки; копирование меток между ними запрещено.

## Хранение

Новый `results/joint_pilot_<date>/`, без перезаписи существующих каталогов:

```text
proposal_snapshot/       # согласованная версия документов и её SHA
plan.json                # полный упорядоченный список попыток и config, заморожен
environment.json         # версии, CPU/OS, питание, thread env, workers=1
source_sha256.json       # реализации/оркестратор/сертификат/данные до и после
identities.csv           # dataset fingerprint, equation, expression, family, split
attempts.jsonl           # все planned/not_run/started/terminal; append-only
trials/<attempt_id>/
  request.json           # параметры до исполнения, не содержит exact-ошибок
  prediction.json        # полный asdict(Result), векторы, hash до оценивания
  score.json             # exact, error vector, E, labels конкретного ответа
  resources.jsonl        # wall, CPU, RSS, состояние worker и среды
  worker.log             # stdout/stderr, исключение/returncode
profiles.csv             # независимые повторы одного equation/mode/policy
comparisons.csv          # одна строка R-policy vs MC-policy, все исходы
ml_targets.csv           # ссылки на comparison, admissibility, censor/missing flags
REPORT.md                # результаты и неполнота, без отбора успешных seed
```

## Строка attempt / trial

| Группа | Поля и смысл |
|---|---|
| Идентичность | schema_version, experiment_id, plan_sha256, attempt_id, dataset_sha256, equation_id, expression_hash, generation_version, family_id, structure_signature, original_split, policy_development_exposed |
| Задание | method, implementation, implementation_sha256, policy_id, policy_sha256, mode, target_x, target_grid_sha256, epsilon, budget_wall_seconds, budget_rss_mib, hardware_id |
| Параметры | Полный asdict(config), фактические n/Q/dx/padding/interp/boundary либо M/dt/chunk, N; T_requested, T_actual, G, gap, сертификат и его SHA, tail_bound, tail_certified |
| RNG | phase, replicate_index, root_seed, seed_entropy, stream_id, point_spawn_keys, rng_scheme, chunk, discovery_seed, seed_role; у детерминированного R — null |
| Процесс | status, started/finished_utc, solver_success, failure_reason, wrapper_reason, returncode, exception/traceback, planned/not_run_reason |
| Ответ | Полный Result; completed_paths на каждую точку, finite_complete, prediction_path, prediction_sha256, prediction_frozen_utc |
| Время | worker_response_seconds, imports/load/certification/preparation/solver/calibration/serialization_seconds, task_process_wall_seconds, CPU_seconds; scoring_seconds отдельно |
| Стоимость | n_path_steps, n_normal_draws, coefficient_calls/values, n_batches, peak_batch_paths либо n_operator_applications, n_spline_builds, work_grid_points, work_domain, outside_query_values, boundary_state_max_abs; счётчики partial или complete |
| Ресурсы | peak_RSS_mib, resource_log, thermal_state, background_load, threads; interrupted=true/false, elapsed_at_kill, partial_counter_flags |
| Exact-оценка | error_vector, max_abs_error, relative_error, exact_amplitude, observed_accuracy_pass, exact_score_utc; вычисляется после prediction_frozen_utc |
| Неопределённость | stderr, pointwise_ci95, simultaneous_grid_ci95, interval_kind, confidence_scope, discretization_error_bound=null, total_error_certified=false |

Время основной метрики `worker_response_seconds` — от старта свежего worker
до записи предсказания, включая import/load, аналитический сертификат и все
стадии политики. Exact-evaluator идёт после этого момента и в latency метода
не входит. Watchdog60 с контролирует этот же участок. Запуск worker происходит
под одним и тем же импортером обоих решателей. Отдельно сохраняется время
самого solver и полный task wall, чтобы не смешивать их с прежними замерами.
Никаких бесплатных повторов/калибровок, скрытого кэша или точечного R-времени,
полученного делением сеточного на 33. Амортизированная серия с повторным
использованием подготовки потребует другой policy, не подменяет cold response.

Статусы разделены:

- `completed`: полный конечный ответ, с отдельными observed accuracy pass/fail.
- `solver_failure`: численный отказ с исходным failure_reason, E=null.
- `inapplicable_certificate`: отсутствующие/противоречивые предпосылки, E=null.
- `external_time_limit`, `external_memory_limit`: отказ оболочки, частичная
  работа и известный elapsed сохранены; E=null, не дорисовывать недостающие точки.
- `worker_exception`, `infrastructure_failure`: причина и returncode, неизвестные
  сведения оставлены null. Исторические три выхода worker не объявлять OOM.
- `not_run_design_subset`, `not_run_after_resource_limit`, `not_run_phase_budget`:
  результат не получен, E/time=null; время 0 не подставлять. Для будущей online
  политики дополнительно `statistical_budget_insufficient` с отдельным
  полем measured/predicted: прогноз превышения — не наблюдавшийся timeout.

JSON не содержит NaN/Infinity: неопределённые поля null с причиной. Начатая
попытка всегда имеет терминальную запись, неначатая — явный not_run.
Пересчёт exact и аналитики не изменяет prediction; новая аналитика версионируется.

## Строка profile

Ключ `(dataset_sha256, expression_hash, equation_id, mode, policy_id, budget, hardware)`.
Хранить requested/completed/evaluable/not_run числа повторов, все seed/attempt_id,
отдельно количество exact-успехов и ресурсных/численных отказов.

Для MC при полной предзаданной последовательности: k/R, односторонние
Clopper–Pearson95% L_p, U_p,`reliability_target=.95`,`reliability_state`:

- `qualified`, если L_p≥p*;
- `below_target`, если U_p<p*;
- `unresolved`, иначе;
- `missing_or_censored_design`, если последовательность не завершена,
  инфраструктура нарушена или остановка зависела от ресурса предыдущего seed.

При неполноте показывать числитель/плановый знаменатель/причину, без
необоснованного CI. Если будущий протокол всё-таки исполнил всю предзаданную
последовательность, ресурсные отказы входят в R как неуспехи. Discovery-семена
не входят в подтверждающую последовательность. При выборе M онлайн единица
испытания — **вся политика**, с её новой discovery и новой финальной оценкой.

Для MC отдельно сохранить mean_error_vector, seed_sd, pointwise_t95_bias_ci,
Bonferroni_grid_bias_ci, RMSE по seed, max observed error, обычные/совместные
статистические полуширины. Mean-bias CI включает усечение и дискретизацию,
а не одну EM; непрерывный хвост известен отдельно и не вычитается из E.
`max_j|mean_error_j|≤.01` не заменяет вероятность успеха новой реализации.
Агрегат нескольких seed — другой ответ с суммарным M и суммарным временем,
а не бесплатный улучшенный ответ одиночного вызова.

Для R: numerical_complete, observed_accuracy_pass, bitwise/numerical repeat
consistency и список E каждого повтора; k/3 не выдаётся за MC-подобную
оценку вероятности. Несогласованные f → `reproducibility_unresolved`.

Для обоих: median/p10/p90/min/max времени, каждый отдельный runtime,
полная потраченная цена, counts иRSS. Прерванные времена right-censored у
фактической границы. Среднее только по успешным ответам не является главной
метрикой: все failure иaccuracy_fail остаются в сравнениях.

## Строка comparison и правило winner

Главный объект — **две замороженные политики в одном mode и бюджете**;
не «лучший когда-либо полученный ответ метода». Порядок решения:

1. Проверить идентичность уравнения, сетку, сертификат, политику, сопоставимость
   среды и полноту данных. При нарушении → `winner=unknown`, конкретная причина.
2. У R требуется полный воспроизводимый детерминированный профиль с E≤.01
   в бюджете; у MC — квалификация по p*. Если MC `unresolved`, победитель
   по надёжному качеству тоже unknown, даже когда все его ответы быстрые.
3. Если квалифицирован только один, а другой **статистически ниже цели**
   или его полная предзаданная серия установила ресурсную непригодность,
   указать `winner=remizov|monte_carlo`, reason=`only_qualified_within_budget`.
   Это заключение об этих политиках, без ускорения относительно неудачного
   метода. Прогноз цены/неисполненные seed такого основания не дают.
4. Если оба квалифицированы и завершены все замеры, сравнить время.
   Descriptive median оставить; для формального правила использовать
   отношение геометрических средних response time `ρ=t_MC/t_R` и Welch95%
   CI лог-времён (приближённо нормальных, учитывать df при малых R).
   Включать времена всех полных повторов, не отбирать только E-pass seed.
   При resource-censoring эту формулу не применять; `unknown_time_censored`.
   Если предположение лог-нормальности не обосновано, оставить предварительное
   эмпирическое сравнение и `unknown_time`, без формального 95% заявления.
5. `CI_low(ρ)>1.10` → R быстрее; `CI_high(ρ)<1/1.10` → MC быстрее.
   Весь CI внутри `[1/1.10,1.10]` → `tie_practical`.
   Иначе → `unknown_time`. Просто недостоверное различие не является ничьей.

Все comparison сохраняют:

`r_accuracy_state, mc_reliability_state, mc_p_lower, mc_p_upper,
r_time_summary, mc_time_summary, time_ratio, time_ratio_ci,
winner, winner_reason, evidence_level, confidence_scope,
both_observed_fail, budget_exceeded, censoring, missingness,
policy_development_exposed, training_eligible`.

`winner` принимает `remizov, monte_carlo, tie_practical, neither, unknown`.
`neither` требует установленных неуспехов обоих профилей; отсутствие данных
не заменяется neither. Причины unknown: `accuracy_unresolved`,
`time_unresolved`, `budget_censored`, `infrastructure`, `not_run`,
`inapplicable`, `incomparable_policy`. Отдельный `best_observed_profile`
может быть описан, но его `evidence_level=exploratory`, он не заменяет winner.

95% здесь относится к одной comparison/профилю. Для перспективного семейства
подтверждающих меток заранее скорректировать α (например, Holm), записать
hypothesis_family иadjusted_confidence. Малый P0 по умолчанию
`evidence_level=diagnostic`,`training_eligible=false`; завершённость и
статистическая мощность проверяются до возможности снять этот флаг.

Дополнительная метрика для бюджетной полезности: **PAR2** — реальное response
time при exact-успехе в бюджете и 2B=120 с при численном/ресурсном/accuracy
неуспехе. Сохранять каждое значение и среднее по всей фиксированной MC-серии,
а не только по успешным seed. Неисполненные/инфраструктурные случаи оставлять
missing, не штрафовать как доказанный численный провал. PAR2 — отдельная
utility label с явным выбранным штрафом, не доказательство p≥.95.

## Передача в ML

Одна строка на equation+mode+budget+policy_pair, все family-группы сохранены.
Отдельные targets: observed accuracy конкретного ответа; биномиальные k/R
профиля; лог-время с right-censoring; winner с reason/uncertainty. Unknown
сохранять как воздержание/маску, а не кодировать самым быстрым методом.
Долю unknown, neither иresource failures показывать в знаменателе отчёта;
удаление трудных задач меняет задачу обучения.

Признаки фиксируются до решения и отделены от outcome-таблиц. Exact,
пострешательные stderr/счётчики/runtime/ошибки не доступны селектору.
Если online discovery используется как признак второго этапа выбора,
нужна отдельная модель каскада с оплаченной discovery и собственным policy_id.
Cross-validation иbootstrap по качеству селектора — по семействам/общим
structure_signature, а не поseed, точкам или 5 образцам одного семейства.
При объединении двух наборов использовать dataset fingerprint; состояние
exposure исходных test-семейств остаётся в метаданных.
