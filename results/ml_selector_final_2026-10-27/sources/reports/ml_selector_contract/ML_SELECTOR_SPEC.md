# Спецификация будущего ML-селектора

Статус: **дизайн, новых моделей и API реализации нет**. Основной target для
рекомендуемого этапа — `observed_fastest_v1`, отдельный optional target —
`reliable_fastest_v1`. Их не смешивать в одной колонке best_method.

## Вход и переиспользуемые компоненты

Python entry point будущего сервиса:
`select_method(coefficients, x_grid, coefficient_metadata, profile_pair, timing_context)`.
coefficients — a, b, c, g на исходной 256-точечной сетке; metadata — whitelist
сертификатов/AST a, b, c; profile_pair задаёт численные параметры и object domain.
Резольвентная постановка и all_methods_v1 — обязательная область применения.
Для произвольного IVP нужны начальные/граничные условия и другой контракт.

Переиспользовать:

| Компонент | Можно использовать | Ограничение |
|---|---|---|
| `ml/remizov_features.py` | 253 input-only признака, порядок/version/hash | `pre_T` и n-derived признаки исторически соответствуют legacy n128; для новых профилей переобозначать как фиксированные descriptors либо версионировать, не подменять молча |
| Семейный split CSV/JSONL/NPZ | 700/150/150; 140/30/30 families | Все варианты и seed одного семейства вместе |
| StandardScaler pipeline | Архитектура train-only fit | Новые targets требуют нового fit на train; старый scaler — не основание передать test в fit |
| Старые CatBoost/quality модели | Сравнительный historical baseline только в совпадающем scope | R taper/MCx0 weights не прогнозируют автоматически zero/grid33 |
| `ml/quality_prototype.py` | Валидация входа/profile/hash, uncertain, структура serving | Сейчас это quality API, не selector; winner=None корректен |
| `diagnostics/predict_quality.py` | Шаблон CLI input JSON/domain | Не использовать ID/exact как скрытые признаки |
| `tests/test_remizov_ml.py`, `test_quality_prototype.py`, `test_mc1000_quality.py` | Leakage/reload/scaler/CLI/status тесты | Потребуются новые contract tests, но в этом этапе они не реализуются |

Запрещены в X: equation/family ID, seed, split, exact/производные/y_shape,
solver f/error/runtime/status/CI и признаки из них. g — заданная forcing,
допустима даже при её manufactured происхождении. Profile/hardware context,
известные **до** выбора, допустимы с отдельной схемой. В двух фиксированных
профилях контекст константен; не выдавать историческую конкуренцию 8/1workers
за свойство уравнения.

## Targets и выход

Training table: одна строка на equation/profile_pair/label_type/episode0,
полный provenance key; другие episodes — повторные наблюдения, не дополнительные
независимые уравнения. Хранить numerical, execution и timing таблицы отдельно
от feature matrix; join только по ключам. `unknown` не обучать как `none`;
при цензурировании явно сообщать coverage и возможное MNAR-смещение.

Будущий ответ:

```json
{
  "schema": "method_selector_v1",
  "label_type": "observed_fastest_v1",
  "profile_pair_id": "frozen-pair-sha256",
  "action": "remizov|monte_carlo|abstain",
  "predicted_label": "remizov|monte_carlo|none|unknown",
  "status": "prediction|insufficient_evidence|unsupported_profile|out_of_domain",
  "probabilities": null,
  "predicted_warm_cost_seconds": null,
  "reliability_status": "not_certified",
  "accuracy_certificate": false,
  "reason": "versioned-reason-code"
}
```

JSON выше — schema example с альтернативами, не реальный численный прогноз.
`none` относится к выбранным бюджетам, action=abstain; `unknown` — отсутствие
оснований выбрать. Модельный p не является CP-lower bound реального solver.
API не использует exact, не запускает дополнительные solver для ответа и
не скрывает стоимость fallback. При изменении профиля/hardware версия модели
и область пригодности проверяются, а не предполагаются.

## Архитектура и порядок обучения

Рекомендуемый первый обучаемый baseline после benchmark — классификация
наблюдаемой допустимости R и MC, регрессия log runtime, и явно описанная политика
решения. Допустим прямой classifier observed_fastest при достаточных targets.
Не усложнять архитектуру ради создания MC-класса; Dummy/Logistic/CatBoost
с фиксированным малым набором моделей достаточны для проверки гипотезы.
Новые fit в текущем задании не выполняются.

Последовательность: freeze numerical pair/label rules → получить train labels
и uniform costs → fit/scaler только train → validation model/threshold/calibration
→ freeze weights/policy → один итоговый test evaluator. Seed-replicas не делить.
Ресурсы ML и стоимость feature extraction/inference измерять отдельно от
solver, но включать в стоимость применения политики. Численные параметры не
подбираются validation/test ML-модели.

Предлагаемый reporting gate для заявления о распознавании MC-winner:
как минимум 20 разных train-семейств и 5 validation-семейств с таким target,
без синтетической балансировки. Это консервативное правило проекта, не теорема
о достаточном размере выборки. Если на test мало/нет MC-class, показать CI и
отсутствие support; не менять модель после просмотра test ради гейта.

## Baselines и oracle

Обязательны constant_R, constant_MC, selector и offline_oracle. Constant baseline
реально выбирает соответствующий frozen профиль; его ошибки/стоимость не
подменять предсказаниями. Oracle смотрит **только для оценки** на сохранённые
test episode0 errors и uniform median4 times; использует те же missing/tie rules.
Oracle labels не подаются в X/обучение/test-настройку. При none/unknown oracle
не выдумывает точный метод, эти случаи отражаются в coverage.

Для cost regret отдельно T_oracle=min median4time среди observed-accurate
профилей. Если practical_tie предпочёл R, его cost regret может быть небольшим
положительным:идеальная стоимость иоперационная tie-label имеют разный смысл.
Для B oracle использует только полностью измеренное frozen Θ.

## Метрики, определённые до test

Для каждого split, всего ипо train-defined complexity/regime:

- Classification accuracy и confusion matrix на известных labels; denominator
  known/total рядом. Macro-F1 по классам с support иполный support vector;
  если MC-class отсутствует, не трактовать нулевой/неопределённый recall как
  успешное обучение этому классу. Не скрывать none/ties/unknown.
- `delivery_accuracy_rate = correct_deliveries / requested_tasks`;
  `conditional_accuracy = correct_deliveries / delivered_finite_outputs`;
  `abstention_rate`, execution failures, timeouts и unknown отдельно.
  Число нарушений E_grid>.01 — только finite outputs; отсутствующий результат
  — delivery failure, а не выдуманное значение E_grid.
- `total_policy_wall = common_input_preprocessing + features + inference +
  all_executed_method_calls + fallback + necessary_setup`. Отдельно warm
  основной вывод, cold setup/JIT истоимость всей исследовательской серии.
  Не сравнивать миллисекундный inference ссекундами полного solver.
- Median/p95 cost по запросам ипо accurate deliveries. Все расходы неудачных
  попыток остаются в total, не только успешные времена.
- Absolute/relative **accuracy-constrained regret**:
  T_policy−T_oracle и(T_policy−T_oracle)/T_oracle только где выбранный ответ
  accurate и oracle существует. На inaccurate/failure/abstain отдельный
  outcome идоля покрытия, не отрицательный «выигрыш скорости».
  Дополнительно operational cost всех задач и accuracy violations совместно;
  произвольный штраф за неточность после просмотра test не вводить.
- Для MC: k/r, индивидуальные CP-CI, scope .95, FWE-поправка если активна;
  вероятность класса ML иизмеренная solver reliability не тождественны.
  Показать episode0 основную оценку и episodes1..3 чувствительность отдельно.

Uncertainty качества ML — bootstrap по **семействам**, с сохранением всех
вариантов/seed, а не по 4000 как независимым задачам. Report confidence level и
число семейных кластеров. Исторический test уже изучался:ограничение
prospective независимости сохраняется даже при хорошем ROC-AUC.

## Если MC нигде не выигрывает

Сохранить наблюдение, не добывать класс настройками, test-поиском или pseudo-labels.
Показать, что constant_R покрывает лучшие observed outcomes в проверенном
бюджете; без данных нельзя утверждать универсальное превосходство R.
Корректный ML-результат — модель риска/допустимости R, а при его высоком риске
fallback **abstain/ручное уточнение** либо MC только при отдельно подтверждённой
его допустимости. Нет основания автоматически запускать неточный MC в роли
«надёжного запасного». Если MC-winner=0, не заявлять, что selector научился
находить превосходство MC; при одной полезной альтернативе сложный classifier
может не давать преимущества над constant_R, что и есть проверяемый результат.
