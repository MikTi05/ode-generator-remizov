# observed_fastest: реализация ML-pipeline

Код читает сохранённые результаты и не вызывает численные методы.
Текущая фиксированная диагностическая пара: R128 cubic/zero/Q513/dx=.0125/
padding=12/common analytic T/tail=1e-4 и MC1 Numba M=4096/h=1/128,
chunk=1024/block_steps=64. R256, taper, x0, MC0/MC2 и mean4 — другие профили;
они не используются для создания дополнительных winner-меток.
Финальный выбор R128/R256 после R0 остаётся отдельным исследовательским решением.
Исходные документы `reports/ml_selector_contract/` сохранены неизменными.
Их запрет fit описывал прежний этап планирования; текущий запрос разрешает ML.

## Запуск

Использовать Python проекта и `requirements-ml.txt` (уже установлены CatBoost
1.2.10 и scikit-learn 1.7.2). Новый выходной каталог обязателен:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/train_method_selector.py \
  --out results/ml_selector_observed_reproduction_2026-10-26
```

Параметры, seed=2488, четыре CV-складки и 200 деревьев фиксированы в коде и
`protocol.json` до анализа. Директория не перезаписывается. `--help` описывает CLI.
Сохранённый итог: `results/ml_selector_observed_2026-10-26_v2/`.
Предыдущая диагностическая итерация также сохранена; повторение подтвердило
идентичность меток, OOF-предсказаний и вероятностей. CBM может содержать разные
служебные метаданные; для каждого запуска хешируется его собственный артефакт.

## Подготовка целей

`ml/selector_evidence.py` проверяет dataset/implementation/plan SHA, ID вместе
с expression_hash, family/split, полные job/config, сетку33, finite f, канонический
exact и пересчитанную E_grid, T, MC seeds/RNG/число шагов и завершённых путей.
Из старой joint-серии берутся только primary R128, контрольные запуски отдельно
перечислены в исключениях. Численные ответы, исполнения, времена, признаки и
таблица индексов записываются раздельно; targets не входят в X.

`ml/selector_labels.py` реализует `observed_fastest_v1` из CONTRACT:

| R episode0 | MC episode0 | label |
|---|---|---|
| точен | неточен | remizov |
| неточен | точен | monte_carlo |
| неточен | неточен | none |
| неизвестен | любой | unknown |
| любой | неизвестен | unknown |
| точен | точен | сравнение измеренных median4 либо unknown |

Точность — E_grid≤0.01 на `[-1+j/16 for j in range(33)]`. Пропущенные/ошибочные
вычисления не заменяются неточностью. Другие seeds не подменяют episode0 и не
становятся независимыми training rows. `none` относится только к этой паре и
бюджетам; `unknown` никогда не обучается как `none`.

Для двух точных нужны четыре конечных нецензурированных времени каждого метода,
совпадающий timing cohort и состав таймера AST/G/T+constructor+solve, воспроизводимость
R и MAD/median≤0.20. Практический победитель требует median_fast≤.90 median_slow
и max(fast)<min(slow); иначе стабильная ничья с приоритетом R. Если точен только
один метод, сравнивать время с неточным не требуется. Метки sole-accurate и none
поэтому достоверны даже при неизвестной сопоставимой стоимости.

## Признаки, fit и проверка

Повторно используется неизменённый `ml/remizov_features.py`: ровно253 input-only
признака на исходных256 узлах, исходный порядок/version/hash. `pre_T` и признаки
с n сохраняют значение фиксированных legacy-descriptors n128/tail1e-8; это не
подстановка фактического T новых профилей. ID, family, split, seed, exact,
дерево решения, ошибки, времена и статусы в X отсутствуют.

Семейные и topology splits остаются700/150/150 (140/30/30 семейств).
StandardScaler обучается только на размеченном train; внутри каждой CV-складки —
заново только на обучающих семействах. CatBoost получает только существующие
классы; oversampling, SMOTE и pseudo-labels отсутствуют.

Для полного R/MC classifier принят предлагаемый gate ML_SELECTOR_SPEC:
≥20 train-семейств с MC-winner и ≥5 validation-семейств с MC-winner, наличие R
в train и размеченной validation; новые validation-классы должны быть обучены
на train. Это правило проекта, не теорема о достаточном размере данных.
Test для оценки нужен отдельно. При недостатке данных выполняется только
train-only StratifiedGroupKFold и отдельно сохраняется диагностический CBM.
Диагностика не выбирает гиперпараметры и не использует original validation/test.
Настоящий test evaluator вызывается только после фиксации полноценной модели.

Метрики содержат support всех трёх обучаемых классов и denominator known/total;
при нулевом MC support его recall=null. Сохранены confusion, macro-F1 по классам
с support, 500 bootstrap выборок целых семейств, baseline constant_R, constant_MC
и offline oracle. Отдельно учитываются observed outputs, нарушения точности,
abstention и unobserved outputs. Без полных сопоставимых расходов policy wall и
accuracy-constrained cost regret остаются null. Из времени inference не делается
вывод об ускорении полного solver. Старый test уже изучался в численных аудитах.

Технические источники: [StratifiedGroupKFold](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
[CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier).
Версии реализации фиксируются в `environment.json`, зависимости — в `requirements-ml.txt`.

## Использование API

```python
from ml.method_selector import MethodSelector
selector = MethodSelector('results/ml_selector_observed_2026-10-26_v2/registry.json')
result = selector.select_method(
    coefficients, x_grid, coefficient_metadata,
    profile_pair_id=selector.registry['profile_pair_id'],
    timing_context=selector.registry['timing_context'],
)
```

Вход — только данные уравнения a,b,c,g на исходных256 узлах и whitelist
коэффициентных сертификатов/AST. Exact и equation ID не нужны. Иная пара/контекст
даёт unsupported_profile, другой domain/grid/некорректные входы — out_of_domain.
При текущем registry ответ — action=abstain, predicted_label=unknown,
status=insufficient_evidence, probabilities=null. Диагностический CBM не становится
полным селектором; registry с двумя классами none/R нельзя объявить trained
R/MC selector. Ни модельные вероятности, ни метки observed не сертифицируют
точность нового MC-вызова или надёжность≥.95.

## Проверки

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_method_selector.py tests/test_remizov_ml.py \
  tests/test_quality_prototype.py tests/test_mc1000_quality.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
  diagnostics/verify_method_selector.py --out results/ml_selector_observed_2026-10-26_v2
```

Последняя команда читает артефакты и создаёт проверки эксклюзивно: в уже
проверенный каталог повторно не пишет. Она не вызывает solver/fit. Исходные
версии кода лежат в `sources/`, исходный статус — в `PROJECT_STATUS.before.md`.
