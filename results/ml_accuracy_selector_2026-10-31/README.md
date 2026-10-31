# ML выбора более точного из двух методов, 2026-10-31

Основной результат: [REPORT.md](run/REPORT.md), [metrics.json](run/metrics.json),
[997 пар x=0](run/comparable_x0.csv), [модель](run/model/model.cbm).
Сохранены mass Remizov n=128 и MC1024 с четырьмя seed; epsilon=0.05.
Фактический семейный split 697/150/150. На test CatBoost всегда выбирает
Remizov и не превосходит постоянный R; средняя ошибка постоянного MC ниже.

- [Независимая финальная сверка](final_independent_verification.json)
- [Сохранность исходных файлов](preservation_verification.json)
- [19 новых тестов](pytest_new.txt); дополнительно прошли 72 прежних ML-теста.
- [Отдельный grid33-анализ](run/grid33.csv): 100 основных train-пар, 900 отсутствует.
- [Дополнительный n=256](run/supplementary_n256.csv), [внешние 20 n256/MC4096](run/external_n256.csv).
- [Первоначальный план](plan.json), [план исполненного fit](run/plan.json).
- [Исходный статус](PROJECT_STATUS.before.md), [исходный README](README.before.md), [SHA до изменений](project_before_sha256.json).

Единственный fit выполнен до анализа test; последующее уточнение текста отчёта
не меняло модель, метрики или параметры. Первоначальный текст и provenance
сохранены в [report_revisions/v1](report_revisions/v1/REPORT.md).

Воспроизведение без численных методов из корня проекта:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/train_accuracy_selector.py --out results/ml_accuracy_selector_reproduction
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/verify_accuracy_selector.py --out results/ml_accuracy_selector_2026-10-31/run
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_accuracy_selector.py tests/test_accuracy_grid_evidence.py
```
