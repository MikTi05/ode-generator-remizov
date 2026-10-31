# Окончательная проверка селектора с MC mean4, 2026-10-31

**Преимущества ML перед постоянным Remizov на test нет.** Модель выбирает
Remizov150/150, пропускает все9 MC-winner и оба исследованных выброса R.
На validation результат ML хуже постоянного R. Дальнейшие эксперименты
автоматически не проводятся.

- [Полный отчёт](run/REPORT.md), [все метрики](run/metrics.json), [997 пар](run/comparable_x0.csv).
- [Сохранённая модель](run/model/model.cbm), [реестр](run/model/registry.json).
- [Разбор выбросов](run/outliers.json), [важности input-only признаков](run/feature_importance.csv).
- [Сравнение с прежней моделью](run/previous_model_comparison.json).
- [План до fit](plan.json), [34 теста](pytest.txt), [финальная независимая проверка](final_independent_verification.json).
- [Независимая проверка двух выбросов](outlier_independent_verification.json).
- [Сохранность исходных файлов](preservation_verification.json), [SHA до изменений](project_before_sha256.json).
- [Прежний отчёт с median seed-errors](../ml_accuracy_selector_2026-10-31/run/REPORT.md) сохранён без изменений.

Только сохранённые mc_seed_f0/y_exact0/E_R; 253 прежних признака скопированы
побайтно. Один фиксированный CatBoost fit на697train, validation150/test150,
без балансировки и подбора. Четыре исходных1024-path оценки объединены средним;
это не новый одиночный4096-path запуск. epsilon=.05, область вывода толькоx=0.
Повторно исследован уже раскрытый набор; нового проспективного подтверждения нет.

Воспроизведение при явном запуске пользователем, из корня проекта:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/train_mean_mc_selector.py --plan results/ml_mean_mc_selector_2026-10-31/plan.json --out results/ml_mean_mc_selector_reproduction
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/verify_mean_mc_selector.py --out results/ml_mean_mc_selector_2026-10-31/run
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_mean_mc_selector.py tests/test_accuracy_selector.py tests/test_accuracy_grid_evidence.py
```

Редакционное уточнение текста отчёта после fit не меняло модель или метрики;
исходный текст и provenance сохранены в report_revisions/v1.
