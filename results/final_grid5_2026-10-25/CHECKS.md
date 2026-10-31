# Проверки

- 26 тестов планирования, биномиальной статистики, отбора и независимого
  guardian: **26 passed in12.61s**, [pytest.log](pytest.log). Дополнительные
  численные решения ОДУ в этих тестах не выполнялись.
- 25 новых ответов проверены по job/plan SHA, production expression_hash,
  train split, сетке33 точек, actual T, M4096, n256/Q513 и счётчикам работы.
  Ошибки E∞ пересчитаны из сохранённых f/exact. Exact совпадает с baseline.
- Baseline n128 скопирован побайтово и сверён с прежним источником. Старые
  MC-повторы сохранены для сравнения, но исключены из нового подтверждения.
- Resume после завершения:25 неизменных SHA,0 новых solver и worker,
  [resume_check.json](resume_check.json).
- Математический код из MIGRATION_MANIFEST и старые научные артефакты не
  редактировались. Аудит16934 путей сохранил исходные SHA. Единственная
  разрешённая ревизия — новый отчётный скрипт текущей задачи; его подготовительная
  версия восстановлена и совпала с before SHA, окончательная также сохранена.
  [История версии](analysis_source_history.json), [исходная](source_history/report_final_grid5_preparation.py),
  [окончательная](source_history/report_final_grid5_final.py).
- Frozen numerical sources/statistical rules не менялись после init; не
  изменялись alpha, биномиальный критерий, число seed, h, M или T по результатам.
- Worker/guardian завершены, глобальные compute/controller locks свободны.
  [verification.json](verification.json).
- График accuracy_cost.png просмотрен. Окружение — requirements-lock.txt;
  численных экспериментов сверх указанных пяти, расчётов900 и ML-fit нет.

Команда контрольного набора:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_final_grid5.py tests/test_joint_grid100.py \
  tests/test_chernov_batch_control.py tests/test_mc_production_x0.py
```
