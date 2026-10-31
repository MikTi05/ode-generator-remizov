# Проверки завершения ML-этапа

Новых ODE/MC solve или fit: 0.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_quality_prototype.py tests/test_remizov_ml.py tests/test_mc1000_quality.py
```

25 passed in 1.91s. Повторная загрузка21 предиктора, все1000 unified inference, все прежние метрики,997 raw R +4000 MC ошибок,7 train-only scaler и независимых non-BLAS checks: model_audit.json.

Сохранены два фактических CLI-ответа: input-only JSON и неподдерживаемые grid33 профили. Проверены все4 графика и локальные ссылки. Старые17049 файлов неизменны. PROJECT_STATUS обновлён отдельно, его прежняя версия сохранена.

В audit.log сохранены известные BLAS RuntimeWarning baseline; независимый einsum совпал с конечными прогнозами (max1.07e-14). Численные регрессии/новые решения уравнений не запускались: схема не менялась, задача запрещает новые расчёты.
