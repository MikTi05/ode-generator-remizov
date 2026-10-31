# Воспроизводимость sentinel-P1

Состояние: серия завершена; worker не запущен.

- `plan.json` — ровно60 исходных stage1 заданий, исходные seeds/configs, порядок и gates.
- `AUTHORIZATION.txt` — пользовательский объём60MC без R0/P2/ML.
- `sources/` — исполненные версии обёртки, анализа и численного кода.
- `environment.json`, `workers/*.warmup.json`, `hardware_power_observed.json` — окружение и ограничения измерения hardware/power.
- `events/` — все60 requests, `records/` — все60 ответов с f33/exact33; повторный запуск не записывал новых requests.
- `analysis/` — CSV/JSON по отдельным вызовам,15 парам, mean4 и будущей стоимости.
- `REPORT.md` — таблицы и результат frozen gates; `INTERPRETATION.md` — выводы и ограничения.
- `independent_verification.json`, `verification.json` — проверки идентичности, ошибок, источников и завершения.

Численный replay требует нового каталога и отдельного задания: повторить тот же60-call plan на сохранённых seeds/версиях. Не запускать его для обычной проверки отчёта. Сама команда `run` на уже законченном каталоге возвращает `new_calls=0,worker_started=False`.

Для анализа без новых траекторий использовать `diagnostics/report_mc_sentinel_p1.py`: он читает только сохранённые записи. Выход `analysis/` должен быть новым; существующий он отказывается перезаписывать. Для полного воспроизведения скопировать plan/records/sessions/workers/control в новый каталог проекта и запустить report с явным `--out`.

`diagnostics/verify_mc_sentinel_p1.py` независимо восстанавливает exact из канонического AST и пересчитывает60 ошибок без solve. Его итоговый JSON также публикуется без перезаписи. Сохранённые источники и исходные данные проверяются по plan.protected_sha256.

Тесты `tests/test_mc_sentinel_p1.py` используют mocks; они не моделируют траектории и не обучают модели.
