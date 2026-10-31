# Воспроизведение

Локальная .venv, команды из корня проекта. Полный протокол plan.json,
источник — неизменный proposal предыдущего этапа; версии пакетов —
environment.json / requirements-lock.txt. Снимки источников — sources/.

Готовый каталог содержит25 полных ответов. Его повторный run пропускает все
записи, не запускает worker и не сбрасывает deadline (resume_check.json).

Для отдельного нового воспроизведения нужны новый путь и новый бюджет:

```bash
.venv/bin/python -B diagnostics/final_grid5.py init --out results/<new-directory>
.venv/bin/python -B diagnostics/final_grid5.py run --out results/<new-directory>
```

Внешний владелец продлевает owner lease300 с через
`diagnostics.chernov_batch_control.renew_owner(Path(...).resolve())`.
При потере владельца guardian завершает группу worker, незавершённый запрос
остаётся pending. Новый run продолжает только отсутствующие записи; общий
абсолютный deadline1200 с от первого старта сохраняется. Его превышение
не запускает повтор или увеличение ресурсов. Терминальные отказы не
повторяются автоматически. Только25 замороженных jobs на5 train-ID.

Отчёт по существующим ответам без численного решения:

```bash
.venv/bin/python -B diagnostics/report_final_grid5.py --out results/final_grid5_2026-10-25
```

Отчётный файл создан в ходе этой задачи до init и позже уточнялся. Поэтому
его исходный preparation-SHA сохранён в protected_before.json, обе версии
в source_history/, разрешённая версия — analysis_source_history.json.
Контроль отклоняет любые другие изменения. Для нового каталога эта история
не нужна, если его отчётный источник не менялся после init.

Baseline R128 и прежние4×1024 MC — только архивные сравнения, не входят в
подтверждающую выборку. Свежие20 trial имеют M4096; их среднее даёт M16384,
не заменяет результат одного4096-path trial. Новых ML-моделей и расчётов900 нет.
