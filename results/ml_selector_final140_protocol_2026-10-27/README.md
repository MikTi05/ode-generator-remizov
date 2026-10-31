# Исполнение финального протокола 140

Сейчас запуск запрещён текущим поручением. Подготовка завершена; бюджет 7464 с
уже выбран для протокола. Следующие команды приведены для единственного
финального исполнения **после отдельного разрешения пользователя**.

Из корня проекта, с локальным `.venv`, без запуска старого 720-controller.
Проверка готовности не вызывает solver/JIT/inference:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/run_final140.py check --out results/ml_selector_final140_protocol_2026-10-27
```

Содержимое `approval_request.json` — запрос с approved=false, не разрешение.
После явной команды старта сохранить отдельный authorization JSON, указав
approved=true, точный SHA `plan.json`, maximum_calls=140, series_seconds=7464
и source_instruction с текстом разрешения. CLI проверит соответствие до запуска:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/run_final140.py run --out results/ml_selector_final140_protocol_2026-10-27 --authorization /absolute/path/to/explicit_start_authorization.json
```

Во время исполнения внешний сопровождающий агент должен не реже чем раз в
300 с (рекомендуется каждые 60 с) проверять прогресс и продлевать owner lease:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/run_chernov_batch.py --output-dir results/ml_selector_final140_protocol_2026-10-27 --keep-alive
```

Эта команда не решает ОДУ: обновляет lease активного guardian. Не запускать
автономный бесконечный self-renewal; прекращение внешнего сопровождения должно
приводить к остановке. Текущая lease хранится в active.json: 300 с.
Оценивать progress.json, thermal.jsonl, workers/*.log и guardian state.
Управляемая остановка:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/run_chernov_batch.py --output-dir results/ml_selector_final140_protocol_2026-10-27 --stop
```

После остановки дождаться finished+cleanup_success у guardian и свободных
глобальных locks. Повторить `check`, затем `run --out …` с уже сохранённым
разрешением. Первоначальный budget.json не удалять и не заменять. Resume
пропускает все reserved requests, включая orphan, независимо от их успеха;
запускает только ещё не запрошенные jobs, если исходный deadline не истёк.
Если deadline истёк, resume запускает 0 workers. Повторные attempts запрещены.
Если обнаружены critical result/несоответствия, продолжение блокируется.

Оценщик читает сохранённые результаты, не запускает solver, inference или fit:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/evaluate_final140.py --out results/ml_selector_final140_protocol_2026-10-27
```

До исполнения он возвращает not_executed. После — пишет analysis.json с точными
знаменателями, unknown/censored и отдельной полнотой основной и MC-оценки.
Если основное сравнение неполно, fixed-N интервалы не строятся. Для полного
основного сравнения нужны 120 оцениваемых R-исходов и учёт всех 140 slots;
полнота MC20 отдельно требует 20 известных исходов.

Научные правила и ограничения: [REPORT.md](REPORT.md),
[analysis_spec.json](analysis_spec.json). Исходный freeze, планы, результаты и
проверенные snapshots не перезаписывать. Production API не разблокировать
автоматически по завершению расчётов.
