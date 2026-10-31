# Воспроизведение

Рабочий каталог — корень проекта; локальная .venv. Полное окружение —
requirements-lock.txt, backend dependencies — requirements-mc-optimized.txt,
ML — requirements-ml.txt. Снимки исходников — sources/, их SHA и исходные
data SHA — plan.json / verification.json / artifact_manifest.json.

Готовая серия полностью завершена. Повторный `run` в её каталоге пропускает
4000 результатов, не создаёт worker и не обновляет бюджет. Проверка сохранена
в resume_check.json. Ниже для нового численного воспроизведения требуется
НОВЫЙ каталог результатов; это отдельные вычисления, здесь повторно не запускались.

```bash
.venv/bin/python -B diagnostics/mc_production_x0.py init --out results/<new-directory>
.venv/bin/python -B diagnostics/mc_production_x0.py smoke --out results/<new-directory>
.venv/bin/python -B diagnostics/mc_production_x0.py run --out results/<new-directory>
```

Во время run внешний владелец продлевает lease (300 с) вызовом
`diagnostics.chernov_batch_control.renew_owner(Path(...).resolve())`.
Контроллер сам свой owner lease не продлевает. Остановка — `request_stop` из
того же модуля. При потере владельца guardian уничтожает worker-группу;
возобновление новым run использует только pending и прежний абсолютный
deadline из budget.json. После14400 с время не добавляется. При resource
failure терминальная запись остаётся, автоматического повторения нет.
Состояние всей серии определяется progress/pending/session, не одним exit code.

Анализ сохранённых ответов, без численного решения:

```bash
.venv/bin/python -B diagnostics/analyze_mc1000_x0.py --out results/<directory>
.venv/bin/python -B diagnostics/report_mc1000_x0.py --out results/<directory>
.venv/bin/python -B diagnostics/verify_mc1000_x0.py --out results/<directory>
```

Воспроизведение обучения требует отсутствующего ml/ в новом каталоге с
копиями зафиксированных plan.json, analysis_protocol.json и analysis/*.csv;
`train_mc1000_quality.py --out <directory>` не перезаписывает существующие
модели. Scaler/train только на исходных train-семействах, stopping/selection
по validation, фиксированный seed и один поток. Для нового обучения сначала
сохранить неизменные targets/features/протокол, затем запускать эту команду.

Существующие выбранные модели можно применять без повторного обучения:

```bash
.venv/bin/python -B diagnostics/predict_mc1000_quality.py \
  --models results/mc1000_x0_2026-10-25/ml \
  --ids AMV1_F0000_S00 AMV1_F0180_S04
```

Предсказания относятся к качеству фиксированных профилей в x=0. Скрипт не
выбирает самый быстрый solver и не сертифицирует полную ошибку .01.
