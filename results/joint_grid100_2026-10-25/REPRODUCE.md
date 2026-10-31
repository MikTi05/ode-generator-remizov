# Воспроизведение зафиксированного pilot

Рабочий каталог — корень проекта, локальная .venv. Схемы исходных solver не
менялись. Окружение закреплено в environment.json и requirements-lock.txt;
снимки исполнявшегося ядра/драйвера — sources/. Plan.json фиксирует выбор,
источники, конфигурации, alpha/looks, cost forecast и SHA до вычислений.

Готовый каталог полностью завершён. Повторный `run` пропускает240 записей,
не сбрасывает абсолютный дедлайн, не запускает compute worker; отдельная
проверка неизменности сохранена в resume_check.json.

Для нового численного воспроизведения требуется отдельный новый каталог:

```bash
.venv/bin/python -B diagnostics/joint_grid100.py init --out results/<new-directory>
.venv/bin/python -B diagnostics/joint_grid100.py run --out results/<new-directory>
```

Драйвер сначала выполняет четыре train-only граничных контроля и проверяет
совпадение с сохранёнными reference-файлами, затем100 основных пар. После
всех пар публикуется immutable adaptive_selection.json и выполняются только
там записанные дополнительные seed. Терминальные отказы автоматически не
повторяются. Прерванный запрос остаётся pending, известные wall/RSS — events/.

Внешний владелец должен продлевать300-секундный lease через
`diagnostics.chernov_batch_control.renew_owner(Path(...).resolve())`.
Контроллер сам owner lease не продлевает. Без действующего владельца guardian
безопасно завершает worker-группу. Для остановки использовать request_stop
из того же модуля; для продолжения — новый run с прежним out и deadline.
При исчерпании четырёх часов pending сохраняются, время не добавляется.

Анализ и проверка готовых данных, без нового решения ОДУ или ML:

```bash
.venv/bin/python -B diagnostics/report_joint_grid100.py --out results/<directory>
.venv/bin/python -B diagnostics/verify_joint_grid100.py --out results/<directory>
```

records/ содержит240 immutable полных ответов; events/ — запросы, один
interrupted event. Analysis/attempts.csv содержит241 физическую попытку,
ключ логической задачи повторяется для прерывания и завершённого исполнения;
использовать вместе с execution_session/source. Comparison.csv содержит
ровно100 разных production-ID, связанные dataset SHA + expression_hash.
Контроли помечены stage=control и не смешиваются с новым common-T профилем.
Статистика — analysis/simultaneous_statistics.json; raw MCResult CI95 остаются
точечными, grid inference использует отдельные33×2 bands.

Mean4 относится лишь к12 целевым повторным случаям. Все100 не считались
приM4096. Training-eligible winner маски оставлены false, нового ML-fit нет.
Проект next_experiment_proposal.json не исполнялся и не является частью
результатов. INTERPRETATION.md содержит научный вывод и его ограничения.
