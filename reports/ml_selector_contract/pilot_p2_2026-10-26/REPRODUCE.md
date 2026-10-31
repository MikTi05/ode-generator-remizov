# P2: воспроизведение и безопасное продолжение

Только 35 оставшихся train-ID из исходного pilot_manifest.csv, профиль MC1,
M=4096, h=1/128, четыре seeds248800020100..103. Максимум140 новых физических
запросов, без повторов. Численные методы и worker/timer относительноP1 неизменны.
Исходные строки execution_authorized=false сохранены как история условного
плана; верхний execution_authorized=true в plan.json фиксирует новый явный
запрос пользователя. Альтернативные MC0/MC2, R0, benchmark1000 и ML запрещены.

Предварительная проверка:
```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_mc_pilot_p2.py tests/test_mc_sentinel_p1.py
```

Контроллер, только для существующего каталога:
```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/mc_pilot_p2.py run --out reports/ml_selector_contract/pilot_p2_2026-10-26
```
После завершения всех140 это проверяет pins и возвращает0 новых вызовов,
без старта worker. При прерывании можно отправить только никогда не отправленные
ключи; все уже запрошенные физические вызовы расходуют потолок140. Утраченный
ответ автоматически не повторяется. budget.json сохраняет исходный deadline,
он не продлевается возобновлением; срок18599с включает паузы.

1 вычислительный worker; все BLAS/Numba threads=1; RSS≤512MiB; watchdog на
каждый вызов взят из исходного job; общий cap18599с=ceil(2×9149.4729+300).
Сценарий9149.47с по измерениямP1 не является гарантией или CI. Состояние worker
обновляется каждые.25с, thermalState каждые10с (2/3 останавливают), guardian
следит за heartbeat контроллера и отдельной lease владельца300с. Агент-владелец
продлевает lease внешними вызовами renew_owner; контроллер сам её не продлевает.
При прекращении связи guardian останавливает worker и освобождает locks.

Отчёты читают только сохранённые массивы, solver/fit не вызывают:
```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/report_mc_pilot_p2.py --out reports/ml_selector_contract/pilot_p2_2026-10-26
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/verify_mc_pilot_p2.py --out reports/ml_selector_contract/pilot_p2_2026-10-26
```
Отчёты создаются эксклюзивно; повторно в тот же каталог не записываются.
Exact используется после остановки таймера; ключевая стоимость включает
prepare_all_methods_mc + MCConfig + solve. Shared loader/JIT и scoring отдельно.
Mean4 только отдельная диагностика с суммой стоимости четырёх вызовов.

Профили Remizov сравниваются раздельно по сохранённым совместимым массивам:
R128 cubic/zero Q513 dx.0125 padding12 common analyticT tail1e-4 (37/40),
R256 тех же настроек n256 (5/40). Недостающие3/35 не запускаются и не заменяются
историческим taper. Времена прежних R не сопоставимы с новым полным таймером;
метка fastest и надёжность≥.95 не объявляются.
