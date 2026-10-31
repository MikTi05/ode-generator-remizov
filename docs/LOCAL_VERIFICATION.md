# Проверка локального переноса

Проект: `/Users/mikti/Documents/vsc projects/ode-generator-remizov`. Исходный код: `generator_DE`.
Исходные SHA-256, состав переноса и адаптации записаны в
[MIGRATION_MANIFEST.json](../MIGRATION_MANIFEST.json).
Полные результаты: [local_verification_results.json](local_verification_results.json).

## Перенесённые файлы: 19

- `generate_all_methods_dataset_v1.py`
- `validate_all_methods_dataset_v1.py`
- `all_methods_dataset_loader.py`
- `remizov_solver.py`
- `src/__init__.py`
- `src/all_methods_dataset_v1.py`
- `src/all_methods_expressions.py`
- `create_remizov_benchmark.py`
- `validate_remizov_benchmark.py`
- `remizov_benchmark_loader.py`
- `verify_remizov_solver.py`
- `data/remizov_benchmark.csv`
- `tests/test_all_methods_expressions.py`
- `tests/test_all_methods_dataset_v1.py`
- `tests/test_remizov_solver.py`
- `tests/test_remizov_loader.py`
- `tests/test_remizov_verification.py`
- `tests/test_remizov_benchmark.py`
- `docs/ALL_METHODS_DATASET_V1.md`

## Новые файлы: 12

- `src/problem_types.py`
- `requirements.txt`
- `requirements-dev.txt`
- `.gitignore`
- `pytest.ini`
- `README.md`
- `examples/solve_generated_equation.py`
- `tests/test_all_methods_loader.py`
- `tests/test_all_methods_remizov_pipeline.py`
- `MIGRATION_MANIFEST.json`
- `docs/LOCAL_VERIFICATION.md`
- `docs/local_verification_results.json`

Дополнительно созданы собственное `.venv`, pytest cache и четыре артефакта
малого набора в `data/generated/`. Они исключены из будущего Git через `.gitignore`.
Старые результаты и полный набор из 1000 уравнений не копировались.

## Изменения существующих файлов

- `all_methods_dataset_loader.py`: только импорт `CauchyProblem` из `src.problem_types`.
- `tests/test_remizov_benchmark.py`: удалён импорт и вызов RK4; сохранена
  символическая проверка невязки обычного представления уравнения.
- `src/__init__.py`: обновлено описание пакета.
- `docs/ALL_METHODS_DATASET_V1.md`: обновлены назначение и документация загрузки.

`CauchyProblem` структурно идентичен исходному, включая декоратор, поля и типы.
Математические блоки документации идентичны. Следующие файлы побайтово сохранены:

- `src/all_methods_dataset_v1.py`
- `src/all_methods_expressions.py`
- `remizov_solver.py`

## Проверки и результаты

Окружение: Python 3.10.11, NumPy 2.2.6, SciPy 1.15.3, SymPy 1.14.0,
pytest 9.1.1. Все зависимости установлены только в `.venv` этого проекта.
`pip check`: конфликтов нет. Все 11 проверенных рабочих/проверочных модулей
загружены из этого проекта. Проверены импорты 21 Python-файла;
зависимостей от старого загрузчика, screening и других численных методов нет.

```bash
.venv/bin/python -B -m pytest -q tests \
  -k 'not test_r1_finite_chernoff_approximations_converge_toward_exact'
```

Результат: **70 passed, 1 deselected, 679 subtests passed**, 17.03 секунды.
Два предупреждения относятся к существующей escape-последовательности `\i`
в docstring математического модуля (включая его повторный AST-разбор).
Этот файл не исправлялся ради сохранения исходных байтов.

Один отложенный тест — исследование сходимости R1 при n=1,2,4,8.
Проверки ручных эталонов R1 при n=1,2, задач R1–R5 при n=1, генератора,
аналитических производных, сертификатов и обработки отказов выполнены.
Новые тесты проверяют повреждённые CSV/JSONL/NPZ, самостоятельность импортов,
неизменность математических файлов и CLI с успешным/неуспешным результатом.

Сгенерирован небольшой набор: seed 2488, 20 семейств, одна выборка на семейство,
ml grid 32, validation grid 257. Записаны CSV, metadata JSONL, NPZ и отчёт.
Генераторная CLI и отдельный полный валидатор завершились успешно;
воспроизводимость из seed/config подтверждена. Загружены все 20 уравнений;
CSV и JSONL содержат по 20 записей с согласованными ID.
Максимальная невязка на проверочной сетке: **0.0**.
В малом наборе train=0, validation=10, test=10: это существующее округление
split при одном семействе в ячейке, а не изменение алгоритма.

## Ремизов: статус отдельно от точности

Две разные задачи выполнялись с одним существующим проверочным бюджетом:
n=1, tail_eps=1e-8, nt_initial=8, dx_initial=0.05, лимиты уточнений 8 и 5.
Библиотечные настройки, математический оператор и бюджеты исходных проверок
не изменялись. Скрытых повторов с увеличенным бюджетом нет.

| Задача | Статус | Ошибка max abs на 33 точках |
|---|---|---:|
| AMV1_F0000_S00 | quadrature_not_converged | не определена; итоговый массив пуст |
| AMV1_F0005_S00 | success | 3.0429668353311623 |

У F0000 последняя квадратура Nt=2048; разность уровней около 4.044e-7.
У F0005 final dx=0.00625, final Nt=512; оценки квадратуры около 9.965e-9,
пространственного уточнения около 3.082e-8. Отдельный расчёт F0005 занял
около 7.85 секунды. Успешное уточнение при конечном n=1 не означает малую
ошибку относительно изготовленного точного решения.

```bash
.venv/bin/python examples/solve_generated_equation.py \
  --input-dir data/generated --equation-id AMV1_F0005_S00 \
  --n 1 --points 33 --verification-budget
```

## Состояние исходного проекта

Сверены SHA-256 и набор путей **28450 файлов**, включая прежнее окружение.
Добавленных и удалённых файлов нет. Код, данные, тесты и старое `.venv`
побайтово совпадают со снимком до переноса.

**Обнаружено изменение только служебного `generator_DE/.DS_Store`.** Его размер
не изменился, SHA-256 изменился. Причина не установлена; скрипты переноса этот
файл не записывают. Он не восстанавливался, чтобы не редактировать исходный
проект. Поэтому полную неизменность абсолютно всех файлов утверждать нельзя;
это исключение отдельно сохранено в JSON с обоими хешами.

## Ограничения и следующие решения

Инфраструктурных блокеров переноса нет. Большая ошибка n=1 и отказ F0000 —
наблюдаемые результаты существующего алгоритма, а не повод менять его при
миграции. Отдельного согласования требуют дальнейшие исследования точности
при больших n и математическая интерпретация tail_eps с учётом нормы g_R.

Полная генерация 1000 уравнений, длительные исследования сходимости,
git commit/push и создание репозитория на GitHub не выполнялись.
Старое виртуальное окружение и Git-история не переносились.
