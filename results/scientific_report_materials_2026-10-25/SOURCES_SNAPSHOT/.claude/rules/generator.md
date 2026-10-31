---
paths:
  - "src/**/*.py"
  - "all_methods_dataset_loader.py"
  - "generate_all_methods_dataset_v1.py"
  - "validate_all_methods_dataset_v1.py"
  - "docs/ALL_METHODS_DATASET_V1.md"
  - "tests/test_all_methods_expressions.py"
  - "tests/test_all_methods_dataset_v1.py"
  - "tests/test_all_methods_loader.py"
---
# Генератор all_methods_v1: подтверждённые свойства

## Конструкция
- Manufactured solution: y = scale·φ((x−μ)/R)·T_y(x), φ(z) = exp(1 − 1/(1−z²)) при |z| < 1, иначе 0;
  μ ∈ [−0.25, 0.25], R ∈ [1.5, 3], так что носитель строго содержит [−1, 1].
  scale нормирует max|y| на validation-сетке [−1, 1] к target_amplitude ∈ [0.5, 1.5].
- y0 = y(−1), v0 = y′(−1); g = a y″ + b y′ + c y. Вне носителя y, y′, y″, g ровно 0.
- Производные: рекурсивный аналитический jet (value, d1, d2) в src/all_methods_expressions.py, без конечных разностей.
  Независимый оракул — SymPy в tests/test_all_methods_expressions.py.
- Сертификаты конструктивны и глобальны: a = A_mid + A_half·tanh(T_a), b = B·tanh(T_b), c_R = C·tanh(T_c), λ = C + gap.
  Проверки на сетках (validation-сетка на [−1, 1], 1025 точек на [−10, 10]) — smoke-тесты, не доказательства.
- Грамматика — whitelist _NODE_SPECS (14 глобально ограниченных гладких операций).
  Исполняемые функции восстанавливаются только из serialized_ast; строки формул — описание, не исполнять (eval/sympify).
- Случайность: адресные пути SeedSequence(master_seed, spawn_key=…) + PCG64 (`_stable_seed`, `_rng`),
  без глобального состояния NumPy, `random` и `hash()`; отказ одной попытки не сдвигает другие потоки.
- Split train/validation/test — по семействам, до генерации образцов, стратифицирован по 20 ячейкам
  (140/30/30 при 200 семействах). Семейство = фиксированная топология четырёх деревьев (structure_signature без чисел).
- Версии: GENERATION_VERSION = "all_methods_v1", METADATA_SCHEMA_VERSION = "all_methods_metadata_v1";
  те же константы продублированы в all_methods_dataset_loader.py.
- validate_dataset_artifacts регенерирует набор из seed/config и сравнивает байты; загрузчик импортирует из корня проекта.

## Развитие
- Изменение, которое меняет байты артефактов при тех же seed/config (грамматика, диапазоны, порядок потребления RNG,
  схема CSV/JSONL/NPZ, формулы), оформлять как новую версию: новые константы, имена файлов и описание в docs/.
  v1 должна оставаться читаемой. Существующие v1-артефакты не перегенерировать на месте.
- Константы версий и схем синхронизировать между генератором и загрузчиком.
  Новая операция AST добавляется в _NODE_SPECS, jet, формульную сериализацию и SymPy-оракул теста.
- Новые сертификаты — только аналитически доказуемые на всей ℝ: решатель использует коэффициенты далеко за [−1, 1].
- Проверка: tests/test_all_methods_*.py и validate_dataset_artifacts на малом наборе во временном каталоге
  (--n-families 20 --samples-per-family 1 --ml-grid-size 32 --validation-grid-size 257).
