# RK4 и Product Integral: отдельные решатели задачи Коши

Модули находятся в `src/numerical_methods/`. Исходные версии и unit-тесты
сохранены в `migration_input/rk4_product_integral_transfer/`. В копиях solver
изменены только ссылка типа на `src.problem_types.CauchyProblem` и связанные
docstring; в копиях тестов — квалифицированные импорты и пути `mock.patch`.
`tests/test_numerical_methods_integration.py` проверяет эти ограничения
побайтовым сравнением текста после перечисленных замен.

## Использование

```python
from pathlib import Path
from all_methods_dataset_loader import load_all_methods_equations
from src.numerical_methods import (
    solve_problem_rk4,
    solve_problem_product_integral_left_exponential,
)

folder = Path("data/all_methods_v1_1000_2026-10-18")
equations = load_all_methods_equations(
    folder / "all_methods_equations_v1.csv",
    folder / "all_methods_equations_v1_metadata.jsonl",
)
equation = next(e for e in equations if e.equation_id == "AMV1_F0001_S00")
problem = equation.cauchy_problem

for solve in (solve_problem_rk4, solve_problem_product_integral_left_exponential):
    result = solve(problem, n_steps=128, a_min=1e-3)
    if not result.success:
        print(result.failure_reason)
        continue
    index = result.x.index(0.0)  # именно x=0, без ближайшего отсчёта NPZ
    error = abs(result.y[index] - equation.y_exact(0.0))
    print(result.equation_id, result.y[index], error, error <= 0.05)
```

Загрузка всех записей сама по себе не решает все уравнения. Пример решает
только одно выбранное уравнение двумя методами. НУ берутся из сохранённых
`problem.y0`, `problem.v0`, а не восстанавливаются из exact. Коэффициенты —
обычные `a,b,c,g`; преобразовывать их в `c_R,g_R` перед этими методами нельзя.
Исполняемые функции восстанавливает только существующий безопасный загрузчик.

## Контракты и ограничения

- Оба метода решают `a*y'' + b*y' + c*y = g` с заданными `y0,v0`.
  В production-наборе это решение совпадает с ограниченным решением на ℝ,
  поскольку НУ заранее согласованы с точной компактно поддержанной функцией.
  Для произвольной глобальной задачи без НУ методы не заменяют резольвенту.
- RK4 — классическая четырёхстадийная схема четвёртого порядка. Product
  Integral — `Z=(y,y',1)`, `Z[k+1]=expm(h*B(x[k])) @ Z[k]`, матрица 3×3,
  коэффициенты фиксируются слева. При переменной полной матрице это схема
  первого порядка; режим `constant` не означает постоянства forcing `g`.
- Шаг фиксированный, интервал только вперёд; число шагов и `a_min` задаются
  явно. Адаптации шага и сертификата полной ошибки нет. `success` означает
  завершённые конечные вычисления, а не достижение ε=0.05.
- Результаты содержат `x,y,dy`; при отказе сохраняются завершённые точки.
  RK4 для части неверных входов возбуждает исключения, PI возвращает статус.
  Этот контракт отличается от пустого `f` при отказе глобальных solver.
- Основная метрика — `abs(y_numeric(0)-y_exact(0))`. Сетка NPZ из 256 точек
  не содержит нуля. При 128 шагах на исходном `[-1,1]` ноль — узел 64.
- Зависимости уже есть в `requirements.txt`: NumPy и SciPy; RK4 использует
  только стандартную библиотеку. Импорты — квалифицированные, без алиасов
  `rk4_solver`, `product_integral_solver` или `dataset_loader` верхнего уровня.

## Фиксированная интеграционная проверка

Пять заранее выбранных train-задач представляют все режимы и уровни L1–L4:
`F0001/S00`, `F0060/S00`, `F0122/S00`, `F0134/S03`, `F0192/S00`.
Для обоих методов: `n_steps=128`, `a_min=0.001`, `h=1/64`, исходный `[-1,1]`.
Ошибки оцениваются в x=0 и на 33 узлах без интерполяции траектории.
Повторов с подбором шага нет. Два дополнительных ожидаемых отказа проверяют
защиту `a_min=a_upper_cert+1` на исходном первом уравнении; это проверка
контракта отказа, не изменение уравнения и не опыт по точности.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  diagnostics/check_numerical_methods_integration.py \
  --output /tmp/numerical_methods_integration.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_rk4_solver.py tests/test_product_integral_solver.py \
  tests/test_numerical_methods_integration.py
```

Драйвер требует новый путь JSON и не перезаписывает существующие результаты.
Он не подключён к массовым экспериментам или ML. План, результаты и проверки
сохранности интеграции находятся в
[`results/rk4_product_integral_integration_2026-11-01`](../results/rk4_product_integral_integration_2026-11-01/REPORT.md).
