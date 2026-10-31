# Источники, конфигурации и ограничения

Все проектные пути отсчитываются от корня `ode-generator-remizov`:
`/Users/mikti/Documents/vsc projects/ode-generator-remizov/`.
Копии ключевых документов и исходников — в SOURCES_SNAPSHOT/ с тем же путём;
численные таблицы — в SUPPORT/. Комплект автономен для написания отчёта.
Для полного воспроизведения solver-исследования и inference нужны исходный
проект и модели. Внешние литературные источники здесь не проверялись.
Снимки исторических документов сохраняют прежние внутренние ссылки;
их большие вложения и веса не все включены. Для включённых файлов
ориентируйтесь на SUPPORT/source_manifest.json и явные пути в этом комплекте.

## Карта утверждений

| Материал | Первичный источник и поля | Ограничение |
| --- | --- | --- |
| M1–M4 | chernov_solver.py: `_geometry`, `_apply`, `_at_t`, `_simpson_weights`, `ChernovSolver._run`; remizov_solver.py: `RemizovProblem`, `compute_truncation_T` | Изучаемый solve не Gauss/streaming |
| M5–M6 | mc_solver.py: `_path_contributions`, `_Moments`; mc_ast_kernels.py; mc_optimized.py | EM и две трапеции, без пространственной интерполяции |
| M7 | mc_solver.py: `auto_T`, `_tail_bound`; all_methods_bounds.py: `certify_all_methods_forcing` | Непрерывный хвост, не total error |
| Генератор | src/all_methods_dataset_v1.py: `_sample_construction_parameters`, `_generate_tree_topology`; src/all_methods_expressions.py: `_NODE_SPECS`, `evaluate_solution` | Глобальные bounds конструктивны, сетки — smoke |
| Датасет и примеры | data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1_metadata.jsonl и *_report.json; results/dataset1000_2026-10-18/REPORT.md | Реальные production-примеры; ID малого набора могут совпадать при иных функциях |
| F3 | results/remizov_ml_v1_2026-10-25/targets.csv; `source_result`, `source_result_sha256` указывают на results/chernov_solve1000_2026-10-18/solutions/*.json | R128/taper/legacyT;3 unknown |
| F4 | results/joint_grid100_2026-10-25/analysis/summary.json: `controls`; records/control*.json | Два train-примера, прежний T |
| F5 | results/mc_optimized_backend_2026-10-25/timing_summary.csv, compilation.json, equivalence_summary.json | M1024, x0, три seed, тёплый solve, малый набор |
| F6 | results/mc1000_x0_2026-10-25/analysis/attempts.csv, equations.csv, summary.json | E0;4000 seed не4000 уравнений |
| F7 | results/joint_grid100_2026-10-25/analysis/comparison.csv, summary.json; plan.json | repr80/stress20,12 адаптивных повторов, train-only |
| F8 | results/final_grid5_2026-10-25/analysis/comparison.csv, statistics.json; records/ и baseline/ | Заранее выбранные5, свежие seed, неизвестная h-ошибка |
| F9 | results/remizov_ml_v1_2026-10-25/metrics.json, feature_schema.json, training_protocol.json; results/mc1000_x0_2026-10-25/ml/metrics.json, feature_schema.json, reproducibility.json | Семейный holdout ранее изучался |
| Итоговый ML | results/ml_research_closure_2026-10-25/registry.json, model_audit.json, verification.json | Quality-surrogate, не winner-ML |

SHA-256 и пути включённых копий: SUPPORT/source_manifest.json. Весь большой
набор raw JSON не продублирован: их пути и хеши доступны в
SUPPORT/remizov_targets.csv и SUPPORT/mc1000_attempts.csv.
Все25 финальных ответов и пять R128 baseline включены в SUPPORT/final5_raw/.
Полные исходные строки формул и AST двух примеров — SUPPORT/examples_metadata.json.

## Конфигурации

- **Исторический R1000:** ChernovSolver.solve, n128,Q513, dx0,0125,
  padding12,cubic/taper,quadratic,legacy autoT,tail_eps1e−8.
  n_steps32768 присутствует в config, но относится к другой ветви;
  число шагов данного solve определяется n и Q.
- **MC1000 x0:** Numba, четыре независимых seed ×1024 пути, h_req1/128,
  autoT по аналитическому G,tail_eps1e−4,chunk1024. Ограничения:
  один worker,512 МиБ,60 с на seed,14400 с на серию.
- **Grid100:** R128,Q513,dx0,0125,padding12,cubic/zero,quadratic;
  MC1024 на точку/seed,h_req1/128. Один общий G-T,33 узла. Сначала один
  seed на всех100, затем три дополнительных на12 выбранных train-задачах.
  Общий бюджет14400 с,512 МиБ,один worker; watchdog зависит от стоимости
  полного33-точечного ответа и не равен автоматически60 с.
- **Grid5:** R256 при прежних остальных параметрах и T; baseline R128
  скопирован. MC4096 на точку,chunk1024,h_req1/128,четыре новых seed
  202610250440–443. Общий бюджет1200 с,512 МиБ,один worker;25/25 завершены.
- **Backend benchmark:** малый датасет, x0,M1024,h1/128,три seed,
  T по G округлён вверх до1/8; reference/NumPy/Numba с одинаковыми RNG.
  Подготовка и первая компиляция отделены от тёплого solve.

Полные config: SOURCES_SNAPSHOT/results/remizov_ml_v1_2026-10-25/models/metadata.json;
SUPPORT/final5_raw/*: `job.config`; первичные plan/protocol в каталоге каждой серии.

## Обязательные разграничения для автора

1. E0 и E∞33 различаются;33 узла не доказывают непрерывный sup.
2. Taper/legacyT, zero/G-T и n256 — разные профили. Старую ML-модель нельзя
   переименовывать в модель исправленного cubic/zero.
3. Один seed1024,mean4×1024,один seed4096,mean4×4096 — разные оценки/бюджеты.
4. MC x0 time и R grid33 time несопоставимы как равнозначная работа.
   Сумма solver wall отличается от active/calendar wall. Повторы оплачиваются
   полностью; одинаковые raw runs, вошедшие в single иmean, дважды не суммировать.
5. CI для дискретного усечённого среднего не является CI полной ошибки;
   неизвестное h-смещение нельзя считать нулём. Одновременные bands требуют поправки.
6. Четыре seed не доказывают надёжность95%; пять целевых train-случаев
   не являются несмещённой оценкой для1000 задач.
7. Нет надёжной модели выбора самого быстрого метода. Quality-surrogate
   не прогнозирует оптимальное время достижения допуска.
8. Отсутствие преимущества MC здесь не универсальная математическая теорема.
9. Отдельное связанное уточнение F0002:
   results/mc_f0002_confirmation_2026-10-24/INTERPRETATION.md.
   h1/128,среднее8192: E0≈0,001554,но лишь2/8 отдельных trial точны;
   bias band [−0,01941;0,02252]. Не переносить этот успех на grid33/весь датасет.

## Неточность вторичного обзора

В `results/ml_research_closure_2026-10-25/REPORT.md` написано ускорение
3,71–21,93×. Первичная `timing_summary.csv` подтверждает **3,71–8,31×**.
В этом комплекте использована первичная таблица. Старый обзор сохранён
без изменений; верхнюю21,93 переносить в научный отчёт нельзя.

## Происхождение рисунков

- Рисунок1: SUPPORT/final5_saved_series.json, F0005/S02,все четыре новых seed.
  Показательный успешный пример, не оценка частоты.
- Рисунок2: SUPPORT/grid100_comparison.csv, раздельные repr/stress ECDF.
- Рисунок3: та же таблица и final5_saved_series.json; mean4 показан с полной стоимостью.
- Рисунок4: SUPPORT/remizov_predictions.csv,mc1000_ml_predictions.csv,
  mc1000_attempts.csv. Только test, положительный класс — риск неточности.

Перестроены ровно четыре фигуры из сохранённых массивов. Подписи и оговорки
в FIGURES/CAPTIONS.md. Плотных новых точек exact или новых решений нет.

## Воспроизведение материалов и проверка

Из корня исходного проекта:
`.venv/bin/python -B diagnostics/build_scientific_materials.py`.
Скрипт требует подготовленного нового каталога с protected_before.json;
читает сохранённые ответы, проверяет факторизацию примеров по AST,
строит4PNG/PDF. Solve/fit не вызывает. Производные bump дополнительно
проверены символически; numerical checks коэффициентов не являются
новым численным решением ОДУ. Шаблоны карточек лежат в
diagnostics/scientific_report_templates/ и включены в SOURCES_SNAPSHOT/.

Окружение: SUPPORT/environment.json; проверки: SUPPORT/verification.json.
Сохранность старых файлов проверена по SHA до/после; прежний PROJECT_STATUS
сохранён отдельно. В исходном проекте остаются большие веса/датасеты;
для написания отчёта все нужные факты, источники и рисунки есть в ZIP.
