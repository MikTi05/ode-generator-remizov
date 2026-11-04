# Выбор R128 / MC mean4 / PI по сохранённым ответам

Эксперимент реально завершён: **997** полных задач из 1000; победители **R128 826 / MC mean4 106 / PI 65**. Все три класса представлены. Выполнены **120 основных + 200 CV = 320 estimator fit**. Новых численных решений, MC-траекторий и уравнений — **0**.

Основная политика зафиксирована по **latest Validation до Test**: `new__augmented__classifier__w1__medium`, `validation_optimal`, biases `[0.7585894996714374, 1.4368175228112068]`. Test: Accuracy **88.667%**, Macro-F1 **0.3144208**, mean regret **1.6552717**, success **147/150**.

## Данные и численная цель

Для каждого метода E_m=|y_m(0)−y_exact(0)|; exact(0) взят из сохранённого аналитического эталона. Winner — минимальная неокруглённая ошибка, при точном равенстве порядок R128→MC mean4→PI. RK4 отсутствует в метке, моделях и выборе. ε=.05 определяет только success и не меняет winner.

R128 — прежний `ChernovSolver.solve`: n=128, num_t=513, quadratic grading, dx=0.0125, padding=12, cubic/taper, tail_eps=1e−8; ответ f[16] на сохранённой сетке из 33 точек. MC mean4 — среднее четырёх уже сохранённых MC1024 ответов при dt=1/128, tail_eps=1e−4; сначала среднее значений, затем модуль ошибки. PI — прежний левоконцевой expm-метод: 128 шагов на [−1,1], h=1/64, a_min=0.001, ответ y[64]. Полные config/seeds/source paths сохранены в labels.csv/dataset.json.

Сохраняются прежние критерии пригодности: успешный конечный ответ и полная идентичность данных/конфигурации. Большая ошибка не исключает задачу. Три отсутствующих R128 ответа остаются в coverage, без искусственных меток:

| ID | Недоступные методы | Статусы |
|---|---|---|
| AMV1_F0134_S03 | remizov | {"remizov": "worker_exception", "monte_carlo": "success", "product_integral": "success"} |
| AMV1_F0136_S01 | remizov | {"remizov": "worker_exception", "monte_carlo": "success", "product_integral": "success"} |
| AMV1_F0139_S00 | remizov | {"remizov": "worker_exception", "monte_carlo": "success", "product_integral": "success"} |

Крупнейшие пригодные R128-ошибки сохранены без отсечения и без повторного решения:

| ID | Split latest | E_R | E_MC | E_PI | Winner |
|---|---|---|---|---|---|
| AMV1_F0141_S00 | train | 842.08658 | 0.082400837 | 0.068543909 | PI |
| AMV1_F0180_S04 | test | 248.32308 | 0.10308096 | 0.32426872 | MC mean4 |
| AMV1_F0184_S01 | train | 28.394332 | 0.16952346 | 0.02358321 | PI |
| AMV1_F0186_S02 | train | 0.84250182 | 0.42087569 | 0.010173915 | PI |
| AMV1_F0189_S02 | train | 0.23588563 | 0.16088483 | 0.096525426 | PI |

## Разбиения и отсутствие утечки

| Карта | Часть | Задач | Семейств | Истинные R/MC/PI |
|---|---|---|---|---|
| old | train | 697 | 140 | 570/84/43 |
| old | validation | 150 | 30 | 120/15/15 |
| old | test | 150 | 30 | 136/7/7 |
| new | train | 747 | 150 | 608/87/52 |
| new | validation | 100 | 20 | 82/12/6 |
| new | test | 150 | 30 | 136/7/7 |

Сохранены обе прежние карты: original — sensitivity, latest — основной сценарий, назначенный до fit. Прежние 150 Test ID неизменны; семейства, топологии и ID не пересекаются между Train/Validation/Test и внутри CV. Исторический Test уже раскрывался в предыдущих исследованиях: это retrospective exploratory оценка, а не новая prospective независимая проверка. Выбор после просмотра нынешнего Test не менялся.

Признаки точно переиспользованы: 253 из `ml.remizov_features.load_inputs/coefficient_features`, +5 прежних `additional_coefficient_features` (258). Разрешены только заданные a,b,c,g, исходная сетка, пять сертификатов коэффициентов и AST a_raw/b_raw/c_raw. Не используются exact/производные/solution AST, y0/v0, численные ответы, ошибки, success/status/time, winner, ID/family/split как признаки. g — заданная правая часть уравнения, несмотря на manufactured происхождение.

## Обучение и подбор

Seed 2488; один процесс и один поток CatBoost/BLAS. Прежняя сетка CatBoost: 120/depth2/l2=10, 240/depth3/l2=10, 300/depth4/l2=20; lr=.05, bootstrap No, random_strength=0. Классификаторы MultiClass с весами [1,w,w], w=1,2,3,5,8,12; это веса loss, без resampling, изменения ошибок или синтетических классов. Дополнительно прежние raw/asinh идеи: две RMSE-головы Δ_MC=E_R−E_MC и Δ_PI=E_R−E_PI; масштабы asinh — медианы абсолютных Δ исключительно соответствующего Train.

Решение — argmax(log raw probabilities+[0,β_MC,β_PI]) для классификации и argmax([0,predictedΔ_MC,predictedΔ_PI]+[0,β_MC,β_PI]) для регрессии. Bias=0 — default. Классификационная coarse-сетка: 100 пар из прежних τ=0.05…0.50 через log((1−τ)/τ); затем два детерминированных старта и ≤3 coordinate sweep всех границ/обоих nextafter-соседей. Для регрессии coarse-сетка — квантильные границы scores. Это локальный 2D поиск, глобальный оптимум не доказан. Все посещённые политики и validation scores сохранены.

Критерий выбора прежний по смыслу: минимальный Validation mean regret (=минимальная средняя выбранная ошибка при фиксированном oracle), затем больший success, Macro-F1, меньше non-R выборов и стабильный ID. Модель не fit-ится на Validation. Веса/конфигурация/feature set/biases выбираются до Test; latest и original имеют отдельные Validation и не ранжируются по Test.

96 основных обучаемых спецификаций дают 120 физических fit (регрессия имеет две головы); семейная CV даёт 200 fit. Все 320 моделей сохранены с ledger конкретных Train ID. Дополнительные tiny unit-fit не входят в experiment ledger.

## Validation: выбранная модель и default

| Политика | Accuracy | Macro-F1 | Balanced acc. | E≤.05 | Mean E | Median E | Mean regret | Max regret | Выборы R/MC/PI |
|---|---|---|---|---|---|---|---|---|---|
| Выбранная latest политика | 83.000% | 0.43432317 | 41.260% | 100/100 | 0.002873959 | 0.0010460174 | 0.00075184309 | 0.028964329 | 96/2/2 |
| Та же модель, default bias0 | 82.000% | 0.3003663 | 33.333% | 99/100 | 0.0044564517 | 0.0010460174 | 0.0023343358 | 0.15161506 | 100/0/0 |

Выборы по каждой карте до Test:

| Карта | Candidate | Policy | Biases | Val mean regret | Val Macro-F1 |
|---|---|---|---|---|---|
| old | old__augmented__classifier__w3__baseline | validation_optimal | [0.2848367613789384, 0.0] | 0.0017860649 | 0.36864772 |
| new | new__augmented__classifier__w1__medium | validation_optimal | [0.7585894996714374, 1.4368175228112068] | 0.00075184309 | 0.43432317 |

## Test: основная зафиксированная политика

| Политика | Accuracy | Macro-F1 | Balanced acc. | E≤.05 | Mean E | Median E | Mean regret | Max regret | Выборы R/MC/PI |
|---|---|---|---|---|---|---|---|---|---|
| ML latest (выбран до Test) | 88.667% | 0.3144208 | 32.598% | 147/150 | 1.6588505 | 0.0011242778 | 1.6552717 | 248.22 | 146/4/0 |
| R128 / majority Train | 90.667% | 0.31701632 | 33.333% | 147/150 | 1.6589621 | 0.0011065436 | 1.6553834 | 248.22 | 150/0/0 |
| MC mean4 | 4.667% | 0.029723992 | 33.333% | 141/150 | 0.01727538 | 0.0099799563 | 0.013696633 | 0.29387748 | 0/150/0 |
| PI | 4.667% | 0.029723992 | 33.333% | 110/150 | 0.054156321 | 0.018461434 | 0.050577574 | 1.865835 | 0/0/150 |
| Oracle по сохранённым ошибкам | 100.000% | 1 | 100.000% | 148/150 | 0.0035787473 | 0.00095408883 | 0 | 0 | 136/7/7 |

Majority baseline определён только по latest Train; это уже показанный постоянный R128. Oracle недоступен при inference.

На Test найдено **0/14** истинных побед MC/PI. Δmean E (ML−постоянный R128)=-0.00011165296; Δmean E (ML−постоянный MC mean4)=1.6415751. Выбранная политика не превзошла лучший постоянный baseline по mean E — **MC mean4** с mean E=0.01727538. Убедительный полезный перенос не установлен. Выбранная до Test модель и её biases сохранены; более удачный Test-кандидат не подменяет основной результат.

Матрица ошибок: строки — истинный лучший метод, столбцы — выбранный, порядок R128/MC mean4/PI.

| Истина \ выбор | R128 | MC mean4 | PI |
|---|---|---|---|
| R128 | 133 | 3 | 0 |
| MC mean4 | 7 | 0 | 0 |
| PI | 6 | 1 | 0 |

| Класс | Support | Precision | Recall | F1 | Предсказаний |
|---|---|---|---|---|---|
| R128 | 136 | 0.9109589 | 0.97794118 | 0.94326241 | 146 |
| MC mean4 | 7 | 0 | 0 | 0 | 4 |
| PI | 7 | 0 | 0 | 0 | 0 |

Macro-F1 усредняет все три объявленных класса; balanced accuracy — только поддержанные истинные классы. Success — доля выбранных ответов с E≤.05; regret=E_selected−min(E_R,E_MC,E_PI). Правильный выбор winner не гарантирует достижения ε, если все три ответа неточны.

Raw-score calibration выбранной модели на Test: Brier(multiclass sum)=0.17841893, logloss=0.38085113, ECE=0.082172782. Brier/logloss/ECE и OVR AUC остальных классификаторов сохранены в calibration.json; они описательные и не удостоверяют калиброванность probabilities, особенно при class weights. Bias-adjusted scores не выдаются за вероятности.

Все заранее замороженные Test-политики приведены в ALL_EXPERIMENTS.md и test_metrics.csv. Их результаты диагностические; лучший по Test вариант не переобъявляется выбранным.

## Семейная CV и устойчивость

Внутри каждого Train выполнены 5 outer StratifiedGroupKFold; внутри outer Train — первый holdout из 4 inner folds. На inner Validation независимо выбираются baseline 253 спецификации (6 весов +raw +asinh), biases и три константы. Соответствующий outer fit оценивается на outer Validation. Основной глобальный Validation selection в CV не переносится. Inner scores, модели, ID и выбранные политики сохранены.

| Latest OOF policy | Accuracy | Macro-F1 | Mean E | Mean regret | Success | SD fold regret | SD fold Macro-F1 |
|---|---|---|---|---|---|---|---|
| always_monte_carlo | 11.647% | 0.069544365 | 0.018903769 | 0.015634678 | 93.307% | 0.0033851744 | 0.013037887 |
| always_product_integral | 6.961% | 0.043387568 | 0.044765459 | 0.041496368 | 76.439% | 0.0072238093 | 0.006457979 |
| always_remizov | 81.392% | 0.29913899 | 1.1703403 | 1.1670713 | 98.527% | 2.5095338 | 0.004993353 |
| nested_selection | 74.297% | 0.35920208 | 1.1731085 | 1.1698394 | 96.653% | 2.5080992 | 0.033695314 |

Это пять зависимых fold-оценок на имеющихся обучающих семьях; редкие большие ошибки R128 могут доминировать mean regret. SD не является доверительным интервалом нового независимого Test. Train-метрики in-sample и отдельно помечены в train_metrics.json.

## Важнейшие признаки

| Признак | Importance |
|---|---|
| b_rms | 10.359927 |
| b_sample_24 | 5.1472676 |
| cert_remizov_lambda | 4.8474854 |
| a_total_variation | 4.6172713 |
| pre_diffusive_length | 4.270501 |
| gR_negative_fraction | 4.106389 |
| research_diffusion_gradient_ratio | 3.6425364 |
| corr_a_b | 3.1549146 |
| cR_median | 3.0688025 |
| pre_drift_length | 3.0188823 |
| gR_sample_03 | 2.6596725 |
| b_sample_11 | 2.411216 |

Для регрессии показано среднее importance двух голов. Это описание обученной модели, не доказательство причинности и не повод менять признаки после Test. Train-only квартильные распределения сохранены в feature_analysis.json; удаления/добавления признаков по этим результатам не было.

## Проверка, артефакты и воспроизведение

Модель после экспорта перезагружена; выбранные Test-предсказания совпали точно (reload_verification.json). Сохранность прежних файлов проверена в preservation_verification.json. Независимая итоговая сверка: verification.json; необходимые pytest: tests.log. Отчёт и рисунки построены только из сохранённых артефактов, без fit/solver/exact-вызовов.

Основные файлы: [labels.csv](labels.csv), [coverage_all1000.csv](coverage_all1000.csv), [plan.json](plan.json), [splits.json](splits.json), [selection_lock.json](selection_lock.json), [selected_model/registry.json](selected_model/registry.json), [fit_ledger.json](fit_ledger.json), [test_metrics.json](test_metrics.json), [cv_summary.json](cv_summary.json), [ALL_EXPERIMENTS.md](ALL_EXPERIMENTS.md), [figure_manifest.json](figure_manifest.json). Все candidate/inner/outer модели находятся в models/ и cv_models/.

Для воспроизведения нужен новый каталог. Из корня проекта, локальная .venv, PYTHONDONTWRITEBYTECODE=1:

```bash
for stage in prepare train cv final; do
  .venv/bin/python -B diagnostics/run_three_method_selector.py "$stage" --out results/NEW_THREE_METHOD_RUN
done
.venv/bin/python -B diagnostics/report_three_method_selector.py --out results/NEW_THREE_METHOD_RUN
.venv/bin/python -B diagnostics/verify_three_method_selector.py --out results/NEW_THREE_METHOD_RUN
```

Исходники pipeline и whitelisted builders зафиксированы в sources/ и plan.json; входные raw-ответы имеют SHA в dataset.json. Прежние эксперименты/датасет/модели сохранены.

## Ограничения вывода

Результат относится к фиксированным профилям и ошибке значения только в x=0. Он не гарантирует точность производной, выполнение НУ приближением R128 или точность на всём интервале. AMV1 согласует точное решение задачи Коши с глобально ограниченным резольвентным решением; алгоритмы используют разные постановки и бюджеты. Универсальное превосходство метода и перенос на новые независимые семейства этим retrospective опытом не доказаны.
