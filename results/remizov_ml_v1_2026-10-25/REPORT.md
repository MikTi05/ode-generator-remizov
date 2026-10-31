# Первая ML-модель Ремизова — 2026-10-25

**Обучены baseline и CatBoost по сохранённым результатам, без новых решений ОДУ.** Модель прогнозирует качество и стоимость конкретного исторического профиля `ChernovSolver.solve`, а не выбирает между методами и не находит оптимальные параметры.

## Данные и защита от утечки

Использованы существующие1000 уравнений production `all_methods_v1` и1000 первичных JSON. Все997 конечных33-точечных ошибок пересчитаны из сохранённых f/exact.871 ответ достиг .01,126 не достигли;3 worker_exception имеют неизвестные математические цели и исключены из fit/метрик. Они сохранены в targets.csv с null, а не превращены в отрицательные примеры. В регрессию входят все997 завершений, включая126 неточных ответов.

Исходные семейные split не менялись: train700/140семейств (697 известных целей),validation150/30,test150/30. Проверено отсутствие пересечения family_id и structure_signature. Прежние numerical audits уже рассматривали test; это сохранённый holdout текущего обучения, но не совершенно нетронутый prospective test выбора численной политики.

В классификаторе 253 признака: статистики и32 отсчёта входных a,b,c_R,g_R, их вариация/спектр/корреляции, аналитические коэффициентные сертификаты, предвычислимый T исторического профиля, размеры/операции только AST a_raw,b_raw,c_raw. Входной ML-grid256, целевой error-grid33 — разные сетки. c_R=c+lambda,g_R=-g. g — заданная правая часть ОДУ, не точное решение.

Запрещены и не подаются в модель: y_exact/производные/y0/v0, y_shape, solution scale/support/target amplitude, полный tree_metrics/complexity, IDs/family/split/hash/seed, exact-ошибки, solver-time и диагностика решения. Regime/complexity используются только в анализе ошибок. scaler обучен только на train; ни feature importance, ни test-ошибки не использованы для отбора признаков или повторной настройки. [Схема](feature_schema.json),[аудит](data_audit.json).

## Обучение

Один заранее зафиксированный набор настроек, без поиска гиперпараметров: LogisticRegression(C=.1)+StandardScaler; Ridge(alpha=100) для log-времени; CatBoost depth4,learning_rate=.03,l2=10,max1200,early stopping80 по validation,seed20261025,thread_count1. Test не участвует в stopping/threshold/model selection. Модели не переобучались на train+validation после выбора.

Число деревьев: {'classifier_catboost': 262, 'regressor_catboost_coeff': 220, 'regressor_catboost_context': 109}. Полный pipeline занял 9.53с без установки библиотек. Выбор по validation: classifier=catboost,regressor=dummy_context. Native CBM и sklearn joblib сохранены в models/. Версии NumPy2.2.6 и SciPy1.15.3 не менялись.

## Классификация на test

Положительный риск — E∞>.01. Threshold0.5 и порог, выбранный наvalidation по balanced accuracy, показаны отдельно. Выбор порога повышает обнаружение неточных решений ценой отказа от части точных.

| Модель | ROC-AUC | AP неточности | Accuracy при0.5 | Balanced accuracy при выбранном пороге | F1 неточности | Пропущено неточных | Ложно отклонено точных |
|---|---:|---:|---:|---:|---:|---:|---:|
| catboost | 0.8801 | 0.5675 | 0.9133 | 0.8106 | 0.4375 | 2 | 34 |
| dummy_prior | 0.5000 | 0.1067 | 0.8933 | 0.5000 | 0.0000 | 16 | 0 |
| logistic | 0.8839 | 0.6022 | 0.9000 | 0.7565 | 0.4878 | 6 | 15 |

CatBoost: failure threshold=0.077054; обнаружено14/16 неточных, ошибочно принято2, ошибочно отклонено34 точных. Среди102 принятых100 точных — наблюдаемая precision98.04%, **не гарантия**. При стандартном0.5 пороге accuracy91.33%, но пропущено11/16 неточных. Полные вероятности и оба operating point сохранены.

Logistic наtest имеет немного больший ROC-AUC/AP, чем CatBoost; CatBoost выбран поvalidation, не переизбран поtest. Нет оснований утверждать строгое превосходство одного learned baseline над другим по этой малой выборке.

95% family-bootstrap интервалы выбранного классификатора: ROC-AUC [0.7821129236175721, 0.9515515478164732],balanced accuracy [0.7096292175434592, 0.9041340425531915].500 resamples целых30 test-семейств; это условная описательная неопределённость, не независимые150 семейств.

## Регрессия времени на test

Target — сохранённый wall `solver_seconds`, а не полное время worker и не минимальная стоимость .01. Первые700 задач планировались с8 внешними workers, последние300 — с1. Поэтому обучены coefficient-only и context-aware варианты. `planned_external_workers` доступен до вычисления, берётся из замороженных pre-run планов, не из времени/PID/готовых результатов; в классификатор он не входит. Нельзя выдавать этот target за изолированную CPU-стоимость или переносить без проверки на новый hardware/режим.

| Модель | Validation MAE,c | Test MAE,c | Test RMSE,c | Test MAPE,% | Test R² |
|---|---:|---:|---:|---:|---:|
| catboost_coeff | 0.8296 | 0.9150 | 1.1762 | 5.520 | 0.5119 |
| catboost_context | 0.5608 | 0.6519 | 0.8319 | 3.833 | 0.7558 |
| dummy_context | 0.5553 | 0.6767 | 0.8681 | 3.943 | 0.7341 |
| dummy_median | 1.2903 | 1.3792 | 1.7413 | 8.759 | -0.0698 |
| ridge_coeff | 1.0389 | 1.4629 | 1.8348 | 8.844 | -0.1878 |
| ridge_context | 0.6828 | 0.8463 | 1.0748 | 4.983 | 0.5924 |

CatBoost с контекстом имеет наtest MAE0.652с против1.379с глобальной train-медианы. Однако контекстная медиана почти столь же хороша (0.677с) и была чуть лучше наvalidation; **serving сохраняет именно её**, а не выбирает CatBoost поtest. Преимущество CatBoost над глобальным constant в основном связано с известным режимом измерения. Без контекста CB MAE0.915с; важность размеров AST в этом варианте частично отражает совпадение сложности с порядком/режимом исторического запуска, а не причинную стоимость AST.

Парные family-bootstrap95% разности: CB−Logistic ROC-AUC [-0.042694783628298205, 0.044802119951765786]; CB−context-median MAE,c [-0.0905942216289355, 0.03985710540643267]. При включении0 преимущество статистически не разрешено. Выбранная поvalidation контекстная медиана: MAE95% [0.5181140085390459, 0.8464509604821987].

## Важность и ошибки

**classifier_catboost**: cert_remizov_lambda (6.04%), a_d2_rms (5.99%), gR_total_variation (4.95%), gR_rms (4.28%), gR_d1_absmax (2.49%), gR_std (2.21%).

**regressor_catboost_context**: planned_external_workers (81.05%), ast_a_raw_nodes (3.80%), ast_b_raw_nodes (1.85%), ast_c_raw_depth (1.27%), ast_a_raw_depth (1.07%), ast_c_raw_op_tanh_affine (0.75%).

Это PredictionValuesChange, не причинный эффект. Дополнительно сохранена permutation importance групп коэффициентов наvalidation (10 повторов); коррелированные признаки и нарушение физических связей при перестановке ограничивают интерпретацию.

Два false accept выбранного CatBoost-порога: F0046/S01(E=.01749385,p_success=.96622),F0084/S03(E=.01251383,p_success=.92357). Ошибки не использовались для повторной настройки. [Все test-ошибки](test_error_analysis.csv),[подгруппы](test_subgroups.csv),[важность](feature_importance.csv),[перестановки](validation_group_permutation.csv).

В sklearn-baseline зарегистрированы RuntimeWarning от BLAS matmul. Отдельная проверка без BLAS (einsum) дала max разницу logistic probabilities1.23e-15,Ridge seconds1.07e-14; все признаки/прогнозы конечны, независимая max-norm градиента logistic8.84e-5 при tol1e-4,39 итераций. Ошибка результата этими проверками не обнаружена; точная причина предупреждений не установлена. Они не были скрыты и не исправлялись подстройкой метрик. [Аудит](numerical_warning_audit.json).

## Использование

```bash
.venv/bin/python -B diagnostics/predict_remizov_ml.py \
  --models results/remizov_ml_v1_2026-10-25/models \
  --equation-id AMV1_F0000_S00 --planned-workers 8
```

CLI читает только входные коэффициенты/сертификаты; targets/results ему не нужны. Возвращает вероятность достижения .01, risk-decision и прогноз секунд. Модель относится к n128/Q513/dx.0125/padding12/**cubic/taper**, legacy T и33 целевым узлам. Для cubic/zero, иного n/dx/T, MC или произвольного ОДУ она не аттестована. Probability не является математическим сертификатом точности.

Воспроизведение обучения в новом каталоге:

```bash
.venv/bin/python -m pip install -r requirements-ml.txt
.venv/bin/python -B diagnostics/train_remizov_ml.py --out results/remizov_ml_v1_reproduction
.venv/bin/python -B diagnostics/report_remizov_ml.py --out results/remizov_ml_v1_reproduction
```

Для независимого report при воспроизведении нужен аналогичный audit предупреждений; основной train сохраняет модели,метрики,прогнозы и importance без report. Полный environment-lock,исходники и проверки сохранены рядом.

## Артефакты

[features.npz](features.npz),[253 input-features](features.csv),[1000 targets/masks](targets.csv),[1000 predictions](predictions.csv),[metrics](metrics.json),[native/sklearn models](models/metadata.json),[protocol](training_protocol.json).

Графики: [ROC/PR](figures/classification.png),[два порога](figures/operating_points.png),[стоимость](figures/runtime.png),[важность](figures/importance.png); PDF рядом.

## MC отложен

После смены приоритета вычисления не возобновлялись. Pilot остался частичным:236 полных логических ответов,2 операторских остановки guardian,25 неначатых заданий; это не завершённое сравнение. Все артефакты сохранены. Отдельный [технический вариант ускорения MC](MC_OPTIMIZATION.md) использует только уже сохранённые измерения и исходный код; MC не менялся и новый profiling-run не выполнялся.
