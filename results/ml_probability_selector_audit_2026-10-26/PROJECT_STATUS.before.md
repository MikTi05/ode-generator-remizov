# Состояние проекта ode-generator-remizov

Обновлено **2026-10-26**: независимая validation-диагностика вероятностного
ML-селектора **завершена — 96/96 вызовов**: 8 исходных validation-семейств,
R128/R256/MC1 ×4. Утверждённый бюджет 5691 с, 96 вызовов максимум,
1 worker/поток, RSS≤512 MiB соблюдён. Фактический wall **2732.966 с (45.55 мин)**,
пик RSS **246.891 MiB**; 10668 RSS-сэмплов, 264 thermal-сэмпла, все state=0.
Guardian/lease, исходные deadlines и безопасное сохранение действовали;
0 retries, execution failures и pending. Completed-resume вернул 0 новых вызовов.

[Отчёт](results/ml_probability_validation_2026-10-26/REPORT.md),
[результаты и метрики](results/ml_probability_validation_2026-10-26/analysis.json),
[все 96 попыток](results/ml_probability_validation_2026-10-26/attempts.csv),
[независимая проверка](results/ml_probability_validation_2026-10-26/independent_verification.json),
[лимиты](results/ml_probability_validation_2026-10-26/resource_verification.json),
[итоговая сохранность](results/ml_probability_validation_2026-10-26/verification.json).

**Основной constant_R256 не превзойдён:** point-политики всех заранее заданных
q=.5/.8/.95 дали **6 точных доставок/8**, **1 неточный ответ/7 выбранных (14.29%)**
и 1 отказ. Constant_R256: **7 точных/8**, 1 неточный, без отказов.
При q=.95 тёплая offline стоимость 119.924 с против 195.127 с (**−38.54%**),
но потеряна одна точная доставка. q=.5/.8: 84.440 с при тех же исходах;
лучший q по outcomes не выбирался. Стоимость включает inference для всех
запросов, включая отказ; загрузка моделей отдельно. Это offline выбор по
замороженным прогнозам и сохранённым медианам, не online dispatch.
Model-spread политика и production research_only guard отказываются на всех
8 задачах: нулевая доставка не подтверждает преимущество по точности.

Подтверждённые ошибки: **F0041/S02** выбран неточный R256 при q=.95
(E=0.01723854, прогноз p=.99217; R128 также неточен), **F0189/S01** отказ
при всех q, хотя R256 точен (E=.00517783, прогноз p=.18916).
F0189/R128 E=.01026625 строго выше .01. R128 точен на 6/8 семействах,
R256 на 7/8; четыре детерминированных повтора не считаются независимыми данными.

**MC1 точен в 4/32 вызовов**, все на F0143/S04; остальные 28 нарушают порог.
4/4 — наблюдение: индивидуальный CP95=[.397635,1], не гарантия точности
нового seed. На F0143 точный R128 12.399 с быстрее MC1 25.309 с;
MC-winner на этих 8 задачах отсутствует, MC ни одной замороженной политикой
не выбран. Поэтому способность получать преимущество именно за счёт MC
остаётся непроверенной. R256 p-head хуже train-constant по Brier/logloss;
median APE времени R128/R256/MC1 =1.37%/2.08%/14.31% (только диагностика 8 семей).

Модели, scalers, q, профильный контекст и прогнозы сохранены по SHA-256;
fit/scaler family overlap пуст. **Нет ML-fit, benchmark1000 или dataset-test**,
численные схемы не менялись. Все 96 ошибок и стоимости политик независимо
пересчитаны; **26 unit/mock/saved-only тестов passed**. Прежние 18328 файлов
проверены, только этот статус изменён с резервными копиями.
Новый код: diagnostics/run_probability_validation.py,
report_probability_validation.py, verify_probability_validation.py;
tests/test_run_probability_validation.py, test_probability_validation_report.py.
Копии исходников, конфигурации, сырые результаты, питание, ресурсы и проверки
сохранены в results/ml_probability_validation_2026-10-26/.

**Следующий шаг:** для новой версии заранее выделить отдельные calibration-
семейства для p_R256/пограничного R128 и неопределённости, добавить реальные
MC-success/failure семейства с независимыми seeds и потенциальной выгодой по
времени. Старый gate20+5 MC-winner не является условием fit вероятностных голов;
эти 8 раскрытых семейств нельзя считать независимыми для следующей версии.
После калибровки и новой заморозки — заранее спланированный нетронутый семейный
тест с достаточным числом MC-выборов. Риск/q не подбирать по текущим outcomes.
Новые расчёты и обучение требуют отдельной следующей задачи.

## Исторический статус до утверждения validation-бюджета

Обновлено **2026-10-26**: validation-пакет96 **не запускался**. Выполнен
технический префлайт, подготовлены полный манифест/jobs/смета и прогнозы.
Причина остановки — явное условие текущего запроса: **нет ранее утверждённого
общего вычислительного бюджета**. Объём96 сегодня запрошен, но предложенный
ниже cap ещё не утверждён; создание сметы не считается согласием пользователя.
Новых solver/MC/benchmark1000 calls и ML-fit **0**.
[Отчёт](results/ml_probability_validation_preflight_2026-10-26/REPORT.md),
[план96](results/ml_probability_validation_preflight_2026-10-26/plan.json),
[префлайт](results/ml_probability_validation_preflight_2026-10-26/independent_verification.json).

8 ID/семейств F0032/F0041/F0069/F0106/F0133/F0143/F0189/F0194 в точности
совпали с prepared design, все original validation; пересечения с fit/scaler
семействами нет.101 frozen ML artifact/54 CBM/scalers и source SHA сохранены,
registry SHA9854a71be60a42f88878da317e02947a9de857fb862e088096661449eb1684d9.
Аналитические G/commonT и configs проверены без solve;32 старых x0 runtime
использованы только для бюджета, не для grid33 accuracy/настройки.
Готовых matching grid33 результатов/requests и конфликтов новых seeds
248800060100..103 нет; locks свободны.96 jobs:32 R128/32 R256/32 MC1,
рабочие лимиты рассчитаны по фактической работе, profiles неизменны.

Сценарий **2695.354с /44.92мин**: R128379.017с, R256771.061с, MC1545.276с.
MC — x0 mean4 runtime×33×4 и сохранённая P2 calibration .99525855;
R — медиана сохранённого full method wall. Это прогноз, не CI/верхняя граница,
будущие setup/import/JIT отдельно не измерены. **Предложен cap5691с
(94мин51с)** =ceil(2×scenario+300),1worker/поток,RSS512MiB,guardian/lease300с,
thermal-stop≥2, без автоматического продления/сброса при resume.
[Бюджет, approved=false](results/ml_probability_validation_preflight_2026-10-26/budget.json).

Прогнозы/действия сохранены до outcomes и независимо воспроизведены.
Point q=.5/.8:7 R128+1 abstain; q=.95:4 R128+3 R256+1 abstain.
Model-spread и production research-only API пока abstain на8/8; это не
доказательство accurate deliveries или выигрыша по стоимости. Ни один q
не выбирался по validation. План оценки фиксирует primary episode0,
MC k/4+Brier/logloss, full median4 cost, нарушения/coverage/абстенцию,
constant_R256 основной и constant_R128 дополнительный baseline; research
96-call cost отдельно от одного запроса политики. Настройка/fit на этих
результатах не разрешаются, test остаётся закрыт.
[Прогнозы](results/ml_probability_validation_preflight_2026-10-26/frozen_predictions.json).

**Следующий шаг: утвердить предложенный бюджет96/cap5691с/RSS512MiB/1worker**
или задать другой явный бюджет; затем подготовить/проверить numerical runner
этого манифеста, повторить актуальные checks и выполнить разрешённый пакет.
Преимущество baseline/ошибки на8 validation пока неизвестны — результатов нет.
8 семейств остаются диагностикой, для окончательной оценки нужны отдельные
calibration/evaluation данные и frozen независимый семейный test.

Новый код diagnostics/plan_probability_validation.py,
verify_probability_validation_plan.py и tests/test_probability_validation_plan.py.
**4 теста пройдены**, независимый audit96 jobs/32 budget sources/frozen predictions
пройден.18,305 прежних путей сохранены, кроме этого статуса. Прежние solver,
модели, policies и экспериментальные результаты неизменны; before/after,
источники и SHA сохранены в каталоге префлайта.

## Исторический статус после вероятностного ML-pipeline

Обновлено **2026-10-26**: реализован и обучен **вероятностный ML-селектор
точность+стоимость**, schema `risk_constrained_typical_cost_v1`, по сохранённым
данным. **Новых численных решений/MC/benchmark1000 —0**. Цель выбора между
Remizov и Monte Carlo сохранена: отдельные p/time-heads обоих методов и общая
политика минимальной типичной warm-стоимости при caller-supplied p≥q.
[Отчёт](results/ml_probability_selector_2026-10-26/REPORT.md),
[описание/API](docs/PROBABILISTIC_METHOD_SELECTOR.md),
[проверки](results/ml_probability_selector_2026-10-26/independent_verification.json).

Достоверные targets: R128103 train-уравнения/75 точных, R25640/33,
MC1 **40 уравнений,160 реальных независимых P1/P2 trials,7 успехов на3 семействах**.
Одна строка equation/profile с реальными k,n; CrossEntropy(target=k/n,weight=n)
равен binomial loss исходов. Синтетических winner-labels/примеров нет;
детерминированные R repeats и MC replay не увеличивают trial count.
Новая группа F0047 не участвует в fit/calibration, хранится отдельным аудитом.
Старые результаты/метки/модели не перезаписывались.

Обучено5 CatBoost-heads: p_R128,p_R256,p_MC1,t_R256,t_MC1. Для t_R128 только
4 семейства с full wall — эмпирическая константа с флагом недостатка данных.
Полные времена доступны для4/36/40 уравнений R128/R256/MC1; solver-only
runtime не подменяет полный timer. У R0 часть time-targets —одиночные full calls,
у MC median4 всех ответов, включая неточные. Исторические cohorts не объявлены
одним подтверждённым deployment-контекстом.
[Targets p](results/ml_probability_selector_2026-10-26/accuracy_targets.json),
[targets wall](results/ml_probability_selector_2026-10-26/full_cost_targets.json).

Сохранены253 input-only признака и legacy-descriptor semantics.1 поток,
фиксированные120 trees/depth2/seed2488 и10 whole-family bootstrap replicas.
3 семейные CV-складки внутри40 paired train-семейств; auxiliary R128103
исключает все held families. Original validation/test outcomes не использовались
для fit, настройки, calibration или оценки; test predictions отсутствуют.

MC probability OOF **Brier .04488 против .04253** у train-fold константы;
logloss .20964/.18889: преимущества не получено. MC cost median APE
**26.29% против51.29%** у константы; сигнал времени обучается.
R128/R256 probability Brier .17720/.10786 против .25250/.15192 у констант,
но условные family-bootstrap intervals разности включают0.
R256 time-регрессия не лучше почти постоянного времени: median APE1.48%/1.41%.
На OOF при q=.5/.8/.95 MC не выбран. При q=.95 тройной point-policy:
15 R128,15 R256,10 abstain;28/30 точных, conditional93.33%, поэтому риск95%
не подтверждён. Полной policy wall/regret нет из-за отсутствующих full costs.
[Все OOF/политики](results/ml_probability_selector_2026-10-26/family_cv.json),
[разброс метрик/calibration bins](results/ml_probability_selector_2026-10-26/metric_uncertainty.json).

На виденном F0047 frozen p_MC=.83779 после fit прежних4/4, отдельные fresh
seeds дали1/4. Это post-hoc known-history seen-family аудит, не слепой holdout.
Bootstrap model spread [.01751,.87107] **не является CI/CP-границей**.
Модель cost_MC13.754с против8.674с измеренных также показывает model bias.
Веса/пороги после аудита не менялись. Реальные методы остаются кандидатами,
но новый registry **research_only_uncalibrated**, deployment action=abstain;
API возвращает прогнозы обеих альтернатив, uncertainty и диагностические решения.
Надёжный оптимальный двухметодный выбор пока не получен.
[Registry](results/ml_probability_selector_2026-10-26/registry.json),
[fresh-seed audit](results/ml_probability_selector_2026-10-26/seen_family_fresh_seed_audit.json).

**Прежний gate20 train +5 validation MC-winner не нужен для fit p/time-heads**:
он остаётся gate прямого winner-classifier. Новые головы учатся на successes/
failures и времени, даже когда MC не выигрывает. Требования готовности —
поддержка outcomes/семейств, независимая calibration при заданном q,
сопоставимость cost и валидация всей политики, затем frozen family test.
Сейчас3 MC-success семейства и0 validation outcomes недостаточны.
Универсальное новое минимальное число не доказано;8 семей для time-regression
в коде — инженерное ограничение, не статистический gate.

**Следующий необходимый эксперимент — независимый validation-пакет**, не
train F0005 для добывания winner. Предложены8 заранее случайно выбранных
исходных validation-семейств F0032/F0041/F0069/F0106/F0133/F0143/F0189/F0194,
по одному ID без просмотра numerical outcomes.4+4 R256/MC1 на ID =64 calls;
для вопроса о самом быстром с учётом R128 добавить4 R128 на ID, **96 calls**.
Модели/q-сетка frozen, новые seeds предложены. Это первый диагностический
пакет, не обещание полной calibration: при отсутствии MC-success сообщить
недостаток поддержки, не добирать seed до удобного класса. Fit/calibration
и её оценка требуют раздельного дизайна. Test не трогать до freeze.
Перед запуском нужны отдельные numerical preflight/seed/work/resource/budget
проверки и разрешение; **пакет не запускался**, execution_authorized=false:
[план](results/ml_probability_selector_2026-10-26/NEXT_VALIDATION.json),
[дизайн](results/ml_probability_selector_2026-10-26/VALIDATION_DESIGN.json).
Предыдущее предложение переключить научную цель на R/abstain заменено этой
двухметодной постановкой; abstain здесь отражает только неопределённость решения.

Pipeline21.12с; **54 теста пройдены** без solve. Независимый verifier проверил
183 accuracy-строки/80 cost-строк,160 уникальных semantic MC-trials,
семейства/scaler means/Brier/361 source SHA и все модели. CBM/scaler reload
даёт идентичные прогнозы;18,197 прежних путей сохранены, кроме этого статуса.
Новый код: ml/probabilistic_evidence.py, probabilistic_selector.py;
diagnostics/train_probabilistic_selector.py, report_probability_diagnostics.py,
verify_probabilistic_selector.py; tests/test_probabilistic_selector.py,
test_probability_run.py; новый docs/PROBABILISTIC_METHOD_SELECTOR.md.
Источники, модели, predictions, targets, before/after статус и SHA сохранены.

## Исторический статус после трёхпрофильного F0047-блока

Обновлено **2026-10-26**: выполнен блок F0047/S00 **4 R128 +4 R256 +4 MC1**,
один worker/cohort; MC-seeds **248800040100..103** новые, зафиксированы до старта.
**Скоростное преимущество MC сохранилось, новая observed-победа — нет**:
primary MC E_grid=.011559691292>.01, точен лишь1/4 новых MC-ответов.
Самый быстрый допустимый профиль нового блока — **R128**, обе новые pair labels
R128/MC1 и R256/MC1 —remizov, sole_episode0_accurate, tie=false.
[Отчёт](results/f0047_three_profile_2026-10-26/REPORT.md),
[сырые12 вызовов](results/f0047_three_profile_2026-10-26/attempts.csv),
[анализ/метки](results/f0047_three_profile_2026-10-26/analysis.json).

Median full warm method wall: **MC1 8.673610с, R12811.678605с,
R25623.146624с**. MC быстрее R128 в1.3465 раза /25.73%, R256 в2.6686 раза;
диапазоны не пересекаются, правило10% по времени выполнено. MAD/median
.656%/.941%/1.611%<20%. R128 E=.001882614416 и R256 E=.000943850740
во всех4, массивы воспроизводимы. MC ошибки по episodes0..3:
.011559691292/.010302801884/.013993653141/.008043264395. Удачный episode3
не заменяет primary.1/4 новых независимых trials: индивидуальный CP95
[.006309,.805880], надёжность≥.95 не подтверждена.

Прежний F0047 timing-блок повторял P1-seeds, поэтому его4/4 не были новыми
accuracy trials. Его метка monte_carlo сохраняет смысл для прежнего episode0
и пары R256/MC1; победа над R128 тогда не измерялась. Старые метки/результаты
не переписывались и не объединялись с новыми seeds для confirmatory CI.
Глобальный train-выбор R256 по покрытию40 не менялся. Исторический gate1+0
сохранён, новая независимая группа не добавила MC-family или устойчивой гарантии.

По сохранённым train-данным повторно проверены **80 R +160 MC1 P1/P2 +20
старых final4096** ответов. Точные MC1 встречаются только на F0047/S00,
F0005/S02,F0136/S01; на всех трёх R128 уже точен. F0005 P1 2/4 и старый final3/4
(описательно5/8), primary P1 неточен; MC median13.806с против R128 native12.077с.
Для10%-победы MC полный R128 должен занять≥15.340с (+27% к native).
Единственный точный вызов P2 — **F0136/S01 episode1**,1/4; primary неточен,
MC median31.103с против R12811.865с и R25624.362с. Его продолжение ради winner
не оправдано. Других случаев одновременно с точными MC1 и выгодной
ориентировочной стоимостью над самым быстрым точным R не найдено.
Exploratory screen minE≤.015 и экономия≥10% новых кандидатов не дал;
это эвристика, не доказательство невозможности других успехов.
[Все40 кандидатов](results/f0047_three_profile_2026-10-26/train_candidates.csv),
[источники/неопределённость](results/f0047_three_profile_2026-10-26/train_candidates.json).

Вероятность полезного продолжения (точный новый primary И сопоставимая победа
над лучшим точным R) по отобранным4-seed выборкам/старым разным таймерам
численно не установлена. Описательные .5/.25 не являются такой вероятностью;
CP после выбора случая/профиля не даёт confirmatory/FWE-вывода. Массовый сбор
MC-класса по этим данным не обоснован, фиксировать отрицательный результат
для устойчивого selector и отдельно планировать семейную проверку R/abstain.

**Необязательный минимальный следующий контроль — только F0005/S02,
4 R128 +4 MC1**, около103.53с метода+setup, предложенный cap508с; приоритет
низкий, ценность — закрыть неопределённость времени, а не ожидаемая MC-победа.
Полный тройной блок там196.25с. Для F0136 минимальные8 вызовов171.87с,
тройной269.32с, не рекомендованы. Предложения **не запускались**:
[точный план](results/f0047_three_profile_2026-10-26/NEXT_EXPERIMENTS.json),
execution_authorized=false. Повтор F0047 до удачного seed недопустим.

12/12 completed,0 отказов/повторов/pending, иных solver-задач/benchmark1000/ML-fit0.
Ядро/численные параметры неизменны; shared imports/load/JIT вне таймеров,
AST/G/T/config/constructor/RNG входят в метод.1worker/поток, RSS512MiB,
guardian/lease300с/thermal-stop; cap657с по сценарию178.1663с.
Method wall173.6331с, controller178.2270с; peakRSS235.40625MiB,
685 RSS/18 thermal nominal. Cleanup/locks/resume0calls проверены.
**54 теста пройдены** без solve/fit; независимый verifier пересчитал12 ошибок/
времён/метки и проверил кандидатов/260 source SHA.18,105 прежних путей сохранены,
кроме намеренного обновления статуса.
[Проверка](results/f0047_three_profile_2026-10-26/independent_verification.json).
Добавлены scripts `f0047_three_profile.py`, `report_f0047_three_profile.py`,
`screen_train_MC_candidates.py`, `verify_f0047_three_profile.py` в diagnostics/
и два tests/test_f0047_three_*.py; источники и before/after статус архивированы.

## Исторический статус двухпрофильного timing-блока F0047

Обновлено **2026-10-26**: подтверждающий timing-блок **F0047/S00 завершён,
ровно4 R256 +4 MC1**, все8 ответов точны, без отказов/повторов/pending.
**observed_fastest_v1 = monte_carlo, tie=false.**
[Отчёт](results/f0047_observed_timing_2026-10-26/REPORT.md),
[метка](results/f0047_observed_timing_2026-10-26/label.json),
[проверка](results/f0047_observed_timing_2026-10-26/independent_verification.json).

Один worker/session/поток, зафиксированные R256/MC1/commonT/grid33, исходные
MC-seeds248800010100..103. Полные method timers: G/T/AST+config/constructor+solve,
shared imports/load/JIT отдельно, exact/scoring/replay вне timer. Численные
методы и прежние результаты не менялись; другие уравнения, benchmark1000 и
ML-fit не запускались. Порядок R/MC определён SHA256(ID|episode)%2 по контракту.
Все R-массивы одинаковы, R/MC воспроизвели прежние массивы тех же параметров/seed.

Медианы **R25623.772812с, MC1 8.681506с**: MC быстрее в **2.7383 раза**,
экономит **15.091306с /63.48%**. Диапазоны23.242298–24.065441 и8.473250–8.790764с
не пересекаются; MAD/median .684%/1.034%<20%. MC median≤.90×R median,
заранее установленное правило10% выполнено. Это операционный вывод для этой
пары/уравнения/окружения, не тест статистической значимости и не перенос на1000.
R E_grid=.000943850740 во всех4; MC episode0=.008585050462,
диапазон4 ошибок .008079956027–.009052895456, все≤.01.
[Все8 результатов](results/f0047_observed_timing_2026-10-26/attempts.csv).

**Надёжность MC≥.95 не установлена.** Перезапущены прежние4 независимых P1-seed,
новых независимых accuracy trials0; нельзя считать это8/8. Исходная4/4 CP-lower95
.397635 сохраняет прежний смысл. Observed eligibility по episode0 не менялась.

Новая копия R256/MC1 labels меняет только F0047/S00 unknown→monte_carlo:
пилот40 **32 remizov,1 monte_carlo,7 none,0 unknown**;
таблица1000 **32/1/7/960**. Все40 известных —40 train-семейств, validation/test
не имеют target. Gate20+5 теперь **1+0**, полноценный ML-селектор не обучался.
[Обновлённая таблица](results/f0047_observed_timing_2026-10-26/selected_labels_1000.json).

Предстартовый cap549с по сценарию124.274862с, RSS512MiB, guardian/lease300с,
thermal-stop≥2; method wall129.480390с, controller134.270720с.
PeakRSS239.796875MiB,516 RSS/14 thermal-сэмплов, thermal nominal;
JIT warmup1.0994с вне таймеров. Guardian cleanup выполнен, locks свободны,
completed resume0 calls/0 startups. **52 теста пройдены** без solve/fit;
независимо пересчитаны8 ошибок/времён, label и изменение только одной строки.
Сохранность18,031 прежних путей проверена SHA, исключение — этот статус.
Новый код: `diagnostics/confirm_f0047_timing.py`, `report_f0047_timing.py`,
`verify_f0047_timing.py`, `tests/test_confirm_f0047_timing.py`; источники и
предыдущий статус архивированы рядом с результатами.

Предыдущий шаг «восемь подтверждающих вызовов» выполнен. **Следующий шаг —
зафиксировать результат текущих бюджетов и принять отдельный научный план:**
сбор ещё19 реальных train- и5 validation-семейств MC-winner с неизменной парой
либо завершение ветки полноценного R/MC-селектора и проверка R/abstain.
Новые seeds одного ID дефицит семейств не устраняют. Подтверждён один реальный
MC-winner; массовый benchmark ради балансировки класса пока не обоснован.

## Исторический статус R0 до подтверждающего timing-блока

Обновлено **2026-10-26**: **R0 завершён —38/38 недостающих клеток**,3 R128 и
35 R256, все конечные. Численное ядро/параметры/старые результаты сохранены;
новых MC, benchmark1000 и ML-fit **0**.
[Отчёт](results/remizov_r0_2026-10-26/REPORT.md),
[проверка](results/remizov_r0_2026-10-26/independent_verification.json),
[ресурсы](results/remizov_r0_2026-10-26/resource_verification.json).

Из42 прежних и38 новых R-ответов получена полная матрица40×2 на исходных
train-семействах. По утверждённому правилу accuracy-only выбран **R256**:
**33/40** точных против **25/40 R128**, E_grid≤.01 на33 фиксированных узлах.
Не выбирать профиль по числу MC-winners. Численные n/Q/T/G/boundary/dx,
идентичности/hash/split и ошибки перепроверены по каноническому exact.

Новая фиксированная пара **R256/MC1**: на пилоте40 **32 remizov,0 monte_carlo,
7 none,1 unknown**. На1000 строках:32/0/7/961;39 достоверных строк —39 разных
train-семейств, validation/test targets отсутствуют. Для диагностической пары
R128/MC1 после3 недостающих ответов:24 remizov,15 none,1 unknown на40.
Both-pass F0047/S00 остаётся unknown по скорости: одного старого solver-only
времени R недостаточно для четырёх полных времён каждого метода в одном cohort.
Sole-accurate/none не требуют сравнительных времён. Старый ML-v2, веса и registry
сохранены; к R256 автоматически не применяются, новое обучение не проводилось.
[Новые метки](results/remizov_r0_2026-10-26/selected_labels_1000.json),
[полный профиль/provenance](results/remizov_r0_2026-10-26/labels_r256_pilot40.json),
[80 R-клеток](results/remizov_r0_2026-10-26/R_observations.csv).

Во время пользовательской паузы guardian остановил worker по owner_lease_expired
после25 completed и одного незавершённого вызова. По CONTRACT §206 и команде
«продолжить» повторён только interrupted F0075/S00 с новым attempt ID, затем12
неисполненных задач. **39 физических попыток:38 completed +1 interrupted**;
старые completed не пересчитывались, исходные plan/results сохранены.
Прежний deadline не продлевался. Операционное дополнение заменило более строгое
самостоятельное ограничение первого драйвера на отсутствие retry:
[resume_amendment](results/remizov_r0_2026-10-26/resume_amendment.json).

1worker/поток, RSS512MiB, guardian/lease300с/индивидуальные watchdog/thermal-stop.
Completed method wall882.9157с; interrupted external wall21.3288с, без
выдуманной частичной численной работы. Два активных controller wall вместе
912.2492с/15.20мин; с паузой1481.828с<исходного cap2103с.
PeakRSS177.390625MiB по3553 сэмплам;90 thermal-сэмплов nominal; guardian cleanup
успешен, locks свободны. Completed work4,784,128 S. Resume после завершения:
0 calls/0 worker-startups. **46 тестов пройдены** без solve/fit; независимый
saved-only verifier пересчитал80 R-ошибок,40 MC episode0 и все1000 labels.
17,876 прежних путей проверены SHA: намеренно изменён только этот статус.

Причина отсутствия MC-класса: MC1 primary точен1/40, все P1/P2 —7/160;
any-seed3/40,4/4 лишьF0047, mean4 отдельно6/40 с суммой4 стоимостей.
Аудит **356 сохранённых grid33 MC-ответов** нашёл16 точных вызовов на трёх
ID F0047/S00,F0005/S02,F0136/S01; на всех трёх точны оба R-профиля.
MC0 совместно joint/P1 —0/156. Реальных accuracy-only MC-winners нет.
Основной кандидат на cost-win **F0047/S00**: MC1 median4 warm8.5216с против
старого R256 solver22.5471с; это разные scope/cohort, метка остаётся unknown.
Удачные дополнительные seed, MC2, historical x0/taper и mean4 не подменяют
MC1 episode0. Наблюдения не доказывают универсального превосходства R.
[Диагностика](results/remizov_r0_2026-10-26/MC_absence_diagnostics.json),
[все архивные кандидаты](results/remizov_r0_2026-10-26/saved_MC_accuracy_candidates.json).

Gate20 train +5 validation MC-winner **0+0**, даже после возможного подтверждения
F0047 будет1+0. Среди уже измеренных40 episode0 больше одного победителя быть
не может. Частоту в популяции по выбранному train-пилоту оценить нельзя.
Сохранённый сценарий1000×4 R256/MC1 **87.04ч** (чувствительность84.91–89.23ч,
не CI/граница, без будущего overhead) не гарантирует нужные классы.
Массовое расширение ради MC-класса сейчас не обосновано.

**Минимальный следующий шаг:** один фиксированный timing-блок F0047/S00,
**4 R256 +4 MC1** в одном cohort с полным method timer, исходными4 MC-seeds
и проверкой воспроизводимости. Сценарий124.27с+setup; блок **не запускался**,
[точное предложение](results/remizov_r0_2026-10-26/NEXT_STEP.json),
execution_authorized=false. Он разрешит последнюю timing-неопределённость.
Затем можно завершить исследование текущих бюджетов отрицательным результатом
для полноценного R/MC ML-селектора; переход к R/abstain требует отдельного
плана валидации. Полноценный selector всё ещё требует реальных20+5 MC-семейств
и итогового семейного test. Предыдущий план «досчитать R0» выполнен.

## Исторический статус ML-v2 до завершения R0

Обновлено **2026-10-26**: реализован saved-only ML-pipeline
**observed_fastest_v1** для фиксированной диагностической пары R128/MC1.
Новых вызовов Remizov/MC и benchmark1000 **0**; численные методы и прежние данные
не менялись. Текущий запрос разрешил ML на сохранённых результатах.
[Итоговый отчёт](results/ml_selector_observed_2026-10-26_v2/REPORT.md),
[руководство](docs/ML_METHOD_SELECTOR.md),
[аудит разметки](results/ml_selector_observed_2026-10-26_v2/data_audit.json),
[независимые проверки](results/ml_selector_observed_2026-10-26_v2/independent_verification.json).

Метки: **22 remizov, 0 monte_carlo, 14 none, 964 unknown** на 1000 уравнениях.
Все 36 известных —36 разных исходных train-семейств; validation/test не имеют
достоверных target. Исходный family/topology split700/150/150 и 140/30/30 сохранён.
Проверены 260 raw-ответов (100 primary R128 zero/commonT и 160 MC1 изP1/P2),
ID+expression_hash, configs/seeds/RNG,33 узла и ошибки по canonical exact.
У sole-accurate и обоих неточных время не меняет label; unknown не обучается
как none. AMV1_F0047_S00: оба episode0 точны, но R имеет одно старое время,
исключающее внешний G/T/factory; label unknown, скорость не домыслена.
Taper/control/x0/MC0/MC2/R256/mean4 не объединялись с выбранной парой.

Повторно использованы **253 прежних input-only признака** без изменения кода,
порядка/version/hash. pre_T/n-derived остаются фиксированными legacy descriptors,
не фактическим T нового профиля. X физически отделён от target/ID/стоимости.
Scaler fit только на labeled train; внутри CV — на train-семействах складки.

Обучен только **диагностический CatBoost remizov/none**:36 реальных задач,
4 train-only семейные CV-складки,200 деревьев/depth3/seed2488/1 поток,
без синтетики и без использования original validation/test.
OOF accuracy **20/36=55.56%** против constant_R **22/36=61.11%**;
macro-F1 по поддержанным классам .5000/.3793. MC support0, recall=null.
Диагностика отказала 10/36, выдала 16 точных и 10 неточных ответов из сохранённых R;
baseline —22 точных и 14 неточных. Преимущество по accuracy не получено.
Family-bootstrap500 OOF:95% interval38.89–72.22%, baseline44.44–77.78%;
это exploratory train-оценка на отобранной выборке, не новый holdout.
[CV/CBM/scaler](results/ml_selector_observed_2026-10-26_v2/diagnostic_cv.json),
[метрики политики](results/ml_selector_observed_2026-10-26_v2/policy_diagnostics.json).

Полный селектор **не обучен**, gate не пройден: MC-winner0 и нет validation labels.
Registry insufficient_evidence; API возвращает abstain/unknown с причиной;
диагностическая модель не подставляется как R/MC selector. Четыре solver seed
и вероятности ML не подтверждают надёжность≥.95. Policy wall/cost regret
остаются null без сопоставимых полных времён; ускорение не заявляется.

Новый код: `ml/selector_labels.py`, `ml/selector_evidence.py`,
`ml/method_selector.py`, `diagnostics/train_method_selector.py`,
`diagnostics/verify_method_selector.py`, `tests/test_method_selector.py`;
добавлено `docs/ML_METHOD_SELECTOR.md`. **62 теста пройдены** без solve,
CBM/scaler reload проверен. Два запуска дали идентичные метки, OOF-предсказания
и вероятности. Первая итерация сохранена в `results/ml_selector_observed_2026-10-26/`,
итоговая — `_v2/`. Источники и исходный статус сохранены рядом с результатами.

Следующий шаг для полноценного ML: зафиксировать окончательную пару/контекст и
собрать реальные достоверные метки в исходных train/validation/test. По gate
спецификации нужны ≥20 train- и ≥5 validation-семейств с MC-winner, примеры R
и нужного none, затем семейный test с явным class support. Для двух точных —
4 сопоставимых полных времени каждого. Если реальных MC-winner не будет,
сохранять задачу R/abstain без создания искусственного MC-класса.
Прежний план R0 (3R128+35R256) остаётся численной работой; сам по себе он не
устранит отсутствие MC-winner и holdout-меток. Новых измерений здесь не было.

## Исторический статус после P2, до реализации ML

Обновлено **2026-10-26**: **P2 завершён — 140/140 новых MC1-вызовов на 35 train-уравнениях,
все конечные,0 отказов/pending/повторов. Вместе с P1 MC1:40 задач,160 вызовов.**
[Отчёт P2](reports/ml_selector_contract/pilot_p2_2026-10-26/REPORT.md),
[сводка](reports/ml_selector_contract/pilot_p2_2026-10-26/analysis/summary.json),
[проверки](reports/ml_selector_contract/pilot_p2_2026-10-26/independent_verification.json).

Профиль и seeds сохранены: MC1 M=4096, h=1/128,4 seed248800020100..103 из исходных
условных P2 jobs; P1 MC1 — 20 вызовов 5 других задач. Перед запуском проверены
идентичности/hash, train/семейства, configs/seeds/G/T/N, все артефакты P1 и budget.
Операционный P2 cap=18599 с основан на сценарии 9149.47 с; потолок 140 соблюдён.
Численные методы не менялись; новых R0/benchmark1000/ML-вызовов 0.

Попадания E_grid≤.01: **P2 1/140; P1 6/20;
вместе 7/160**. Episode0:1/40, хотя бы 1/4:
3/40, все 4/4:1/40. Coverage20:0/80,
stress20:7/80. Медиана ошибки P2 0.045270798,
всех 160 0.041618837. Mean4 отдельно:6/40,
с суммой стоимости 4 вызовов; не подменяет single-call. Это выбранный train-пилот,
P1 участвовал в выборе MC1; точность на 1000 и надёжность≥.95 не установлены.

Сохранённый совместимый Remizov R128 accurate **23/37**,
MC1 episode0 на тех же 37:**1/37**.
R256 точен 5/5 на sentinel; для 3 R128/35 R256 ответов нет. Missing не считается
численной ошибкой. Taper/старые timings не подменяют новый профиль/полный таймер;
fastest/reliable метки пока не подтверждены.

1 worker/поток; сумма MC1-P2 warm AST/G+config+solve **9301.828 с**,
wall **9310.534 с/2.586 ч**;
peak RSS 278.609 MiB<512; thermal{'0': 896};
59,047,329,792 updates. JIT отдельно, внутри каждого solve compilation=0.
Guardian завершён, locks свободны; resume:0 новых calls/startups и неизменные
хеши 140 записей. Power записана до старта; hardware sysctl не объявляется
доказанным. Все прежние файлы, кроме намеренного обновления статуса, сохранены;
старый статус и исходники восстановимы в каталоге P2. 13 mock-тестов пройдены,
160 ошибок сверены с exact; независимо 140 новых ошибок и 5280 RNG-адресов.

Свежий 1000×4 сценарий(не запуск/не CI/не гарантия): MC1≈59.53 ч;
с R128≈73.28 ч(чувствительность 71.16–75.47 ч),
с R256≈87.04 ч; новые R-preparation и будущие overhead не измерены.
[Расчёт](reports/ml_selector_contract/pilot_p2_2026-10-26/analysis/benchmark1000_cost_scenarios.csv).
Следующий рекомендуемый шаг: закрыть 38 недостающих R0 и исследовать причины
ошибок MC1 на train, после этого отдельно решить о профилях и бюджете benchmark.
Полный benchmark и ML не разрешались и не выполнялись.

Предыдущий план продолжить P2 на 35 задачах выполнен; ниже сохранён статус до P2.

## Исторический статус после P1 (до текущего P2)

Обновлено: **2026-10-26**. **Sentinel-P1 завершён: ровно60/60 MC-вызовов,
60 конечных ответов,0 отказов/тайм-аутов/pending. Следующий этап не запускался.**
[Отчёт](reports/ml_selector_contract/sentinel_p1_2026-10-26/REPORT.md),
[научный итог](reports/ml_selector_contract/sentinel_p1_2026-10-26/INTERPRETATION.md),
[машинная сводка](reports/ml_selector_contract/sentinel_p1_2026-10-26/analysis/summary.json),
[все60 вызовов](reports/ml_selector_contract/sentinel_p1_2026-10-26/analysis/attempts.csv),
[15 пар ошибок/вариативности](reports/ml_selector_contract/sentinel_p1_2026-10-26/analysis/errors.csv).

Пользователь разрешил только stage1 на5 train-семействах F0047/S00,F0005/S02,
F0042/S00,F0059/S04,F0125/S04. Сохранены прежние configs/seeds из контракта:
MC0 M1024,h1/128;MC1 M4096,h1/128;MC2 M4096,h1/256,по4 независимых seed.
Новые R0/P2/benchmark1000/ML-вызовы **0**. Численные методы не менялись.

Попадания E_grid≤.01 на33 узлах: **MC0 0/20,MC1 6/20,MC2 4/20**;
точность хотя бы раз достигнута на **2/5** задачах. F0047/S00:0/4,4/4,3/4;
F0005/S02:0/4,2/4,1/4;наостальных трёх задачах успехов нет. Все ошибки —
конечные численные значения, не отсутствие расчёта. Медианы ошибки20 вызовов:
.03813764/.02217078/.01933788. MC1 лучше по заранее выбранному числу успехов,
MC2 — по общей медиане ошибки,но вдвое дороже и с меньшим числом успехов.

M×4 уменьшило медианную ошибку на5/5 иmax-stderr примерно в2 раза,
при примерно4-кратной стоимости. h/2 улучшило медианную ошибку на3/5,
при примерно2-кратной стоимости;монотонность/порядок/95% надёжность не доказаны.
Mean4 сохранён только как отдельная диагностика с полной суммой стоимости:
его успехи не подменяют single-call counts. Даже4/4 уF0047/MC1 дают
индивидуальную двустороннюю95% нижнюю CP-границу .397635,не .95.

Frozen gates применены без изменений:failures0;median20 стоимости относительно
MC0:1/4.00004/7.97291≤8;уMC1/MC2 есть successes. **Выбран кандидат MC1 дляP2**.
Это train-выбор observed-профиля,не подтверждённый reliable_fastest/winner-ML.
Рекомендовано только отдельно разрешённое продолжение35 задач сMC1 для
наблюдаемого benchmark. Сценарий140 вызовов — **9149.47с/2.54ч**,не измерение
и не гарантированный прогноз;предлагаемый cap18599с/5.17ч не принят/не потрачен.
[Сценарии стоимости](reports/ml_selector_contract/sentinel_p1_2026-10-26/analysis/remaining35_cost.csv).

1worker/поток;warm end-to-end AST/G+config+solve **1645.5669с**,wall сессии
**1652.5848с/27.54мин**. JIT/import прогреты отдельно за1.19442с;внутри всех
60 вызовов compilation0. PeakRSS274.4375MiB<512;160 thermal-сэмплов nominal.
Максимальный внутрипарный MAD/median времени0.814%. Guardian завершён,
locks свободны;resume-проверка дала0 новых вызовов и0 worker-startups.
Подробный sysctl hardware-запрос запрещён sandbox;power settings записаны
во время серии,не на старте. Перед P2 фиксировать доступную power-метаинформацию
до запуска;не объявлять аппаратные параметры независимо доказанными.

Новые scripts: `diagnostics/mc_sentinel_p1.py`, `report_mc_sentinel_p1.py`,
`verify_mc_sentinel_p1.py`;новые10 mock-тестов проверили scope/timer/failure/gates
без траекторий. Независимо пересчитаны60 ошибок/идентичностей/параметров,
все16,602,550,272 updates совпали с планом. Сохранность17273 прежних путей:
кроме намеренного обновления этого статуса изменений нет;ядро,данные,модели,
прежние результаты и договорные документы сохранены. Исходный/промежуточный
статус сохранены в каталоге серии.
[Проверки](reports/ml_selector_contract/sentinel_p1_2026-10-26/independent_verification.json),
[итоговая сохранность](reports/ml_selector_contract/sentinel_p1_2026-10-26/verification.json).
Работа остановлена послеP1;35 задач,R0,benchmark1000 иML не выполнялись.

Предыдущий этап **2026-10-25**: **подготовлен контракт ML-селектора и точный план
завершения benchmark №2488; новые численные расчёты и обучение не запускались.**
Следующий шаг предыдущего аудита «зафиксировать контракт» выполнен:
[CONTRACT](reports/ml_selector_contract/CONTRACT.md),
[матрица профилей](reports/ml_selector_contract/PROFILE_MATRIX.md),
[пилот](reports/ml_selector_contract/MC_PILOT_PLAN.md),
[benchmark1000](reports/ml_selector_contract/BENCHMARK_PLAN.md),
[ML-spec](reports/ml_selector_contract/ML_SELECTOR_SPEC.md),
[решения и ответы](reports/ml_selector_contract/DECISIONS.md).

Машинные артефакты: [plan.json](reports/ml_selector_contract/plan.json),
[40 train-ID](reports/ml_selector_contract/pilot_manifest.csv),
[условные задания](reports/ml_selector_contract/pilot_jobs.jsonl),
[недостающие клетки1000](reports/ml_selector_contract/benchmark_requirements.csv).
Планировщик [plan_ml_selector_contract.py](diagnostics/plan_ml_selector_contract.py)
читает сохранённые данные; solver/fit и команды исполнения у него отсутствуют.

В v1 разделены observed_fastest (заранее назначенный episode0 и медианная
стоимость4 вызовов) и reliable_fastest. Рекомендован первый фиксированный
benchmark A; p(success)>=.95 не введён автоматически. Для индивидуальной
95% нижней границы нужны минимум59/59 успехов, при одновременной поправке
1000 задач —194/194, для двух сторон207/207. Четыре повторения такой гарантии
не дают. CI по полной Bernoulli E_grid учитывает и фиксированное h-смещение;
CI дискретного среднего ответа не подменяет вероятность нового вызова.

Новые кандидаты: R128/R256 cubic/zero,Q513,dx.0125,padding12,common analyticT
tail1e-4; MC0/1/2 Numba single1024/h128,single4096/h128,single4096/h256.
Taper остаётся историческим. Выбор по train-точности/стоимости, не по победам
MC или test. Mean4 диагностический; h512/coupling не включены. Более узкое
предложение h512 прежнего аудита заменено этим планом, не считается выполненным.

Точный пилот:40 разных train-семейств,20 coverage-ячеек+20 stress,включая
три worker_exception,два train taper-артефакта и final5. R0 требует3 R128
и35 R256;37/5 совместимых массивов сохраняются. P1:5 sentinel×3MC×4seed=60
новых вызовов; P2:только один выбранный профиль на35 задачах×4=140.
Максимальная активная ветка —200MC+38R;518 JSONL строк содержат альтернативы,
а не разрешение выполнить их все. Все execution_authorized=false.

Новый timer включает G/T/AST каждого метода, JIT вынесен отдельно;1worker/
поток,4 повтора,median/MAD/ranges,ничья10% с заранее заданным приоритетомR.
Только сценарии по сохранённым solver wall: R0+P1≈42.80мин без накладных;
предлагаемый первый compute cap5437с,512MiB,не принят и не потрачен.
Самая дешёвая полная A-пара с4 повторами —≈28.71 solver-ч,не измеренный
прогноз нового benchmark;более дорогие MC могут стоить намного больше.
Ресурсы и цель observed/reliable требуют отдельного решения до будущего запуска.

Проверки:40 train-ID/семейств,все20 ячеек,518 config.validate без solve,
три условные ветки,границы59/194/207,disjoint pilot/final seeds,1000 строк
недостачи;8 машинных выгрузок повторились побайтово в новом каталоге.
[validation_checks.json](reports/ml_selector_contract/validation_checks.json),
[сохранность](reports/ml_selector_contract/verification.json).
Прежние методы,данные,модели,аудит и результаты сохранены; изменён только
этот статус, добавлены документы/планировщик. Работа остановлена без запуска.

Предыдущий этап **2026-10-25**: **завершён локальный аудит готовности ML-селектора
№2488**, без новых solver calls, MC-траекторий, ML-fit или сетевых обращений:
[подробный отчёт](reports/ml_selector_readiness/AUDIT_REPORT.md),
[1000 строк по уравнениям](reports/ml_selector_readiness/audit_by_equation.csv),
[машинная сводка](reports/ml_selector_readiness/audit_summary.json),
[реестр исходных ответов/профилей](reports/ml_selector_readiness/run_evidence.csv).
Воспроизводимый read-only скрипт:
[audit_ml_selector_readiness.py](diagnostics/audit_ml_selector_readiness.py).

CSV/JSONL/NPZ подтвердили 1000 уникальных ID, 200 семейств по 5 вариантов,
seed=2488, splits 700/150/150 строк и 140/30/30 семейств без family/topology
leakage. Важный риск: все 20 ID старого `data/generated/` встречаются в
production, но совпадающих expression_hash нет. Старые малые MC-пилоты нельзя
объединять с production только по ID.

По raw и первичным request/result повторно подтверждены массовые R:
1000 запусков, 997 конечных ответов, 871 точный/126 неточных на grid33,
3 worker_exception (F0134/S03,F0136/S01,F0139/S00), без математических
отрицательных меток для отсутствующей ошибки. MC: 4000 конечных seed-ответов
на всех 1000 ID, но только x=0; 1426 seed-успехов, среднее4 точно на535 задачах.
На общей grid33 есть лишь100 train-пар zero/commonT: R73 точных, первый MC0.
Адаптивные средние MC точны на2; final5 R256 —5/5, MC4096 —5/20 свежих seed.
Они не подменяют массовый профиль и не подтверждают надёжного MC-winner.

**DATA COVERAGE=INCOMPLETE для grid33** (MC нет у900:600train/150val/150test),
при **COMPLETE по наличию запусков в любой пространственной области**.
**READY FOR IMPLEMENTATION=PARTIAL**, **READY FOR TRAINING AND EVALUATION=NO**
для надёжного fastest-селектора и выбора минимального измеренного времени.
Описательно сравнимы100 прогретых solver-пар; внешний MC AST/G preprocessing,
JIT и историческая конкуренция8/1 workers требуют явной политики стоимости.
Консервативные winner-labels: unknown1000; отдельно сохранены наблюдаемые
primary исходы73/0/27/900, которые не являются надёжными целями политики.

253 input-only признака восстановлены точно;7 scaler проверены на train-only.
Сохранённые selected AUC воспроизведены: R33 .8801306, R x0 .8946759,
MCsingle x0 .7350065; это quality-модели, не best_method. Проверены6 хешей
selected артефактов, NPZ replay исходного/Numba MC и точные seed-интервалы final5.
Подтверждена прежняя неточность вторичного speedup3.71–21.93×:
первичный диапазон3.71–8.31×; старые результаты не переписаны.

Проверки:33 ML/serving/protocol теста и17 AST-тестов с56 SymPy-subtests прошли;
новый аудит повторён в новом каталоге,7 основных выгрузок совпали побайтово.
Все17237 прежних файлов проверены: кроме этого намеренного обновления статуса
изменений нет; исходный статус сохранён в каталоге аудита. Подробности —
[verification.json](reports/ml_selector_readiness/verification.json).

Следующий минимальный шаг — отдельным заданием зафиксировать контракт
grid/profile/reliability/cost/unknown и prospective оценки. В отчёте предложена
ограниченная будущая h-диагностика на трёх уже изученных train-ID; она иMC900
**не запускались**. Повторять выполненную final5 серию не требуется.
После аудита работа остановлена; новый селектор не реализовывался.

Предыдущий этап **2026-10-25**: **подготовлены проверенные материалы для научного
отчёта на5–7страниц, сам отчёт не написан**:
[комплект](results/scientific_report_materials_2026-10-25/README.md),
[ZIP](results/scientific_report_materials_2026-10-25.zip),
[формулы и реальные примеры](results/scientific_report_materials_2026-10-25/EQUATIONS.md),
[факты](results/scientific_report_materials_2026-10-25/FACTS.md),
[две таблицы](results/scientific_report_materials_2026-10-25/TABLES.md),
[источники и ограничения](results/scientific_report_materials_2026-10-25/SOURCES.md).

Сверены реализованные знаки c=c_R−lambda,g_R=−g, резольвента, формула
Чернова, **неравномерный Simpson в ChernovSolver.solve**, две трапеции/EM
в MC и аналитический хвост. Не смешивались с другой реализацией Gauss/streaming.
Для production F0005/S02 иF0047/S00 выписаны настоящие коэффициенты,
точное решение и полная факторизованная правая часть; числовые параметры
сохранены с17значащими цифрами. Проверены topology ианалитические формулы
на3078точках каждого примера (maxgразница2.67e-15); производные bump
также проверены символически. Это вычисления заданных функций, не solve ОДУ.

Четыре итоговых рисунка: solution/error,grid33 error distributions,
accuracy/cost иML ROC; каждый PNG400dpi ивекторныйPDF. PDF дополнительно
отрисованы системнымsips ивизуально проверены. Сохранены подписи, точные
таблицы,30финальныхraw/baselineответов, снимки исходников/отчётов, SHA иZIP.
Повторно сверены997rawRошибок и4000MCseedошибок; новыхsolve/MC/ML-fit **0**.

Обнаружена неточность вторичного ml_research_closure/REPORT.md:
диапазон3.71–21.93× неверен по первичному timing_summary.csv. В материалы
включены подтверждённые **3.71–8.31×**; исторический обзор не переписан.
Областиx0/grid33,taper/zero,MCsingle/mean4 истоимости раздельны;
ограниченияwinner-ML сохранены. Следующий шаг — написать краткий научный
отчёт по этому комплекту, без необходимости новых численных расчётов.

Предыдущий этап — **текущий ML-этап завершён как воспроизводимый
прототип качества фиксированных профилей с отказом от неподтверждённого
выбора метода.** [Единый итоговый отчёт](results/ml_research_closure_2026-10-25/REPORT.md),
[научное заключение](results/ml_research_closure_2026-10-25/INTERPRETATION.md),
[таблица экспериментов](results/ml_research_closure_2026-10-25/experiments.csv),
[все ML-метрики](results/ml_research_closure_2026-10-25/ml_metrics.csv),
[аудит моделей](results/ml_research_closure_2026-10-25/model_audit.json).

Новые `ml/quality_prototype.py` и `diagnostics/predict_quality.py` загружают
только прежние validation-selected веса: R CatBoost grid33 cubic/taper,
R CatBoost x0 accuracy/error, MC CatBoost x0 single1024 accuracy и mean4x1024
log-error; время R — историческая context-median для планировавшихся1/8
workers. Реестр закрепляет SHA весов/feature-code, сетки, параметры и цели.
R cubic/zero n128/n256 и MC grid33 single1024/single4096/mean16384 явно
`unsupported_no_trained_model`; x0/taper/бюджеты не переносятся на них.
`method_selection.status=uncertain`,winner=null; времён MC x0 и R grid33
в искусственном рейтинге нет. Probability/10**logerror не являются
сертификатами; prediction intervals и h-bias bound отсутствуют явно.
API принимает заданные a,b,c,g на исходной256-точечной сетке и коэффициентные
сертификаты/AST a/b/c без exact,ID или solver outputs; CLI поддерживает
production IDs и отдельный input JSON. OOD/принадлежность distribution не доказаны.

Повторно загружены **21** сохранённый предиктор, включая baseline;
максимальное отличие прогнозов **3.56e-15**. Сверены все1000 feature-строк
обоих архивов, инвариантность к запрещённой metadata,997 raw R grid/point
ошибок и4000 MC seed-ошибок. Исходные700/150/150,140/30/30 семейств и
topology-disjoint split сохранены; seed-реплики не разделяются. У7 scaler
проверены train-only mean/variance/count. Все сохранённые ML-метрики и
validation-выбор воспроизведены, исходники обучения совпадают с архивными.
Прежние BLAS RuntimeWarning повторились у baseline; независимый einsum
всех7 линейных пайплайнов совпал (max1.07e-14), причина warnings не установлена.
Лог сохранён. Object names в старом MC NPZ не unpickle-ились; используется JSON schema.

Test AUC: R grid33 taper **.8801**, R x0 **.8947**, MC single1024 x0 **.7350**.
MC logloss **.5863** противdummy **.6743**; R x0 **.1182** против **.1775**.
R grid33 runtime выбранная context-median MAE **.6767с**; лучший test CB
не переизбирался по test. MAE log10 error R x0 **.3579**, MC mean4 **.3624**.
Это описательные метрики прежнего семейного holdout; test ранее изучался
численно и не объявляется нетронутым prospective test выбора solver.

Сохранены4 сводных графика, исходники, реестр,1000 новых inference-ответов,
пример входа без exact, окружение, воспроизведение и хеши. **25 тестов прошли**
без решений ОДУ/MC и обучения. Аудит17049 прежних путей: изменений нет;
модели, данные, математические методы и исторические результаты сохранены.

Решение: MC900 и автоматическое повышение M остаются необоснованными;
winner-ML не обучать на нынешних неопределённых/односторонних метках.
Для первоначальной цели нужны конкурентный MC с контролем h/статистики,
прямые сопоставимые grid33 стоимости, достаточные оба класса надёжных исходов,
семейные holdout/calibration данные cubic/zero и новый prospective test политики.
В этой задаче новых solver calls/ML-fit **0**; будущие расчёты не запускались.

Предыдущий этап — **выполнена заключительная диагностика пяти train-
уравнений** из прежнего next_experiment_proposal:
[отчёт и график](results/final_grid5_2026-10-25/REPORT.md),
[итоговые решения](results/final_grid5_2026-10-25/INTERPRETATION.md),
[таблица точности/стоимости](results/final_grid5_2026-10-25/analysis/comparison.csv).
Исторический proposal не переписывался; его план теперь выполнен этой серией.

**25/25** новых полных ответов, pending0: пять ChernovSolver.solve n256 и
четыре свежих независимых Numba MC4096 seed на каждое из5 уравнений.
Общая сетка33 точки, прежний аналитически сертифицированный T (tail1e-4).
R:Q513,dx.0125,padding12,cubic/zero; изменён только n128→256.
MC:M4096 на точку,chunk1024,h1/128; seed202610250440–443. Никакой настройки
по первым результатам. Baseline n128 скопирован побайтово, не пересчитывался;
прежние MC seed исключены из подтверждающей выборки.

Ремизов улучшился на всех5, успехи **2/5→5/5**. E∞ n128→n256:
F0047/S00 **.00188261→.00094385**; F0005/S02 **.00037902→.00019149**;
F0042/S00 **.01366233→.00711630**; F0059/S04 **.01331025→.00532767**;
F0125/S04 **.01870600→.00917467**. Уменьшение ошибки1.92–2.50×;
два n не устанавливают порядок или строгую границу остатка. Эти5 целевых
train-случаев не дают оценки частоты успеха R256 на1000 уравнениях.

MC4096 свежие успехи соответственно **2/4,3/4,0/4,0/4,0/4** — всего5/20.
MC-only при неуспехе R256 **0**. Пять успешных MC-trial действительно быстрее
также точного R256: медианы8.44/13.60 с против22.55/23.18 с на первых двух.
Однако p(success)>=.95 не подтверждена ни на одной задаче. Для первых двух
точные семейные95% CP интервалы p **[.0294,.9706]/[.1109,.9987]**.
Для трёх0/4: **[0,.7341]**, exact binomial p=6.25e-6 отвергает H0:p>=.95
при заранее заданном alpha.01 на задачу (поправка на5 задач).

Увеличение M1024→4096 снизило наблюдаемый stderr примерно вдвое (1.995–2.006×).
Среднее4 свежих trial — отдельный бюджет **16384 пути на точку**: ошибки
**.004468/.004807/.014753/.014586/.042859**. На первых двух этот агрегат точен,
но стоит33.74/54.39 с против22.55/23.18 с у R256; одновременные upper-error
bands **.01410/.01388** всё ещё выше.01. По F0125 глобальный одновременный
интервал ошибки дискретного среднего **[.00736,.08253]** не доказывает
барьер h-ошибки выше.01. Временное смещение не ограничено отдельно.
Сохранены Bonferroni bands на5 задач×5 оценок×33 точки, Hoeffding и seed-t;
асимптотические bands не выдаются за полную сертификацию точности.

Один worker,512 МиБ,жёсткий бюджет1200 с. Фактический wall **628.03 с/10.47 мин**,
peak RSS **281.91 МиБ**. Solver wall **622.81 с**: R117.78 с, все20 MC505.03 с.
Отказов/прерываний нет, budget не расширялся. Worker/guardian завершены,
locks свободны. Resume-проверка:25 неизменных SHA,0 новых solver и worker.

**MC900 с текущим профилем не запускать, M автоматически не наращивать.**
Данные подтверждают уменьшение шума, но не конкурентную высокую надёжность;
на3 задачах даже16384-path средние неточны. Это не универсальный вывод
об отсутствии преимущества MC. Надёжных меток выбора быстрейшего метода
сейчас недостаточно; есть полезные наблюдения риска/качества и стоимости
фиксированных профилей. Следующий ML-шаг — сохранять раздельную постановку
риска/точности/стоимости Ремизова, а к массовой сравнительной разметке
возвращаться после обоснования конкурентного MC-профиля и критерия надёжности.
Новых900, ML-fit или дополнительных численных экспериментов не выполнялось.

**26** тестов планирования/статистики/guardian прошли за12.61 с без новых
решений ОДУ. Все25 метрик/профили/идентичности проверены, baseline сохранён.
Аудит16934 путей: исторические и численные файлы не менялись. Новый отчётный
скрипт этой задачи уточнялся; обе версии восстановимы и SHA проверены в
analysis_source_history.json. Численные workers,схемы и статистические
правила оставались замороженными. Сохранены источники/окружение/манифест.

Предыдущий этап — **завершён первый общий33-точечный pilot на100
train-уравнениях**: [отчёт](results/joint_grid100_2026-10-25/REPORT.md),
[научный вывод и конкретный следующий эксперимент](results/joint_grid100_2026-10-25/INTERPRETATION.md),
[вся сравнительная таблица](results/joint_grid100_2026-10-25/analysis/comparison.csv).

Выбор до вычислений:80 репрезентативных строк (4/20 complexity×regime,
80 разных train-семейств, равная вероятность включения4/35, вес8.75)
и20 стресс-случаев (10 по прежним train R33-ошибкам,5 по стоимости MCx0,
5 по сложности AST); всего88 семейств. Validation/test не использовались
для отбора, нового численного профиля или адаптации. Стресс-часть и12
дополнительно исследованных случаев смещены и анализируются отдельно.

До основных сравнений четыре train-контроля F0141/S00 и F0184/S01 побитово
воспроизвели сохранённые taper/zero f. Переход boundary устранил катастрофы:
E∞19704.50→.0187193 и444.589→.0535681, при прежнем T и прочих параметрах.
Это не гарантия .01. Основной новый профиль: ChernovSolver.solve,
cubic/zero,n128,Q513,quadratic,dx.0125,padding12 и общий с MC аналитический T
по G при tail_eps1e-4. MC — существующий Numba backend,M1024 на точку,
h1/128,chunk1024. Исторические cubic/taper/legacyT результаты не смешивались
с новыми. Exact использован для метрик после solve, не для h/M/T.

**100/100** основных пар завершены. Затем по замороженному правилу выбраны12
неопределённых случаев, каждому добавлены3 независимых seed. Всего **240**
полных результатов:100 основных R,100 основных MC,36 повторов MC и4
граничных R-контроля. Есть один физически прерванный MC-запрос (всего241
попытка исполнения), итоговые partial solver-счётчики неизвестны, wall/RSS
сохранены. Guardian остановил worker по owner-lease во время паузы; команда
«продолжить» возобновила только pending без сброса абсолютного дедлайна.
Готовые записи не пересчитывались. Итог pending=0, ресурсных/численных
отказов нет, worker завершён и locks свободны.

Общий бюджет14400 с, один worker,512 МиБ, обоснованные индивидуальные
watchdog для33 точек (MC минимум180 с, R минимум120 с, зависят от прежней
стоимости); старый60-секундный лимит не переносился. Активный wall
**3561.57 с /59.36 мин**, календарный с паузой **11406.99 с /190.12 мин**,
peak RSS **286.39 МиБ**. Известный solver wall **3474.70 с**; MC первый
проход1833.88 с, повторы349.13 с; R основной1238.33 с, контроли53.36 с.
Медианы прямых33-точечных времён R12.34 с, MC11.94 с; MC максимум102.04 с.
Это время фиксированных профилей, не оптимальное время достижения .01.

Репрезентативные80: R точен **66/80 (82.5%)**, первый MC **0/80**.
Стресс20: R **7/20**, первый MC **0/20**. Primary: R-only73, neither27,
both/MC-only0. После адаптации: R-only71,both2,neither27,MC-only0; final
объединяет88 одиночных ответов и12 средних4×1024, а не100 одинаковых4096.
Оба наблюдаемых MC-успеха уже имеют точный R: F0005/S02 (MC E∞.009038,
стоимость13.938 с vs R12.077 с) и F0047/S00 (.008792,8.812 с vs12.087 с).
Одновременные нормальные upper-error оценки **.024246/.024518** не
подтверждают .01, поэтому F0047 не объявляется надёжным победителем.

Bonferroni учитывает33 точки и2 заранее разрешённых просмотра. Нормальные
bands асимптотические; отдельно сохранены консервативные Hoeffding bands
для ограниченных дискретных вкладов и seed-t диагностика. Для дискретного
среднего12 нормальных интервалов поддерживают превышение .01,88 неопределённы;
поддержанных успехов0. Hoeffding неопределённы100/100. Временное смещение
EM/трапеций и полная точность не сертифицированы;4 seed не подтверждают
p(success)>=.95. Reliable-fastest/ML-winner метки остаются unconfirmed.

**Оставшиеся900 с этим профилем не запускать автоматически.** Первый MC
профиль дал вырожденный класс успеха0/100; 100 train-случаев полезны для
стоимости/ошибок, но не дают надёжных положительных меток выбора MC.
ML не переобучалась. Предложен, но **не выполнен**, следующий контролируемый
эксперимент на5 train-ID (F0047/S00,F0005/S02,F0042/S00,F0059/S04,F0125/S04):
R n256 при остальных прежних настройках и четыре свежих MC4096 запуска
на точку при прежних h/T. Это проверка воспроизводимости уже испытанного
бюджета4096 и дешёвого уточнения R, без автоматического роста M/h.
Прогноз635 с, фиксированный бюджет20 мин, один worker512 МиБ;
[машинный proposal](results/joint_grid100_2026-10-25/next_experiment_proposal.json).

Регрессия **373 passed,1 deselected,679 subtests passed**,44.39 с; исключён
длительный R1. Четыре новых теста проверяют выбор train-only, профили,
поправку33×2 и pooled variance. Сохранность16379 прежних файлов, контроль
профилей, статистики и очистки — verification.json нового каталога.
Численные схемы, защищённое ядро, генератор, датасеты, прежние результаты
и ML-модели не изменены. Исторические этапы ниже относятся к своим профилям.

Предыдущий этап — **завершён массовый MC1000 в x=0 и обучены первые
предварительные модели качества обоих методов**:
[отчёт, таблицы и графики](results/mc1000_x0_2026-10-25/REPORT.md),
[все4000 попыток](results/mc1000_x0_2026-10-25/analysis/attempts.csv),
[пары и статистика по уравнениям](results/mc1000_x0_2026-10-25/analysis/equations.csv),
[метрики ML](results/mc1000_x0_2026-10-25/ml/metrics.json).

Production `data/all_methods_v1_1000_2026-10-18/`: **4000/4000** полных
ответов, **1000/1000** уравнений, **pending=0**. Каждый ответ — Numba MC,
x=0, M1024, h1/128, четыре независимых seed на уравнение, T=None по
аналитическому G с tail_eps1e-4, без индивидуальной настройки. Схема,
RNG и сертификаты не изменены. 50 последовательных партий по20 уравнений,
один compute worker; ограничения60 с/попытку,14400 с/серию,512 МиБ.
Перед массовой серией пройдены два отдельных smoke на production-уравнениях,
проверены атомарная публикация и запрет перезаписи результатов.

Активный wall двух массовых сессий **1716.77 с /28.61 мин**; календарный
от первого старта до завершения с учётом паузы **1875.67 с /31.26 мин**.
Сумма solver wall **1633.25 с**, peak worker RSS **285.06 МиБ**,
в завершённых результатах **12 138 684 416** обновлений. Ресурсных и
численных отказов нет. После2493 ответов guardian остановил worker по
истечении owner lease во время подготовки анализа. Незавершённый запрос
остался pending и возобновлён в новой сессии с прежним общим дедлайном;
2493 завершённых записи не пересчитывались. Частичные итоговые solver-счётчики
этого запроса недоступны, известные wall/RSS сохранены в events. Проверка
resume после завершения:4000 неизменных SHA,0 новых попыток,0 worker-startups.
Все guardian завершены, locks свободны. Старый прерванный pilot не затронут.

Сопоставлены **997** корректных исторических `ChernovSolver.solve`:
production SHA + expression_hash, implementation, обе сохранённые сетки,
единственный узел x=0 (index16), совпадение exact. Три старых worker_exception
остаются без R-цели. Это сравнение в одной точке, не прежнего E∞33.
Допуск .01: MC отдельный seed **1426/4000 (35.65%)**; среднее четырёх seed
**535/1000 (53.50%)**; Ремизов **930/997 (93.28%)**. В парной выборке
MC-aggregate **533/997**, разность accuracy MC−R **−.39819**, family-bootstrap95
**[−.42708,−.36377]** (условно по наблюдённым seed). Медианы ошибок:
R **.001186**, MC-aggregate **.009188**. Редкие R-выбросы сохранены:
максимум **842.087**, test F0180_S04 **248.323**; MC-aggregate максимум **.569764**.
4/4 успехов MC лишь56 уравнений; даже4/4 дают одностороннюю нижнюю95%
границу p≈.473, не .95. Pooled stderr4096, межseed sd и t95(df3) сохранены
отдельно. Временное смещение полной ошибкой не сертифицировано.

Сравнительных exploratory меток слишком мало: MC mean closer7, R closer60,
uncertain930. У train только5 MC-меток против44 R, поэтому заранее заданный
гейт не пройден; **comparative classifier не обучался**. Вместо него
обучены Dummy/Logistic/CatBoost вероятности успеха каждого метода отдельно
и Dummy/Ridge/CatBoost log10-error. MC probability — успех одного1024-path
seed; MC error-regression — среднее4×1024. Это разные цели, не fastest solver.

253 input-only признака прежнего whitelist, идентичная прежней матрица X;
exact/y_shape/решения/ошибки/IDs/seed/split не входят в признаки.
Исходные700/150/150 строк и140/30/30 семейств сохранены; seed-реплики целиком
в subset семейства. Fit/scaler на train, stopping/selection на validation,
test не настраивал параметры. Pipeline **13.81 с**,12 моделей сохранены,
воспроизводимый код и serving `diagnostics/predict_mc1000_quality.py`.

Выбраны CatBoost. Test MC (600 seed-меток150 уравнений30 семейств):
ROC-AUC **.7350**, logloss **.5863** против Dummy **.6743**, Brier **.2015**
против **.2405**. R x0: AUC **.8947**, logloss **.1182** против **.1775**,
Brier **.0296** против **.0395**; всего6 неточных test-ответов, ранговые
метрики и recall предварительны. Family-bootstrap95 для разности logloss
CB−Dummy: MC **[−.1251,−.0515]**, R **[−.1219,−.0184]**. Это поддерживает
улучшение относительно постоянной базы при данных seed, не подтверждает
95% надёжность solver. Test MAE log10-error CB: MC **.3624**, R **.3579**.
R-выброс248.323 классификатор отметил риском (p(success)≈.242), но регрессор
дал лишь≈.0368; величину редкой катастрофы модель не воспроизводит.

Все модели повторно загружены и сверены с предсказаниями,18 наборов метрик
пересчитаны; logistic проверен независимым einsum. Прежние BLAS matmul
RuntimeWarning повторились в sklearn, конечность и reload/einsum проверены,
причина предупреждений не установлена. Регрессия **369 passed,1 deselected,
679 subtests passed**,44.65 с; исключён только длительный R1. Защищённое
ядро, существующие методы, генератор, датасеты и исторические результаты
сохранены; подробные SHA/сохранность — verification.json нового каталога.

Получены данные и рабочие предварительные модели качества в x=0. Метки
самого быстрого метода не создавались: MC x0 wall и исторический R33 wall
неравнозначны. Следующий шаг к полноценному сравнению — отдельный общий
33-точечный профиль с одновременной статистической политикой и прямыми
измерениями обоих методов. Успех x=0 на E∞33 не переносится. Новой массовой
33-точечной серии в этой задаче не было; прежние планы ниже исторические.

Предыдущий этап — **реализован и проверен отдельный оптимизированный
backend Монте-Карло**: [отчёт, таблицы и графики](results/mc_optimized_backend_2026-10-25/REPORT.md),
[API и ограничения](docs/MC_OPTIMIZED.md). Техническое предложение из
`results/remizov_ml_v1_2026-10-25/MC_OPTIMIZATION.md` выполнено в части
value-only/fused AST и Numba; пространственная интерполяция не вводилась.

Новые `mc_optimized.py` и `mc_ast_kernels.py` используют существующие
RemizovProblem/MCConfig и аналитический сертификат G, сохраняют
Фейнмана–Каца, Эйлера–Маруяму, обе трапеции, устойчивую статистику и адресный
PCG64. Коэффициентные AST без ненужных jets; необходимые производные y_shape
для заданной g_R сохранены. Коэффициенты в узле вычисляются совместно,
a/b переиспользуются на следующем шаге. Numba — один поток, fastmath=False,
нормальные векторы генерируются NumPy в прежнем порядке, блоки по 64 шага.
Фабрика принимает безопасную all_methods_v1 metadata; произвольные callbacks
остаются у исходного MC. Доступен fused NumPy без optional Numba.

Короткое cProfile подтвердило затраты coefficient/solution jets: inclusive
evaluate_jet2 около 39/69/87/91% на F0000/F0002/F0014/F0017. Затем отдельная
непрофилированная серия: M=1024, h=1/128, x=0, три независимых seed на задачу,
общие сертифицированные T с хвостом 1e-4. Медианы эталон→Numba:
F0000 **0.2301→0.0620 с (3.71×)**; F0002 **4.1259→0.7304 с (5.65×)**;
F0014 **2.7180→0.4065 с (6.69×)**; F0017 **14.9279→1.9351 с (7.71×)**;
F0018 **6.1021→0.7345 с (8.31×)**. Без JIT fused NumPy даёт 1.50–3.31×.
Подготовка с G 0.38–1.78 мс; первый импорт/JIT отдельно **0.947 с** на процесс.
Пять полных replay-серий на одинаковых нормалях: max разность траекторных
вкладов **4.0e-15**, средних **2.23e-16**; допуски зафиксированы до измерений.

На настоящей сетке 33 точки (один seed, M=1024 на точку) Numba:
F0000 **2.132 с**, F0002 **24.640 с**, F0017 **63.941 с**; отношение к x=0
33.0–34.4. Наблюдаемые E∞ соответственно **.02433/.04809/.09752**:
все расчёты завершены, ни один не достиг .01. Ускорение не контролирует
временное смещение и не уменьшает статистический шум. Constant y=.5:
ошибка **1.0116e-5** при h=1/128,T=8, детерминированный дискретный оракул
совпадает до округления. Gaussian x=0: ошибка **.004104**, CI±.008526;
аналитическое дискретное смещение/усечение +.001868 отделено от шума.
Его диагностический T=4 не удовлетворяет tail_eps1e-4 (bound .000671),
что явно сохранено. Полный допуск .01 по этим CI не сертифицируется.

Серия заняла **238.16 с**, один worker, 804.5 млн обновлений, process peak
**202.34 МиБ**. Независимый guardian, owner lease300 с, memory limit768 МиБ;
без 60-секундных лимитов попытки и без автоматических повторов. Worker
завершён, locks свободны. Регрессия **363 passed, 1 deselected, 679 subtests
passed** (46.51 с), включая 38 новых тестов; исключён только длительный R1.
Проверены **8071** прежних файлов — изменений нет. Ядро из
MIGRATION_MANIFEST, эталон mc_solver.py, генератор, загрузчик, данные,
Ремизов и исторические результаты сохранены. NumPy2.2.6/SciPy1.15.3 прежние,
optional Numba0.65.1/llvmlite0.47.0 закреплены в requirements-mc-optimized.txt.

Для всех 1000 выполнен только аналитический preflight, **0 численных
решений**. M1024×33, h1/128, один seed: **100.396 млрд обновлений**,
max T125.375, max N16048. Сценарный прогноз **3.63 часа**, ориентировочная
чувствительность **1.55–6.96 часа**, не доверительная/верхняя граница.
75 AST сложнее измеренного диапазона; 345 задач превышают default100 млн
max_path_steps. Четыре seed дают примерно14.5 ч центрального сценария;
8 seed×2048 —58 ч. Это стоимость фиксированного профиля, не достижения .01.
Новые массовые MC1000 и обучение ML не запускались.

Следующий шаг — использовать новый backend явно и с сохранением его SHA в
контролируемом pilot точности/надёжности. Инфраструктура ускорена и проверена;
политика M/h/seed для достоверных ML-меток выбора метода пока не установлена.
Прежние выводы связанного F0002 о неподтверждённой полной .01 сохраняются.
Исторические планы и результаты ниже относятся к предыдущим этапам.

Предыдущий этап — **обучена первая ML-модель качества Ремизова**
по существующим 1000 уравнениям и сохранённым `ChernovSolver.solve`:
[отчёт, метрики и графики](results/remizov_ml_v1_2026-10-25/REPORT.md),
[модели и serving-метаданные](results/remizov_ml_v1_2026-10-25/models/metadata.json),
[план ускорения MC без новых запусков](results/remizov_ml_v1_2026-10-25/MC_OPTIMIZATION.md).

253 признака построены только из заданных a,b,c,g, коэффициентных сертификатов,
предвычислимого T и AST a/b/c. Exact, его производные, y_shape, metadata
solution scale/support, IDs, split, solver-ошибки/время/диагностика в признаки
классификатора не входят. Исходные train/validation/test: 700/150/150 строк,
140/30/30 семейств. Из 1000 строк 997 имеют цели, 3 worker_exception оставлены
неразмеченными и не считаются математическими неуспехами. Известные train-цели:
697; validation и test по 150. Точность относится к конечному ответу на 33 узлах,
историческому cubic/taper n128/Q513/dx.0125/padding12 и legacy T.

Обучены Dummy, Logistic, Ridge и CatBoost, один фиксированный набор настроек,
seed=20261025, один поток. Scaler и fit только на train, stopping/threshold/model
selection на validation; test не использовался для настройки. По validation
выбран CatBoost-классификатор (262 дерева). Test ROC-AUC **0.8801**,
AP неточности **0.5675**; при стандартном пороге accuracy **0.9133**.
При выбранном на validation пороге риска 0.07705: balanced accuracy **0.8106**,
обнаружено **14/16** неточных, пропущено 2, ложно отклонено 34 точных.
Logistic test AUC **0.8839**, AP **0.6022**; family-bootstrap не подтверждает
строгое превосходство CatBoost. 95% AUC-интервал CatBoost примерно [0.782,0.952].
Модель — прогноз, не сертификат точности и не предсказатель worker failures.

Runtime-target — исторический solver wall: первые 700 задач планировались
при 8 внешних workers, последние 300 — при 1. Для стоимости обучены варианты
с коэффициентами и с известным до запуска planned_external_workers; это не
PID/наблюдённое время. CatBoost с контекстом: test MAE **0.6519 с**, R² **0.7558**;
глобальная train-медиана: MAE **1.3792 с**, контекстная: **0.6767 с**.
По validation чуть лучше контекстная медиана, поэтому serving использует её;
CatBoost-регрессоры также сохранены. Разница CB и контекстной медианы не
подтверждена парным family-bootstrap. Важность контекста у CB около 81%:
не выдавать модель за изолированную CPU-стоимость или переносимый прогноз
всех 1000 задач при одном worker. Исторический test уже изучался в численных
аудитах; исходный split сохранён, prospective независимость не заявляется.

Данные/модели/код: features.npz/CSV, targets.csv с масками, predictions.csv,
CBM/joblib, exact requirements-lock, feature schema, importance и анализ
ошибок в новом каталоге. Обучение всего pipeline заняло около **9.53 с**
без установки зависимостей. NumPy 2.2.6 и SciPy 1.15.3 сохранены; добавлены
ML-зависимости через requirements-ml.txt. 8 тестов утечек/меток/split/serving
прошли без численных решателей. Повторно проверены 997 старых ошибок и
8011 защищённых файлов, изменений нет. BLAS RuntimeWarning в sklearn
проверены независимым einsum: прогнозы совпадают до 1.3e-15/1.1e-14,
все конечны, logistic-градиент соответствует штатному tol; причина предупреждений
не установлена. Итоговые SHA/метрики: verification.json в каталоге ML.

По смене приоритета **дорогой MC-pilot отложен и не завершён**. Сохранены
236 полных логических ответов, 2 операторские остановки guardian и 25 ещё
не начатых заданий (ещё одна логическая попытка осталась прерванной).
Обе прежние сессии завершены, worker отсутствует, compute-lock свободен.
[Исходная сессия](results/joint_pilot_2026-10-25/summary.json),
[продолжение](results/joint_pilot_2026-10-25/recovery_01/summary.json).
Эти частичные данные не добавлялись к обучающим 1000 примерам.
В исходниках выявлены ненужные derivative jets при value-only вычислении
коэффициентов и повторные a/b/c внутри g_R. Рекомендуется отдельный
value-only/fused AST nopython-backend после малого профилирования;
табуляция коэффициентов требует самостоятельного контроля новой
пространственной ошибки и точного fallback вне таблицы. Сейчас MC и схемы
не менялись, новых MC/ODE-запусков или массовых решений не было.

Следующий шаг — использовать сохранённый baseline как прогноз риска
данного профиля; улучшения сравнивать по неизменному семейному протоколу,
не подбирать модель по test. Для переносимой модели времени нужны
однородные измерения; их запуск в эту задачу не входил. Ускорение MC пока
остаётся техническим предложением, массовый расчёт MC1000 не запускается.

Предыдущий этап: подготовлен **проект совместного pilot Ремизов / MC**
без новых решений ОДУ, benchmark или ML:
[протокол и решения перед запуском](results/joint_benchmark_plan_2026-10-25/PLAN.md),
[G, T, операции и прогнозы всех 20 задач](results/joint_benchmark_plan_2026-10-25/RESOURCE_FORECAST.md),
[структуры результатов и ML-метки](results/joint_benchmark_plan_2026-10-25/RESULTS_AND_LABELS.md).
Предложены все 20 задач в x=0: MC с M=2048 на 8 независимых seed и
3 замера Ремизова; затем 6 задач на 33 узлах: MC с M=1024 на точку
и 4 другими seed, 3 замера Ремизова. MC: h=1/128. Ремизов: именно
ChernovSolver.solve, n=128, Q=513, dx=.0125, padding=12, cubic/zero.
Обеим ветвям предлагается общий аналитический T для хвоста 1e-4,
округлённый вверх до 1/8. Это новые профили; прежние 20/20 и 871/1000
не являются их измерениями. Прогноз: 4.450 млрд обновлений MC,
около 54.4 мин ядерного времени до накладных расходов. Предложен бюджет
90 мин, по 45 мин на фазу, 60 с / 384 МиБ на попытку, один worker.
Перенос скорости F0002 на другие AST — сценарий, не измерение или граница.
Полная сеточная серия всех 20 вместе с точечной оценивается в 104.5 мин
до накладных расходов и в рекомендуемый pilot не включена.
Пересчитаны 997 сохранённых ошибок на 33 точках из 1000 первичных записей:
871 с E∞≤.01, 126 выше порога, 3 прежних worker_exception; новых решений нет.
Аудит идентичности: 20 общих строковых ID двух наборов, но 0 общих
expression_hash; JOIN требует dataset SHA + ID + expression hash.
В малом наборе нет train: 10 validation и 10 test. Он остаётся диагностическим;
исходные split сохранены. В production — 140/30/30 семейств train/validation/test.
Прежние аудированные test-результаты не являются нетронутыми для выбора
численной политики. Даже 8 успехов MC из 8 дают одностороннюю нижнюю
95% границу p≈.688; для подтверждения p≥.95 нужны минимум 59/59 без
поправки на множество профилей. Pilot не обещает таких меток: unknown,
цензурирование и недостигнутый допуск сохраняются; training_eligible=false
по умолчанию. Этот план не подтверждает точность .01 новых профилей.
Проверки: 10 чистых тестов планирования без численных решателей,
пересчёт прежних ошибок и итоговая SHA/ссылочная проверка в каталоге плана.
Production-методы, генератор, данные и все исторические результаты сохранены.
Следующий шаг — принять перечисленные решения о реализации и профилях,
бюджете, надёжности и доступных признаках; затем отдельно подготовить
и заморозить исполняемый план с явными dataset/target_x и отдельным
exact-evaluator. Нынешний proposal не разрешает запуск.

Предыдущий этап: завершена ограниченная независимая проверка F0002
после связанного пилота: [отчёт и три графика](results/mc_f0002_confirmation_2026-10-24/REPORT.md),
[научный итог и стоимость](results/mc_f0002_confirmation_2026-10-24/INTERPRETATION.md).
Новый контроль 8 seed×1024 на паре 1/32,1/64 прошёл прежние критерии:
относительная t95-полуширина 0.20234<=0.5, выигрыш дисперсии 19.3917>=2.
После отдельного плана выполнена тройка 1/32,1/64,1/128 на других
8 seed×1024; старые пилоты и новые фазы не объединялись. Везде x=0,T=63.625.
Обе парные разности тройки положительны в одновременных 95% интервалах.
На 1/128 наблюдаемая ошибка агрегированного ответа 0.001554, но интервал
среднего смещения [-0.019410,0.022518] не подтверждает устойчивый допуск 0.01.
Порядок не установлен. Статистический CI±0.005 на 1/128 ориентировочно стоит
751.54 млн обновлений на точку; 33 точки — 24.80 млрд для точечных CI
или 64.95 млрд для совместного покрытия, при одинаковой дисперсии как в x=0.
Это стоимость только статистики, не гарантия полной ошибки. Большие сценарии
не запускались; ограничение текущего MC-профиля зафиксировано, ресурсы
автоматически не увеличивать. 16/16 попыток успешны, 166.79 млн обновлений,
77.99 с, RSS 81.5 МиБ. Регрессия: 291 passed, 1 deselected, 679 subtests.
Production-код, генератор, данные и все прежние результаты сохранены.
Следующий шаг — учитывать этот результат в отдельном плане сравнения методов;
ML и массовые/сеточные расчёты не выполнялись.

Предыдущий этап: выполнен отдельный связанный пилот временного
уточнения MC на F0002: x=0, сертифицированный T=63.625, h=1/32→1/64,
4 независимых seed×1024 пути. [Отчёт и три графика](results/mc_f0002_coupled_h_2026-10-24/REPORT.md),
[выводы и причина остановки](results/mc_f0002_coupled_h_2026-10-24/INTERPRETATION.md).
Парная разность 0.020955±0.013290 (t95), корреляция 0.94634,
выигрыш в дисперсии разности ≈18.46. Смещение на h=1/64:
0.012472±0.019710; точность 0.01 и практический шаг не подтверждены.
Относительная полуширина разности 0.634 превысила объявленный порог 0.5:
подтверждение и h=1/128 не запускались, 8 попыток not_run. Это более
строгий критерий точности, чем обнаружение ненулевой разности; t95 уже
исключает ноль. Реально 25.018 млн обновлений, 12.318 с, RSS 79.72 МиБ;
отказов/повторов нет. Регрессия: 283 passed, 1 deselected, 679 subtests.
Предложенный тогда контроль той же пары и переход к трём уровням выполнены
новой независимой серией выше; увеличение M само по себе не устраняет смещение.
Production MC, Ремизов, генератор, данные и прежние результаты сохранены.

Предыдущий этап: завершена контролируемая диагностика Монте-Карло
по M/h/T в одной точке x=0: два аналитических примера и F0000–F0002.
[Таблицы и пять графиков](results/mc_convergence_2026-10-24/REPORT.md),
[анализ и рекомендации](results/mc_convergence_2026-10-24/INTERPRETATION.md).
295/295 заранее объявленных попыток успешны, 8 независимых seed на каждую
стохастическую конфигурацию; 223610232 обновления, 45.60 с всей серии,
пик RSS 98.91 МиБ. Статистические полуширины согласуются с M^(-1/2).
F0002 имеет существенное временное смещение: при h=1/32 и сертифицированном
T=63.625 среднее отклонение 0.05503±0.01678 (t95, M=2048, 8 seed),
хвост <=9.833e-5; увеличение только M недостаточно для 0.01.
Порядок на F0000/F0001 не установлен из-за шума. Следующий шаг — отдельный
контроль временного смещения и стоимости выбранных профилей в нескольких
точках; полная сетка, сравнение методов и ML пока не запускались.
Основная регрессия: 268 passed, 1 deselected, 679 subtests; финальные
7 тестов протокола/анализа прошли отдельно. Схемы, генератор, данные,
Ремизов и все прежние результаты сохранены.

Предыдущий этап: добавлен автоматический аналитический сертификат
G>=sup|g_R| для all_methods_v1: [доказательство](docs/ALL_METHODS_FORCING_BOUNDS.md),
[полная таблица 20 задач, горизонтов и стоимости](results/forcing_bounds_2026-10-24/survey/REPORT.md).
CLI и сравнительный MC worker теперь выводят G из существующей AST.
При tail_eps=1e-4 получены G=25.1183…5397.5704, T=8.3357…63.5599;
обзор не моделировал траектории. Финальная регрессия: 262 passed,
1 deselected, 679 subtests passed. Генератор, данные, Ремизов и прежние
результаты сохранены. План изолированных h/T/M выполнен малой диагностикой
ниже; полная точность и политика сравнения методов остаются открытыми.

Предыдущая интеграция Монте-Карло на RemizovProblem:
[API и ограничения](docs/MONTE_CARLO.md),
[отчёт пяти проверок](results/monte_carlo_integration_2026-10-24/REPORT.md).

Предыдущий численный этап: завершён [один контроль временной квадратуры F0180/S04](results/chernov_time_quadrature1025_2026-10-24/REPORT.md):
513 → 1025 узлов при прежних n=128, padding=12, dx, T и cubic/zero.
E∞ уменьшилась с 0.07045188952 до 0.07044935797; D_Q=0.0001354983,
около 0.192% прежней ошибки. Квадратура по этому сравнению не выглядит
главным источником остатка; сходимость и истинная ошибка не установлены.
Следующий минимальный контроль предложен только для n=256 при num_t=1025;
он не выполнен. Production-код, данные и прежние результаты сохранены.

Предыдущий [контроль расширения окна](results/chernov_window_padding24_2026-10-24/REPORT.md)
padding 12 → 24 дал разность ответов 1.08e-13 при прежней E∞≈0.07045;
дальнейшее расширение окна имеет низкий диагностический приоритет.

Непосредственно предшествующая [граничная диагностика четырёх задач](results/chernov_boundary_diagnosis_2026-10-24/REPORT.md)
побайтово воспроизвела baseline: cubic/zero устранил три катастрофических выброса,
но оставил ошибки 0.01872–0.07045. Контроль сохранился побайтово, linear/zero
оказался менее точным на всех четырёх задачах. Та серия включала 12 решений.

Предыдущий этап **2026-10-20** — [научно-технический отчёт](results/remizov_technical_report_2026-10-20/REPORT.md)
по двум реализациям, ускорению, 126 превышениям и трём историческим
worker_exception. На том этапе выполнялся только анализ сохранённых данных.

Серия **ChernovSolver.solve на 1000 задачах завершена**:
**1000 терминальных записей**, **997 конечных численных ответов**,
**871 с E∞≤0.01**, **126 выше порога**, **3 прежних worker_exception**, pending=0.
Каталог ответов: `results/chernov_solve1000_2026-10-18/solutions/`.
Отчёт/таблица/сверка: `results/chernov_managed_completion_2026-10-18/final/`.
Последние 300 задач выполнены автоматически: 15 последовательных партий по 20,
между ними паузы ≥60 с; 238 достигли порога, 62 выше порога, новых отказов нет.
Прежние 700 итогов не пересчитывались и побайтово сохранены. Данные и численное
ядро не менялись. Монитор завершился с кодом 0; процессов расчёта нет,
backend-блокировки свободны. ML, streaming и другие методы не запускались.

`diagnostics/chernov_monitored_batch.py --run --all-pending --batch-size 20`
включает управляемую серию с одним worker, guardian и телеметрией каждые 2 с.
В этом явно разрешённом режиме внешний монитор раз в 20 с продлевает
120-секундный lease только для живого принадлежащего ему worker/guardian/session
после свежей здоровой телеметрии. Просроченный lease не восстанавливается.
При потере/зависании монитора продления прекращаются; guardian независим.
Тепловая/процессная проблема останавливает всю серию. Режим одной малой партии
по-прежнему не продлевает разрешение. Инструкция: `docs/CHERNOV_BATCH_EXECUTION.md`.

45 коротких тестов прошли за 14.84 с, без численных решателей. Основной запуск
занял **5292.375 с (88.21 мин)**, включая **845.484 с** пауз; наблюдался максимум
один worker, CPU до 100.2% одного ядра, RSS до 179.5 МиБ; все 2572 сэмпла nominal.
Температура в °C не измерялась; пользователь согласовал macOS thermalState
вместе со своим наблюдением за нагревом. Протоколы/резервные копии/журналы:
`results/chernov_managed_completion_2026-10-18/`, основной `managed_run_v2/`.
Первая управляющая попытка `managed_run/` остановлена без новых ответов:
исправлено повторное полное сканирование результатов для каждого pending ID;
обе версии сохранены, добавлена регрессия ограничения сканирований.
Следующий шаг отдельно определяется пользователем; три прежних отказа
автоматически не повторять, обучение ML не начинать.
Предыдущая работа — подготовка и проверка production
all_methods_v1: 1000 уравнений / 200 семейств в
`data/all_methods_v1_1000_2026-10-18/`, отчёт — `results/dataset1000_2026-10-18/`.
На этапе подготовки численное решение этих 1000 задач не запускалось;
оно начато следующей задачей, описанной в разделе текущего пакета ниже.
Предыдущая полностью завершённая серия решения — обе схемы Chern на всех
20 уравнениях с едиными параметрами: `results/chernov_dataset20_2026-10-18/`.
Адаптация/первое сравнение и прежние серии 2026-10-12 сохранены.
Документ сводит README, документацию, код, тесты и артефакты; результаты прошлых
запусков отделены от проверки файлов 2026-10-17 и от ещё не выполненных планов.
Постоянные инструкции находятся в [AGENTS.md](AGENTS.md).

## Откуда продолжать

### Завершение ограниченной проверки F0002 — 2026-10-24/25

Источники: [полный отчёт](results/mc_f0002_confirmation_2026-10-24/REPORT.md),
[краткий научный итог](results/mc_f0002_confirmation_2026-10-24/INTERPRETATION.md),
[предварительная политика](results/mc_f0002_confirmation_2026-10-24/policy.json),
[план контроля](results/mc_f0002_confirmation_2026-10-24/control/plan.json),
[решение о переходе](results/mc_f0002_confirmation_2026-10-24/gate.json),
[отдельный план тройки](results/mc_f0002_confirmation_2026-10-24/refinement128/plan.json),
[ошибки/CI всех запусков](results/mc_f0002_confirmation_2026-10-24/attempt_levels.csv),
[ковариации и агрегаты](results/mc_f0002_confirmation_2026-10-24/aggregates.json),
[сценарии стоимости](results/mc_f0002_confirmation_2026-10-24/cost_scenarios.csv),
[регрессия](results/mc_f0002_confirmation_2026-10-24/pytest.txt),
[сохранность и хронология](results/mc_f0002_confirmation_2026-10-24/final_verification.json).

- Новые `diagnostics/mc_f0002_confirmation.py` и отчётный скрипт
  переиспользуют прежние coupled_paths и production-ядро без изменений.
  До запуска зафиксированы 8 новых seed×1024 пути для контроля пары и
  независимые 8×1024 для возможной тройки. Все seed отличаются от прежних,
  включая ранее не выполненные попытки. Старые выборки не входят в статистику.
- x=0,T=63.625 фиксированы, хвост <=9.832333e-5. Контроль дал парную
  разность 0.02099403±0.00424796 (t95), относительную полуширину 0.202341
  и Var reduction 19.3917. Прежние пороги <=0.5 и >=2 сохранены и выполнены.
  После записи gate.json отдельный plan.json тройки записан до её worker.
- В независимой тройке средние смещения 0.03435729/0.01389283/0.001554095
  для h=1/32,1/64,1/128. Парные разности в одновременных Bonferroni95
  интервалах 0.02046446±0.00693182 и 0.01233874±0.00690880 положительны.
  Это статистическая поддержка временного уточнения на испытанном диапазоне;
  условие — приближённая нормальность средних по независимым seed.
- Различать наблюдаемую точность и свойство профиля: ошибка агрегированного
  ответа 8192 путей на 1/128 равна 0.001554095 в этой реализации, однако
  одновременный интервал среднего смещения [-0.01941005,0.02251824]
  выходит за ±0.01. RMSE восьми отдельных запусков M=1024 равно 0.01780,
  их средняя CI-полуширина 0.04747. Вероятность повторного достижения
  допуска и точность сетки не установлены. На 1/32 интервал смещения
  [0.01228983,0.05642475] целиком выше 0.01: этот шаг недостаточен.
- Корреляции соседних интегралов тройки 0.9498/0.9810, выигрыши в
  дисперсии разностей 19.72/52.42. Абсолютный шум уровня остаётся.
  Условный показатель отношения разностей 0.730 с bootstrap95
  [0.122,1.560] слишком неопределёнен для надёжного показателя.
  Асимптотический порядок не установлен, степенная экстраполяция для
  выбора h или коррекции численных ответов не использовалась.
- При h=1/128 и sigma(0)=0.77495 статистический CI±0.005 потребовал бы
  около 92282 путей / 751544608 обновлений на точку, около 351 с при
  последовательных chunk=1024. Для 33 точечных CI — 24.80 млрд обновлений
  / 3.22 ч; совместный Bonferroni95 (z=3.17177) — около 241661 пути
  на точку / 64.95 млрд обновлений / 8.35 ч. Предполагается одинаковая
  дисперсия как в x=0, остальные точки не проверены. Временное смещение
  этим не контролируется; полной стоимости гарантии 0.01 нет.
  Вариант всего допуска на статистику (±0.01) уже требует около
  187.89 млн обновлений на точку, без запаса на смещение.
- Фактический предел до запуска: 170 млн обновлений / 240 с / 384 МиБ,
  один worker/BLAS=1; M и число повторов не адаптируются, h<1/128 и
  повторов отказов нет. Факт: 16 success, 0 отказов/not_run/повторов,
  166789120 обновлений, 266862592 нормальных генераций с replay,
  ядра 77.243 с, серия 77.987 с, RSS 81.5 МиБ. Большие сценарии не запускались.
- Проверки: 8 новых тестов неизменности порогов, числа повторов и
  точечной/совместной стоимости; с прежним coupling-набором 22 passed.
  Обычная регрессия **291 passed, 1 deselected, 679 subtests** за 35.78 с.
  Прежний длительный R1 исключён, два escape-warning сохранены.
  Старые диагностические файлы тоже защищены SHA. Production MC,
  Ремизов/Chern, генератор, данные, manifest и прежние результаты
  не менялись; PROJECT_STATUS до изменения сохранён в before/.

Решение: ограниченная проверка F0002 завершена. h=1/128 — перспективный
диагностический кандидат, практический профиль с подтверждённой полной
точностью 0.01 не выбран. Стоимость и неподтверждённое смещение — ограничения
для будущего benchmark/ML. Не выводить сеточную или воспроизводимую точность
из одного удачного агрегированного значения; наблюдаемую exact-ошибку
сохранять отдельно. Автоматически увеличивать ресурсы не продолжать.
Будущий план сравнения должен фиксировать бюджет, качества/неуспехи и
проверку полной ошибки; победитель между MC и Ремизовым здесь не установлен.

### Связанный временной пилот F0002 — 2026-10-24

Источники: [полный отчёт](results/mc_f0002_coupled_h_2026-10-24/REPORT.md),
[выводы и ограничения](results/mc_f0002_coupled_h_2026-10-24/INTERPRETATION.md),
[план до запуска](results/mc_f0002_coupled_h_2026-10-24/plan.json),
[решение об остановке](results/mc_f0002_coupled_h_2026-10-24/decision.json),
[все статусы](results/mc_f0002_coupled_h_2026-10-24/run_summary.json),
[ответы и интегралы парных путей](results/mc_f0002_coupled_h_2026-10-24/records/),
[ошибки и CI](results/mc_f0002_coupled_h_2026-10-24/attempt_levels.csv),
[статистика и ковариации](results/mc_f0002_coupled_h_2026-10-24/aggregates.json),
[независимый пересчёт](results/mc_f0002_coupled_h_2026-10-24/decision_recheck.json),
[регрессия](results/mc_f0002_coupled_h_2026-10-24/pytest.txt),
[итоговая сохранность](results/mc_f0002_coupled_h_2026-10-24/final_verification.json).

- Новый `diagnostics/mc_f0002_time_refinement.py` использует неизменённый
  `_path_contributions` production MC. Отдельный CoupledNormals выдаёт
  sum(z_fine)/sqrt(q), q=1,2,4: coarse ΔW равен сумме fine ΔW, каждый
  маргинальный уровень сохраняет EM и обе трапеции. Полная временная
  матрица не хранится; поток воспроизводится на каждом уровне, добавочные
  нормальные генерации учтены. Сохранены все отдельные интегралы путей.
- План зафиксирован до запуска: пилот 4×1024 на h=1/32,1/64;
  подтверждение независимыми 8×2048, при необходимости с h=1/128.
  Максимум 258523136 обновлений / 433651712 нормальных значений с replay,
  300 с / 384 МиБ / один worker и BLAS=1, 20 с на связанный повтор.
  Условия перехода: ненулевая парная t95-разность, относительная
  полуширина <=0.5, выигрыш в дисперсии >=2. Пилот не объединяется
  с подтверждением, автоматических повторов и изменения порогов нет.
- Пилот: exact(0)=−0.387122082343043; средние f=−0.3536949572 и
  −0.3746502350. Средние отклонения 0.03342713±0.02399206 и
  0.01247185±0.01971025 (t95 по четырём seed). Парная разность
  0.02095528±0.01329021, CI [0.00766506,0.03424549], p≈0.01524.
  В малом предварительном опыте снижение оценки различимо; общей
  сходимости абсолютной ошибки и независимого подтверждения нет.
- Корреляция интегралов 0.946343, оценённое снижение дисперсии разности
  18.4581 раза: SE по путям 0.004034 против 0.017333 для независимых
  уровней при том же числе путей. Неопределённость каждого отдельного
  уровня этим связыванием не устраняется.
- Условие относительной точности не выполнено: 0.634218>0.5.
  Расширение остановлено согласно плану, хотя t95-интервал разности уже
  исключает ноль. 4 success, 8 not_run; h=1/128 не запускался.
  Временной порядок по двум уровням не оценивается. Прежние уровни
  при T=8 не смешивались с новым T=63.625.
- На h=1/64 одновременный 95% интервал среднего смещения
  [−0.013395,0.038339] не помещается в ±0.01. При хвосте <=9.833e-5
  достижение точности 0.01 не подтверждено, недостижимость не доказана.
  Практический production h пока не выбран. 1/64 — диагностический
  кандидат; достаточность более мелкого h остаётся гипотезой.
- Реально 25018368 обновлений, 33357824 нормальных значений с replay,
  4096 независимых связанных путей; ядра 11.513 с, серия 12.318 с,
  пик RSS 79.71875 МиБ. Медианы уровней 0.9657 и 1.9149 с на seed.
  Программных/ресурсных отказов нет, все счётчики сверены с preflight.
- Первичный матричный расчёт ковариации выдал NumPy FP-warning при
  конечных результатах. Лог сохранён; отчётный код перешёл на math.fsum
  центрированных произведений. Независимый скалярный пересчёт подтвердил
  показатели и исходное решение об остановке без новых траекторий.
  Причина предупреждений не установлена. Исходная и итоговая версии
  анализа сохранены раздельно в sources/ и sources_final/.
- Проверки: 14 новых тестов coupling, предписанных приращений, переменных
  коэффициентов, совпадения с production, воспроизводимости, отказов,
  устойчивой статистики и выбора уровней. Обычная регрессия:
  **283 passed, 1 deselected, 679 subtests** за 35.80 с; прежний
  длительный R1 исключён, два прежних escape-warning сохранены.
  Production-код, генератор, датасеты, manifest и все прежние результаты
  побайтово сохранены; PROJECT_STATUS до изменения — в before/.

Решение: малая связанная диагностика выполнена с остановкой по объявленному
критерию. Выбор практического h и вероятность достижения 0.01 остаются
открытыми. Предложенный контроль той же пары на 8 новых seed×1024 пути
выполнен в новой серии выше, затем выполнена независимая тройка с 1/128.
Прежний прогноз CI-полуширины 0.00675 остаётся оценкой того этапа;
фактическая полуширина нового контроля — 0.004248. Большой запуск ради CI±0.005
на h=1/64 потребовал бы около 91410 путей / 372.22 млн обновлений
и здесь не выполнялся. Сравнение с Ремизовым, полная сетка и ML не запускались.

### Контролируемая сходимость Монте-Карло — 2026-10-24

Источники: [полный отчёт](results/mc_convergence_2026-10-24/REPORT.md),
[выводы с разделением достоверности](results/mc_convergence_2026-10-24/INTERPRETATION.md),
[замороженный план](results/mc_convergence_2026-10-24/plan.json),
[все попытки CSV](results/mc_convergence_2026-10-24/attempts.csv),
[агрегаты и аналитические оракулы](results/mc_convergence_2026-10-24/aggregates.json),
[проверки значимости](results/mc_convergence_2026-10-24/statistical_findings.json),
[24 Welch-сравнения и Holm](results/mc_convergence_2026-10-24/differences.json),
[бюджет следующих профилей](results/mc_convergence_2026-10-24/recommendations.json),
[регрессия](results/mc_convergence_2026-10-24/pytest.txt),
[финальные тесты анализа](results/mc_convergence_2026-10-24/pytest_analysis_final.txt),
[SHA до/после](results/mc_convergence_2026-10-24/verification.json).

- Новый `diagnostics/mc_convergence.py` фиксирует все конфигурации до MC,
  запускает по одному изолированному worker на задачу, сохраняет каждый
  MCResult, CI, exact-ошибки, затраты, время, телеметрию и окружение.
  `diagnostics/report_mc_convergence.py` читает только сохранённые ответы,
  формирует CSV/JSON, пять PNG/PDF, точечные интервалы и анализ стоимости.
- Только x=0. M=512/2048/8192 при h=1/32,T=8; h=1/8,1/16,1/32 при
  M=8192,T=8; T=1,2,4,8 и сертифицированный при M=2048,h=1/32.
  На Gaussian/F0000/F0001/F0002 по 8 независимых seed на конфигурацию,
  общие baseline решены один раз. У постоянного примера M=2, SE=0;
  там отдельные детерминированные h/T-контроли. Броуновского сопряжения нет.
- Ресурсы объявлены до запуска: <=250 млн обновлений, 360 с всей серии,
  384 МиБ RSS, один worker/BLAS=1; на попытку <=5 млн обновлений и
  10 с cooperative. Реально 295 success, 0 отказов/повторов/not_run,
  223610232 обновления, solver 43.342 с, серия 45.603 с, RSS 98.906 МиБ.
  Счётчики работы сверены с preflight; эти обновления не приравниваются
  к одинаковому числу FLOP для разных AST.
- Подтверждено в испытанных условиях: M*16 уменьшает средние статистические
  полуширины в 3.98–4.01 раза. Разброс независимых средних уменьшился на
  всех четырёх задачах (приближённый F-тест/Holm по четырём сравнениям).
  Точный показатель и покрытие CI по восьми seed не установлены.
- Независимые аналитические ожидания отделяют временное смещение и хвост.
  При T=8,h=1/32 ошибка постоянного примера 0.000162694; при T=4.375 —
  0.0000834934. Для Gaussian временное смещение 0.00194918, RMSE восьми
  M=8192 запусков 0.00503544. Детерминированные порядки трапеций стремятся
  к 2; это не порядок общей EM-схемы с переменными коэффициентами.
- На F0000/F0001 соседние различия h не выдерживают Holm по всем 24
  сравнениям h/T: общий временной порядок не установлен. На F0002 обе
  разности h значимы; средние отклонения 0.19412/0.08856/0.03656 при
  h=1/8,1/16,1/32 и M=8192,T=8. Наблюдаемый p=1.021 с приближённым
  интервалом [0.663,1.379] совместим с первым порядком, асимптотика не доказана.
- Сертифицированные горизонты округлены вверх до кратного 1/8, чтобы h
  оставался строго фиксированным: 4.375/5/8.5/57.375/63.625 для
  constant/Gaussian/F0000/F0001/F0002. Длинный T для F0001/F0002 стоит
  примерно в 7–8 раз дороже T=8, но эквивалентность короткого горизонта
  с допуском 0.01 не установлена; границы хвоста при T=8 равны 37.14/183.89.
  На всех четырёх стохастических задачах T=1→2 различим после Holm.
  Сертификат непрерывного хвоста не включает статистику и дискретизацию.
- Предварительные M для статистического CI±0.005, оценённые на длинном T:
  Gaussian 24259, F0000 22305, F0001 58700, F0002 99230.
  Gaussian: кандидат M≈25000,h=1/32,T=5 с известным аналитическим
  временным смещением. F0000/F0001 требуют отдельного уточнения h.
  Для F0002 h=1/32 недостаточен: даже на длинном T среднее отклонение
  0.05503±0.01678 при хвосте <1e-4. Гипотетические h=1/256–1/512
  для точности 0.01 требуют 1.62–3.23 млрд обновлений на точку и
  не запускались; перенос наблюдаемого первого порядка пока гипотеза.
- Проверки: основная регрессия **268 passed, 1 deselected, 679 subtests**
  за 35.97 с; после добавления анализа Holm — **7 passed** за 0.30 с.
  Исключён прежний длительный R1, два escape-warning сохранены.
  4498 защищённых файлов до/после не изменились; исходный PROJECT_STATUS
  сохранён в before/. Генератор, оба прежних метода, MC solver, датасеты,
  MIGRATION_MANIFEST и предыдущие результаты не менялись.

Решение: малая диагностика h/T/M выполнена; общая точность 0.01 на сетке
и датасете, покрытие CI и сравнение с Ремизовым не установлены.
Прежний следующий шаг F0002 продолжен связанным пилотом 1/32→1/64 выше;
его правило остановки оставило 1/128 невыполненным. Разрешение малых
разностей F0000/F0001 на фоне шума в нескольких заранее заданных точках
остаётся отдельным шагом с предварительным бюджетом. Сразу 33 точки
или массовый расчёт не обоснованы. Production-схемы, RNG и данные сохранены.

### Автоматический сертификат G all_methods_v1 — 2026-10-24

Источники: [доказательство и API](docs/ALL_METHODS_FORCING_BOUNDS.md),
[отчёт с таблицей всех 20 задач](results/forcing_bounds_2026-10-24/survey/REPORT.md),
[предварительный план](results/forcing_bounds_2026-10-24/survey/plan.json),
[сертификаты и стоимость каждого профиля](results/forcing_bounds_2026-10-24/survey/certificates.json),
[отдельные сеточные наблюдения](results/forcing_bounds_2026-10-24/survey/observations.json),
[аналитические примеры](results/forcing_bounds_2026-10-24/survey/analytical_examples.json),
[регрессия](results/forcing_bounds_2026-10-24/pytest.txt),
[сохранность и независимый пересчёт](results/forcing_bounds_2026-10-24/verification.json).

- Новый `all_methods_bounds.py` переиспользует decoder и проверки конструкции
  безопасного загрузчика. Глобальные нормы AST-значения и двух производных
  выводятся для всех 14 операций правилами произведения/цепочки. Bump имеет
  доказанные границы 1, 3/R, 1809/(64*R²). Для y=s*bump*T_y получены Y0,Y1,Y2,
  затем G=A*Y2+B*Y1+(lambda+C)*Y0. Границы коэффициентов следуют из той же
  конструкции tanh; в MC остаются прежние lambda и C из RemizovProblem.
- Положительная арифметика границ — точные Fraction с округлением float64 вверх;
  корень проверяется точным сравнением квадрата. Сертификат содержит версию,
  SHA serialized_ast с construction, компоненты и MCForcingBound. Сеточные
  максимумы и формульные строки не используются для вывода G.
- CLI и ветка monte_carlo сравнительного драйвера получают сертификат
  автоматически. Прямой MC API сохранён. --uncertified-tail с явным T либо
  forcing_bound=null в плане сохраняют явно неподтверждённый режим. При
  непредставимом сертификате отказ, без сеточного fallback и без ограничения G.
- Тест выявил прежний дефект округления auto_T: F0002 давал вычисленную границу
  0.00010000000000000009 при tail_eps=1e-4. Исправлена арифметика горизонта и
  хвоста: точная разность lambda-C округляется вниз; ln/exp и деление —
  направленно с Decimal, 80 значащих цифр. Проверено независимым 200-значным
  эталоном. Допуск не увеличивался. EM, обе трапеции и статистика не менялись.
- Детальный обзор: все 20 сертификатов получены до диагностических сеток
  полного носителя (4097/16385 точек). Все наблюдаемые |g_R| ниже G, внешние
  значения равны нулю; это smoke-проверки, не доказательства. Обзор содержит
  **0 траекторий MC**. G=25.1183219…5397.5704166; T=8.33569545…63.55987156
  при tail_eps=1e-4, все вычисленные границы хвоста <=1e-4.
- Аналитические примеры: y=1/2 даёт точный G=1, T=4.2585965957;
  exp(-x²) даёт точный G=4, T=4.9517437763 (a=1,b=c_R=0,lambda=2).
  Доказанный стресс-пример сокращения одинаковых синусов увеличивает G
  с 13.0664 до 2.87333e13 для того же решения: запас не менее 2.199e12,
  горизонты 5.5436 и 19.7531. Сокращения пока не отслеживаются.
- Стоимость оценена по K*M*N и числу коэффициентных вызовов, без пересчёта
  операций в секунды. В бюджете 1e8 path steps: point_default
  (K=1,M=20000,dt=.005) — 14/20; curve_default (K=33) — 0/20;
  curve_moderate (K=33,M=2000,dt=.02) — 15/20.
  Умеренный профиль превышают F0001/F0002/F0015/F0016/F0017; у четырёх,
  кроме F0016, даже оптимистический ориентир от сеточного максимума превышает
  бюджет. Этот ориентир не является сертификатом и в MC не используется.
  Большие горизонты связаны также с малым gap; нельзя приписывать всё запасу G.
- Новые тесты: независимые SymPy-производные 14 операций, символическая
  формула bump, рациональные округления, аналитические нормы, сокращение,
  отсутствие выборок в proof, все 20 сертификатов и отказ MC до траекторий,
  автоматический CLI/worker. Один малый интеграционный тест использует реальные
  пути только F0000 (M=16, одна точка, dt=.1); общей точности он не утверждает.
- Финальная регрессия: **262 passed, 1 deselected, 679 subtests passed**,
  37.07 с; исключён прежний длительный R1. Два прежних escape-warning сохранены.
  Необходимые прежние файлы скопированы в before/ нового каталога. Ядро
  генератора, загрузчик, Ремизов/Chern, данные и все прежние результаты
  сохранены побайтово; проверка — verification.json. MIGRATION_MANIFEST не менялся.

Решение: прежний план автоматического G выполнен. Ограничения — запас
треугольных оценок, отсутствие учёта сокращений и полной временной ошибки.
Сертификат относится к неизменённой паре problem/metadata штатного загрузчика.
Он контролирует только хвост; total_error_certified=False, статистика отдельно.
Прежний следующий шаг h/T/M выполнен описанной выше малой диагностикой;
строгая временная точность и окончательные профили сравнения не установлены.
Массовые решения, сравнение победителей и ML не запускались.

### Интеграция Монте-Карло — 2026-10-24

Источники: [отчёт](results/monte_carlo_integration_2026-10-24/REPORT.md),
[предварительный план](results/monte_carlo_integration_2026-10-24/plan.json),
[полные ответы, конфигурации и окружение](results/monte_carlo_integration_2026-10-24/runs/summary.json),
[проверка сохранности и пересчёт ошибок](results/monte_carlo_integration_2026-10-24/verification.json),
[финальная регрессия](results/monte_carlo_integration_2026-10-24/pytest_final.txt).

- Новый `mc_solver.py`: MCConfig, MCForcingBound, MCResult, auto_T,
  solve_monte_carlo и pilot-оценка paths_for_tolerance. Схема исходного MC
  сохранена: Эйлер — Маруяма, трапеции потенциала и временного интеграла,
  среднее независимых траекторий. Вход — существующий RemizovProblem;
  подтверждена фактическая векторная работа функций загрузчика.
- Для автоматического горизонта нужен доказанный глобальный G>=sup|g_R|:
  T=max(0,log(G/(gap*tail_eps))/gap), gap=lambda-C. Численного сертификата G
  в all_methods_v1 нет; автоматический вывод из AST не реализован. Без G
  нужен явный T, tail_error_bound=None; никаких сеточных оценок нормы нет.
  Явный T не обходит глобальные условия коэффициентов и lambda>C.
- Статистика — центрированные пакетные моменты и объединение Chan; M>=2,
  без вычитания квадратов и искусственного порога дисперсии. PCG64/SeedSequence
  адресуется по seed, problem_id, точке и stream_id. Размер chunk — часть
  конфигурации воспроизводимости. seed=None сохраняет энтропию для повтора.
  Проверяются значения/границы коэффициентов, конечность состояний,
  реальные массивы, параметры и бюджеты; при отказе f и интервалы пусты.
- stderr и ci95 относятся только к среднему дискретной усечённой схемы;
  интервалы точечные асимптотические, без общей гарантии покрытия.
  discretization_error_bound=None, total_error_certified=False даже с G.
  Регулярность произвольных callable программно не доказывается.
- Новый CLI `examples/solve_mc_equation.py` выводит JSON без записи данных.
  Существующий `diagnostics/compare_chernov.py` расширен веткой monte_carlo;
  переиспользованы worker, watchdog, RSS и протокол попыток. Старые ветки
  численных методов не менялись. Production batch не расширялся.
- Выполнено ровно пять запусков, seed=2488, один worker/BLAS=1, без подбора
  и повторов. На 33 точках [-2,2]: для y=1/2 при M=8, dt<=.01,
  tail_eps=1e-6 получено E∞=1.5621812643e-5; для exp(-x²) при M=8000,
  dt<=.01, tail_eps=1e-5 — E∞=0.0070094606362. Использованы аналитические
  G=1 и G=4. Предварительные наблюдаемые пороги 2e-5 и .03 выполнены.
- Три задачи малого набора: AMV1_F0000_S00 (L1/constant), F0001
  (slow_varying), F0002 (moderate_varying). M=2000, dt=.02, T=4,
  33 точки [-1,1], без G. Все вернули конечный результат; наблюдаемые E∞:
  0.03799808403 / 0.02195797957 / 0.05750379201. Хвост не подтверждён;
  достижения общего допуска и преимущества над Ремизовым не заявлены.
- Внешний watchdog — 60 с / 512 МиБ на попытку, внутренний лимит 30/45 с;
  ресурсных отказов нет. Максимальный RSS worker — 81.8125 МиБ.
  Все пять попыток сохранены протоколом без скрытых повторов. Версии MC и
  сравнительного драйвера, использованные в эксперименте, сохранены рядом
  с отчётом и сверены с SHA из summary. После запусков добавлена отдельная
  защита underflow положительного T/dt и непредставимого полушага; она
  не затрагивает обычные режимы пяти попыток. Повторных расчётов не было.
- Финальная регрессия: **237 passed, 1 deselected, 679 subtests passed**,
  36.17 с; исключён прежний длительный R1 n=1,2,4,8. Два прежних warning
  про escape-последовательность сохранены. Проверены независимая трапеция,
  ручной шаг переменных коэффициентов, гауссово дискретное ожидание,
  статистика при большом среднем, воспроизводимость, отказы и лимиты,
  CLI и путь генераторного загрузчика через сравнительный worker.
- Прежние README, PROJECT_STATUS и сравнительный драйвер сохранены в новом
  каталоге; остальные 4492 прежних файла, включая всё математическое ядро,
  данные и прежние результаты, сохранены побайтово. Три исходных MC-файла
  скопированы в comparison_inputs/monte_carlo с provenance и SHA; внешние
  оригиналы не менялись. MIGRATION_MANIFEST не менялся.

Ограничения: единичные seed не устанавливают вероятность достижения допуска;
порядок и полная временная ошибка не доказаны; границы и регулярность
произвольных callable остаются ответственностью постановки. Исторический план
аналитического G выполнен следующим этапом, описанным выше; отдельные проверки
h/T/M и согласование общего сравнения остаются. Массовый benchmark и обучение
ML на этапе интеграции не выполнялись.

### Один контроль временной квадратуры F0180/S04 — 2026-10-24

Источники: [отчёт](results/chernov_time_quadrature1025_2026-10-24/REPORT.md),
[результат/оба ответа/exact](results/chernov_time_quadrature1025_2026-10-24/result.json),
[план и хеши](results/chernov_time_quadrature1025_2026-10-24/plan.json),
[узлы, веса и проверки](results/chernov_time_quadrature1025_2026-10-24/quadrature.json).
Драйвер — `diagnostics/chernov_time_quadrature1025.py`; используется прежняя
проверенная обёртка наблюдения и исходная `_apply`. Научный код не редактировался.

- Один новый полный запуск AMV1_F0180_S04, num_t=1025 вместо 513.
  Baseline взят из сохранённого cubic/zero/padding=12, не пересчитывался.
  Только num_t изменён в конфигурации. n=128, dx=.0125, padding=12,
  T=18.692237908953764, lambda, 2081 рабочий узел [-13,13], коэффициенты,
  forcing и 33 target_x сохранены. Хеши/версии/metadata сверены с production
  и обеими предыдущими диагностиками.
- Составной неравномерный Simpson интегрирует квадратичные интерполянты
  на тройках. Узлы T*(j/(num_t-1))². Старые узлы побайтово равны чётным новым;
  отрицателен только первый вес: -4.75368192e-5 → -1.18842048e-5.
  Это свойство формулы, не обнаруженный баг. Суммы весов равны T в float64;
  аналитические моменты 1,t,t² и сравнение со scipy.simpson проверены.
- E513=0.07045188951985172, E1025=0.07044935796912224,
  D_Q=0.0001354983426354872, Delta E=-0.000002531550729476.
  Улучшение E∞ около 0.00359%; D_Q составляет 0.1923% прежней ошибки.
  Максимумы ошибок находятся при x=-1, максимум разности — при x=-0.5625.
  Порог .01 не достигнут. Граничная production-метрика совпала: 1.9026940933.
- Все 131072 фактических шага наблюдались; нарушений оценки шага не обнаружено,
  максимум отношения к (1+tau*C)||u|| — 0.9999138613. Полный граничный максимум,
  включая последние u_128, — 1.9592138688; максимум состояний с u_0 — 176.6927711.
  Четыре общих узла прежних zero-трасс воспроизвели final-global нормы точно.
  Runtime с диагностикой 23.9492 с, внутри _apply 22.9976 с, RSS 182.30 МиБ.
  Один процесс/BLAS=1; три квадратурных теста прошли. Сохранены оба f, exact,
  обе временные сетки/веса, агрегаты 1025 узлов, метаданные и SHA.
- Малое D_Q — эмпирическое свидетельство против доминирования квадратуры,
  не доказательство малой истинной квадратурной ошибки. Два уровня не дают
  порядка сходимости; D_Q/15 и экстраполяция без проверок не обоснованы.
  Пространственная ошибка и конечное n ещё не разделены.
- Следующий минимальный контроль **предложен, не выполнен**: один n=256
  при num_t=1025, padding=12, dx=.0125, прежнем T и cubic/zero; сравнить с новым
  сохранённым ответом. Если влияние n мало, приоритет h/2 возрастёт.
  Третий уровень квадратуры полезен для её отдельного исследования сходимости,
  но имеет меньший приоритет для объяснения большого остатка .07045.
  Статус обновлён только после сохранения и проверки отчёта; прежняя версия
  находится в `PROJECT_STATUS.before.md` нового каталога.

### Один контроль размера окна F0180/S04 — 2026-10-24

Источники: [отчёт](results/chernov_window_padding24_2026-10-24/REPORT.md),
[новый ответ и exact](results/chernov_window_padding24_2026-10-24/result.json),
[план/хеши/проверка сетки](results/chernov_window_padding24_2026-10-24/plan.json).
Драйвер — `diagnostics/chernov_window_padding24.py`; исходный решатель не менялся.

- Выполнен ровно один новый полный запуск AMV1_F0180_S04, cubic/zero,
  padding=24, область [-25,25], 4001 узел. Padding=12 взят из сохранённого
  результата предыдущей серии: E∞=0.07045188951985172, время 11.9556 с.
  Новая E∞=0.0704518895199231, max|f24-f12|=1.0808021144725899e-13.
  Порог .01 не достигнут в обоих случаях.
- T=18.692237908953764, lambda, n=128, dx=.0125, num_t=513/quadratic,
  узлы/веса квадратуры и target_x сохранены. Хеши научных исходников и
  production-данных совпали с планами обеих прежних серий. Коэффициенты и
  forcing на общей части сетки совпали побайтово.
- Все 2081 старых float64-координаты сохранены внутри новой сетки побайтово.
  Независимый np.linspace для нового окна отличался бы на <=3.553e-15 в
  812 общих координатах. Диагностическая обёртка добавила внешние узлы,
  сохранив старые координаты, номинальный шаг и исходные _apply/сплайн/Simpson.
- Граничный максимум в сопоставимой production-метрике снизился с 1.90269409
  до 0.08913288. Новый полный максимум с учётом последних u_128 — 0.09554613;
  production-метрика наблюдает k=0..127 и конечный интеграл. Максимум всех
  новых промежуточных состояний, включая u_0=g_R, равен 176.6927711.
  Outside-запросов: 5698944 → 4670720. Полный максимум padding=12 не сохранён,
  его не восстанавливали повторным решением.
- На всех 65536 новых шагах проверено отношение ||u_(k+1)|| /
  ((1+tau*C)||u_k||); максимум 0.9996633141, нарушений нет. Это наблюдение
  траекторий, а не доказательство нормы оператора cubic/zero для любых данных.
  Короткий независимый тест подтверждает возможность сплайн-выброса на
  простом граничном профиле; универсальная сжимаемость не предполагается.
- Новый runtime 19.8464 с включает наблюдатель; время внутри _apply — 19.3005 с,
  RSS 178.61 МиБ, один процесс/BLAS=1. Три коротких теста прошли.
  Сохранены агрегаты всех 513 временных узлов, полный новый f/exact,
  baseline, конфигурация, хеши и резерв прежнего статуса.
- Вывод: в этом парном сравнении расширение окна не влияет существенно на
  остаточную ошибку на 33 узлах. Общая достаточность окна и причина остатка
  не установлены. Дальнейшее расширение имеет низкий приоритет.
  Предложенный здесь контроль num_t=1025 **выполнен следующей задачей** при
  прежних n=128, dx, T и padding=12; результат приведён в разделе выше.
  Пространственная ошибка не исключена.

### Граничная диагностика четырёх задач — 2026-10-24

Источники: [отчёт](results/chernov_boundary_diagnosis_2026-10-24/REPORT.md),
[таблица](results/chernov_boundary_diagnosis_2026-10-24/comparison.csv),
[проверяемые данные анализа](results/chernov_boundary_diagnosis_2026-10-24/analysis.json).
Драйвер — `diagnostics/chernov_boundary_diagnosis.py`, отдельный отчётный скрипт —
`diagnostics/report_chernov_boundary_diagnosis.py`; исходный решатель не исправлялся.

| ID | Baseline E∞ | cubic/zero E∞ | linear/zero E∞ |
|---|---:|---:|---:|
| AMV1_F0141_S00 | 19704.5019109 | 0.01871927854 | 0.06228970391 |
| AMV1_F0180_S04 | 19134.4051868 | 0.07045188952 | 0.13239608240 |
| AMV1_F0184_S01 | 444.5892546 | 0.05356807322 | 0.10279976759 |
| AMV1_F0180_S03 | 0.00718266032 | 0.00718266032 | 0.07723601189 |

- Проверены хеши пяти научных исходников и четырёх production-артефактов,
  версии Python/NumPy/SciPy/SymPy, metadata и первичные attempt-файлы.
  Все четыре повторных baseline дали побайтово прежние f и точно прежние
  boundary maxima. Варианты имеют одинаковые SHA массивов коэффициентов,
  forcing, рабочей сетки, target_x, узлов и весов квадратуры; T совпадает точно.
- Сохранён бюджет solve: n=128, num_t=513/quadratic, dx=0.0125,
  окно [-13,13], 2081 узел, target_x — прежние 33 точки. Эффективный taper=2.6.
  Единственное изменение cubic/zero — boundary; у linear/zero затем только interp.
- В 16 реальных узлах t (около .25T/.5T/.75T/T) прослежены обе кубические
  цепочки. Наблюдатель побайтово сверял исходную `_apply` на всех 4096 шагах.
  При t=T первые нарушения нормы шага для трёх выбросов — k=29/31/36;
  разность на цели >.01 появляется при k=47/59/65, после распределённого
  веса — k=96/99/101. Итоговая взвешенная разность в этом узле около
  160011 / 33671 / 8005. Это значения цепочек в узле, не ошибки интеграла.
- Причинно подтверждено: при фиксированных остальных элементах дискретизации
  выбор taper создаёт катастрофический вклад в целевой ответ трёх задач.
  Данные не разделяют дальнейшее распространение сдвигами и глобальную связь
  кубического сплайна. В контрольной задаче огромные необвешенные состояния
  подавляются экспонентой, а cubic/zero сохраняет итоговый f побайтово.
- Для linear/zero аналитически ||S_h(tau)||∞<=1+tau*C: линейные сдвиги с нулём
  вне окна не увеличивают норму. Это не гарантия точности: linear/zero не достиг
  .01 ни на одной задаче, cubic/zero достиг только на исходном успешном контроле.
  Нулевая граница остаётся приближением; остаток может включать конечное n,
  квадратуру, пространственную интерполяцию и окно, их доли не установлены.
- Все 12 solve конечны; суммарное время решателей 117.64 с, пиковый RSS 186.92 МиБ,
  один вычислительный процесс, BLAS=1. Плотные матрицы не строились.
  Тесты диагностики и два прежних операторных контроля: 11 passed,
  12 deselected; предупреждение только о прежней escape-последовательности docstring.
  Сверены 4334 прежних защищённых файла; после расчётов изменений не было.
  Статус обновлён отдельно с резервом `PROJECT_STATUS.before.md`.
- Предложенный здесь минимальный контроль F0180/S04 cubic/zero с padding=24
  **выполнен следующей задачей** при прежних n, dx, T и квадратуре; результат
  приведён в разделе выше. Генератор и production-схема не переписывались.
  Три исторических worker_exception не повторялись.

### Рисунки двух примеров — 2026-10-20

Для AMV1_F0000_S00 и AMV1_F0141_S00 подготовлены коэффициенты a,b,c,
сопоставление точного/сохранённого численного решений и абсолютная ошибка.
Финальные файлы: [remizov_example_figures_2026-10-20_v2](results/remizov_example_figures_2026-10-20_v2/).
Семь векторных PDF и семь PNG 300 dpi; для F0141 дополнительно логарифмическая
ошибка и вставка точного решения в собственном масштабе. Численные значения
и ошибки показаны только в 33 сохранённых узлах; коэффициенты и точная кривая
оценены штатным безопасным AST-загрузчиком на 2001 точке. Показан c=c_R−λ.
Скрипт: `diagnostics/plot_remizov_examples.py`; готовы `figures.tex` и архив
`overleaf_figures.zip`. Сверены идентичность задач, ошибки и значения y_exact,
SHA источников до/после; все PDF отрисованы и проверены визуально.
Новых решений ОДУ нет. Первая версия рисунков сохранена отдельно в каталоге
без суффикса `_v2`; в финальной версии улучшен масштаб малой ошибки F0000.

### Анализ сохранённых результатов — 2026-10-20

- Проверены все 1000 терминальных записей и их побайтовое совпадение с первичными
  `attempts/.../result.json`; ошибки по 997 сохранённым массивам пересчитаны:
  итог **871/126/3** подтверждён. Полные таблицы, рисунок и реестр отказов:
  `results/remizov_technical_report_2026-10-20/analysis/` и `REPORT.md`.
- Ускорение нового solve **1.30–34.63 раза** подтверждается прежним сравнением
  четырёх задач по три повтора каждого метода (36 запусков суммарно), при
  достигнутом E∞≤0.01. Оно связано с сочетанием иной дискретизации/бюджета
  и переиспользования неизменных массивов; вклад каждого изменения отдельно
  не измерен. Старый решатель не запускался на всей тысяче с тем же бюджетом.
- Три ошибки примерно **19704.50, 19134.41, 444.59** дают **99.9741%** суммы
  E∞ всех конечных ответов. Наиболее проблемная ячейка L4/oscillatory:
  34/50 выше порога. Однозначные численные причины 126 превышений не установлены:
  нет независимых уточнений production-задач; boundary/outside встречаются
  и у ответов, достигших порога, и сами по себе причину не доказывают.
- Архив контроллера подтверждает: три исторических `worker_exception` —
  обнаруженный выход процесса до записи результата и синтетическая запись
  контроллера. Traceback, returncode/сигнал отсутствуют, worker-логи пусты;
  адресный запрос системного журнала новых сведений не дал. Первичная причина
  выхода не установлена; ошибкой численного метода это не доказано.
- Добавлены только скрипты обработки сохранённых материалов:
  `diagnostics/analyze_remizov_saved_results.py`,
  `diagnostics/plot_remizov_saved_analysis.py`. Численные методы, датасет и
  прежние результаты сохранены; резерв статуса и SHA до работы — в каталоге отчёта.
  Проверки: 1000 идентичностей записей, 997 ошибок, 36 сравнительных запусков;
  тесты с вызовом решателя и новые численные эксперименты не проводились.
- Следующий шаг требует отдельной задачи: небольшой план до 12 контролей
  n/квадратуры/h/окна на трёх задачах с фиксированными T и шириной taper
  описан в разделе 10 отчёта. Он **не выполнен**. Три исторических отказа
  автоматически не повторять; ML не начинать.

### Контекст предшествующих серий

Проект перенесён из `generator_DE` как независимый генератор и решатель одного
метода. Малый датасет, безопасная загрузка, эталоны R1–R5, проверки и четыре
исследовательские серии существуют. Математическое ядро пока совпадает с исходным
снимком переноса. 2026-10-18 добавлена отдельная адаптация `chernov_solver.py`
с двумя схемами предоставленного Chern, без изменения исходного решателя.
Сравнение трёх вариантов на R1/F0005/F0006/F0007 при E∞≤0.01 завершено;
подробности и ограничения — в разделе адаптации и сохранённом отчёте.
Последующий обзор всех 20 записей без индивидуального подбора завершён:
solve прошёл 20/20, streaming — 18/20. Вычислительных отказов нет, однако
success новых схем не контролирует полную ошибку. Для переноса на 1000 задач
solve перспективнее по покрытию; streaming быстрее, но данный общий бюджет
не обеспечивает порог на двух уже известных задачах. Подробности — в новом
разделе «Chern на всём малом датасете» ниже. После этого массовая серия была
начата и принудительно остановлена; её частичные результаты и исправление
надзора описаны в разделе пакета ниже.
Production-набор теперь создан и проверен отдельно штатным генератором:
200 семейств × 5, split 700/150/150. Все 1000 задач загружаются и совместимы
с входом chernov_solver.py. Полная серия solve теперь завершена: 871/1000
достигли порога на 33 точках, 126 выше порога, три прежних отказа сохранены.

Последняя из прежних серий дала успешный F0006 при n=4 и timeout F0007 при n=4.
Сохранённый тогда план F0006 при n=1,2, затем возможно F0007 при n=1,2,
заменён текущей задачей адаптации. Именно эти старые конфигурации не запускались.
Старые планы профилировать F0000, сделать контроль без профиля и проверить
F0006/F0007 при n=4 уже выполнены более поздними сериями.

2026-10-17 настроены общие инструкции и передача контекста между Claude Code и
Codex. В этой задаче научный код, тесты, данные и старые результаты не менялись;
решатель, генератор и численные тесты не запускались.

## Реализованные компоненты

| Компонент | Фактическая реализация |
|---|---|
| [src/all_methods_expressions.py](src/all_methods_expressions.py) | Whitelist AST из 14 гладких ограниченных операций, рекурсивные аналитические jet-производные до второго порядка, bump с компактным носителем, сериализация формул и топологий. |
| [src/all_methods_dataset_v1.py](src/all_methods_dataset_v1.py) | Manufactured solutions, 4 сложности × 5 режимов, структурные семейства, split по семействам до генерации, адресные RNG-потоки, аналитические сертификаты, запись четырёх артефактов и полная проверка с регенерацией. |
| [all_methods_dataset_loader.py](all_methods_dataset_loader.py) | Строгая проверка CSV/JSONL, AST, формул, параметров, сертификатов и seeds; исполняемые функции восстанавливаются из AST. Загрузка одного ID проверяет весь набор. |
| [src/problem_types.py](src/problem_types.py) | Самостоятельный тип `CauchyProblem`, без зависимости от старого проекта. |
| [remizov_solver.py](remizov_solver.py) | Последовательная конечная аппроксимация Чернова, квадратура Gauss–Legendre с удвоением Nt, CubicSpline на расширенных рабочих сетках, уточнение dx/2 с полным пересчётом квадратуры. |
| [chernov_solver.py](chernov_solver.py) | Отдельная адаптация Chern на RemizovProblem: фиксированное n + Simpson и отдельная streaming-цепочка. Кэш коэффициентов/forcing, фиксированное окно, taper/zero, накопление квадратуры, spawn по узлам t. |
| [mc_solver.py](mc_solver.py), [examples/solve_mc_equation.py](examples/solve_mc_equation.py) | Монте-Карло Фейнмана — Каца на RemizovProblem: EM и трапеции, устойчивые моменты, адресный RNG, глобальные сертификаты, диагностика и лимиты. Автоматический T требует сертификата G; без него нужен явный T с неизвестным хвостом. |
| [all_methods_bounds.py](all_methods_bounds.py), [diagnostics/certify_forcing_bounds.py](diagnostics/certify_forcing_bounds.py) | Аналитический G из v1 AST, производных решения и bump; направленное округление, автоматическое подключение к MC, детерминированный обзор сертификатов/стоимости всех 20 задач без траекторий. |
| [examples/solve_chernov_equation.py](examples/solve_chernov_equation.py) | CLI обеих новых схем на существующем CSV/JSONL; JSON результата и E∞ на 33 точках. |
| [diagnostics/compare_chernov.py](diagnostics/compare_chernov.py), [diagnostics/report_chernov_comparison.py](diagnostics/report_chernov_comparison.py) | Явные планы, изолированные замеры, RSS, все попытки и отдельный отчёт из сохранённых данных. |
| [diagnostics/plan_chernov_dataset20.py](diagnostics/plan_chernov_dataset20.py), [diagnostics/report_chernov_dataset20.py](diagnostics/report_chernov_dataset20.py) | Фиксация единого протокола всех 20 записей перед запуском через compare_chernov; проверка покрытия, сохранности, ошибок и отдельная классификация точности по сохранённым результатам. |
| [diagnostics/chernov_monitored_batch.py](diagnostics/chernov_monitored_batch.py), [diagnostics/report_chernov_managed_completion.py](diagnostics/report_chernov_managed_completion.py) | Управляемые последовательные партии по 10–20 с паузами, тепловым контролем и проверяемым продлением lease; отдельный отчёт и сверка сохранённых ответов без вызова решателя. |
| [diagnostics/check_all_methods_production.py](diagnostics/check_all_methods_production.py) | Дополнительная проверка всех 1000 записей публичного загрузчика: векторные функции против NPZ, exact/производные/начальные условия, условия Ремизова и конструктор ChernovSolver. Решатели не вызывает. |
| [remizov_benchmark_loader.py](remizov_benchmark_loader.py), [create_remizov_benchmark.py](create_remizov_benchmark.py), [validate_remizov_benchmark.py](validate_remizov_benchmark.py) | Создание, загрузка и проверка пяти эталонов R1–R5 с точным решением `exp(-x²)`. |
| [verify_remizov_solver.py](verify_remizov_solver.py) | Ручные оракулы R1; Stage A: R1 при n=1,2,4,8,16,32; текущая константа Stage B задаёт R2–R5 при n=4,8. Прежнее описание n=1,2 было устаревшим; исходник не менялся. Наличие драйвера не подтверждает выполнение полной серии. |
| [examples/solve_generated_equation.py](examples/solve_generated_equation.py) | CLI: загрузка → Ремизов → JSON со статусом, конфигурацией и ошибкой относительно exact. Exit code 0 при success, 1 при штатном отказе; при отказе ошибка `null`. |
| [diagnostics/](diagnostics/) | Драйверы прежних четырёх серий и сравнения Chern, watchdog, профилирование и отдельные скрипты отчётов/графиков из сохранённых данных. |

API загрузки: `load_all_methods_equations(csv_path, metadata_path)` и
`load_all_methods_equation(...)`. Объект `AllMethodsEquation` содержит
`cauchy_problem`, `remizov_problem`, `y_exact`, `dy_exact`, `d2y_exact` и metadata.
`solve_remizov` возвращает `RemizovResult`: значения `f`, статус, причины отказа,
истории уточнения, время и счётчики работы. Производной численного решения нет;
при штатном отказе массив `f` пуст.

RK4, Product Integral и модели машинного обучения в этот проект не перенесены.
Названия `all_methods`, `_ml.npz`, поля split и ML-сетки сохранены как контракт v1.
Описание формата: [docs/ALL_METHODS_DATASET_V1.md](docs/ALL_METHODS_DATASET_V1.md).

## Математическая постановка и принятые соглашения

- Обычная форма: `a y″ + b y′ + c y = g`. Для Ремизова:
  `a y″ + b y′ + (c_R − λ)y = −g_R`, где `c = c_R − λ`, `g_R = −g`.
  Оператор `L = a∂² + b∂ + c_R`; решается `(λ − L)y = g_R` на всей ℝ.
- Искомое решение ограничено на ℝ. Решатель не принимает произвольные начальные
  условия. В генераторе `y0=y_exact(-1)`, `v0=y_exact′(-1)` получены из того же
  гладкого решения с компактным носителем, поэтому обе постановки согласованы.
- В `S(τ)` входит `c_R`, λ используется в резольвентном весе и усечении;
  знак сдвига дрейфа положительный. Ручные оракулы R1 при x=0:
  n=1 → 1.48, n=2 → 95/81; точное решение в этой точке равно 1.
- Глобальные условия: `0 < a_lower ≤ a ≤ a_upper`, `|b|≤B`, `|c_R|≤C`, `λ>C`.
  Сертификаты `analytic/all_real_x` следуют из конструкции; проверка на конечной
  сетке служит проверкой реализации, а не доказательством глобальных условий.
- Точное решение генератора — масштабированный bump × выражение AST; носитель
  строго содержит [-1,1]. Производные и forcing конструируются аналитически.
  Строки формул описательные: загрузчик не исполняет их через eval/sympify.
- Байтовая воспроизводимость v1 при одинаковых seed/config — часть контракта.
  Меняющие артефакты изменения требуют новой версии с сохранением чтения v1;
  константы версии в генераторе и загрузчике должны совпадать.

Подробности и правила изменений:
[генератор](.claude/rules/generator.md), [Ремизов](.claude/rules/remizov.md),
[методология экспериментов](.claude/skills/experiment/SKILL.md).

## Окружение и сохранённые данные

Локальное `.venv` существует; его Python при проверке 2026-10-17 — **3.10.11**.
Зафиксированы NumPy 2.2.6, SciPy 1.15.3, SymPy 1.14.0, pytest 9.1.1 в
[requirements.txt](requirements.txt) и [requirements-dev.txt](requirements-dev.txt).
Это также окружение прежних отчётов; повторный `pip check` в настройке контекста
не выполнялся. Команды запускаются из корня, `pytest.ini` задаёт `pythonpath = .`.
Каталога `.git` нет; Git-история и архив версий ядра ещё не созданы.

[MIGRATION_MANIFEST.json](MIGRATION_MANIFEST.json) фиксирует происхождение,
19 перенесённых файлов и адаптации. Три файла `unchanged_mathematical_core`:
`src/all_methods_dataset_v1.py`, `src/all_methods_expressions.py`,
`remizov_solver.py` — сейчас совпадают с `source_sha256`.
Остальные `source_sha256` описывают исходный перенос и не означают, что каждый
адаптированный файл обязан совпадать с исходной версией.

В [data/generated/](data/generated/) сохранён полный малый комплект:

- `all_methods_equations_v1.csv`, `_metadata.jsonl`, `_ml.npz`, `_report.json`;
- seed 2488, 20 семейств × 1 образец, ID `AMV1_F0000_S00`…`AMV1_F0019_S00`,
  ML-сетка 32, validation-сетка 257;
- по 5 уравнений каждого L1–L4, по 4 каждого режима; 20 структурных сигнатур,
  нет утечек семейств, точных дубликатов и отклонённых попыток по отчёту;
- split **train=0 / validation=10 / test=10** — следствие существующего округления
  при одном семействе в каждой из 20 ячеек. Это проверочный набор, не production split;
- `X` имеет форму `[20,4,32]`, каналы `[a,b,c,g]`, exact и dy_exact — `[20,32]`,
  массивы числовые/Unicode без object/pickle. Максимальная сохранённая невязка — 0.0.

Стандартная конфигурация **200 × 5 = 1000** теперь сохранена отдельно в
[data/all_methods_v1_1000_2026-10-18/](data/all_methods_v1_1000_2026-10-18/README.md).
До задачи подготовки production-набора полного комплекта в репозитории не было.
В [data/remizov_benchmark.csv](data/remizov_benchmark.csv) находятся только R1–R5,
целевой интервал [-2,2].

### Production-набор, подготовленный 2026-10-18

Источники: [отчёт подготовки](results/dataset1000_2026-10-18/REPORT.md),
[канонический report](data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1_report.json),
[полная валидация](results/dataset1000_2026-10-18/validation.stdout.json),
[проверка загрузки](results/dataset1000_2026-10-18/loader_check.json).
Сохранены CSV, JSONL, NPZ, канонический JSON-report и README в новом каталоге.
Малый набор `data/generated/`, прежние результаты, генератор и математическое
ядро не изменялись. Defaults путей остались прежними — новый набор надо указывать
явно. ID относятся к конкретному набору/конфигурации; совпадение ID с малым
набором не устанавливает тождество уравнения.

Использован существующий `generate_all_methods_dataset_v1.py`, без разработки
нового генератора или изменения формул/диапазонов. Штатный seed 2488,
200 семейств × 5 = **1000**, ML-сетка 256, validation-сетка 2049;
max_attempts=100, max_abs_safety=1e6, a_min=1e-3, residual_rtol=1e-10.
Полная конфигурация, команды, окружение и SHA-256 сохранены до запуска в
[plan.json](results/dataset1000_2026-10-18/plan.json) и before_sha256.json.
BLAS=1, процессы последовательные. Генератор со встроенной валидацией занял
16.717 с; отдельный валидатор — 12.480 с; проверка всех функций загрузчика —
30.491 с (wall time команд; это не время численного решения).

- Каждая сложность L1–L4: **50 семейств / 250 уравнений**.
- Каждый режим constant, slow_varying, moderate_varying, oscillatory,
  high_contrast: **40 семейств / 200 уравнений**.
- Каждая из 20 ячеек: **10 семейств / 50 уравнений**; 200 различных
  структурных сигнатур, по пять численных параметризаций каждой.
- Train/validation/test: **700/150/150 уравнений**, **140/30/30 семейств**,
  без пересечений семейств; точных символьных и числовых дубликатов нет.
- NPZ: X `[1000,4,256]`, каналы `[a,b,c,g]`, exact/dy_exact `[1000,256]`,
  y0/v0 `[1000]`, числовые/Unicode массивы без object/pickle.
- Начальные условия для других методов сохранены в CSV/JSONL/NPZ:
  y0=y_exact(-1), v0=y_exact′(-1). Ремизов их не использует;
  `c=c_R−λ`, `g_R=−g` и ограниченное на ℝ точное решение сохранены.

Встроенная и отдельная полная валидация `--expect-production` прошли.
Сертификаты для всех 1000 восстановлены из аналитической конструкции;
плотные сетки [-1,1] и [-10,10] являются smoke-проверками, не доказательством
глобальных границ. Проверены exact, аналитические производные, носитель,
нормализация, знаки, начальные условия, структура AST, split, уникальность
ID/seeds и безопасная сериализация. Побайтовая регенерация всех четырёх
канонических файлов из seed/config совпала. Максимальная наблюдаемая невязка
на обеих проверочных сетках — 0.0. Один кандидат near_zero_solution отклонён
штатно: AMV1_F0159_S00 принят со второй попытки, пропусков в итоговом наборе нет.

Публичным `load_all_methods_equations` успешно загружены все 1000 задач.
Дополнительный драйвер сопоставил векторные коэффициенты, exact/dy_exact
со всеми значениями NPZ, проверил начальные условия и невязку, условия
`_problem_conditions_valid` и конструктор ChernovSolver для каждой задачи.
Список по всем ID — [loaded_equations.csv](results/dataset1000_2026-10-18/loaded_equations.csv).
**Вызовов solve/solve_streaming — 0**. Набор готов как вход решателя;
успешность и точность вычислений на 1000 ещё не проверены. ML не запускался.

Профильные тесты AST, генератора и загрузчика:
[pytest.txt](results/dataset1000_2026-10-18/pytest.txt) — **44 passed,
619 subtests passed**, 3.73 с; одно прежнее предупреждение `\i` в docstring.
Новые тесты численных схем не требовались, сами схемы не менялись.
Хеши четырёх файлов — artifacts_sha256.json; итоговая сохранность — integrity.json
в том же каталоге результатов. Предыдущий статус сохранён как
[PROJECT_STATUS.before.md](results/dataset1000_2026-10-18/PROJECT_STATUS.before.md).
Следующий этап — отдельно заданный пользователем план численного расчёта,
с учётом того, что success не проверяет полную ошибку; здесь расчёт не начат.

## Что проверено

### Завершение solve на 1000 управляемыми партиями, 2026-10-18

Источники: [протокол v2](results/chernov_managed_completion_2026-10-18/plan_v2.json),
[итоговый отчёт](results/chernov_managed_completion_2026-10-18/final/REPORT.md),
[все 1000 итогов](results/chernov_managed_completion_2026-10-18/final/all_equations.csv),
[сверка](results/chernov_managed_completion_2026-10-18/final/verification.json).
Перед запуском --status подтвердил 700 терминальных записей:
633 accuracy_pass, 64 accuracy_not_met, 3 worker_exception; pending=300.
Это включает три дополнительные задачи после проверки возобновления
с 697 записями ниже. Список оставшихся ID зафиксирован до запуска.
Использован только ChernovSolver.solve, прежние единые параметры из серии 20:
n=128, num_t=513, quadratic, dx=0.0125, padding=12, cubic+taper,
tail_eps=1e-8, n_jobs=1, BLAS=1, T по сертификатам, E∞ на 33 точках [-1,1].
Exact участвует только в последующей оценке. Ядро и параметры не менялись.

Доработан `diagnostics/chernov_monitored_batch.py`: явный --all-pending,
партии по 10–20 (здесь 20), единая блокировка на всю серию и паузы, --stop
останавливает всю серию также во время охлаждения. ThermalState>nominal,
потеря телеметрии, лишний/непринадлежащий worker, CPU>125%, RSS>512 МиБ,
неудачная очистка или новый вычислительный отказ останавливают всю серию.
Разрешение guardian остаётся 120 с, проверяемое внешнее продление раз в 20 с
разрешено только в all-pending; истёкший lease не оживляется. Потеря backend
по-прежнему обнаруживается по EOF/5-секундному heartbeat, потеря/зависание
внешнего монитора прекращает продления. Автоповторов терминальных итогов нет.

Новые `tests/test_chernov_managed_batch.py` вместе с существующими тестами
монитора/guardian: **45 passed, 14.84 с**, короткие тестовые процессы,
без численных решателей. Проверены здоровое/отклонённое продление,
запрет оживления истёкшего lease, SIGKILL внешнего renewer при живом
контроллере с очисткой worker/потомка, паузы/блокировки/stop/resume,
штатная гонка ps с финальной записью guardian. Прежние тесты EOF/сигналов
контроллера/owner lease также прошли. Тестовые времена lease укорочены только
в фикстурах; production по-прежнему 120 с.

Первый управляющий запуск остановлен через команду монитора до новых ответов:
повторное полное сканирование результатов для каждого pending ID нагружало
монитор. Исправлено одним снимком на партию, добавлена регрессия; исходник,
оба протокола и журналы сохранены.
Первая сессия сохранила один interrupted-запрос AMV1_F0139_S03 (2.543990 с)
без численного ответа; при продолжении создана новая успешная попытка.
Всего новых запросов 301, терминальных ответов 300; опубликованные итоги
не пересчитывались. После повторного прохождения тестов
основная серия managed_run_v2 закончилась штатно, exit code 0:
**15 партий / 300 новых ответов / 238 accuracy_pass / 62 accuracy_not_met**,
новых вычислительных отказов нет. Wall **5292.375 с**, паузы **845.484 с**,
209 проверяемых продлений lease. Максимум worker=1, CPU=100.2% одного ядра,
RSS по сэмплам=179.5 МиБ, 2572 сэмпла nominal. Температура в °C недоступна.

Итог: **1000 терминальных записей, 997 конечных ответов, 871 accuracy_pass,
126 accuracy_not_met, 3 worker_exception, pending=0**. Прежние отказы
AMV1_F0134_S03, AMV1_F0136_S01, AMV1_F0139_S00 сохранены без повторов;
точные сигналы старый драйвер не записал. Все прежние 700 ответов и сырые
results совпали по SHA-256. Сверены 3181 прежний файл; численные исходники,
датасеты, исходный plan.json и safety_policy_v2 не изменились. Резервные
версии изменённых файлов — before/, revisions/.

Распределение E∞ по 997 ответам: median=0.00274797, p90=0.01260561,
p95=0.02270658, max=19704.50191. Очень большие конечные ошибки входят в
126 accuracy_not_met, а не в worker_exception; причины численной ошибки
в этой серии не исследовались. Новые solve: median=14.4169 с,
mean=14.4582 с, min=13.2953 с, max=16.2118 с. Времена прежней параллельной
и новой последовательной частей не являются контролируемым сравнением скорости.
Порог относится к фиксированной сетке, не к непрерывной области.
Процессы последней партии и монитора отсутствуют по проверке ОС,
backend-блокировки свободны. ML и другие методы не запускались.
Следующее исследование — только по отдельному заданию пользователя.

### История: первоначальный пакет solve прерван с 694 записями

Протокол до вычислений: [plan.json](results/chernov_solve1000_2026-10-18/plan.json),
геометрия и сертификаты: [preflight.json](results/chernov_solve1000_2026-10-18/preflight.json).
Реализация только `ChernovSolver.solve`, прежние численные схемы не менялись.
Полная конфигурация взята из успешной серии 20: n=128, num_t=513, quadratic,
dx=0.0125, padding=12, cubic+taper, tail_eps=1e-8, n_jobs=1. Проверка E∞≤0.01
на 33 точках [-1,1]. T по сертификатам, диапазон 10.2973…155.7748.
В исходной прерванной сессии было до 8 worker, BLAS=1; лимит задачи 120 с, startup 120 с,
RSS 512 МиБ/worker (до 4096 МиБ суммарно). Загрузка всех задач один раз на worker,
решения в нём последовательные. Начальная оценка — 15494 с суммарного времени
решателя, 2400–5400 с параллельного wall time; это оценка, не результат.

Добавлены `diagnostics/run_chernov_batch.py`, `diagnostics/report_chernov_batch.py`,
`tests/test_chernov_batch.py`. Атомарные отдельные результаты после каждой задачи,
журналы всех запросов/попыток, контрольные записи, блокировка параллельных
контроллеров в одном каталоге, watchdog времени/RSS, проверка родителя потоком
worker. Последняя не была независимым механизмом надзора и не связывала
жизнь процесса с остановкой Codex; запуск/остановка исправлены ниже.
Завершённые отказы не повторяются; незавершённая попытка сохраняется и при
продолжении получает новый запрос. **Не запускать без отдельного указания.**
Историческая команда следующего шага была заменена управляемым --all-pending
(серия выше уже завершена; полный текущий протокол — в инструкции):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/chernov_monitored_batch.py \
  --output-dir results/chernov_solve1000_2026-10-18 \
  --run --max-new 3 --thermal-state-only
```

Пилот: F0000/S00, F0060/S00, F0120/S00, F0180/S00, F0199/S04 — все конечные,
3/5 достигли порога, F0180/S00 E∞=0.41538645 и F0199/S04 E∞=0.01949062 не
достигли. Это не стало основанием менять конфигурацию. Повторный pilot-запуск
не создал новых расчётов и сохранил хеши пяти результатов: resume_check.json;
ошибки/схема/счётчики проверены в pilot_check.json. 28 тестов пакетного механизма
и прежней регрессии Chern прошли (1.37 с), tests.json. Один прежний warning `\i`.

Файлы solutions/<ID>.json содержат x/f, exact, ChernovResult с фактическими
параметрами, время, ошибку, флаг порога, вычислительный и точностный статусы.
Exact вызывается только после solve для проверки. progress.json — число
сохранённых результатов/статусы; sessions/ — история запусков; workers/ — RSS
и логи; attempts/ — все запросы и ответы. Итоговый отчёт будет создан только
после наличия 1000 результатов. ML, streaming и старый solve_remizov не запускаются.

#### Исправление надзора после принудительной остановки 2026-10-18

Источники: [отчёт](results/chernov_batch_safety_2026-10-18/REPORT.md),
[аудит ответов](results/chernov_batch_safety_2026-10-18/results_audit.json),
[инструкция](docs/CHERNOV_BATCH_EXECUTION.md),
[политика v2](results/chernov_solve1000_2026-10-18/safety_policy_v2.json).
Восемь процессов были явно заданы `workers=8` в исходном драйвере/плане;
n_jobs=1 не ограничивал внешнюю пакетную параллельность. Остановка Codex
прерывала наблюдение, но не обязательно сам ОС-процесс контроллера. Закрытие
VS Code не отслеживалось. Поток внутри worker наблюдал только за непосредственным
Python-родителем; независимого guardian и команды stop не было. Сессия контроллера
не записала finished_utc и успела заменить три завершившихся worker новыми.
Точные сигналы принудительного завершения старые логи не фиксируют.

Полная сверка solutions и attempts подтвердила **694** идентичных терминальных
записи, без непубликованных ответов. **691** содержит f/exact и прошёл проверку
ID, исходного плана, конфигурации, схемы solve и пересчёт E∞ на 33 точках:
**629 accuracy_pass, 62 accuracy_not_met**. Ещё **3 worker_exception** не имеют
численного ответа: F0134/S03, F0136/S01, F0139/S00. Без терминальной записи
**306** задач: **8** оборванных запросов и **298** ещё не запрошенных.
Список pending и таблица всех 694 записей сохранены в каталоге исправления.
Ни один старый результат, запрос, журнал, план или файл датасета не изменён.

Исправления:

- Драйвер жёстко запускает максимум один compute worker даже при `workers=8`
  в историческом плане; n_jobs должен быть 1, BLAS=1. Прямой внутренний worker
  без guardian отклоняется. Глобальная блокировка контроллера и отдельная
  блокировка compute-слота препятствуют второму запуску, включая другой каталог.
- Новый `diagnostics/chernov_batch_control.py` запускает независимый guardian.
  Он контролирует EOF/heartbeat контроллера, stop-файл и внешнее разрешение,
  останавливает всю отдельную группу worker (TERM→KILL), ожидает процесс.
  Сигналы SIGINT/SIGTERM/SIGHUP обрабатываются. Guardian удерживает run-locks
  до очистки; compute-FD наследуется worker и не освобождает второй слот при
  смерти guardian, пока старый worker ещё существует.
- Внешнее разрешение истекает через **120 с** без отдельного `--keep-alive`.
  Контроллер его не продлевает. Это ограничивает продолжение после исчезновения
  Codex даже при живом контроллере. Его собственный heartbeat ограничен 5 с.
  `--stop`, `--status`, `--check-runtime` не запускают вычисления.
- Атомарная публикация ответов и подхват ответа из attempts после сбоя сохранены.
  Сохранённые терминальные записи пропускаются; оборванные запросы сохраняются
  и получают новую попытку. Три терминальных отказа автоматически не повторяются.
- Старые версии драйвера, отчёта, тестов и статуса сохранены в
  `results/chernov_batch_safety_2026-10-18/before/`. Исходный plan.json и его хеш
  не менялись. safety_policy_v2.json фиксирует старые/новые хеши управляющего кода;
  математические источники/данные должны совпадать с исходным планом. Новые
  сессии/ответы записывают действующую политику и происхождение runtime-кода.

Проверки: [pytest_complete.txt](results/chernov_batch_safety_2026-10-18/pytest_complete.txt)
— **29 passed**, 12.43 с, только короткие спящие/фиктивные процессы, без
численных решателей. Проверены SIGKILL/SIGSTOP контроллера, SIGINT/TERM/HUP,
потомки, owner lease/продление, stop, обе глобальные блокировки, legacy workers=8,
запрет nested pool/прямого worker, атомарная запись, продолжение без повторов.
Повторная проверка выявила macOS PermissionError у signal-0 probe; исправлено
консервативной обработкой без прекращения TERM/KILL/reap, добавлена регрессия.
Уточнён тест ожидания освобождения блокировки после полного выхода guardian;
ранние неудачные логи сохранены. Проверки `--status/--stop/--check-runtime`:
[control_commands.json](results/chernov_batch_safety_2026-10-18/control_commands.json).
Активных процессов серии при проверке не обнаружено. **Массовый расчёт не
возобновлён**, все 694 старые записи и научное ядро сохранены; следующий шаг —
дождаться отдельного решения пользователя о продолжении.

#### Проверка безопасного возобновления, 2026-10-18

Источники: [план до запуска](results/chernov_resume_check_2026-10-18/plan.json),
[отчёт](results/chernov_resume_check_2026-10-18/REPORT.md),
[сверка](results/chernov_resume_check_2026-10-18/summary.json).
Выбраны первые три pending без подбора по exact: AMV1_F0138_S01, S02, S03.
Использован только `chernov_solver.ChernovSolver.solve`, исходная конфигурация
серии не менялась. Ни один из 691 прежних конечных ответов не пересчитывался,
три прежних worker_exception пропущены.

Первая партия (max-new=2) сохранила S01 и остановилась во время S02 через
`request_stop`; незавершённый запрос и interrupted.json сохранены, терминального
ответа S02 тогда не было. Контроллер/guardian/worker завершились, что отдельно
проверено по PID. Через 79.927 с после окончания первой партии вторая
(max-new=2) продолжила S02 и решила S03; S01 пропущен и сохранён без изменений.
Время решателя: 13.916 / 14.583 / 14.765 с; E∞ на прежних 33 точках:
0.00341514 / 0.00264064 / 0.00345236. Все три достигли 0.01.
Это проверка жизненного цикла на трёх параметризациях одного семейства,
а не исследование покрытия точности всего набора.

Добавлены `diagnostics/chernov_monitored_batch.py` и
`tests/test_chernov_monitored_batch.py`: партии строго 1–3 задачи, отдельная
блокировка монитора, телеметрия каждые 2 с, остановка при тепловом состоянии
выше nominal/потере телеметрии/лишнем worker/CPU>125%/RSS>512 МиБ, обязательная
пауза 60 с между партиями. Никаких автоматических продлений owner lease и
переходов к следующим партиям. Защитная политика v2 и её исходники не менялись.
Наблюдался максимум один worker, CPU до 100.1% одного ядра, RSS до 173.44 МиБ,
все 30 сэмплов nominal; температура в °C не измерялась. Пользователь явно
принял macOS thermalState вместе с собственным наблюдением за нагревом.
13 тестов оболочки прошли за 0.05 с, без численных вычислений.

Состояние на момент этой проверки: **697 терминальных записей / 303 pending**, из них
**694 конечных ответа** (632 accuracy_pass, 62 accuracy_not_met), прежние
3 worker_exception. Все прежние данные/ответы/логи/численные исходники
сохранены по SHA-256. Численные процессы сейчас остановлены, блокировки свободны.
Тогда следующий шаг после указания пользователя — одна наблюдаемая партия до трёх
pending командой выше, затем проверка нагрева и пауза. Весь остаток без
подтверждения не запускать. Этот план заменён явным разрешением на управляемое
завершение всех pending с проверяемым внешним продлением, описанным выше.

**Сохранённая первоначальная проверка**:
[docs/LOCAL_VERIFICATION.md](docs/LOCAL_VERIFICATION.md) и
[docs/local_verification_results.json](docs/local_verification_results.json).

- `70 passed, 1 deselected, 679 subtests passed`, 17.03 с. Отложен
  `test_r1_finite_chernoff_approximations_converge_toward_exact` (R1, n=1,2,4,8).
  Полный pytest и длинная серия `verify_remizov_solver.py` не подтверждены этим запуском.
- Проверены аналитические производные независимым SymPy-оракулом, AST, носитель,
  детерминизм, split, сертификаты, форматы и повреждения CSV/JSONL/NPZ,
  изоляция импортов, хеши ядра, оператор и знаки, ручные R1 n=1,2,
  R1–R5 при n=1, отказы и CLI F0000/F0005.
- Малый генератор, отдельный полный валидатор и загрузка всех 20 уравнений
  завершились успешно; воспроизводимость из seed/config подтверждена.
  Подтверждены отсутствие конфликтов зависимостей и независимые импорты проекта.
- Два предупреждения об escape-последовательности `\i` в математическом docstring
  оставлены ради сохранения исходных байтов. Отдельно зафиксировано прежнее
  изменение служебного `.DS_Store` исходного `generator_DE`; неизменность буквально
  всех его файлов в отчёте не утверждается.

**Проверка файлов 2026-10-17, без численных запусков**:

- Статически сверены 8 тестовых файлов: 64 определения тестов, 71 случай с учётом
  существующей параметризации. Это инвентаризация, не новый результат pytest.
- Сверены SHA-256 трёх файлов ядра с манифестом и трёх файлов данных с
  `output_sha256` сохранённого отчёта: совпадают. Полный валидатор не перезапускался.
- Четыре успешные ошибки ниже пересчитаны из сохранённых `f` и `y_exact` на 33
  точках: совпадают с JSON. Это проверка арифметики артефактов, не новый расчёт
  решателя и не независимое восстановление точного решения.
- Сверены 46 файлов трёх ранних каталогов `results/` с хешами, записанными в
  последнем `summary.json`: расхождений нет.

## Исторические численные результаты (до адаптации Chern)

Во всех четырёх сериях ниже целевая сетка `linspace(-1,1,33)` и проверочная
конфигурация: `tail_eps=1e-8`, `nt_initial=8`, `dx_initial=0.05`,
квадратурные/пространственные rtol=1e-7 и atol=1e-9, пределы уточнения **8/5**,
`max_work_grid_points=100000`. Библиотечный `RemizovConfig(n)` имеет пределы
**4/4**; это другой бюджет. Исторические внешние пределы серий — 45 с / 512 МиБ,
потоки BLAS=1. Они не являются общими ограничениями будущих задач.

### Последовательность серий

| Каталог / отчёт | Фактически выполнено |
|---|---|
| [remizov_n_study_2026-10-11](results/remizov_n_study_2026-10-11/REPORT.md) | F0000 n=1,2: отказ квадратуры; F0005 n=1,2: успех; F0000 n=4: внешний timeout. F0005 n=4 в этой серии записан как `not_run`. |
| [remizov_n4_followup_2026-10-12](results/remizov_n4_followup_2026-10-12/REPORT.md) | F0005 n=4 без профиля: успех; F0000 n=4 с cProfile: timeout, сохранён частичный профиль. |
| [remizov_f0000_n4_control_2026-10-12](results/remizov_f0000_n4_control_2026-10-12/REPORT.md) | Один контрольный F0000 n=4 без профиля: timeout. |
| [remizov_variable_n4_2026-10-12](results/remizov_variable_n4_2026-10-12/REPORT.md) | F0006 n=4: успех; F0007 n=4: timeout; оба без профиля. |

Итого в этих четырёх исторических сериях **10 фактических попыток**: 4 success, 2 `quadrature_not_converged`,
4 `external_time_limit`. Не включены первоначальная проверка в `docs/` и `not_run`.
`external_time_limit` — статус оболочки; его нельзя выдавать за причину штатного
отказа решателя. При внешней остановке `RemizovResult` не получен, E∞ неизвестна.

### Успешные расчёты

Здесь `E∞ = max|f-y_exact|` только на общей сетке 33 точек; время — внутри решателя,
из исследовательских серий, без смешения с первоначальным CLI-запуском в `docs/`.

| Задача | n | E∞ | final dx | final Nt | Время решателя, с | Применений S |
|---|---:|---:|---:|---:|---:|---:|
| AMV1_F0005_S00 | 1 | 3.0429668353311623 | 0.00625 | 512 | 8.8910 | 10208 |
| AMV1_F0005_S00 | 2 | 0.731014101169212 | 0.003125 | 512 | 21.1404 | 22448 |
| AMV1_F0005_S00 | 4 | 0.3807441844324305 | 0.003125 | 512 | 30.4348 | 36704 |
| AMV1_F0006_S00 | 4 | 0.09481647848327257 | 0.00625 | 1024 | 22.7129 | 32640 |

- **Наблюдение:** у F0005 ошибка убывает на n=1,2,4; отношения E1/E2≈4.163,
  E2/E4≈1.920. Асимптотический порядок и предел при n→∞ не установлены.
  При n=4 ошибка всё ещё ≈57.70% амплитуды exact на тех же 33 точках.
- F0006 — сохранённый успех с переменными коэффициентами (`slow_varying`, L2);
  ошибка ≈6.42% амплитуды. Зависимость от n пока не исследована.
- F0005/F0006/F0007 различаются не только режимом, но также семействами, AST,
  forcing, амплитудами, носителями, сертификатами, gap и рабочей геометрией.
  Их сравнение не устанавливает причинное влияние одного режима коэффициентов.

### F0000 и F0007: отказы и стоимость

- F0000 n=1,2: квадратура не принята уже при первом dx=0.05, final Nt=2048,
  `f=[]`; последние разности ≈4.0437e-7 и ≈9.0269e-8. Ошибки решения нет.
- F0000 n=4: три внешних остановки примерно на 45.263 с (первая серия),
  45.014 с (профиль), 45.012 с (контроль без профиля). Отключение cProfile
  не обеспечило завершение в историческом бюджете; накладные расходы профиля
  этим сравнением не измерены, так как обе попытки незавершённые.
- Только у профилируемой попытки известна стадия остановки: Nt=2048,
  dx=0.003125. Квадратура на четырёх более грубых dx принята; три завершённых
  пространственных сравнения ещё не приняты. Последнее ≈1.0115e-6 при пороге
  ≈7.0771e-8. Это частичная диагностика, не итоговый результат решения.
- Профиль показывает многократные S, построение/вычисление CubicSpline и полный
  пересчёт квадратуры при dx/2 как источники затрат. Стадии остановки запусков
  без профиля неизвестны; переносить на них частичную историю нельзя.
- F0007 n=4: внешний timeout на ≈45.017 с, наблюдаемый RSS ≈205.33 МиБ;
  значения, ошибка, финальные Nt/dx и внутренние истории неизвестны.
  Во всех сериях наблюдаемый RSS ниже прежнего порога; периодический опрос
  не исключает пиков между измерениями.

Каждый каталог содержит `summary.json`, исходные записи попыток, отчёт и
`verification.json`; более поздние серии также сохраняют preflight,
ресурсные ряды и сравнения. Отчётные скрипты читают сохранённые файлы и не решают
уравнения повторно. Единственная диагностическая адаптация последней серии —
добавление F0006/F0007 в argparse choices `diagnostics/remizov_n4_followup.py`;
математические worker/watchdog-функции при этом не менялись.

## Адаптация Chern и сравнение 2026-10-18

Источники: [отчёт](results/chernov_integration_2026-10-18/REPORT.md),
[таблица](results/chernov_integration_2026-10-18/comparison.csv),
[все попытки](results/chernov_integration_2026-10-18/all_attempts.csv),
[проверка сохранности](results/chernov_integration_2026-10-18/verification.json).
API и запуск описаны в [docs/CHERNOV_ADAPTATION.md](docs/CHERNOV_ADAPTATION.md).

Предоставленные Chern.py и example.ipynb сохранены побайтово в comparison_inputs/;
SHA-256 и источник — provenance.json. Исходный remizov_solver.py, генератор,
движок AST, загрузчик и все исходные артефакты данных сохранены побайтово.
README и прежний статус до этой задачи скопированы в
archive/chernov_integration_before_2026-10-18/. Манифест ядра не менялся.

Новый ChernovSolver принимает RemizovProblem с правильными c_R и g_R и
глобальными сертификатами. Исправлены поддержка скалярного forcing, построение
концов сетки, валидация и контракт отказа f=[]. Коэффициенты/forcing кэшируются,
геометрия сдвигов переиспользуется при одном τ. Simpson накапливается без
матрицы «время × пространство»; неравномерные веса сверены с SciPy. Параллельный
solve передаёт spawn-процессам только массивы, не пользовательские замыкания.

Схемы различаются: solve использует одно n во всех узлах t;
solve_streaming использует u_k=S(τ)^k g_R, τ=T/K. Фиксированное окно и
продолжение taper/zero вводят отдельное пространственное приближение.
Автоматических уточнений новых схем нет; success означает конечный расчёт,
не сертификат точности. boundary_state_max_abs также не граница ошибки.

В сравнительных сериях сохранены 107 завершённых попыток (включая подбор и
контроли) и одна инфраструктурная попытка без результата: sandbox запретил ps.
Наблюдение заменено worker self-reporting через resource.getrusage, исходные
файлы неудачной попытки сохранены. Это не отказ численного решателя.

Итог: 36 успешных попыток = 3 повтора × 4 задачи × 3 варианта. Ошибка
E∞ проверялась на общей для методов сетке: R1 — 257 точек [-2,2], датасет —
33 точки [-1,1]. Ниже медиана времени решателя, включая подготовку и проекцию,
без загрузки CSV/JSONL/стартового импорта Python. BLAS=1, n_jobs=1;
wall time и пиковый RSS также сохранены. Повторы дали идентичные массивы f.

| Задача | Старый: E∞ / с | Новый solve: E∞ / с | streaming: E∞ / с |
|---|---:|---:|---:|
| R1 | 0.00949155 / 0.3478 | 0.00949163 / 0.2676 | 0.00990161 / 0.0659 |
| F0005 | 0.00781008 / 5.7923 | 0.00782526 / 1.0367 | 0.00944366 / 1.0565 |
| F0006 | 0.00546596 / 10.8410 | 0.00535553 / 2.1153 | 0.00402738 / 0.5333 |
| F0007 | 0.00646274 / 39.3280 | 0.00645704 / 1.1357 | 0.00729074 / 0.2847 |

Параметры фиксированного n: 8/32/64/32; K streaming: 512/8192/4096/2048.
Новые схемы: dx=0.025, padding=8, cubic+taper; solve — quadratic, 257 узлов.
Исходный решатель намеренно получил более мягкий бюджет, чем verification:
обе пары rtol/atol=1e-3/1e-6, уточнения 7/4, dx_initial=0.05, nt_initial=8,
max_grid=200000. Полная ошибка независимо проверена по exact; одинаковая
цель E∞≤0.01 выполнена всеми. Это не сравнение при стандартных допусках 1e-7.

Контроли: dx/2 изменил новые ответы не более чем на 1.61e-4, удвоенное окно —
менее 1e-9, 513 узлов fixed-n Simpson — не более 3.17e-5. При streaming
уменьшение tail_eps с фиксированным K одновременно увеличивает τ и меняет
временную ошибку. Отдельный контроль с K∝T сохранил τ приближённо и изменил
ответы менее чем на 4.02e-5. Удвоение K уменьшало полную наблюдаемую ошибку,
но порядок сходимости не заявлен. Контроли — индикаторы, не строгие границы.

Проверки: [pytest.txt](results/chernov_integration_2026-10-18/pytest.txt) —
**84 passed, 1 deselected, 679 subtests passed**, 19.16 с; исключён прежний
длительный тест R1 n=1,2,4,8. Два прежних escape-warning оставлены в ядре.
Новые 14 случаев проверяют оператор, ручные R1, эквивалентность исходному Chern
для обеих схем и интерполяций, веса Simpson, отказы, spawn и запуск обеих схем
на всех 20 существующих записях (последнее — только smoke, не точность).
Обе CLI-команды на F0005 прошли с E∞≤0.01: cli_check.json.

Подтверждённый вывод: при этой цели и на этих сетках обе адаптированные схемы
быстрее исходного решателя; fixed-n — 1.30–34.63×, streaming — 5.27–138.13×.
На F0005 обе новые схемы сопоставимы, на F0006/F0007 streaming заметно выгоднее.
Отдельный вклад кэширования, квадратуры и геометрии не измерялся; универсальное
превосходство на других задачах/точностях не установлено. Подбор параметров не
включён в итоговое время и полностью сохранён. Полные 1000 уравнений и ML не
запускались. Намеченная расширенная проверка диапазона задач выполнена следующей
серией ниже; автоматический контроль полной ошибки по-прежнему не реализован.

## Chern на всём малом датасете 2026-10-18

Источники: [REPORT.md](results/chernov_dataset20_2026-10-18/REPORT.md),
[attempts.csv](results/chernov_dataset20_2026-10-18/attempts.csv),
[statistics.json](results/chernov_dataset20_2026-10-18/statistics.json),
[runs/summary.json](results/chernov_dataset20_2026-10-18/runs/summary.json).
В отчёте — полная таблица всех 20 уравнений; сырые результаты содержат f,
y_exact, полную конфигурацию, относительную ошибку, счётчики и время.

Задача — проверить покрытие E∞≤0.01 при ограниченных ресурсах перед решением
о массовом запуске. До вычислений записаны
[plan.json](results/chernov_dataset20_2026-10-18/plan.json) и
[preflight.json](results/chernov_dataset20_2026-10-18/preflight.json): ровно
40 попыток, по одной на пару ID/схема, без повторов и подстроек по exact.
Для всех уравнений solve: n=128, quadratic Simpson, 513 узлов; streaming:
K=32768, равномерный шаг. Общие dx=0.0125, padding=12, cubic+taper,
tail_eps=1e-8, 2081 рабочая точка на [-13,13], целевая сетка — 33 точки [-1,1].
T=11.6272…88.3899 получается из сертификатов λ−C по неизменному общему правилу;
y_exact оценивается только после решения и не участвует в выборе параметров.
Это больший фиксированный бюджет относительно прежнего сравнения, а не поиск
минимальной стоимости достижения порога.

Один worker за раз, n_jobs=1, BLAS=1; лимит каждой попытки — 60 с / 512 МиБ.
Watchdog автоматически завершает превышение и продолжает серию, не повторяя
попытку. Оценка до запуска: 65536/32768 применений S; ожидаемые времена
10–25/5–15 с, верхний суммарный бюджет worker — 2400 с плюс завершение процессов.
Ресурсные сэмплы getrusage и логи сохранены; исключений и ресурсных остановок
в этой серии не было. Окружение и SHA-256 — в summary.json.

| Показатель | solve | streaming |
|---|---:|---:|
| Конечный результат | 20/20 | 20/20 |
| E∞≤0.01 | **20/20 (100%)** | **18/20 (90%)** |
| Максимальная E∞ | 0.00893517 (F0002) | 0.0174607 (F0018) |
| Время решателя: среднее / медиана, с | 15.494 / 15.493 | 7.783 / 7.847 |
| min…max / p95, с | 13.864…17.146 / 16.553 | 6.821…8.720 / 8.248 |
| Суммарное время решателя, с | 309.884 | 155.668 |
| Полное время worker: среднее / сумма, с | 15.917 / 318.349 | 8.238 / 164.769 |
| Максимальный RSS, МиБ | 83.52 | 82.97 |

Времена — одиночные измерения на каждой задаче; медиана в таблице относится
к 20 разным уравнениям. Время решателя включает его подготовку и проекцию,
исключает загрузку датасета и старт Python; wall time включает процесс целиком.
Streaming в среднем быстрее в 1.99 раза, но точнее не на каждой задаче.
Единственный наблюдавшийся тип отказа по цели — конечный ответ с E∞>0.01:
AMV1_F0002_S00 (L1/moderate_varying), 0.01181397, и AMV1_F0018_S00
(L4/oscillatory), 0.01746071. Оба streaming-вызова вернули success=True.
Вклад K, интерполяции, окна, квадратуры и хвоста в эти ошибки не разделялся;
причину сверх наблюдаемого недостижения точности эта серия не устанавливает.

Проверки: [pytest.txt](results/chernov_dataset20_2026-10-18/pytest.txt) —
**22 passed**, 1.51 с: 8 новых случаев протокола/статусов/статистики и 14
прежних случаев обеих схем (оператор, оракулы, исходный Chern, отказы, spawn,
smoke всех записей). Один прежний DeprecationWarning для `\i` в AST-модуле.
Отчётный скрипт пересчитал E∞ по сохранённым массивам и проверил все 40 пар,
полное совпадение конфигураций и отсутствие повторов.
[verification.json](results/chernov_dataset20_2026-10-18/verification.json)
подтвердил неизменность 649 прежних файлов, включая ядро, данные, старые результаты
и численные реализации. Схемы, их интерфейсы и общие defaults не менялись.
Добавлены только два диагностических скрипта, тест протокола, новые результаты
и обновление статуса. Прежний статус сохранён в
[PROJECT_STATUS.before.md](results/chernov_dataset20_2026-10-18/PROJECT_STATUS.before.md).

Решение: инфраструктура пригодна для ограниченного пакетного исполнения и учёта
отказов; solve — более перспективный кандидат для будущего большого набора при
этом бюджете. Однако 20/20 на 33 точках не гарантирует точность на новых 1000
уравнениях или между узлами, а streaming уже не обеспечивает полное покрытие.
Бесконтрольный массовый расчёт с success в роли метки точности преждевременен.
Дальнейшая политика контроля точности остаётся отдельной задачей пользователя;
дополнительные уточнения, полный набор 1000 и ML в этот обзор не входили и не
запускались. После этого обзора работа остановлена до дальнейших указаний.

## Открытые вопросы и ограничения

1. **Полная ошибка.** В исходном решателе `success` означает принятие внутренних критериев, не малую
   ошибку точного решения. Разности Nt и dx — индикаторы, а не доказанные границы;
   их нельзя вычитать из E∞ для получения «чистой ошибки конечного n». В адаптации
   Chern success означает только конечный расчёт; проверки уточнений внешние.
2. **Усечение хвоста.** `T=max(0, log(2/(gap*tail_eps))/gap)` не использует
   глобальную норму `g_R`. В оценке хвоста идеального оператора остаётся множитель
   `‖g_R‖∞`; универсальная абсолютная трактовка `tail_eps` не обоснована.
3. **Первоисточник.** Локальной копии статьи и полного математического аудита
   дискретных схем в проекте нет. В предыдущем сравнительном анализе использован
   [первоисточник, Theorem 6](https://doi.org/10.46698/a3908-1212-5385-q);
   его оценки для точного S не переносятся автоматически на сплайны и taper.
4. **Сходимость и производительность.** Нет установленного порядка сходимости
   решателя на сгенерированных задачах,
   строгого разделения ошибок n/T/Nt/dx нет; CubicSpline не гарантирует сохранения
   операторной оценки нормы. В отдельной адаптации реализованы кэширование и
   Simpson, выполнены внешние контроли уточнений; вклад каждого изменения в
   ускорение не измерялся. Экстраполяция по n не реализована. Timeout сам по себе
   не доказывает математическую недопустимость.
5. **Запись и пути.** Генератор/загрузчик по умолчанию используют `data/`, малый
   набор лежит в `data/generated/`. Генератор заменяет одноимённые файлы до полной
   проверки комплекта; генератор эталонов и верификатор тоже имеют существующие
   пути по умолчанию. Для новых работ нужны явные пути и сохранность прежних версий.
6. **Устаревший текст.** В docstring `all_methods_dataset_loader.py` осталось
   «all three implemented methods» из исходного проекта. Здесь реализован только
   Ремизов (теперь с отдельной адаптацией Chern); исправление этого docstring не
   входило в задачи настройки контекста и адаптации.

## Принятые решения и следующие шаги

- Прежние серии сохраняют исходную математику и критерии; отказы не скрывались
  увеличением бюджета. В отдельной серии 2026-10-18 использован явно записанный
  более мягкий бюджет под цель E∞≤0.01, проверенную по exact. Это не основание
  менять общие библиотечные допуски или переписывать исторические результаты.
- После контроля F0000 исследование переключено на другие задачи. Автоматически
  повторять ту же остановленную попытку без нового вопроса не планировалось.
- **Исторический план, заменённый адаптацией Chern 2026-10-18:** F0006 n=1,2, сравнение с сохранённым
  n=4 на той же сетке и с теми же численными критериями. План и ресурсы следующей
  серии определяются её задачей; прежние 45 с/512 МиБ не обязательны для всех работ.
- **Возможное продолжение:** отдельно F0007 n=1,2. F0003 (L1, oscillatory) только
  предварительно оценён геометрически, не решался в сохранённых сериях.
- Математический аудит хвоста/первоисточника, независимые эталоны конечного n,
  разделение источников ошибки и дальнейшая оптимизация остаются направлениями;
  адаптация Chern и её контроли теперь реализованы отдельно. До 2026-10-18 серии
  решателя на сгенерированных задачах при n=8,16,32
  не были выполнены; адаптация покрыла n=8,32,64 на трёх задачах, затем
  обзор всего малого датасета — n=128 на всех 20 без индивидуального подбора.
  Подготовка production-набора выполнена следующей задачей 2026-10-18;
  численное решение 1000 и полная длительная верификация не запускались.
  Создание Git-репозитория, commit/push также не выполнялись.

Исторические формулировки «после согласования» и лимиты в старых отчётах описывают
условия тех задач. Они сохранены как история и не вводят новых разрешительных
барьеров: текущий порядок работы определён `AGENTS.md` и конкретной задачей.

## Передача контекста, настроенная 2026-10-17

Созданы `AGENTS.md` и этот статус. В README добавлено описание передачи контекста,
а пример генерации направлен в отдельный каталог вместо сохранённого набора.
`CLAUDE.md` содержит только `@AGENTS.md`;
общие правила не дублируются. Оба математических файла `.claude/rules/` сохранены
побайтово; `AGENTS.md` направляет к ним и Codex. В experiment skill сохранена
методология, убраны обязательное согласование каждого малого опыта и трактовка
исторических ресурсных лимитов как универсальных.

Новый чат из корня проекта получает общие инструкции напрямую (Codex) или через
импорт в `CLAUDE.md` (Claude Code). По ним агент читает `PROJECT_STATUS.md` и нужные
исходники в начале задачи, а после существенной работы обновляет статус в той же
задаче. Передаётся сохранённое состояние исследования, а не полная история чатов.
При параллельной работе агентов перед записью нужно перечитать актуальный статус
и сохранить существенные дополнения другого участника.

Статическая проверка настройки пройдена: импорт `@AGENTS.md`, 36 локальных ссылок,
сохранение frontmatter skill, согласованность инструкций и SHA-256 математического
ядра. Сравнение со снимком до настройки подтвердило: изменены только `CLAUDE.md`,
README и experiment skill, добавлены два документа; остальные 109 файлов снимка,
включая научный код, тесты, rules, данные и результаты, не изменились. Окружение
и кеши не включались в снимок. Отдельные новые сессии CLI Claude Code и Codex для
проверки фактической загрузки не запускались.
