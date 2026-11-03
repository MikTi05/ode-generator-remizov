"""Compose the evidence dossier from already audited artifacts; no fitting/solving."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
USED = json.loads((OUT / 'used_sources.json').read_text())
parts = []


def source(path, title=None, line=None):
    p = ROOT / path
    assert p.is_file(), path
    USED[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    suffix = '' if line is None else f'#L{line}'
    return f'[{title or path}](../../{path}{suffix})'


def read(path):
    source(path)
    return json.loads((ROOT / path).read_text())


def frame(path):
    source(path)
    return pd.read_csv(ROOT / path, float_precision='round_trip')


def add(value):
    parts.append(str(value).strip() + '\n')


def fmt(value, digits=9):
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return '—'
    if isinstance(value, (bool, np.bool_)):
        return 'да' if value else 'нет'
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return format(float(value), f'.{digits}g')
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(headers, rows, digits=9):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join('---' for _ in headers) + '|'] +
                     ['| ' + ' | '.join(fmt(v, digits) for v in row) + ' |' for row in rows])


def figure(number):
    m = json.loads((OUT / 'figure_manifest.json').read_text())[number - 1]
    return (f'![Рисунок {number}. {m["title"]}](figures/{m["file"]})\n\n'
            f'Рисунок {number}. {m["notes"]} [PDF](figures/{m["pdf"]}). '
            'Источники и параметры визуализации: [figure_manifest.json](figure_manifest.json), '
            '[plot_evidence.py](plot_evidence.py).')


def main():
    H = 'results/ml_selector_hypotheses_2026-10-31/'
    F = 'results/four_method_accuracy_1000_2026-11-01/'
    D = 'data/all_methods_v1_1000_2026-10-18/'
    comparison = frame(str((OUT / 'tables/main_model_comparisons.csv').relative_to(ROOT)))
    policies = frame(str((OUT / 'tables/model_policy_metrics_all.csv').relative_to(ROOT)))
    cases = frame(str((OUT / 'tables/diagnostic_258_cases.csv').relative_to(ROOT)))
    distributions = frame(str((OUT / 'tables/four_method_distributions.csv').relative_to(ROOT)))
    grouped = frame(str((OUT / 'tables/four_method_group_distributions.csv').relative_to(ROOT)))
    cv = frame(str((OUT / 'tables/cv_policies_22.csv').relative_to(ROOT)))
    config = read(H + 'plan.json')
    data_report = read(D + 'all_methods_equations_v1_report.json')
    add('''# Доказательная база: ML выбора метода и сравнение четырёх численных схем

Дата среза: **2026-11-01**, проект `ode-generator-remizov`. Это сбор и проверка материалов, а не окончательный научный отчёт. Описываются существующие артефакты, включая отрицательные результаты. Ни одна модель не обучалась заново, решатели и генератор не запускались.

## 0. Как читать документ и что проверено

**Факт** ниже означает значение, подтверждённое исходником, сохранённой записью или пересчётом её арифметики. **Наблюдение** — описание данного набора. **Гипотеза** — объяснение или перенос за его пределы, для которого нет достаточного подтверждения. Средние, квантили, таблицы ошибок и графики вновь вычислены только из уже сохранённых данных. Это не новые численные эксперименты.

Три результата, которые нельзя смешивать:

- Исторический ML качества/стоимости решал задачу достижения **E_grid33 ≤ 0.01** фиксированными профилями R128/R256/MC1. Его вероятность успеха, порог риска q и время относятся к конкретному профилю.
- Новый ML двух методов предсказывает, у кого **меньше абсолютная ошибка в x=0**, сравнивая исторический R128 и среднее четырёх MC1024. Здесь **ε=0.05** — отдельная оценка приемлемости, а не правило winner.
- Сравнение четырёх методов добавляет RK4 и левый экспоненциальный Product Integral при **N=128 по x**. Историческое `n=128` Ремизова и MC `dt=1/128` имеют другой смысл и другой бюджет.

Числа в обзорных таблицах округлены только для чтения. Полные float64-значения сохранены в CSV; разметка и проверки использовали неокруглённые значения. Пропуск результата не заменяется нулём, бесконечностью или отрицательным классом.
''')
    add(table(['Проверка в этой работе', 'Объём', 'Доказательство'], [
        ['Инвентаризация прежнего дерева', '24897 файлов с SHA-256; архивные копии выделены', '[repository_inventory.csv](repository_inventory.csv), [before_sha256.json](before_sha256.json)'],
        ['Индекс активного Python-кода', '249 файлов; 2006 AST-символов; ошибок разбора 0', '[code_index.csv](code_index.csv), [checks/inventory.json](checks/inventory.json)'],
        ['Восстановление 258 признаков', '997 × 258 = 257226 значений, побитовое равенство', '[checks/feature_values.json](checks/feature_values.json)'],
        ['Повторное чтение сохранённых CBM', '96 моделей; 96 predict; fit=0', '[checks/model_scores.json](checks/model_scores.json)'],
        ['Метрики основных политик', '844 × 3 split; сверено 1688 прежних Val/Test строк', '[checks/metrics.json](checks/metrics.json), [полные 2532 строки](tables/model_policy_metrics_all.csv)'],
        ['Сохранённый family-CV', '110 fold-метрик и 22 pooled-политики', '[checks/cv_saved_arithmetic.json](checks/cv_saved_arithmetic.json)'],
        ['Первичные численные записи', '2000 новых IVP-записей, 1000 R-записей, 4000 MC-записей', '[checks/raw_saved_records.json](checks/raw_saved_records.json)'],
        ['Графики', '14 PNG + 14 PDF; визуальная проверка контактных листов', '[figure_manifest.json](figure_manifest.json), [checks/visual_qa.json](checks/visual_qa.json)'],
    ]))
    add('''Train-метрики 96 сохранённых моделей получены **сейчас только посредством inference** на их собственном Train, без обучения. Они являются in-sample и не заменяют OOF. Для Validation и Test scores сохранённых CBM в точности совпали с прежними scores; метрики независимо сверены. Для более старых ML-этапов ниже используются сохранённые таблицы/аудиты; их полное повторное обучение не выполнялось.

Структура всего репозитория проиндексирована; смысловая проверка сосредоточена на действующих источниках данных, математических контрактах и цепочке ML-артефактов. Архивы `before/`, `sources/`, резервные копии и рисунки не считаются отдельными экспериментами. Индекс не означает независимого доказательства каждой строки всех архивных реализаций.
''')
    add('## 1. Репозиторий, постановка и реальный датасет')
    add(table(['Компонент', 'Назначение и проверяемый источник'], [
        ['Контекст/правила', source('AGENTS.md') + '; ' + source('PROJECT_STATUS.md') + '; ' + source('MIGRATION_MANIFEST.json')],
        ['Генератор all_methods_v1', source('src/all_methods_dataset_v1.py', 'AllMethodsConfig / generate_dataset / validate_record', 99)],
        ['Безопасные выражения и exact', source('src/all_methods_expressions.py', 'SafeExpressionNode / evaluate_solution', 657)],
        ['Загрузка', source('all_methods_dataset_loader.py', 'AllMethodsEquation / load_all_methods_equations', 144) + '; ' + source('src/problem_types.py', 'CauchyProblem', 10)],
        ['Ремизов/Чернов', source('remizov_solver.py') + '; ' + source('chernov_solver.py')],
        ['MC', source('mc_solver.py') + '; ' + source('mc_optimized.py')],
        ['RK4 / Product Integral', source('src/numerical_methods/rk4_solver.py', 'solve_rk4', 35) + '; ' + source('src/numerical_methods/product_integral_solver.py', 'solve_product_integral_left_exponential', 262)],
        ['ML-входы/метрики/политики', source('ml/remizov_features.py') + '; ' + source('ml/selector_research_features.py') + '; ' + source('ml/accuracy_selector.py') + '; ' + source('ml/selector_hypotheses.py')],
        ['Экспериментальные драйверы', '[code_index.csv](code_index.csv), строки diagnostics/; ' + source('diagnostics/run_selector_hypotheses.py') + '; ' + source('diagnostics/run_rk4_product_integral_mass.py')],
        ['Сохранённые исследования', '[report_index.json](report_index.json) — 59 текущих отчётов/README; [historical_sources.json](historical_sources.json)'],
    ]))
    add(f'''### 1.1. Уравнения и точное решение

Реально хранящаяся задача: `a(x)y'' + b(x)y' + c(x)y = g(x)` на рабочем интервале **[-1,1]**. Сырые безопасные AST `a_raw`, `b_raw`, `c_raw` параметризуются в семействах; затем `a=A_mid+A_half*tanh(a_raw)`, `b=B*tanh(b_raw)`, `c_R=C*tanh(c_raw)`, `λ=C+gap`, `c=c_R−λ`, `g_R=−g`. В частности `a≥A_minus>0`, `λ−C>0`, `q=−c>0` на всей прямой по конструкции, а не только на сетке.

Генератор сначала задаёт известное точное решение `y_exact=scale*bump(mu,radius)*y_shape`. Нормированный bump гладок, имеет компактный носитель; носитель строго покрывает рабочий [-1,1]. Правая часть производится как `g=a*y_exact''+b*y_exact'+c*y_exact`. Аналитические производные вычисляются безопасным jet2. Начальные `y0=y_exact(-1)` и `v0=y_exact'(-1)` **сохранены**. Их не требуется повторно получать для RK4/PI. Источники: {source('src/all_methods_dataset_v1.py', '_coefficient_values / _all_values', 567)}, {source('src/all_methods_expressions.py', 'normalized_bump_jet2 / evaluate_solution', 597)}, {source('all_methods_dataset_loader.py', '_validate_numeric_reconstruction', 699)}.

Глобальные R/MC решают ограниченную резольвентную задачу `(λ−L)y=g_R` на ℝ, где `L=a*d²/dx²+b*d/dx+c_R`. RK4/PI решают задачу Коши по сохранённым НУ. На manufactured-наборе эти постановки задают одно и то же известное решение; для произвольного нового уравнения глобальная ограниченность сама по себе не поставляет начальные условия. Для будущего выбора между четырьмя методами нужны корректные доступные НУ либо отдельно обоснованный способ их задания. Использование сохранённых НУ здесь допустимо, но является существенным условием эксперимента.

Безопасный загрузчик восстанавливает scalar/vector callable из whitelist AST, проверяет форму метаданных, конечность, сертификаты, остаток и согласованность НУ. Формульные строки CSV не исполняются через произвольный eval. Сохранённый report сообщает `global_max_residual=0.0` на проверочной сетке 2049 узлов: это согласованность manufactured-вычисления, **не независимое доказательство точности решателя** и не замена аналитических сертификатов. Источники: {source('all_methods_dataset_loader.py')}, {source(D+'all_methods_equations_v1_report.json')}.
''')
    add(table(['Параметр/артефакт', 'Фактическое значение'], [
        ['Версия / seed', 'all_methods_v1 / 2488'],
        ['Объём', '1000 уравнений = 200 структурных семейств × 5 параметризаций'],
        ['Сложности', 'L1/L2/L3/L4: по 250; в каждой из 20 ячеек сложность × режим по 50'],
        ['Режимы', 'constant, slow_varying, moderate_varying, oscillatory, high_contrast: по 200'],
        ['Первичный split', 'Train700 / Validation150 / Test150; семьи140/30/30'],
        ['CSV', source(D+'all_methods_equations_v1.csv') + ': equation_id,a_expr,b_expr,c_expr,g_expr,x0,x1,y0,v0'],
        ['JSONL', source(D+'all_methods_equations_v1_metadata.jsonl') + ': AST, construction, exact/derivatives, certificates, identity, topology, regime, complexity, split, НУ'],
        ['NPZ', source(D+'all_methods_equations_v1_ml.npz') + ': X[1000,4,256], channel_order=[a,b,c,g], x_grid; дополнительные exact/метаданные не являются ML-входами'],
        ['ML-сетка', '256 узлов [-1,1]; x=0 отсутствует. Нельзя считать соседний отсчёт exact(0) или численным решением в нуле'],
        ['CSV SHA-256', '940d6c10a8cdd0933bb5c663c681efd39bac5ed450ea37cbb95be1ca481136ad'],
        ['Минимальный a_lower_cert по 1000', data_report['certified_ranges']['a_lower_cert']['min']],
        ['Минимальный gap по 1000', data_report['certified_ranges']['remizov_gap_cert']['min']],
    ]))
    add(f'''Источники всей таблицы: {source(D+'all_methods_equations_v1_report.json')}, {source(D+'README.md')}, {source('results/dataset1000_2026-10-18/REPORT.md')}. Реальные поля проверены по файлам, а не предположены по названию.

**Почему 997:** у трёх прежних R-записей `AMV1_F0134_S03`, `AMV1_F0136_S01`, `AMV1_F0139_S00` статус `worker_exception`, пригодного ответа нет. Это не математический отрицательный класс. MC и оба новых метода имеют ответы на всех 1000, но метки двух-/четырёхметодного сравнения требуют общего покрытия. Исключаются только эти три строки; сами задачи не удалены. Источник: {source(F+'release/incomplete_coverage.csv')}. Причину конкретных worker exceptions этот набор не позволяет превращать в доказательство неприменимости R к уравнению.
''')
    add('## 2. История ML: разные цели, проверенные улучшения и отрицательные результаты')
    add('''### 2.1. 25 октября: качество Ремизова и прогноз времени

Изначальная идея — до решения предсказать точность фиксированного профиля Ремизова и его стоимость, затем выбирать быстрый надёжный метод. Это ещё не classifier «R или MC». Для R на 33 узлах при допуске .01 известны 997 целей: 871 успешная и 126 неточных; три ресурсные/worker-записи неизвестны. Используются 253 входа и семейный split 697/150/150. Проверены Dummy, LogisticRegression (C=.1, StandardScaler), Ridge (α=100 для log-времени), CatBoost (максимум1200, depth4, lr=.03, l2=10, seed20261025, один поток, early stopping80 по Validation). Фактическое число деревьев CB: classifier262; time coefficients220; time with context109.

Модели выбирались по Validation; порог вероятности **неуспеха** .07705429174905876 максимизировал Validation balanced accuracy с развязкой по failure recall. Это другой знак/смысл score, чем MC probability ниже. На Test CB ROC-AUC=.880130597, failure AP≈.56745; при .5 CM=[[132,2],[11,5]] (good,bad), при выбранном пороге CM=[[100,34],[2,14]], balanced accuracy=.810634328, accuracy=.76: найдены14/16 плохих, два ложных допуска и34 ложных отказа. Logistic Test AUC≈.8839 не является основанием выбрать его после просмотра Test. Сохранённый family-bootstrap CI разности CB−Logistic AUC включает ноль.

По времени проверялись coefficient-only и дополнительный предрешательный контекст `planned_external_workers` (1/8), константные медианы и ML. По Validation выбрана context-median: MAE≈.5553с против CB-context≈.5608с; на Test CB-context≈.6519с и median≈.6767с. Эта стоимость зависит от исторической нагрузки, не является CPU-временем или честным временем отдельного x=0. У контекстного CB planned_workers имеет≈81.05% важности. В текущие 258 признаков он не входит.
''')
    add('Источники: ' + source('results/remizov_ml_v1_2026-10-25/REPORT.md') + '; ' + source('results/remizov_ml_v1_2026-10-25/metrics.json') + '; ' + source('results/remizov_ml_v1_2026-10-25/training_protocol.json') + '; ' + source('diagnostics/train_remizov_ml.py', 'choose_threshold / main', 108) + '. Полные Train/Val/Test всех этих сохранённых вариантов приведены в приложении B.')
    add('''### 2.2. 25 октября: quality-surrogates R и MC в x=0

Существующая серия MC1000 содержит 4000 завершённых запусков: четыре независимых seed по1024 пути, EM dt=1/128, auto-T по аналитической границе forcing G, tail_eps=1e-4. Эти же ответы используются во всех последующих mean4-таблицах; повторно считать их независимыми экспериментами нельзя. Исторический R cubic/taper использует другой T/tail_eps и ранее вычислен на33 узлах. x=0 извлекается как узел16.

При историческом допуске .01 точны1426/4000 отдельных MC (35.65%),535/1000 средних четырёх (53.5%),930/997 R в нуле. Это **не** доли при нынешнем ε=.05. Обучены отдельные модели вероятности успеха single-MC1024 и R-point, а также регрессии log10(point-error) R и mean4MC с floor1e-12. Проверены Dummy, Logistic/Ridge и CatBoost; scaler/fit на Train, early stopping и выбор по Validation. Каталог моделей и точные параметры находятся в исходном protocol. На Test600 seed-меток MC соответствуют лишь150 уравнениям из30 семейств.

Test MC: CB AUC=.735006519, logloss=.586306127, Brier≈.2015; Dummy .5/.6743/.2405; Logistic AUC≈.6557, logloss≈.6790. Test R-point: CB AUC=.894675926, logloss=.118239837 против Dummy .1775; оба имеют accuracy=.96, потому что неуспехов лишь6. MAE log10 R: CB .357941753 против Dummy≈.4844; MC mean4: .362436707 против Dummy≈.4121. Raw-error MAE R≈1.65801 остаётся большой из-за редких выбросов, хотя log-loss регрессии выглядит умеренным.

Сравнительные интервальные метки были экспериментальными:7 «MC mean closer»,60 «R closer»,930 uncertain. Полноценный comparative classifier не обучен из-за gate недостатка разделённых целей. Семейный bootstrap условен на четырёх наблюдённых seed, не ограничивает временное смещение и не гарантирует точность нового seed. Регистрированный API этапа closure возвращал `method_selection.status=uncertain`; неподдержанные профили не подменялись похожими моделями.
''')
    add('Источники: ' + source('results/mc1000_x0_2026-10-25/REPORT.md') + '; ' + source('results/mc1000_x0_2026-10-25/ml/metrics.json') + '; ' + source('results/mc1000_x0_2026-10-25/plan.json') + '; ' + source('results/mc1000_x0_2026-10-25/analysis_protocol.json') + '; ' + source('diagnostics/train_mc1000_quality.py') + '; ' + source('results/ml_research_closure_2026-10-25/REPORT.md') + '; ' + source('results/ml_research_closure_2026-10-25/registry.json') + '.')
    add('Точная CB-конфигурация quality-surrogates x0 из analysis_protocol.models.catboost: максимум1200 деревьев, depth4, lr=.03, l2=10, seed2026102502, thread_count1, early_stopping_rounds80 по Validation; Logistic C=.1/max_iter3000, Ridge α100, Dummy prior/median. Она отличается от seed первой R-grid модели и от120-tree моделей позднего selector.')
    add('''### 2.3. 26 октября: observed-fastest и вероятностный селектор

Для цели «самый быстрый точный на33 узлах при .01» сопоставимые observed labels имелись лишь на36 Train-задачах:22 R и14 none, **MC-побед 0**;964 неизвестны. Диагностический CB200/depth3/lr.05/l2=5/seed2488 проверен 4-fold family-CV для R-vs-none. OOF accuracy20/36=.5556 против always-R22/36=.6111; macro-F1=.5 против≈.3793. Это не обученный полноценный R/MC-селектор; Val/Test labels для такой цели отсутствовали.

Следующая гипотеза: отдельно прогнозировать вероятность .01 и log-cost профилей R128/R256/MC1 и выбирать минимальное предсказанное время при p≥q. q=.5/.8/.95 были заранее фиксированными диагностическими политиками; q — вероятность, а не численная погрешность. На разрешённом Train: R128103 ответов (75 точных), R25640 (33), MC40 уравнений×4 seed=160 испытаний (7 точных). Полные cost targets:4/36/40 задач соответственно.

Binomial CrossEntropy учитывал k/n по уравнению, RMSE-head — log-cost; CB120/depth2/lr.05/l2=10/seed2488/bootstrapNo, один поток; scaler внутри Train. Обучены5 голов (R128 cost constant) и10 семейных bootstrap-реплик ансамбля. Проверена 3-fold paired family-OOF, без исходных Validation/Test outcomes. Brier R128=.177199835 vs constant .252498453; R256=.107857841 vs .151922265; MC=.044883361 vs constant .042527588 (хуже). MC logloss=.209643199 vs .188889196. Для MC medianAPE времени .262853493 vs constant .512895046; для R256 .014832960 vs constant .014097230 (чуть хуже). MC не выбирался при q≥.5.
''')
    add('Источники: ' + source('results/ml_selector_observed_2026-10-26_v2/REPORT.md') + '; ' + source('results/ml_selector_observed_2026-10-26_v2/diagnostic_cv.json') + '; ' + source('results/ml_probability_selector_2026-10-26/REPORT.md') + '; ' + source('results/ml_probability_selector_2026-10-26/family_cv.json') + '.')
    add('''### 2.4. Валидация8, аудит, калибровка и timing5

На восьми заранее выбранных старых Validation-семействах сохранены96 вызовов (8×3 профиля×4 повтора), без переобучения. Primary точность: R1286/8, R2567/8, MC1/8; повторы дают24/32,28/32,4/32, но не32 независимые семьи. Point-q95 политика доставила6 точных,1 неточный ответ и1 отказ; always-R256 —7 точных,1 неточный. Warm cost119.924с против195.127с: сокращение≈38.54% сопровождалось потерей одной точной доставки и не является безусловным улучшением. Все96 solver-вызовов завершены.

Аудит выявил неверно высокие вероятности на F0041, ложное недоверие F0189, лишь три MC positive-family и неполноту timing targets. Разброс bootstrap-моделей не откалиброван как CI вероятности. План дополнительного pilot не является выполненным результатом.

Проверена монотонная регуляризованная sigmoid-калибровка `σ(a*logit(p)+b)`, a≥0, L2=2 к identity;6 внешних блоков и3 внутренних family-разбиения. Сохранены38 узлов моделей/scaler и11 calibrators: **число узлов не равно числу CatBoost-fit**. R128 family-Brier .131669→.146498 ухудшился; R256 .107858→.096009 описательно улучшился, но интервал разности включал0. При q95 политика выбрала R25633/40, переключила5 на R128, все пять в одном outer fold;33/40 точны. Предсказанная экономия6.25% первоначально не была измеренной из-за отсутствующих времён.

Отдельный старый timing5 подтвердил локальную экономию≈49.92% на этих пяти известных Train-OOF переключениях (40 R-вызовов), без ML-fit. Это не новая независимая оценка распространённости полезных переключений. Последующий input-only scan250 ещё не fit-семей показал все250 действий R256; глобальный максимум score данной MC-модели≈.837788412<.95. Это ограничение конкретного замороженного score, а не верхняя граница реальной надёжности MC.
''')
    add('Источники: ' + '; '.join(source('results/'+p+'/REPORT.md') for p in ['ml_probability_validation_2026-10-26','ml_probability_selector_audit_2026-10-26','ml_nested_calibration_2026-10-26','ml_switch5_r_timing_2026-10-26','ml_selector_generalization_plan_2026-10-26']) + '; ' + source('results/ml_nested_calibration_2026-10-26/report_summary.json') + '.')
    add('''### 2.5. Development8, Train-revision, заморозка и завершённые140 вызовов

На восьми новых development-семействах прежний пилот сделал96 вызовов. По уникальным задачам R128/R256 точны7/8, MC0/8 (все32 MC seed неточны). Замороженный nested-q95-default выбрал R2568/8:7 точных,1 нарушение, cost195.616296с с overhead против195.525433с baseline; преимуществ нет, семь возможностей более дешёвого точного R128 пропущены. Эти восемь новых семейств нельзя соединять с production по одному canonical ID: ID повторяются, ключ должен включать dataset SHA и expression/topology. В последующей ревизии все96 outcomes **карантинированы**, а не добавлены в Train, согласно запрету повторно использовать раскрытые исходы.

Ревизия27 октября новых ОДУ не решала. Разрешённые targets: R128103/90семейств, R25640/40, MC12/180 семантически независимых испытаний на40семьях;20 старых допустимых seed добавлены,4 дубликата удалены. MC-положительных семей всё ещё3. Проверены binary_raw, binary_sigmoid и continuous_error: последний моделирует log(n*E_grid), переводит residual family-OOF ECDF в вероятность; масштабирование1/n является ML-гипотезой, не доказанным порядком R. Pooled log(t/n) также отдельная гипотеза стоимости. CB120/depth2/lr.05/l2=10, scaler только внутри узла.

Изучены20 политик: constantsR128/R256, для трёх вариантов q95 default/risk, parity δ=0/.02/.05, cost/success. Полный строго вложенный outer3/inner-family выбор дал R25640/40,33 точных и7 нарушений: выигрыша над R256 нет. Сохранены312 model/scaler-узлов,13 bundles,39 sigmoid-calibrators; без отдельного журнала их нельзя назвать312 fit.

Исследовательский кандидат `binary_sigmoid_parity_0.02` выбран по общей Train-OOF:11R128/29R256,33/40 точных, на9 cost-complete задачах cost-per-accurate21.280157с vs27.339410с (−22.16%). Это оценка после выбора по той же OOF-таблице, не основной nested результат. Brier raw/sigmoid/continuous: R128 .18016/.18434/.16602, R256 .10786/.12177/.08268, MC .04086/.04297/.04297; constant соответственно≈.17685/.15192/.04040. Калибровка не дала универсального выигрыша.

Первоначальный план final720 вызовов был заменён бюджетным вариантом140C; **720 не выполнены**. Реально завершённый final140 использовал60 свежих стратифицированных семейств,60R128+60R256+20MC1, без ML-fit и последующей перенастройки. Замороженный выбор:21R128/39R256/0MC. На60: R12853 точных/7 нарушений, R25657/3, ML57/3; отказов и solver failures0. Warm cost R128706.407125875с, R2561411.006868253с, ML1164.270215714с (включая .103670212с inference proxy). Экономия ML относительно R25617.4866% при0 наблюдённых индивидуальных потерь; против R128 цена выше64.8158%, получены4 дополнительные точные доставки.

MC точен0/20, ещё40 семей MC не измерены; политика MC не выбирала. Поэтому качество выбора R-vs-MC и полный трёхпрофильный oracle60 **не подтверждены**. Нулевые потери на60 не доказывают популяционную noninferiority: сохранённая консервативная95% верхняя граница loss≈.1580, при поправке на два сравнения≈.1753. Исходный holdout был свежим на момент заморозки; теперь раскрыт и не может повторно служить независимым после новой настройки.
''')
    add('Источники: ' + source('results/ml_development8_2026-10-26/REPORT.md') + '; ' + source('results/ml_train_revision_2026-10-27/REPORT.md') + '; ' + source('results/ml_selector_final_2026-10-27/training/REPORT.md') + '; ' + source('results/ml_selector_final_2026-10-27/result_summary.json') + '; ' + source('results/ml_selector_budget_review_2026-10-27/REPORT.md') + '; ' + source('results/ml_selector_final140_protocol_2026-10-27/FINAL_REPORT.md') + '; ' + source('results/ml_selector_final140_protocol_2026-10-27/final_summary.json') + '.')
    add('''### 2.6. 31 октября: от медианы абсолютных MC-ошибок к ошибке среднего ответа

Первый accuracy-selector использовал медиану четырёх **абсолютных** seed-ошибок MC. На997 целях976R/21MC; Train697=686R/11MC, Test150=146R/4MC. Один CB120/depth2/lr.05/l2=10/seed2488,253 признака. Test default-политика выбрала R150/150: accuracy=.973333333, macro-F1=.493243243, ε-success147/150. Высокая accuracy объясняется дисбалансом.

Далее математически зафиксирован другой агрегат: `E_MC=abs(mean(f_seed)-exact0)`, а не median(abs(...)), и не mean(abs(...)). После изменения targets103winner-метки сменились; получилось881R/116MC. Один такой же CB на исходном split снова всегда R на Test: accuracy=.94, macro-F1=.484536082, meanE=1.658962113, ε-success147/150. Validation ML хуже постоянного R; это отрицательный результат без поиска порога. Медианный и mean4 эксперименты нельзя прямо сравнивать по accuracy как одну цель.
''')
    add('Источники: ' + source('results/ml_accuracy_selector_2026-10-31/run/REPORT.md') + '; ' + source('results/ml_mean_mc_selector_2026-10-31/run/REPORT.md') + '; ' + source('results/ml_mean_mc_selector_2026-10-31/run/coverage.csv') + '; ' + source('ml/accuracy_selector.py', 'accuracy_target / selection_metrics', 32) + '.')
    add('## 3. Пять гипотез A–E: проверенная сетка и сравнение Train/Validation/Test')
    add(f'''Источники протокола: {source(H+'plan.json')}, {source(H+'candidates.json')}, {source(H+'selection_lock.json')}, {source('diagnostics/run_selector_hypotheses.py', 'fit_model / train / final', 169)}, {source('ml/selector_hypotheses.py', 'threshold_scan / selection_key', 80)}.

Общая цель: `E_R=abs(saved_R128(0)−exact0)`, `E_MC=abs(mean4(saved_MC1024(0))−exact0)`, `Δ=E_R−E_MC`. MC winner iff `Δ>0`; ties R. Ошибка политики `E_selected`, regret=`E_selected−min(E_R,E_MC)≥0`. При фиксированной выборке минимум среднего regret равносилен минимуму meanE. Успех `E_selected≤.05` оценивается отдельно; winner может быть неточным, а ошибочный winner-choice — достаточно точным.
''')
    add(table(['Сценарий', 'Split', 'n', 'R / MC', 'Семейств'], [
        ['old', 'Train', 697, '608 / 89', 140], ['old', 'Validation', 150, '132 / 18', 30], ['old', 'Test', 150, '141 / 9', 30],
        ['new', 'Train', 747, '653 / 94', 150], ['new', 'Validation', 100, '87 / 13', 20], ['new', 'Test', 150, '141 / 9', 30],
    ]))
    add(f'''Split new оставляет неизменными150 Test ID. Из development-семей20 полных по5 уравнений выбраны в Validation, три неполных семейства остаются Train. Страты167 полных семей по числу MC-побед0/1/2/3 содержат91/52/18/6семей; Val-квоты11/6/2/1 по largest remainder, перестановка seed2488. Family и topology между split не пересекаются. Это стратификация по development labels, не по предсказаниям или Test. Источник: {source(H+'splits.json')}, {source('ml/selector_hypotheses.py', 'stratified_family_split', 14)}.
''')
    add(table(['Гипотеза', 'Что реально проверено', 'Ограничение вывода'], [
        ['A: class weights', 'R=1, MC=1/2/3/5/8/12; baseline120/depth2/l2=10, medium240/depth3/l2=10, deeper300/depth4/l2=20', 'Больше MC-выборов не равнозначно меньшей ошибке'],
        ['B: thresholds', 'classifier .05:.05:.50 и все Val-score partitions; regression0 и все partitions; строгий score>t', 'Только Val используется для порога; .2 — не вероятность гарантии точности'],
        ['C: regression Δ', 'RMSE на raw Δ и asinh(Δ/s); s=max(median_train abs(Δ),1e-12); обратное s*sinh(score)', 'Меняет ML-loss, не ошибки/метки/численную схему'],
        ['D: interactions', '253 исходных vs258=253+5 коэффициентных взаимодействий', 'Нет отдельной five-way ablation; вклад каждого нового признака не установлен'],
        ['E: split / CV', 'Вся main-сетка повторена old/new; 5-fold family-CV original253, nested inner-quarter threshold/spec', 'Нельзя приписывать258-модели полный nested-CV; CV main253 only'],
        ['Ёмкость/калибровка', '3 заранее фиксированных config; Val Brier/logloss/ECE classifier scores', 'Новая calibration-fit в A–E не выполнялась; weighted scores не считаются calibrated'],
    ]))
    add('''Все main модели: lr=.05, seed2488, bootstrapNo, random_strength0, thread1, без early stopping, resampling и scaler; регрессоры RMSE, classifier Logloss. 2split×2feature-set×(6weights×3classifiers+2regressions×3config)=96fit (72classifier,24regressor), плюс4constants. Имеется844 сохранённых политики и12337 строк Val-threshold scans. Исторически CV160fit и независимый baseline audit1fit:257fit этой прежней серии; **в настоящем сборе fit=0**. Верхняя плановая оценка266 не является фактическим числом.

Выбор: minimum Val mean_regret; затем больше ε-success, macro-F1, меньше MC, стабильный ID. Внутри полного threshold scan при равенстве предпочтён больший порог. Lock создан до итогового прохода Test. Однако прежний Test раскрыт в предыдущих исследованиях, поэтому даже корректный новый lock не восстанавливает prospective независимость.
''')
    add('### 3.1. Основные политики: все три split')
    add('Обозначения: baseline253=.5; selected253 new — глобально выбранный по Val `new__X__classifier__w1__baseline`, t=.25; diagnostic258 — не выбранный глобально `new__augmented__classifier__w1__baseline`, t=.2006671072130637; selected258 old — `old__augmented__classifier__w12__baseline`, t=.8130974275754173. Train является in-sample при окончательном Val-пороге.')
    rows = []
    for r in comparison.to_dict('records'):
        if r['label'] in ('always R', 'always MC', 'oracle') and r['scenario'] == 'old':
            continue
        rows.append([r['label'], r['split'], int(r['n']), r['accuracy'], r['macro_f1'], r['mc_precision'],
                     r['mc_recall'], f"{int(r['mc_correct'])}/{int(r['mc_false'])}/{int(r['mc_missed'])}",
                     r['mean_error'], r['mean_regret'], f"{int(r['success_count'])}/{int(r['n'])}"])
    add(table(['Политика', 'Split', 'n', 'Accuracy', 'MacroF1', 'P_MC', 'Recall_MC', 'TP/FP/FN', 'MeanE', 'Meanregret', 'E≤.05'], rows))
    add(f'''Источники: [main_model_comparisons.csv](tables/main_model_comparisons.csv), {source(H+'validation_metrics.csv')}, {source(H+'test_metrics.csv')}; текущая сверка [checks/metrics.json](checks/metrics.json). Baseline old имеет свои697/150 Train/Val; references в этой таблице приведены для new747/100. Test для всех один и тот же150. «Oracle» использует наблюдённые ошибки только как идеальный lower bound, не является доступной до решения политикой.

Новый выбранный253 на Test имеет CM=[[136,5],[8,1]]:1/9 найденных MC,5FP,8FN. Средний regret на .000833143621 хуже always-R, ε-success146/150 против147/150. Старый выбранный258/weight12 также не лучше always-R на Test. По Val у diagnostic258 regret=.000876030131, у выбранного253=.000710351281:258 не прошёл основной критерий. Выбирать258 после просмотра сильного Test было бы новым post-hoc model selection.
''')
    add(figure(2))
    add('### 3.2. Веса, регрессии и устойчивость по family-CV')
    weights = frame(str((OUT / 'tables/weights_new253_val_selected.csv').relative_to(ROOT)))
    weight_rows=[]
    for r in weights.to_dict('records'):
        va=policies[(policies.candidate_id==r['candidate_id'])&(policies.policy=='validation_optimal')&(policies.split=='validation')].iloc[0]
        weight_rows.append([r['weight'],str(r['candidate_id']).split('__')[-1],r['threshold'],r['val_regret'],
                            f'{int(va.mc_correct)}/{int(va.mc_false)}',r['test_regret'],f"{int(r['TP'])}/{int(r['FP'])}"])
    add(table(['ВесMC', 'Config', 't', 'Val regret', 'Val TP/FP', 'Test regret', 'Test TP/FP'],weight_rows))
    add('Для каждого веса показана конфигурация original253/new, ранее выбранная по Val среди сохранённых политик. Это не подбор по текущему Test. Источник: [weights_new253_val_selected.csv](tables/weights_new253_val_selected.csv) и исходные validation_metrics/test_metrics. Regret и TP/FP зависят от выбранного порога, не только от веса.')
    add(figure(8))
    regression_rows = []
    for kind in ('raw', 'asinh'):
        own = policies[(policies.scenario == 'new') & (policies.kind == kind) & (policies.split == 'validation')]
        for feature_set in ('X', 'augmented'):
            subset = own[own.feature_set == feature_set].sort_values(['mean_regret','success_count','macro_f1','mc_predictions','candidate_id','policy'], ascending=[True,False,False,True,True,True])
            chosen = subset.iloc[0]
            t = policies[(policies.candidate_id == chosen.candidate_id) & (policies.policy == chosen.policy) & (policies.split == 'test')].iloc[0]
            regression_rows.append([kind, feature_set, chosen.candidate_id, chosen.threshold, chosen.mean_regret, t.mean_regret, f'{int(t.mc_correct)}/{int(t.mc_false)}'])
    add(table(['Target', 'Features', 'Val-selected saved ID', 't', 'Val regret', 'Test regret', 'Test TP/FP'], regression_rows))
    add('Исходные и asinh-регрессии не показали переноса преимущества над always-R. Эта таблица лишь извлекает зафиксированный Val-выбор из сохранённой сетки; все остальные результаты доступны в [model_policy_metrics_all.csv](tables/model_policy_metrics_all.csv).')
    add(table(['Scenario', 'OOF policy', 'n', 'Accuracy', 'MacroF1', 'TP/FP/FN', 'Pooled regret', 'Fold regret mean ± SD'], [
        [r.scenario,r.policy,int(r.n),r.accuracy,r.macro_f1,f'{int(r.mc_correct)}/{int(r.mc_false)}/{int(r.mc_missed)}',r.mean_regret,f'{r.fold_regret_mean:.9g} ± {r.fold_regret_sd:.9g}'] for r in cv.itertuples()
    ]))
    add(f'''Источники: {source(H+'cv_summary.json')}, {source(H+'cv_folds.csv')}, {source(H+'cv_predictions.csv')}, [cv_policies_22.csv](tables/cv_policies_22.csv). Current audit проверил арифметику110fold-строк/22pooled; исходный {source(H+'independent_verification.json')} подтверждает family separation и ranking сохранённых inner-кандидатов. Inner scores не сохранены, поэтому **оптимальность внутренних порогов независимо не устанавливается** без нового fit; в этой работе он запрещён. Fixed classifier threshold=.5; raw/asinh threshold0; nested выбирает внутри outer Train. Описательные mean±SD пяти коррелированных folds не являются CI.

Old fixed weight12 имеет pooled regret≈.00579441, значительно ниже always-R≈1.24972544, но new weight12≈1.17093090 vs R≈1.16603007; nested old/new≈1.25167420/1.17082676 также не улучшает R. Нестабильность выбора между семейными folds и зависимость от редких выбросов не поддерживают вывод о надёжном общем выигрыше. Сравнивать абсолютные pooled old/new напрямую как одинаковые samples нельзя.
''')
    add(figure(7))
    add('## 4. CatBoost с258 признаками: точная диагностическая политика и все ошибки')
    params = read(str((OUT / 'model258_all_parameters.json').relative_to(ROOT)))
    add(f'''Сохранённый candidate: **`new__augmented__classifier__w1__baseline`**. {source(H+'models/new__augmented__classifier__w1__baseline.cbm', 'Файл CBM')}; SHA-256 `255c29a2caa087538ae2b589cd0fdc2ae693c7f73de3babe4ac7803ff9e46c18`. Обучен прежде на747 Train-строках,258input-only признаках. Class order=[remizov,monte_carlo], score=P_model(MC), действие MC iff **score>.2006671072130637**; равенство→R.

Гиперпараметры: iterations120 (tree_count120), depth2, learning_rate=.05 (в CBM float32≈.05000000074505806), l2_leaf_reg10, Logloss, seed2488, bootstrap_typeNo, random_strength0, thread_count1, allow_const_labelTrue, allow_writing_filesFalse, verboseFalse. При weight1 `class_weights` не передавался, auto_class_weightsNone. Нет scaler, ранней остановки или новых fit калибровки. CBM также хранит border_count254, leaf_estimation_iterations10, boosting_typePlain, grow_policySymmetricTree. Полный перечень сериализованных параметров: [model258_all_parameters.json](model258_all_parameters.json); metadata/имена/число деревьев: [model258_metadata.json](model258_metadata.json). `thread_count` отсутствует в get_all_params: значение1 подтверждается plan и {source('diagnostics/run_selector_hypotheses.py', 'fit_model',169)}, а не выводится из CBM.
''')
    diagnostic = comparison[comparison.label == 'diagnostic258 new']
    add(table(['Split','n','CM: [[TN,FP],[FN,TP]]','Accuracy','Balanced acc','MacroF1','P_MC','R_MC','F1_MC','MeanE','MedianE','Meanregret','ε-success'], [
        [r.split,int(r.n),r.confusion_matrix,r.accuracy,r.balanced_accuracy,r.macro_f1,r.mc_precision,r.mc_recall,r.mc_f1,r.mean_error,r.median_error,r.mean_regret,f'{int(r.success_count)}/{int(r.n)}'] for r in diagnostic.itertuples()
    ], digits=12))
    add('Строки CM — истинный winner R/MC, столбцы — выбор R/MC. Test:130R+20MC choices; из20 MC только6 правильных,14ложных; из9 истинных MC-побед3пропущены. 127TN. Accuracy ниже always-R94%, но средняя ошибка намного ниже из-за обнаруженного большого R-выброса. Это различные критерии, противоречия нет. Train показан с тем же Val-порогом и не является оценкой обобщения.')
    add('Дополнительные сохранённые вероятностные метрики diagnostic258 на Validation: Brier=.10617309246414589, logloss=.35977853406730176, ECE=.059394761427063285. Это описательные scores на100строках, не гарантия calibration или вероятности ошибки решения; источник: '+source(H+'calibration.json')+'.')
    add(figure(3))
    add('### 4.1. Все6 найденных,14 ложных и3 пропущенных MC-победы Test')
    add('В следующей таблице **Δ=E_R−E_MC**: плюс означает преимущество MC. Regret выбранного метода равен abs(Δ) только при FP/FN, при TP он0. Последний столбец показывает **обе** ошибки относительно ε=.05, чтобы не путать ошибку классификации с недопустимой численной ошибкой. Все23 строки относятся только к неизменному Test150. Полная точность чисел в [diagnostic_258_cases.csv](tables/diagnostic_258_cases.csv); scores и все997 строк — [diagnostic_258_all_splits.csv](tables/diagnostic_258_all_splits.csv).')
    for category, title in [('TP','Найденные MC:6'),('FP','Ложные MC:14'),('FN','Пропущенные MC:3')]:
        add('#### ' + title)
        rows = []
        for r in cases[cases.category == category].sort_values('equation_id').itertuples():
            note = 'оба≤.05' if r.R_accurate and r.MC_accurate else 'R≤.05; MC>.05 — новый отказ по ε' if r.R_accurate else 'оба>.05' if not r.MC_accurate else 'R>.05; MC≤.05'
            rows.append([source('results/chernov_solve1000_2026-10-18/solutions/'+r.equation_id+'.json',r.equation_id),r.E_R,r.E_MC,r.delta_R_minus_MC,r.regret,note])
        add(table(['ID → исходная R-запись','E_R','E_MC','Δ=R−MC','Regret выбора','Отношение к .05'],rows,digits=17))
    add(f'''Источники для каждой строки: указанный R JSON (`target_x[16]=0`, `result.f[16]`, `y_exact[16]`); четыре MC-записи `results/mc1000_x0_2026-10-25/attempts/<ID>_seed0…3.json` (`result.f[0]`, scalar y_exact); {source('results/ml_mean_mc_selector_2026-10-31/run/coverage.csv')} и {source(H+'test_predictions.csv')}, фильтр candidate+policy=`validation_optimal`. Текущее чтение первичных записей подтвердило ошибки без вычисления нового exact или численного решения.

**Существенность:** TP дают сумму уменьшений против R248.2523548967466; почти всё приходится на `AMV1_F0180_S04` (248.21999576573202). У этого случая E_R≈248.323, E_MC≈.103081: даже удачный выбор winner **не достигает .05**. Второй такой TP `AMV1_F0070_S04`: .082588→.056320, также выше допуска. Остальные4TP оба метода точны по .05. Во всех3FN обе ошибки≤.05; сумма упущенных улучшений=.007545289608040247, наибольшее≈.003499886.

У14FP суммарный ущерб=.248645740919119. В12 случаях оба ответа остаются≤.05, хотя MC хуже. Два новых нарушения: `AMV1_F0180_S01` .0317484014→.0965483750 (regret .06479997355309741) и `AMV1_F0180_S02` .0033794861→.0604145634 (regret .05703507735703593). Новых спасений порога0. Поэтому selected145/150 против R147/150. Общий regret=.2561910305271593, /150=.0017079402035143952. Это численная величина ущерба, а не только17неверных меток.
''')
    add(figure(4))
    add(figure(5))
    add('''### 4.2. Чувствительность среднего и объяснение выброса

Для всех150 Test always-R meanE=1.6589621134157981, diagnostic258=.005604052376948204. Только для описательного post-hoc sensitivity исключим крупнейшую R-ошибку `F0180/S04`: на остальных149 always-R meanE=.0034982569431481238, diagnostic258=.004949844929084751 — преимущество меняет знак. **Основные результаты остаются на150**, строка не удаляется и новая политика по этому анализу не выбирается. Среднее обладает другой чувствительностью, чем медиана/доля ε-success; нельзя описывать улучшение средней как одинаковое улучшение всех задач.

Причина конкретного исторического R-выброса частично исследована ранее, не устанавливается заново ML-важностью. В сохранённой boundary-диагностике смена только cubic/taper→cubic/zero на четырёх известных задачах устранила три катастрофических E_grid33-выброса. Для F0180/S04 E_grid33≈19134.4052→.07045189 при том же n/T/dx/квадратуре. Это сильное свидетельство роли граничного продолжения в **том** профиле; величина19134 — максимум на33 узлах, не E_R(0)=248.323. Последующее padding12→24 изменило ответ лишь≈1.08e-13, квадратура513→1025 —≈1.355e-4 в zero-профиле. Эти controls не разрешают заменять старые R-метки исправленными значениями и не доказывают точность R для всех уравнений. Остаточные доли ошибки по n/h/квадратуре глобально не установлены.
''')
    add('Источники: [checks/case_summary.json](checks/case_summary.json), [figure06_sensitivity.csv](tables/figure06_sensitivity.csv); ' + source('results/chernov_boundary_diagnosis_2026-10-24/REPORT.md') + '; ' + source('results/chernov_window_padding24_2026-10-24/REPORT.md') + '; ' + source('results/chernov_time_quadrature1025_2026-10-24/REPORT.md') + '.')
    add(figure(6))
    add('## 5. Полный каталог258 признаков: определения и источники')
    add(f'''Порядок взят из **augmented_names** сохранённого {source(H+'features.npz')}; feature_names_ в CBM совпадают. Первые253 — отсортированный словарь `coefficient_features`, затем5 `FEATURE_NAMES` в заданном порядке. Ничего не опущено:5cert+10pre+4×14stat+4×32sample+6corr+3×16AST+5research=258. Машиночитаемый каталог: [all_258_features.csv](tables/all_258_features.csv); проверка полноты: [checks/feature_catalog.json](checks/feature_catalog.json). Все257226 значений на997 строках восстановлены из реального X/whitelist и **побитово** равны сохранённой матрице.

### 5.1. Общие обозначения и соглашения

`x_i` — реальная256-точечная входная сетка [-1,1]; математически −1+2i/255. Каналы для исходных253: `a`, `b`, **`cR=c+λ`**, **`gR=−g`**. Это преобразование **ML-дескрипторов**; оно не означает подмену входов RK4/PI, которым передавались исходные c,g. `A_minus`, `A_plus`, `B`, `C`, `λ` — сохранённые глобальные сертификаты. pre_T — legacy-формула с1e-8, без множителя G; это описательный признак прежнего профиля, не универсальная граница полной ошибки.

Для статистик `d1_i=(v_(i+1)−v_i)/(x_(i+1)−x_i)` (255 значений), `d2` — разность d1, делённая на расстояния между серединами соседних интервалов (254). RMS и std невзвешенные, std ddof0. `P_k=abs(rfft(v−mean(v)))²`; spectral_high_fraction использует bins8…128 и0 при нулевой энергии. Strict zero_crossings не считает переход через точный нулевой отсчёт. Samples берут **индексы** `linspace(0,255,32,dtype=int)`, без интерполяции; для каждого имени ниже указан индекс и фактическая координата. Corr — центрированный скалярный product/произведение норм,0 при нулевом знаменателе.

AST-дескрипторы относятся к **сырому** coefficient-tree до tanh, не к exact/y_shape. Для каждого из a_raw/b_raw/c_raw: nodes, root-depth1 и14counts операций. Подробные формулы безопасных операций: {source('src/all_methods_expressions.py', '_evaluate_array',455)}. Константные параметры узлов сами не становятся отдельными AST-признаками.

Для research5: `q=−c>0`, `F=−g`, значения a0,b0,q0 получены **линейной интерполяцией коэффициентов** в0; это не интерполяция ответа и не exact как вход. `G(v)=np.gradient(v,x,edge_order=2)`, `G2=G(G(v))`. RMS реализован через масштабирование на absmax для устойчивости. F предварительно нормирован на RMS(F); записанные ниже формулы математически эквивалентны реализации, но буквальная последовательность float-операций — в коде. Интегралы — составные трапеции по входной сетке. При F=0 три forcing-признака равны0. Есть проверки increasing grid, four finite channels, interval[-1,1], a>0,q>0 и конечности результата.

Exact, y_shape, y0/v0, solution derivatives, ID/family/split/seed, solver outputs, errors/success/time и postdiagnostics в258 не входят. g допустим как заданный input, хотя manufactured-происхождение g связано с exact. Такая связь — свойство генерации; перенос на иную RHS-distribution отдельно не проверен. Сертификаты/коэффициентные AST доступны нынешнему датасету; на внешнем наборе одного списка sampled-значений может быть недостаточно для253-ветки.

### 5.2. Все258 строк в порядке CBM
''')
    add((OUT / 'feature_catalog.md').read_text())
    add('### 5.3. Что можно сказать о полезности признаков')
    importance = frame(str((OUT / 'tables/diagnostic_258_feature_importance.csv').relative_to(ROOT)))
    add(table(['Research feature','PredictionValuesChange','Позиция в отсортированном списке258'], [[r.feature,r.importance,int(r.rank)] for r in importance[importance.feature.str.startswith('research_')].itertuples()]))
    add(f'''Три research-признака имеют нулевую сохранённую важность этой модели; номера среди нулевых — технический порядок сортировки, не доказательство разных эффектов. Diffusion-gradient≈2.58505 и forcing-cancellation≈1.17306 участвуют в деревьях, но отдельное причинное улучшение ими не доказано. Исходный drift-сигнал описательно различает классы: Train/new AUC `b_absmax`≈.670, `b_rms`≈.669; классы перекрываются. AUC здесь исходный roc_auc(MC=1, feature), без разворота знака; ranking в feature_analysis использовал удалённость от.5. Это однофакторный анализ Train/Val, не доказательство физического критерия применимости. Источники: {source(H+'feature_importance.csv')}, {source(H+'feature_analysis.json')}, {source('diagnostics/analyze_selector_hypotheses.py', 'features',41)}.
''')
    add(figure(9))
    add('## 6. Существующий эксперимент четырёх методов: протокол, ошибки и997 побед RK4')
    add(f'''Используется завершённая серия {source(F+'REPORT.md')}, а не новый запуск. План до расчёта: {source(F+'plan.json')}.2000production-вызовов,1000RK4+1000PI, **один фиксированный шаговый профиль**; R и MC заново не считались. Current arithmetic audit прочитал все7000первичных JSON и подтвердил записи/НУ/значения/ошибки, не оценивая exact заново.

### 6.1. Четыре математических профиля
''')
    add(table(['Метод','Алгоритм/данные','Фиксированные параметры','Источник'], [
        ['R', 'bounded resolvent через конечные аппроксимации Чернова, историческое cubic/taper, значение узла0 из grid33', 'n128, Q513 quadratic, dx=.0125, padding12, legacy auto-T, tail_eps1e-8', source('chernov_solver.py') + '; ' + source('results/chernov_solve1000_2026-10-18/plan.json')],
        ['MC mean4', 'среднее4 независимых сохранённых EM/Feynman–Kac ответов; два trapezoidal накопления; G-normalized auto-T', 'M1024×4, dt1/128, seeds202610250101…104, tail_eps1e-4, target0', source('mc_optimized.py') + '; ' + source('results/mc1000_x0_2026-10-25/plan.json')],
        ['RK4','U=(y,v), U′=(v,(g−bv−cy)/a); классические k1,k2,k3,k4 и веса1/6,1/3,1/3,1/6; точки x,x+h/2,x+h/2,x+h','[-1,1], N128, h1/64, a_min.001;129узлов, index64=0; исходныеa,b,c,g,y0,v0',source('src/numerical_methods/rk4_solver.py', 'solve_rk4',35)],
        ['PI','Z=(y,v,1); Z′=B(x)Z, B=[[0,1,0],[-c/a,-b/a,g/a],[0,0,0]]; Z_(k+1)=expm(hB(x_k))Z_k','тот же интервал/N/guard/НУ; коэффициенты строго в левой точке; scipy.linalg.expm3×3',source('src/numerical_methods/product_integral_solver.py', 'solve_product_integral_left_exponential',262)],
    ]))
    add(f'''RK4 действительно классический четырёхстадийный порядок4; это свойство схемы при достаточной гладкости/устойчивости, а не оценённый по одному N порядок. Левый экспоненциальный PI в переменных коэффициентах обычно даёт первый порядок глобальной ошибки: точная expm решает замороженный локальный B, не устраняет ошибку заморозки. Теоретическая оценка порядка не является измеренной сходимостью данной серии. Даже regime=constant имеет переменную g, поэтому расширенная B с forcing в общем случае не постоянна и PI не обязан быть точным. Численные формулы не менялись в интеграции; исходные переносимые файлы и сравнение: {source('docs/NUMERICAL_METHODS.md')}, {source('results/rk4_product_integral_integration_2026-11-01/REPORT.md')}.

Ошибки: `E0=abs(result.y[64]−saved_exact0)` для RK4/PI; x[64] точно0. Для R target_x[16]=0; для MC target_x=[0], ошибка среднего ответов. Нет ближайшего NPZ-отсчёта, интерполяции численного ответа, нового восстановления НУ или exact в solver state. Сохранённый safe-loader до solver по прежнему проверяет manufactured-контракт; g как вход вычисляется из исходного представления задачи. Winner — минимум E0 по четырём пригодным ответам; точные равенства float64 развязываются порядком[R,MC,RK4,PI], но в997строках равенств0. ε=.05 не изменён.

Ресурсы прежней серии: один процесс, BLAS/OpenMP1, Python3.10.11, NumPy2.2.6, SciPy1.15.3. Цикл с расчетом/оценкой/записью308.432394с без первоначальной загрузки, peakRSS219.578125MiB. Счётчики512000RK4-RHS и128000PI-expm. Resume прочитал2000завершённых записей и сделал0вызовов, сохранив bytes/mtime. Источники: {source(F+'execution.log')}, {source(F+'resume_verification.json')}, {source(F+'independent_verification.json')}. Эти исторические времена разных профилей/эпох/числа точек **не являются** новым честным сравнением скорости.
''')
    add('### 6.2. Полное доступное покрытие и общее пересечение')
    add(table(['Scope','Метод','n','E≤.05','MeanE0','MedianE0','q90','q95','q99','MaxE0'], [
        [r.scope,r.method,int(r.n),int(r.accurate),r.mean,r.median,r.q90,r.q95,r.q99,r.max] for r in distributions.itertuples()
    ], digits=12))
    add(f'''Источник значений: {source(F+'release/coverage_all1000.csv')} и {source(F+'release/ml_four_methods.csv')}, пересчёт [four_method_distributions.csv](tables/four_method_distributions.csv) содержит также min,q01,q05,q25,q75,std_population,zero_count. Quantiles — NumPy linear interpolation; это описательные квантили задач, не CI. У всех E0>0, floor/замена нулей для логарифмических графиков не применялись. На доступных1000RK4 имеет1000точных, PI760; на общих997—997/758; R982/997, MC935/997. Сравнивать доли из разных scopes как парные нельзя.
''')
    add(figure(1))
    add(figure(10))
    add('### 6.3. Сложность, режимы, семейства и победители')
    add(f'''Уровни L1…L4 задают диапазоны глубины/узлов AST и максимальной частоты примитивов: L1(1–2,3–7,1.5), L2(2–3,6–12,3), L3(3–4,10–18,5), L4(4–5,14–24,8). В constant-режиме коэффициентные деревья принудительно константные, но y_shape остаётся структурно богатым. Поэтому L — категориальная конструкция генератора, а не измеренная stiffness или универсальная трудность ОДУ. Источники: {source('src/all_methods_dataset_v1.py', '_COMPLEXITY_SPECS',76)}, {source('src/all_methods_dataset_v1.py', '_generate_families',439)}.
''')
    for group_type, title in [('complexity_level','Сложность'),('regime','Режим')]:
        add('#### ' + title + ': общее пересечение997')
        add(table(['Группа','Метод','n','MedianE0','IQR[q25,q75]','MeanE0','MaxE0','E≤.05'], [
            [r.group,r.method,int(r.n),r.median,f'[{r.q25:.9g}, {r.q75:.9g}]',r.mean,r.maximum,int(r.accurate)] for r in grouped[grouped.group_type == group_type].itertuples()
        ]))
    add(f'''Суммы RK4-побед: L1=250,L2=250,L3=247,L4=250; constant/slow/moderate/high_contrast по200, oscillatory197. Во всех20ячейках сложности×режима winnerRK4; по200структурным семействам —197семей×5 иF0134/F0136/F0139×4, все RK4. По original split697/150/150, latest747/100/150 — также только RK4. Источники: {source(F+'release/winners_by_complexity.csv')}, {source(F+'release/winners_by_regime.csv')}, {source(F+'release/winners_by_complexity_regime.csv')}, {source(F+'release/winners_by_family.csv')}, {source(F+'release/winners_by_original_split.csv')}, {source(F+'release/winners_by_latest_split.csv')}.

Полные распределения всех200семейств без пропусков: [four_method_group_distributions.csv](tables/four_method_group_distributions.csv), group_type=family_id. Их медианы показаны на рисунке13; статусы/все E0/идентичности всех1000 — исходная coverage. Список997индивидуальных winner — {source(F+'release/winners.csv')}. На трёх неполных случаях RK4 также лучший из доступных трёх ответов, но четырёхметодная метка им не назначается из-за отсутствия R.
''')
    add(figure(11))
    add(figure(12))
    add(figure(13))
    add('### 6.4. Худшие ошибки и33 сохранённых узла')
    common = frame(str((OUT / 'tables/four_method_common997.csv').relative_to(ROOT)))
    top_rows = []
    for method in ('rk4','product_integral'):
        own = common.sort_values('E_' + method, ascending=False).head(3)
        for r in own.to_dict('records'):
            top_rows.append([method,r['equation_id'],r['complexity_level'],r['regime'],r['E_'+method],r['E_remizov'],r['E_monte_carlo']])
    add(table(['Метод сортировки','ID','L','Режим','E0 метода','E_R0','E_MC0'],top_rows,digits=12))
    add('Источник: [four_method_common997.csv](tables/four_method_common997.csv); полный top10 каждого нового метода и исходные отказы — ' + source(F+'release/problem_equations.csv') + '.')
    grid = frame(str((OUT / 'tables/new_methods_grid33_summary.csv').relative_to(ROOT)))
    add(table(['Метод','n','E_grid33≤.05','MeanE_grid33','MedianE_grid33','MaxE_grid33'],[[r.method,int(r.n),int(r.accurate_grid33),r.mean,r.median,r.maximum] for r in grid.itertuples()],digits=12))
    add('''Вспомогательная E_grid33 — max abs ошибки на каждом четвёртом из129 узлов, не непрерывная E∞. У PI эта метрика существенно хуже точки0:760/1000 точны в0 и404/1000 на33узлах. У RK4 худший E0≈4.59e-5, но max grid33≈4.58e-4. Нельзя из одной точки заключать равномерную точность. Существующие trajectories и exact_grid33 повторно прочитаны, новых exact-вычислений нет. Для MC имеется только x0 на1000, поэтому общего четырёхметодного E_grid33-сравнения на1000 не существует.

Большие PI-ошибки согласуются с ограничением левой заморозки при переменной RHS/коэффициентах и возможным усилением ошибки в IVP. Но один N не разделяет локальную погрешность заморозки, чувствительность IVP и накопленное округление; причинная декомпозиция для каждого случая не установлена. Для RK4 найденные малые ошибки не доказывают устойчивость при другом масштабе, stiffness или длине интервала. Никакие параметры не подбирались для улучшения этих значений.
''')
    add(figure(14))
    add('''### 6.5. Что означает997/997

Это **эмпирический факт для этих997задач, данных НУ, фиксированных реализаций и бюджетов, метрики E0**. Он не доказывает универсального превосходства RK4, не оценивает порядки сходимости по N, оптимальные бюджеты и стоимость достижения заданного допуска, не сравнивает среднюю будущую ошибку случайного нового MC-seed. Mean4MC — один конкретный агрегат конечных seed. R использует глобальное условие, IVP-методы дополнительно используют сохранённые НУ; для иной постановки этого может не быть.

Все997 four-method labels — один классRK4. Обучение многоклассового ML на такой цели не даст проверки распознавания областей применимости: константа always-RK4 уже воспроизводит100% этих меток. Валидный следующий научный вопрос — отдельное, заранее спроектированное исследование точности/бюджетов/устойчивости и иной области задач, а не объявление ML-селектора доказанным. В настоящей работе новые N/задачи/подборы не запускались.
''')
    add('## 7. Реестр подтверждённых выводов и границ доказательств')
    add(table(['Утверждение','Статус','Основание / чего недостаёт'],[
        ['997пар и3пропуска R','Подтверждено','coverage +7000raw-записей; пропуски не отрицательные классы'],
        ['258признаков доступны до solve','Подтверждено в нынешнем контракте','whitelist+точное восстановление997×258; внешним данным нужны certificates/AST'],
        ['258 на Test находит6MC,14FP,3FN','Подтверждено','сохранённые scores+CBM+23строки; это diagnostic, не глобальный Val-winner'],
        ['258 лучше во всех смыслах','Не подтверждено','winner accuracy и ε-success хуже R; mean определяется крупным выбросом'],
        ['Дополнительные5 обеспечивают переносимое улучшение','Гипотеза','нет свежего независимого подтверждения и покомпонентной ablation'],
        ['Веса/пороги/большая глубина устойчиво улучшают селектор','Не подтверждено','Val выигрыш не переносится стабильно; CV нестабилен'],
        ['Nested thresholds независимо проверены','Частично','outer metrics и inner ranking да; inner scores отсутствуют'],
        ['Final140 был выполнен','Подтверждено','140raw records, final_summary, FINAL_REPORT; первоначальный720 план не выполнен'],
        ['Final140 доказал качество выбора MC','Не подтверждено','0MC-actions; MC0/20,40неизмерены'],
        ['Общее четырёхметодное сравнение на997:RK4winner','Подтверждено','все индивидуальные E0,997argmin без ties'],
        ['RK4 универсально лучше PI/R/MC','Не подтверждено','один N/набор/метрика; бюджеты и условия различаются'],
        ['Четырёхметодные E_grid33 на1000','Отсутствуют','MC сохранён только в0'],
        ['Полная ошибка MC ограничена CI/stderr/tail_eps','Не подтверждено','CI sampling-only; tail отдельно; EM bias и whole-grid error не покрыты'],
        ['Старый Test снова независим после новой заморозки','Нет','повторное раскрытие и множество просмотренных моделей/политик'],
        ['Перспективная четырёхклассовая ML-цель уже готова','Нет информативной проверки классов','997одинаковыхwinnerRK4'],
    ]))
    add('''Семейства являются единицами группировки и неопределённости; пять параметризаций и четыре MC-seed не независимы как новые семьи. Сохранённые bootstrap-интервалы воспроизведены как исторические результаты, новые bootstrap-эксперименты или проверки статистической значимости не проводились. Ни один Test-оптимум по844политикам не объявляется новой выбранной моделью. Просмотр всех политик, ошибок и feature-важностей увеличивает риск дальнейшей адаптации к этому Test; для подтверждения будущего решения нужен новый независимо отобранный и заранее замороженный набор.

Для итоговой статьи допустимы описания зафиксированных наблюдений с указанными контрактами и отрицательными результатами. Недопустимы заявления о95%надёжности любого нового solver-call, универсальной применимости learned selector, причинном влиянии признаков, одинаковой стоимости профилей или непрерывной точности всего интервала, если требуемых источников нет.
''')
    add('## Приложение A. Все96 сохранённых моделей A–E: сопоставление трёх split')
    add('Для каждой модели здесь её **собственный ранее определённый Val-optimal порог**, не новый выбор по Test. Это не сравнение всех моделей на одном Train: old/new разные. Train in-sample; Val использован для порога; Test уже раскрыт. Fixed .05… .50 политики также доступны без пропусков в [полной2532-строчной таблице](tables/model_policy_metrics_all.csv). Точные config/SHA/model paths — [models_96.csv](tables/models_96.csv). В ID X=253, augmented=258; w — вес MC; baseline/medium/deeper определены в разделе3.')
    model_rows=[]
    for cid in sorted(policies[policies.kind != 'constant'].candidate_id.unique()):
        own=policies[(policies.candidate_id==cid)&(policies.policy=='validation_optimal')].set_index('split')
        tr,va,te=(own.loc[k] for k in ('train','validation','test'))
        model_rows.append([cid,va.threshold,tr.mean_regret,va.mean_regret,te.mean_regret,
                           tr.macro_f1,va.macro_f1,te.macro_f1,f'{int(te.mc_correct)}/{int(te.mc_false)}/{int(te.mc_missed)}',f'{int(te.success_count)}/150'])
    assert len(model_rows)==96
    add(table(['ID','t','Train regret','Val regret','Test regret','Train MacroF1','Val MacroF1','Test MacroF1','Test TP/FP/FN','Test ε-success'],model_rows))
    add(f'Источник: {source(H+"candidates.json")}, {source(H+"validation_metrics.csv")}, {source(H+"test_metrics.csv")}, [checks/model_scores.json](checks/model_scores.json). Train получен исключительно inference сохранённых моделей; обучения и новой настройки нет.')
    add('## Приложение B. Все63 сохранённые split-метрики первоначальных quality/time-моделей')
    old=frame(str((OUT/'tables/historical_63_split_metrics.csv').relative_to(ROOT)))
    add(f'''Перенесены все63строки {source('results/ml_research_closure_2026-10-25/ml_metrics.csv')}, без отбора удачных. Исходники целевых таблиц: {source('results/remizov_ml_v1_2026-10-25/metrics.json')}, {source('results/mc1000_x0_2026-10-25/ml/metrics.json')}. Здесь AUC/logloss относятся к вероятностному качеству; MAE_seconds — к стоимости; MAE_log10 и MAE_error — к регрессии численной ошибки. «—» означает неприменимо/отсутствует. Rows=n для данного target, а MC single-seed n600 не равно600независимым уравнениям. В historical_taper_grid33 classifier accuracy/recall/threshold в полном CSV относятся к прежнему Validation-порогу (не MC winner). В таблице сохранены все модели; поле selected и остальные метрики — в CSV. Тренировочные оценки не считаются OOF.
''')
    add(table(['Scope','Target','Model','Split','n','AUC','Logloss','Brier','MAE_seconds','MAE_log10','MAE_error'],[
        [r.scope,r.task,r.model,r.split, int(r.rows if pd.notna(r.rows) else r.n) if pd.notna(r.rows) or pd.notna(r.n) else None,
         r.roc_auc,r.logloss,r.brier,r.mae_seconds,r.mae_log10,r.mae_error] for r in old.itertuples()
    ]))
    add('## Приложение C. Источники, воспроизведение анализа и сохранность')
    add('''Все новые материалы находятся в этом каталоге. Вне него обновлён только PROJECT_STATUS.md по AGENTS.md; исходник прежнего статуса сохранён в [PROJECT_STATUS.before.md](PROJECT_STATUS.before.md). Старые исходники, датасеты, модели и результаты не заменены. Контроль SHA всех24897 прежних файлов и перечень допустимых изменений — [checks/integrity.json](checks/integrity.json); хеши математического ядра/manifest также проверены. Соседний generator_DE не используется.

Точное соответствие источников текущему срезу:

- [used_sources.json](used_sources.json): SHA-256 прочитанных источников, включая7000raw-записей, модели, исходники и таблицы; ссылки в этом документе ведут к конкретным артефактам/функциям.
- [repository_inventory.csv](repository_inventory.csv), [code_index.csv](code_index.csv), [report_index.json](report_index.json): карта репозитория и истории.
- [historical_metrics_ledger.csv](tables/historical_metrics_ledger.csv), [historical_sources.json](historical_sources.json): машинный реестр скалярных JSON-метрик и JSON pointers; отсутствие поля не восстанавливается предположением.
- [figure_manifest.json](figure_manifest.json):14названий,источников,оговорок; PNG и PDF — в figures/. Контактные листы и visual QA — в checks/.

Скрипты нового каталога воспроизводят только анализ существующих данных:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B results/scientific_evidence_2026-11-01/collect_evidence.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B results/scientific_evidence_2026-11-01/audit_saved_tables.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B results/scientific_evidence_2026-11-01/build_feature_catalog.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B results/scientific_evidence_2026-11-01/plot_evidence.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B results/scientific_evidence_2026-11-01/build_dossier.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B results/scientific_evidence_2026-11-01/verify_dossier.py
```

collect_evidence запрещает импорт solver/generator/training-driver и подменяет CatBoost.fit запретом. Разрешены только load_model/predict, чтение входных признаков и арифметика метрик. audit_saved_tables не импортирует solver/ML вообще. Эти команды не являются командами повторного обучения/решения; не следует запускать старые train/solve-команды из прежних REPORT.md ради этого документа. Полный pytest в этой работе не запускался, поскольку он содержит численные и обучающие проверки; отсутствие нового запуска тестов не подменяется историческим результатом. Текущие проверки относятся к целостности доказательной базы и существующей арифметике, их результат — [checks/report_validation.json](checks/report_validation.json).

Все запрошенные разделы либо снабжены источниками, либо имеют явную отметку недостаточности данных. Новые гипотезы не переведены в подтверждённые выводы.
''')
    document='\n'.join(parts)
    # Improve spacing in narrative without altering identifiers or feature names.
    replacements={'на33':'на 33','на1000':'на 1000','на997':'на 997','на150':'на 150','на60':'на 60',
                  'всех997':'всех 997','всех1000':'всех 1000','всех200':'всех 200','все997':'все 997',
                  'всех7000':'всех 7000','все23':'все 23','все258':'все 258','все63':'все 63',
                  'из9':'из 9','из20':'из 20','из129':'из 129','два сравнения≈':'два сравнения ≈'}
    for before,after in replacements.items():
        document=document.replace(before,after)
    (OUT/'EVIDENCE.md').write_text(document,encoding='utf-8')
    (OUT/'used_sources.json').write_text(json.dumps(USED,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(report='EVIDENCE.md',bytes=(OUT/'EVIDENCE.md').stat().st_size,
                          features=258,cases=len(cases),models=len(model_rows),historical_metric_rows=len(old))))


if __name__=='__main__':
    main()
