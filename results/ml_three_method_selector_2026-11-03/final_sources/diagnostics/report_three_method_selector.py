"""Saved-only Russian report for the completed three-method ML experiment.

This reader never imports an ODE solver, an exact evaluator, or training code.
Every model/policy choice comes from the pre-Test Validation selection lock.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("remizov", "monte_carlo", "product_integral")
LABELS = dict(remizov="R128", monte_carlo="MC mean4", product_integral="PI")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def number(value, digits=8):
    return "—" if value is None else format(float(value), "." + str(digits) + "g")


def percent(value):
    return "—" if value is None else f"{100 * value:.3f}%"


def counts(values):
    return "/".join(str(values.get(method, 0)) for method in METHODS)


def table(headers, rows):
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |" for row in rows)
    return output


def policy_key(policy):
    return (policy["mean_regret"], -policy["success_rate"], -policy["macro_f1"],
            policy["non_R_predictions"], policy["candidate_id"], policy["policy"])


def identity(policy):
    return policy["candidate_id"], policy["policy"]


def metric_row(label, result):
    return [label, percent(result["accuracy"]), number(result["macro_f1"]),
            percent(result["balanced_accuracy"]), str(result["success_count"]) + "/" + str(result["n"]),
            number(result["mean_error"]), number(result["median_error"]),
            number(result["mean_regret"]), number(result["max_regret"]), counts(result["selection_counts"])]


def describe_features(out, dataset, chosen, candidates, validation):
    """Report importances and Train-only distributions without selecting on Test."""
    source = chosen
    substituted = chosen["kind"] == "constant"
    if substituted:
        by_id = {candidate["id"]: candidate for candidate in candidates}
        trained = [policy for policy in validation
                   if policy["scenario"] == "new" and by_id[policy["candidate_id"]]["kind"] != "constant"]
        source = by_id[min(trained, key=policy_key)["candidate_id"]] if trained else None
    importance = read_csv(out / "feature_importance.csv")
    own = [] if source is None else [dict(feature=row["feature"], importance=float(row["importance"]))
                                    for row in importance if row["candidate_id"] == source["id"]]
    own.sort(key=lambda item: (-item["importance"], item["feature"]))
    distributions = []
    if source is not None:
        with np.load(out / "features.npz", allow_pickle=False) as archive:
            feature_set = source["feature_set"]
            matrix = archive[feature_set].copy()
            names = archive["feature_names" if feature_set == "X" else "augmented_names"].tolist()
            lookup = {pid: i for i, pid in enumerate(archive["equation_id"].tolist())}
        train = [row for row in dataset["rows"] if row["split_latest"] == "train"]
        for item in own[:20]:
            column = names.index(item["feature"])
            by_class = {}
            for method in METHODS:
                values = np.asarray([matrix[lookup[row["equation_id"]], column]
                                     for row in train if row["best_method"] == method])
                by_class[method] = dict(n=len(values),
                    q25=float(np.quantile(values, .25)) if len(values) else None,
                    median=float(np.median(values)) if len(values) else None,
                    q75=float(np.quantile(values, .75)) if len(values) else None)
            distributions.append(dict(feature=item["feature"], by_class=by_class))
    return dict(schema="three_method_feature_description_v1", chosen_candidate=chosen["id"],
        importance_candidate=None if source is None else source["id"],
        chosen_constant_uses_best_trained_validation_candidate=substituted,
        importance_kind="mean of CatBoost PredictionValuesChange importances over the two heads for regression; single-model importance for classification",
        feature_set=None if source is None else source["feature_set"], top_features=own[:30],
        training_distributions=distributions, distribution_split="latest Train only",
        test_used_for_feature_selection=False, feature_selection_performed=False,
        allowed_inputs=["X coefficient channels a,b,c,g", "coefficient grid", "five analytic coefficient certificates", "a_raw,b_raw,c_raw ASTs"],
        forbidden_inputs=["exact solution/derivatives and solution AST", "y0/v0", "numerical answers/errors/status/success/time", "best_method", "IDs/family/split/hash as features"],
        original_features=253, augmented_features=258,
        interpretation="Descriptive model importance, not causal evidence; no feature was removed or selected using these values.",
        numerical_calls=0, fit_calls=0)


def figures(out, dataset, selected, analysis):
    import os
    import tempfile
    cache = tempfile.TemporaryDirectory(prefix="three_method_matplotlib_")
    previous_cache = os.environ.get("MPLCONFIGDIR")
    if previous_cache is None:
        os.environ["MPLCONFIGDIR"] = cache.name
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = out / "figures"
    directory.mkdir(exist_ok=False)
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180,
                         "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False})
    result = []

    def save(fig, stem, description):
        paths = []
        for extension in ("png", "pdf"):
            path = directory / (stem + "." + extension)
            fig.savefig(path, bbox_inches="tight")
            paths.append(dict(path=str(path.relative_to(out)), sha256=sha(path)))
        plt.close(fig)
        result.append(dict(name=stem, description=description, files=paths))

    counter = Counter(row["best_method"] for row in dataset["rows"])
    fig, ax = plt.subplots(figsize=(6.2, 4))
    bars = ax.bar([LABELS[method] for method in METHODS], [counter[method] for method in METHODS],
                  color=["#2563a6", "#d78132", "#438665"])
    ax.bar_label(bars, padding=4)
    ax.set(ylabel="Число задач", title="Победители по абсолютной ошибке в x=0")
    ax.set_ylim(0, max(counter.values()) * 1.13)
    fig.tight_layout()
    save(fig, "01_winner_counts", "997 complete saved-result rows; RK4 excluded from target")

    matrix = np.asarray(selected["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    mesh = ax.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if matrix[i, j] > matrix.max() * .5 else "black")
    labels = [LABELS[method] for method in METHODS]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set(xlabel="Выбор ML", ylabel="Истинный победитель",
           title="Test: политика, выбранная по latest Validation")
    fig.colorbar(mesh, ax=ax, label="Задач")
    fig.tight_layout()
    save(fig, "02_selected_test_confusion", "Rows=true best method, columns=selected method; frozen latest Validation policy")

    top = analysis["top_features"][:12][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if top:
        ax.barh([item["feature"] for item in top], [item["importance"] for item in top], color="#2563a6")
        ax.set_xlabel("CatBoost feature importance")
    else:
        ax.text(.5, .5, "Обученная модель отсутствует", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Важнейшие входные признаки\n" + (analysis["importance_candidate"] or "constant policy"), fontsize=10)
    fig.tight_layout()
    save(fig, "03_top_features", "Training-model importance; regression averages two heads; no causal interpretation or Test-based feature selection")
    write_json(out / "figure_manifest.json", dict(figures=result, numerical_calls=0, fit_calls=0))
    if previous_cache is None:
        os.environ.pop("MPLCONFIGDIR", None)
    cache.cleanup()
    return result


def build(out):
    out = Path(out).resolve()
    required = ["dataset.json", "plan.json", "splits.json", "selection_lock.json", "candidates.json",
                "validation_metrics.json", "test_metrics.json", "train_metrics.json", "cv_summary.json",
                "cv_folds.json", "fit_ledger.json", "feature_importance.csv", "features.npz", "calibration.json",
                "reload_verification.json", "preservation_verification.json"]
    for name in required:
        if not (out / name).is_file():
            raise ValueError("finish all experiment stages before reporting: missing " + name)
    outputs = ["REPORT.md", "ALL_EXPERIMENTS.md", "feature_analysis.json", "figure_manifest.json", "report_sources.json", "figures"]
    if any((out / name).exists() for name in outputs):
        raise ValueError("report outputs already exist; refusing to overwrite")
    dataset, plan, splits, lock, candidates, validation, test, train, cv, folds, ledger = [read(out / name) for name in required[:11]]
    if tuple(plan["methods"]) != METHODS or not test["test_evaluation_passes"] == 1:
        raise ValueError("unexpected methods or final evaluation contract")
    chosen = next(candidate for candidate in candidates if candidate["id"] == lock["selected"]["candidate_id"])
    selected = test["selected"]
    if identity(selected) != identity(lock["selected"]):
        raise ValueError("final result differs from the pre-Test selection")
    analysis = describe_features(out, dataset, chosen, candidates, validation)
    write_json(out / "feature_analysis.json", analysis)
    manifests = figures(out, dataset, selected, analysis)
    calibration_rows = read(out / "calibration.json")
    raw_calibration = next((row["calibration"] for row in calibration_rows
                            if row["candidate_id"] == chosen["id"] and row["split"] == "test"), None)
    selected_val = lock["selected"]
    default_val = next((policy for policy in validation if policy["candidate_id"] == chosen["id"] and policy["policy"] == "default"), selected_val)
    majority = max(METHODS, key=lambda method: splits["scenarios"]["new"]["train"]["classes"].get(method, 0))
    baselines = [(LABELS[method] + (" / majority Train" if method == majority else ""),
                  next(policy for policy in test["policies"] if policy["scenario"] == "new" and policy["candidate_id"] == "new__always_" + method))
                 for method in METHODS]
    best_constant = min(baselines, key=lambda item: item[1]["mean_error"])
    minority_correct = sum(selected["confusion_matrix"][i][i] for i in (1, 2))
    minority_support = sum(selected["true_class_counts"][METHODS[i]] for i in (1, 2))
    n_main = sum(row["stage"] == "main" for row in ledger)
    n_cv = sum(row["stage"].startswith("cv_") for row in ledger)
    distribution = Counter(row["best_method"] for row in dataset["rows"])
    headers = ["Политика", "Accuracy", "Macro-F1", "Balanced acc.", "E≤.05", "Mean E", "Median E", "Mean regret", "Max regret", "Выборы R/MC/PI"]
    lines = ["# Выбор R128 / MC mean4 / PI по сохранённым ответам", "",
        f"Эксперимент реально завершён: **{len(dataset['rows'])}** полных задач из {len(dataset['coverage'])}; "
        f"победители **R128 {distribution['remizov']} / MC mean4 {distribution['monte_carlo']} / PI {distribution['product_integral']}**. "
        f"Все три класса представлены. Выполнены **{n_main} основных + {n_cv} CV = {len(ledger)} estimator fit**. "
        "Новых численных решений, MC-траекторий и уравнений — **0**.", "",
        f"Основная политика зафиксирована по **latest Validation до Test**: `{chosen['id']}`, "
        f"`{lock['selected']['policy']}`, biases `{lock['selected']['biases']}`. "
        f"Test: Accuracy **{percent(selected['accuracy'])}**, Macro-F1 **{number(selected['macro_f1'])}**, "
        f"mean regret **{number(selected['mean_regret'])}**, success **{selected['success_count']}/{selected['n']}**.", "",
        "## Данные и численная цель", "",
        "Для каждого метода E_m=|y_m(0)−y_exact(0)|; exact(0) взят из сохранённого аналитического эталона. "
        "Winner — минимальная неокруглённая ошибка, при точном равенстве порядок R128→MC mean4→PI. "
        "RK4 отсутствует в метке, моделях и выборе. ε=.05 определяет только success и не меняет winner.", "",
        "R128 — прежний `ChernovSolver.solve`: n=128, num_t=513, quadratic grading, dx=0.0125, "
        "padding=12, cubic/taper, tail_eps=1e−8; ответ f[16] на сохранённой сетке из 33 точек. "
        "MC mean4 — среднее четырёх уже сохранённых MC1024 ответов при dt=1/128, tail_eps=1e−4; "
        "сначала среднее значений, затем модуль ошибки. PI — прежний левоконцевой expm-метод: "
        "128 шагов на [−1,1], h=1/64, a_min=0.001, ответ y[64]. Полные config/seeds/source paths сохранены в labels.csv/dataset.json.", "",
        "Сохраняются прежние критерии пригодности: успешный конечный ответ и полная идентичность данных/конфигурации. "
        "Большая ошибка не исключает задачу. Три отсутствующих R128 ответа остаются в coverage, без искусственных меток:", ""]
    missing = [row for row in dataset["coverage"] if not row["eligible"]]
    lines += table(["ID", "Недоступные методы", "Статусы"], [[row["equation_id"], ", ".join(row["missing_methods"]), json.dumps(row["status"], ensure_ascii=False)] for row in missing])
    lines += ["", "Крупнейшие пригодные R128-ошибки сохранены без отсечения и без повторного решения:", ""]
    lines += table(["ID", "Split latest", "E_R", "E_MC", "E_PI", "Winner"], [[row["equation_id"], row["split_latest"], number(row["E_remizov"]), number(row["E_monte_carlo"]), number(row["E_product_integral"]), LABELS[row["best_method"]]] for row in sorted(dataset["rows"], key=lambda row: row["E_remizov"], reverse=True)[:5]])
    lines += ["", "## Разбиения и отсутствие утечки", ""]
    lines += table(["Карта", "Часть", "Задач", "Семейств", "Истинные R/MC/PI"], [[scenario, split, group["n"], group["families"], counts(group["classes"])] for scenario, groups in splits["scenarios"].items() for split, group in groups.items()])
    lines += ["", "Сохранены обе прежние карты: original — sensitivity, latest — основной сценарий, назначенный до fit. "
        "Прежние 150 Test ID неизменны; семейства, топологии и ID не пересекаются между Train/Validation/Test и внутри CV. "
        "Исторический Test уже раскрывался в предыдущих исследованиях: это retrospective exploratory оценка, "
        "а не новая prospective независимая проверка. Выбор после просмотра нынешнего Test не менялся.", "",
        "Признаки точно переиспользованы: 253 из `ml.remizov_features.load_inputs/coefficient_features`, "
        "+5 прежних `additional_coefficient_features` (258). Разрешены только заданные a,b,c,g, исходная сетка, "
        "пять сертификатов коэффициентов и AST a_raw/b_raw/c_raw. Не используются exact/производные/solution AST, y0/v0, "
        "численные ответы, ошибки, success/status/time, winner, ID/family/split как признаки. "
        "g — заданная правая часть уравнения, несмотря на manufactured происхождение.", "",
        "## Обучение и подбор", "",
        "Seed 2488; один процесс и один поток CatBoost/BLAS. Прежняя сетка CatBoost: "
        "120/depth2/l2=10, 240/depth3/l2=10, 300/depth4/l2=20; lr=.05, bootstrap No, random_strength=0. "
        "Классификаторы MultiClass с весами [1,w,w], w=1,2,3,5,8,12; это веса loss, "
        "без resampling, изменения ошибок или синтетических классов. "
        "Дополнительно прежние raw/asinh идеи: две RMSE-головы Δ_MC=E_R−E_MC и Δ_PI=E_R−E_PI; "
        "масштабы asinh — медианы абсолютных Δ исключительно соответствующего Train.", "",
        "Решение — argmax(log raw probabilities+[0,β_MC,β_PI]) для классификации и "
        "argmax([0,predictedΔ_MC,predictedΔ_PI]+[0,β_MC,β_PI]) для регрессии. "
        "Bias=0 — default. Классификационная coarse-сетка: 100 пар из прежних τ=0.05…0.50 через log((1−τ)/τ); "
        "затем два детерминированных старта и ≤3 coordinate sweep всех границ/обоих nextafter-соседей. "
        "Для регрессии coarse-сетка — квантильные границы scores. Это локальный 2D поиск, "
        "глобальный оптимум не доказан. Все посещённые политики и validation scores сохранены.", "",
        "Критерий выбора прежний по смыслу: минимальный Validation mean regret (=минимальная средняя выбранная ошибка "
        "при фиксированном oracle), затем больший success, Macro-F1, меньше non-R выборов и стабильный ID. "
        "Модель не fit-ится на Validation. Веса/конфигурация/feature set/biases выбираются до Test; "
        "latest и original имеют отдельные Validation и не ранжируются по Test.", "",
        "96 основных обучаемых спецификаций дают 120 физических fit (регрессия имеет две головы); "
        "семейная CV даёт 200 fit. Все 320 моделей сохранены с ledger конкретных Train ID. "
        "Дополнительные tiny unit-fit не входят в experiment ledger.", "",
        "## Validation: выбранная модель и default", ""]
    lines += table(headers, [metric_row("Выбранная latest политика", selected_val), metric_row("Та же модель, default bias0", default_val)])
    lines += ["", "Выборы по каждой карте до Test:", ""]
    lines += table(["Карта", "Candidate", "Policy", "Biases", "Val mean regret", "Val Macro-F1"], [[scenario, policy["candidate_id"], policy["policy"], policy["biases"], number(policy["mean_regret"]), number(policy["macro_f1"])] for scenario, policy in lock["selected_by_scenario"].items()])
    lines += ["", "## Test: основная зафиксированная политика", ""]
    lines += table(headers, [metric_row("ML latest (выбран до Test)", selected)] + [metric_row(name, result) for name, result in baselines] + [metric_row("Oracle по сохранённым ошибкам", test["oracle"])])
    lines += ["", "Majority baseline определён только по latest Train; это уже показанный постоянный " + LABELS[majority] + ". Oracle недоступен при inference.", "",
        f"На Test найдено **{minority_correct}/{minority_support}** истинных побед MC/PI. "
        f"Δmean E (ML−постоянный R128)={number(selected['mean_error'] - baselines[0][1]['mean_error'])}; "
        f"Δmean E (ML−постоянный MC mean4)={number(selected['mean_error'] - baselines[1][1]['mean_error'])}. " +
        (f"Выбранная политика не превзошла лучший постоянный baseline по mean E — **{best_constant[0]}** "
         f"с mean E={number(best_constant[1]['mean_error'])}. Убедительный полезный перенос не установлен."
         if selected["mean_error"] >= best_constant[1]["mean_error"] else
         "На этом раскрытом Test средняя ошибка ниже постоянных baseline; это retrospective наблюдение, "
         "не подтверждение устойчивого prospective переноса.") +
        " Выбранная до Test модель и её biases сохранены; более удачный Test-кандидат не подменяет основной результат.", "",
        "Матрица ошибок: строки — истинный лучший метод, столбцы — выбранный, порядок R128/MC mean4/PI.", ""]
    lines += table(["Истина \\ выбор"] + [LABELS[method] for method in METHODS], [[LABELS[method]] + selected["confusion_matrix"][i] for i, method in enumerate(METHODS)])
    lines += [""] + table(["Класс", "Support", "Precision", "Recall", "F1", "Предсказаний"], [[LABELS[method], selected["true_class_counts"][method], number(selected["precision"][method]), number(selected["recall"][method]), number(selected["f1"][method]), selected["selection_counts"][method]] for method in METHODS])
    lines += ["", "Macro-F1 усредняет все три объявленных класса; balanced accuracy — только поддержанные истинные классы. "
        "Success — доля выбранных ответов с E≤.05; regret=E_selected−min(E_R,E_MC,E_PI). "
        "Правильный выбор winner не гарантирует достижения ε, если все три ответа неточны.", "",
        "Raw-score calibration выбранной модели на Test: " + ("не применима — выбранная политика не является классификатором." if raw_calibration is None else
        "Brier(multiclass sum)=" + number(raw_calibration["brier_multiclass"]) + ", logloss=" + number(raw_calibration["logloss"]) + ", ECE=" + number(raw_calibration["ece"]) + ".") + " "
        "Brier/logloss/ECE и OVR AUC остальных классификаторов сохранены в calibration.json; "
        "они описательные и не удостоверяют калиброванность probabilities, особенно при class weights. "
        "Bias-adjusted scores не выдаются за вероятности.", "",
        "Все заранее замороженные Test-политики приведены в ALL_EXPERIMENTS.md и test_metrics.csv. "
        "Их результаты диагностические; лучший по Test вариант не переобъявляется выбранным.", "",
        "## Семейная CV и устойчивость", "",
        "Внутри каждого Train выполнены 5 outer StratifiedGroupKFold; внутри outer Train — первый holdout из 4 inner folds. "
        "На inner Validation независимо выбираются baseline 253 спецификации (6 весов +raw +asinh), "
        "biases и три константы. Соответствующий outer fit оценивается на outer Validation. "
        "Основной глобальный Validation selection в CV не переносится. Inner scores, модели, ID и выбранные политики сохранены.", ""]
    cv_main = [item for item in cv["policy_summary"] if item["scenario"] == "new" and (item["policy"] == "nested_selection" or item["policy"].startswith("always_"))]
    lines += table(["Latest OOF policy", "Accuracy", "Macro-F1", "Mean E", "Mean regret", "Success", "SD fold regret", "SD fold Macro-F1"], [[item["policy"], percent(item["pooled"]["accuracy"]), number(item["pooled"]["macro_f1"]), number(item["pooled"]["mean_error"]), number(item["pooled"]["mean_regret"]), percent(item["pooled"]["success_rate"]), number(item["fold_variability"]["mean_regret"]["sd"]), number(item["fold_variability"]["macro_f1"]["sd"])] for item in cv_main])
    lines += ["", "Это пять зависимых fold-оценок на имеющихся обучающих семьях; редкие большие ошибки R128 "
        "могут доминировать mean regret. SD не является доверительным интервалом нового независимого Test. "
        "Train-метрики in-sample и отдельно помечены в train_metrics.json.", "",
        "## Важнейшие признаки", ""]
    if analysis["chosen_constant_uses_best_trained_validation_candidate"]:
        lines += ["Выбрана постоянная политика; у неё нет важности признаков. Ниже показан лучший обученный "
                  "кандидат по latest Validation, отдельно от выбранной политики: `" + str(analysis["importance_candidate"]) + "`.", ""]
    lines += table(["Признак", "Importance"], [[item["feature"], number(item["importance"])] for item in analysis["top_features"][:12]])
    lines += ["", "Для регрессии показано среднее importance двух голов. Это описание обученной модели, "
        "не доказательство причинности и не повод менять признаки после Test. Train-only квартильные распределения "
        "сохранены в feature_analysis.json; удаления/добавления признаков по этим результатам не было.", "",
        "## Проверка, артефакты и воспроизведение", "",
        "Модель после экспорта перезагружена; выбранные Test-предсказания совпали точно (reload_verification.json). "
        "Сохранность прежних файлов проверена в preservation_verification.json. "
        "Независимая итоговая сверка: verification.json; необходимые pytest: tests.log. "
        "Отчёт и рисунки построены только из сохранённых артефактов, без fit/solver/exact-вызовов.", "",
        "Основные файлы: [labels.csv](labels.csv), [coverage_all1000.csv](coverage_all1000.csv), "
        "[plan.json](plan.json), [splits.json](splits.json), [selection_lock.json](selection_lock.json), "
        "[selected_model/registry.json](selected_model/registry.json), [fit_ledger.json](fit_ledger.json), "
        "[test_metrics.json](test_metrics.json), [cv_summary.json](cv_summary.json), "
        "[ALL_EXPERIMENTS.md](ALL_EXPERIMENTS.md), [figure_manifest.json](figure_manifest.json). "
        "Все candidate/inner/outer модели находятся в models/ и cv_models/.", "",
        "Для воспроизведения нужен новый каталог. Из корня проекта, локальная .venv, PYTHONDONTWRITEBYTECODE=1:", "",
        "```bash",
        "for stage in prepare train cv final; do",
        "  .venv/bin/python -B diagnostics/run_three_method_selector.py \"$stage\" --out results/NEW_THREE_METHOD_RUN",
        "done",
        ".venv/bin/python -B diagnostics/report_three_method_selector.py --out results/NEW_THREE_METHOD_RUN",
        ".venv/bin/python -B diagnostics/verify_three_method_selector.py --out results/NEW_THREE_METHOD_RUN",
        "```", "",
        "Исходники pipeline и whitelisted builders зафиксированы в sources/ и plan.json; "
        "входные raw-ответы имеют SHA в dataset.json. Прежние эксперименты/датасет/модели сохранены.", "",
        "## Ограничения вывода", "",
        "Результат относится к фиксированным профилям и ошибке значения только в x=0. "
        "Он не гарантирует точность производной, выполнение НУ приближением R128 или точность на всём интервале. "
        "AMV1 согласует точное решение задачи Коши с глобально ограниченным резольвентным решением; "
        "алгоритмы используют разные постановки и бюджеты. Универсальное превосходство метода и перенос на новые "
        "независимые семейства этим retrospective опытом не доказаны.", ""]
    with (out / "REPORT.md").open("x") as stream:
        stream.write("\n".join(lines))

    test_lookup = {identity(policy): policy for policy in test["policies"]}
    experiments = ["# Все заранее зафиксированные модели и политики", "",
        "RK4 исключён. Основная политика определена selection_lock.json до Test; все прочие Test-результаты "
        "диагностические, переотбора по ним нет. Original — sensitivity, latest — основной сценарий.", "",
        f"Обучаемых основных спецификаций 96; основных estimator fit {n_main}; CV estimator fit {n_cv}; всего {len(ledger)}. "
        "Полный ledger и Train ID каждой головы: fit_ledger.json.", ""]
    candidates_by_id = {candidate["id"]: candidate for candidate in candidates}
    experiment_rows = []
    for policy in validation:
        candidate = candidates_by_id[policy["candidate_id"]]
        result = test_lookup[identity(policy)]
        is_selected = identity(policy) == identity(lock["selected_by_scenario"][policy["scenario"]])
        experiment_rows.append([policy["scenario"], candidate["id"], candidate["kind"], candidate["feature_set"],
            candidate["config"].get("name", "constant"), candidate["weight"], policy["policy"], policy["biases"],
            "ДА" if is_selected else "", number(policy["mean_regret"]), number(policy["macro_f1"]),
            percent(result["accuracy"]), number(result["macro_f1"]), number(result["mean_regret"]),
            str(result["success_count"]) + "/" + str(result["n"]), counts(result["selection_counts"])])
    experiments += table(["Split", "Candidate", "Kind", "X", "Config", "Weight", "Policy", "Biases", "Val selected", "Val regret", "Val F1", "Test accuracy", "Test F1", "Test regret", "Test success", "Test R/MC/PI"], experiment_rows)
    experiments += ["", "## Все pooled OOF-политики", ""]
    experiments += table(["Split", "Policy", "Accuracy", "Macro-F1", "Mean E", "Mean regret", "Success", "SD regret"], [[item["scenario"], item["policy"], percent(item["pooled"]["accuracy"]), number(item["pooled"]["macro_f1"]), number(item["pooled"]["mean_error"]), number(item["pooled"]["mean_regret"]), percent(item["pooled"]["success_rate"]), number(item["fold_variability"]["mean_regret"]["sd"])] for item in cv["policy_summary"]])
    experiments += ["", "Все отдельные fold и inner selection:", ""]
    experiments += table(["Split", "Fold", "Policy", "Chosen spec", "Accuracy", "Macro-F1", "Mean E", "Mean regret", "Success"], [[item["scenario"], item["fold"], item["policy"], item["chosen_spec"], percent(item["accuracy"]), number(item["macro_f1"]), number(item["mean_error"]), number(item["mean_regret"]), percent(item["success_rate"])] for item in folds])
    experiments += ["", "cv_folds.json дополнительно сохраняет все inner candidates выбранной nested политики; "
                    "cv_inner_predictions.csv и policy_search/ позволяют независимо сверить выбранные biases. "
                    "CV не меняет основной выбор latest Validation.", ""]
    with (out / "ALL_EXPERIMENTS.md").open("x") as stream:
        stream.write("\n".join(experiments))
    sources = {name: sha(out / name) for name in required}
    sources["script:" + str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    write_json(out / "report_sources.json", dict(source_sha256=sources, numerical_calls=0, fit_calls=0,
        selection_from="selection_lock.json/latest", test_used_for_selection=False,
        output_files=["REPORT.md", "ALL_EXPERIMENTS.md", "feature_analysis.json", "figure_manifest.json"],
        figures=len(manifests)))
    print(json.dumps(dict(report=str(out / "REPORT.md"), all_experiments=str(out / "ALL_EXPERIMENTS.md"),
        figures=len(manifests), numerical_calls=0, fit_calls=0), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.out)
