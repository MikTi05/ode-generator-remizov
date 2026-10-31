"""Report one saved mean4 CatBoost fit; never fit or run a numerical method.

The only file written is the new run's REPORT.md, created exclusively. The
previous median-error experiment is read through its saved comparison metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("remizov", "monte_carlo")


def _read(path):
    return json.loads(Path(path).read_text())


def _number(value):
    return "—" if value is None else f"{value:.10g}"


def _percent(value):
    return "—" if value is None else f"{100 * value:.3f}%"


def _ml(metrics, split):
    own = metrics[split]
    return own.get("ML", own)


def _path(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def _boolean(value):
    return "да" if value else "нет"


def build_report(out):
    """Read only saved artifacts and exclusively create ``out/REPORT.md``."""
    out = Path(out).resolve()
    evidence = _read(out / "evidence.json")
    metrics = _read(out / "metrics.json")
    registry = _read(out / "model/registry.json")
    audit = _read(out / "data_audit.json")
    comparison = _read(out / "previous_model_comparison.json")
    outliers = _read(out / "outliers.json")
    if isinstance(outliers, dict):
        outliers = outliers.get("outliers", outliers.get("rows", []))
    rows = evidence["main_rows"]
    coverage = evidence.get("coverage_rows", rows)
    support = audit["paired_summary"]["winners"]
    old_metrics = comparison["old_metrics_under_old_target"]
    old_on_new = comparison["old_policy_under_mean_target"]
    old_support = Counter()
    for split in ("train", "validation", "test"):
        old_support.update(_ml(old_metrics, split)["true_class_counts"])

    lines = [
        "# Выбор метода в x=0: ошибка среднего четырёх MC-ответов, ε=0.05", "",
        "Выполнено **одно фиксированное обучение CatBoost** на сохранённых данных. "
        "Сравниваются массовый **Remizov n=128, cubic/taper** и **среднее четырёх "
        "сохранённых MC1024 оценок в x=0**. Новых решений ОДУ, траекторий, уравнений "
        "и поиска параметров нет. Прежний эксперимент с медианой seed-ошибок сохранён отдельно.", "",
        "Основная метка теперь определяется по", "",
        "```text",
        "f_MC_mean(0) = (f_seed0(0) + f_seed1(0) + f_seed2(0) + f_seed3(0)) / 4",
        "E_MC = abs(f_MC_mean(0) - y_exact(0))",
        "E_R  = abs(f_Remizov(0) - y_exact(0))",
        "best_method = remizov, если E_R <= E_MC; иначе monte_carlo",
        "success = E_selected <= 0.05",
        "```", "",
        "Среднее берётся **до** абсолютной ошибки. Прежняя медиана "
        "`median(abs(f_seed(0)−exact(0)))` остаётся исторической постановкой и "
        "не используется в новой разметке. Оба целевых события сохраняют непрерывные "
        "ошибки; сравнения с ε выполняются без округления. Четыре исходных seed по "
        "1024 траектории сохранены: это агрегат уже вычисленных ответов, без нового "
        "запуска с иным M, h или seed. Скорость в выборе не участвует.", "",
        "## Данные и целевые метки", "",
        f"Сопоставимы **{len(rows)}/{len(coverage)}** уравнений: Remizov точнее "
        f"**{support['remizov']}**, средний MC точнее **{support['monte_carlo']}**. "
        f"В прежней разметке соответствующие числа были **{old_support['remizov']}** "
        f"и **{old_support['monte_carlo']}**. Это изменение определения MC-оценки, "
        "а не новые численные решения.", "",
        "| Split | Исходно | Сопоставимо | Нет пар | R точнее | MC точнее | R успех | MC успех | Oracle успех |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        saved = audit["by_split"][split]
        own, original = saved["summary"], saved["original"]
        lines.append(
            f"| {split} | {original} | {own['n']} | {original-own['n']} | "
            f"{own['winners'][METHODS[0]]} | {own['winners'][METHODS[1]]} | "
            f"{own['remizov_success_count']} | {own['mc_success_count']} | "
            f"{own['oracle_success_count']} |")
    lines += [
        "", "Исходный split **700/150/150** и раздельные семейства сохранены. "
        "Три исходных worker_exception Remizov не имеют вычисленной ошибки: "
        "они исключены из обучения и парных метрик, без синтетических winner-меток. "
        "Большие конечные ошибки Remizov сохранены без замены диагностическими "
        "ответами другого профиля. Строки n=256 и внешней серии 60 уравнений "
        "в этот fit и приведённые метрики не входят.", "",
        "## Одно обучение и ограничения проверки", "",
        f"Fit использует только **{len(registry['train_equations'])}** сопоставимых train-строк. "
        "Использованы прежние 253 признака из заданных a,b,c,g, аналитических "
        "сертификатов и AST коэффициентов. Exact, численные ответы, ошибки, "
        "success и фактическое время в признаках отсутствуют. Параметры CatBoost "
        "зафиксированы до fit: 120 итераций, depth=2, learning_rate=0.05, "
        "l2_leaf_reg=10, Logloss, seed=2488, один поток, bootstrap=No, "
        "random_strength=0. Балансировка, resampling, early stopping, поиск "
        "конфигураций и выбор по validation/test не проводились.", "",
        "Train-метрики описывают подгонку на обучающих уравнениях. Validation и "
        "test содержат отдельные семейства, однако их численные ответы уже были "
        "раскрыты предыдущими исследованиями. Этот выпуск — **повторная "
        "ретроспективная проверка новой заранее заданной MC-агрегации**, "
        "без заявления нового проспективного независимого подтверждения. "
        "Успех на train, в том числе распознавание train-выброса, не доказывает "
        "перенос на validation/test.", "",
        "## Метрики всех стратегий", "",
        "Accuracy — доля совпадений с методом меньшей ошибки. Balanced accuracy "
        "усредняет recall поддержанных классов; macro-F1 использует оба "
        "фиксированных класса. Regret = E_selected − min(E_R,E_MC). Oracle "
        "использует фактические ошибки после расчётов. Его chosen_success "
        "отличается от успеха метода, выбранного ML.", "",
        "| Split | Стратегия | Accuracy | Balanced accuracy | Macro-F1 | E≤0.05 | Средняя E | Медианная E | Средний regret | Медианный regret | Max regret |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        for strategy in ("ML", "always_remizov", "always_monte_carlo", "oracle"):
            m = metrics[split][strategy]
            lines.append(
                f"| {split} | {strategy} | {_percent(m['accuracy'])} | "
                f"{_percent(m['balanced_accuracy'])} | {_number(m['macro_f1'])} | "
                f"{m['success_count']}/{m['n']} ({_percent(m['success_rate'])}) | "
                f"{_number(m['mean_error'])} | {_number(m['median_error'])} | "
                f"{_number(m['mean_regret'])} | {_number(m['median_regret'])} | "
                f"{_number(m['max_regret'])} |")
    lines += ["", "| Split | Выбрано R | Выбрано MC | Доля R | Доля MC |",
              "|---|---:|---:|---:|---:|"]
    for split in ("train", "validation", "test"):
        m = metrics[split]["ML"]
        lines.append(f"| {split} | {m['selection_counts']['remizov']} | "
                     f"{m['selection_counts']['monte_carlo']} | "
                     f"{_percent(m['selection_fractions']['remizov'])} | "
                     f"{_percent(m['selection_fractions']['monte_carlo'])} |")
    cm = metrics["test"]["ML"]["confusion_matrix"]
    lines += ["", "Confusion matrix test: строки — фактический winner, столбцы — выбор ML.", "",
              "| Факт / ML | remizov | monte_carlo |", "|---|---:|---:|",
              f"| remizov | {cm[0][0]} | {cm[0][1]} |",
              f"| monte_carlo | {cm[1][0]} | {cm[1][1]} |", ""]
    test = metrics["test"]
    new, constant_r, constant_mc = (test[k] for k in ("ML", "always_remizov", "always_monte_carlo"))
    if new["selection_counts"]["monte_carlo"] == 0:
        lines += [f"**На test CatBoost всегда выбирает Remizov: {new['n']}/{new['n']}. "
                  "Распознавание более точного MC на этом test не подтверждено.**", ""]
    if all(new[k] == constant_r[k] for k in ("accuracy", "success_rate", "mean_error", "median_error")):
        lines += ["**Преимущества перед постоянным Remizov на test нет:** "
                  "accuracy, достижение ε, средняя и медианная ошибки совпадают.", ""]
    lines += [
        f"На test ML достигает ε в **{new['success_count']}/{new['n']}** случаев. "
        f"Против постоянного R: Δaccuracy **{100*(new['accuracy']-constant_r['accuracy']):+.3f} п.п.**, "
        f"Δуспех **{100*(new['success_rate']-constant_r['success_rate']):+.3f} п.п.**, "
        f"Δсредняя E **{new['mean_error']-constant_r['mean_error']:+.10g}**, "
        f"Δмедианная E **{new['median_error']-constant_r['median_error']:+.10g}**. "
        f"Средняя E постоянного MC — **{_number(constant_mc['mean_error'])}**, "
        f"его медианная E — **{_number(constant_mc['median_error'])}**, "
        f"успех — **{constant_mc['success_count']}/{constant_mc['n']}**. "
        "Доля правильных winner-меток, достижение ε и средняя численная ошибка "
        "описывают разные свойства; высокая accuracy на преобладающем классе "
        "сама по себе не подтверждает преимущества селектора.", "",
        "## Сопоставление с прежней моделью", "",
        "Прежние веса повторно не обучались. Их сохранённые решения переоценены "
        "по новому E_MC. Результаты прежней модели на прежней цели приводятся "
        "отдельно: прямое сравнение accuracy разных целей не измеряет улучшение модели.", "",
        "| Split | Модель / цель | Accuracy | Balanced accuracy | Macro-F1 | E≤0.05 | Средняя E | Медианная E | Средний regret |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        for label, collection in (("прежняя / median seed-errors", old_metrics),
                                  ("прежняя / mean4", old_on_new),
                                  ("новая / mean4", metrics)):
            m = _ml(collection, split)
            lines.append(f"| {split} | {label} | {_percent(m['accuracy'])} | "
                         f"{_percent(m['balanced_accuracy'])} | {_number(m['macro_f1'])} | "
                         f"{_percent(m['success_rate'])} | {_number(m['mean_error'])} | "
                         f"{_number(m['median_error'])} | {_number(m['mean_regret'])} |")
    lines += ["", "## Два сохранённых выброса", ""]
    for r in outliers:
        previous = r.get("previous_selected_method", "—")
        lines += [
            f"### {r['equation_id']} ({r['split']})", "",
            f"E_R = **{_number(r['E_R'])}**; E_MC(mean4) = **{_number(r['E_MC'])}**. "
            f"Среднее MC f(0) = {_number(r['mc_mean_f0'])}, exact(0) = {_number(r['y_exact0'])}. "
            f"Прежний выбор: **{previous}**; новый выбор: **{r['selected_method']}**. "
            f"Новая выбранная ошибка = **{_number(r['E_selected'])}**, "
            f"достижение ε: **{_boolean(r['selected_success'])}**, "
            f"regret = {_number(r['regret'])}. "
            f"P(MC): прежняя {_number(r.get('previous_p_monte_carlo'))}, "
            f"новая {_number(r.get('p_monte_carlo'))}.", "",
        ]
        if r["split"] == "train":
            lines += ["Это прогноз на обучающем уравнении. Его правильный выбор не "
                      "считается validation/test-подтверждением.", ""]
        else:
            if r["selected_method"] == "remizov" and r["E_R"] > r["E_MC"]:
                lines += ["**Этот test-выброс пропущен:** выбран Remizov с большей "
                          "ошибкой; преимущество по предотвращению этого случая отсутствует.", ""]
            elif r["selected_method"] == "monte_carlo" and r["E_MC"] < r["E_R"]:
                lines += ["В этом test-случае выбран MC с меньшей ошибкой. "
                          "Это наблюдение на повторно используемом test, без нового "
                          "проспективного подтверждения обнаружения выбросов.", ""]
            else:
                lines += ["Этот случай учтён в исходном test, повторно используемом "
                          "в исследовании; нового проспективного статуса он не получает.", ""]
        if min(r["E_R"], r["E_MC"]) > .05:
            lines += ["**Оба метода не достигают ε=0.05.** Выбор меньшей ошибки "
                      "не превращает этот случай в успешное решение.", ""]
        features = r.get("features", [])
        if features:
            lines += ["| Входной признак | Значение | Перцентиль в train | Глобальная важность |",
                      "|---|---:|---:|---:|"]
            for f in features[:5]:
                lines.append(f"| `{f['name']}` | {_number(f['value'])} | "
                             f"{_number(f['train_percentile'])} | {_number(f['importance'])} |")
            lines += [""]
    if outliers:
        lines += ["Глобальная важность признаков и их положение в train описывают "
                  "входы модели. Они не являются доказательством причины отдельного "
                  "прогноза или сертификатом обнаружения численной неустойчивости. "
                  "Важности всех 253 признаков сохранены в `feature_importance.csv`.", ""]
    lines += [
        "## Математическая и пространственная область", "",
        "Оба метода аппроксимируют одно ограниченное на ℝ решение "
        "(λ−L)y=g_R, L=a∂²+b∂+c_R, c=c_R−λ, g_R=−g. "
        "Chernoff использует c_R и вес e^(−λt); FK использует "
        "dX=b dt+√(2a)dW и exp(∫(c_R−λ)dt)g_R. "
        "Сохранены исторические дискретизации и разные горизонты: у R "
        "tail_eps=1e−8 и окно [-13,13] с taper, у MC forcing-aware "
        "tail_eps=1e−4. Это аппроксимации одной глобальной задачи, без "
        "утверждения равенства усечённых интегралов или бюджетов.", "",
        "**Все новые ML-метрики относятся только к реально вычисленному x=0.** "
        "Они не оценивают максимальную ошибку на 33 узлах или между ними. "
        "Прежние отдельно сохранённые сеточные исследования этой MC-агрегацией "
        "не заменяются. Вероятности CatBoost не являются гарантией точности "
        "нового решения или следующего seed-запуска.", "",
        "## Артефакты и воспроизведение", "",
        "- `evidence.json`: непрерывные ошибки, четыре исходных MC f(0), "
        "их среднее, identity и источники; `comparable_x0.csv` и `coverage.csv`: таблицы.",
        "- `model/model.cbm`, `model/registry.json`: единственная новая модель и контракт.",
        "- `metrics.json`, `predictions.csv`: все split/strategy-метрики и прогнозы.",
        "- `previous_model_comparison.json`, `outliers.json`, "
        "`feature_importance.csv`: переоценка прежней политики и разбор выбросов.",
        "- `plan.json`, `data_audit.json`, `source_sha256.json`: конфигурация, "
        "охват и происхождение данных.", "", "```bash",
        "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/train_mean_mc_selector.py \\",
        "  --plan results/ml_mean_mc_selector_2026-10-31/plan.json \\",
        "  --out results/ml_mean_mc_selector_reproduction",
        "```", "",
        f"Этот выпуск: `{_path(out)}`. Повтор команды создаёт новый каталог; "
        "прежний run не перезаписывается. Новых численных вызовов **0**, "
        f"fit новой модели **{audit.get('fit_count', registry.get('fit_count', 1))}**.",
    ]
    path = out / "REPORT.md"
    with path.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build_report(args.out))
