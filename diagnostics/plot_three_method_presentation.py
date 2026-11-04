"""Six simple presentation figures from the completed saved ML experiment.

Reads only the specified result directory. No model loading, fitting, inference,
ODE evaluation, or hyperparameter selection. Previous figures/manifests remain
historical artifacts; REPORT.md is backed up before the presentation is added.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

METHODS = ("remizov", "monte_carlo", "product_integral")
METHOD_LABELS = ["R128", "MC mean4", "PI"]
COLORS = ["#7653a6", "#28649a", "#db8b35", "#43866b"]  # ML, R, MC, PI
FEATURE_LABELS = {
    "b_rms": "Среднеквадратичное b",
    "b_sample_24": "Значение b (b_sample_24)",
    "cert_remizov_lambda": "Параметр λ",
    "a_total_variation": "Изменчивость a",
    "pre_diffusive_length": "Диффузионная длина",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def displayed(value, digits=4, percent=False):
    return f"{value:.{digits}f}".replace(".", ",") + ("%" if percent else "")


def saved_values(result):
    """Resolve the frozen latest model and already measured scalar metrics."""
    test = read(result / "test_metrics.json")
    lock = read(result / "selection_lock.json")
    selected = test["selected"]
    assert lock["primary_scenario"] == "new"
    for key in ("candidate_id", "policy", "biases"):
        assert selected[key] == lock["selected"][key]
    assert selected["class_order"] == list(METHODS) and selected["n"] == 150
    baseline = {}
    for method in METHODS:
        own = [p for p in test["policies"] if p["candidate_id"] == "new__always_" + method and p["policy"] == "constant"]
        assert len(own) == 1 and own[0]["n"] == selected["n"]
        baseline[method] = own[0]
    dataset = read(result / "dataset.json")
    counts = Counter(r["best_method"] for r in csv_rows(result / "labels.csv"))
    assert counts == Counter(r["best_method"] for r in dataset["rows"])
    assert len(dataset["rows"]) == sum(counts.values()) == 997
    assert all(counts[m] == dataset["summary"]["classes"][m] for m in METHODS)
    importance = [dict(feature=r["feature"], importance=float(r["importance"]))
                  for r in csv_rows(result / "feature_importance.csv")
                  if r["candidate_id"] == selected["candidate_id"]]
    importance.sort(key=lambda r: (-r["importance"], r["feature"]))
    top = importance[:5]
    assert len(top) == 5 and len({r["feature"] for r in top}) == 5
    analysis = read(result / "feature_analysis.json")
    assert analysis["importance_candidate"] == selected["candidate_id"]
    assert top == analysis["top_features"][:5]
    for row in top:
        row["label"] = FEATURE_LABELS.get(row["feature"], row["feature"])
    mean_order = [selected] + [baseline[m] for m in METHODS]
    graphs = [
        dict(name="01_winners", title="Распределение победителей", labels=METHOD_LABELS,
             values=[counts[m] for m in METHODS], display_values=[str(counts[m]) for m in METHODS],
             sample="Все 997 полных задач", source="labels.csv; dataset.json/summary/classes"),
        dict(name="02_accuracy", title="Accuracy на Test", labels=["ML-модель", "Всегда R128"],
             values=[selected["accuracy"], baseline["remizov"]["accuracy"]],
             display_values=[displayed(p["accuracy"] * 100, 2, True) for p in (selected, baseline["remizov"])],
             plotted_values=[selected["accuracy"] * 100, baseline["remizov"]["accuracy"] * 100],
             sample="Test, 150 задач", source="test_metrics.json: selected/accuracy; policies/new__always_remizov/accuracy"),
        dict(name="03_macro_f1", title="Macro-F1 на Test", labels=["ML-модель", "Всегда R128"],
             values=[selected["macro_f1"], baseline["remizov"]["macro_f1"]],
             display_values=[displayed(p["macro_f1"]) for p in (selected, baseline["remizov"])],
             sample="Test, 150 задач", source="test_metrics.json: selected/macro_f1; policies/new__always_remizov/macro_f1"),
        dict(name="04_mean_error", title="Средняя ошибка в x=0", labels=["ML-модель", "Всегда R128", "Всегда MC mean4", "Всегда PI"],
             values=[p["mean_error"] for p in mean_order],
             display_values=[displayed(p["mean_error"], 6) for p in mean_order],
             sample="Test, 150 задач", source="test_metrics.json: selected/mean_error; policies/new__always_{method}/mean_error"),
        dict(name="05_confusion_matrix", title="Матрица ошибок на Test", labels=METHOD_LABELS,
             matrix=selected["confusion_matrix"], rows="Истинный лучший метод", columns="Предсказанный метод",
             sample="Test, 150 задач", source="test_metrics.json: selected/confusion_matrix"),
        dict(name="06_top5_features", title="Пять важнейших признаков", labels=[r["label"] for r in top],
             values=[r["importance"] for r in top], features=[r["feature"] for r in top],
             display_values=[displayed(r["importance"], 2) for r in top],
             sample="Сохранённая важность выбранной модели", source="feature_importance.csv: selected candidate; descending importance, first 5; not renormalized"),
    ]
    return dict(candidate_id=selected["candidate_id"], policy=selected["policy"],
                biases=selected["biases"], class_order=list(METHODS), winners_n=997,
                test_n=150, top_features=top, graphs=graphs,
                numerical_calls=0, fit_calls=0, inference_calls=0, hyperparameter_search_calls=0)


def draw_figures(folder, data):
    # The temporary font cache avoids modifying the result inputs or user cache.
    with tempfile.TemporaryDirectory(prefix="three_method_simple_plots_") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        if previous is None:
            os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
            "axes.titlesize": 15, "axes.labelsize": 12, "axes.spines.top": False,
            "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 180,
            "pdf.fonttype": 42})
        manifests = []
        for spec in data["graphs"]:
            name = spec["name"]
            fig, ax = plt.subplots(figsize=(7.4, 4.9))
            if name == "05_confusion_matrix":
                matrix = np.asarray(spec["matrix"], dtype=int)
                assert matrix.sum() == data["test_n"]
                ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=16,
                                color="white" if matrix[i, j] > matrix.max() / 2 else "black")
                ax.set_xticks(range(3), spec["labels"])
                ax.set_yticks(range(3), spec["labels"])
                ax.set_xlabel("Предсказанный метод")
                ax.set_ylabel("Истинный лучший метод")
                ax.tick_params(length=0)
            elif name == "06_top5_features":
                bars = ax.barh(spec["labels"], spec["values"], color=COLORS[0], height=.65)
                assert np.array_equal([b.get_width() for b in bars], spec["values"])
                ax.bar_label(bars, labels=spec["display_values"], padding=6, fontsize=12)
                ax.invert_yaxis()
                ax.set_xlim(0, max(spec["values"]) * 1.18)
                ax.set_xlabel("Важность, %")
                ax.tick_params(axis="y", length=0)
            else:
                values = spec.get("plotted_values", spec["values"])
                labels = spec["labels"]
                if name == "04_mean_error":
                    labels = ["ML-модель", "Всегда\nR128", "Всегда\nMC mean4", "Всегда\nPI"]
                colors = COLORS[1:] if name == "01_winners" else COLORS[:len(values)]
                bars = ax.bar(labels, values, color=colors, width=.58)
                assert np.array_equal([b.get_height() for b in bars], values)
                ax.bar_label(bars, labels=spec["display_values"], padding=6, fontsize=12)
                ax.set_ylim(0, {"02_accuracy": 100, "03_macro_f1": 1}.get(name, max(values) * 1.18))
                ax.set_ylabel({"01_winners": "Число задач", "02_accuracy": "Верных выборов, %", "03_macro_f1": "Macro-F1", "04_mean_error": "Средняя ошибка"}[name])
                ax.set_xlabel("Метод")
                ax.tick_params(axis="x", length=0)
            ax.set_title(spec["title"], pad=18)
            fig.tight_layout(pad=1.3)
            files = []
            for extension in ("png", "pdf"):
                dest = folder / (name + "." + extension)
                assert not dest.exists()
                fig.savefig(dest, bbox_inches="tight", facecolor="white")
                files.append(dict(path=dest.name, sha256=sha(dest)))
            plt.close(fig)
            manifests.append(dict(name=name, title=spec["title"], source=spec["source"],
                                  sample=spec["sample"], files=files))
        if previous is None:
            os.environ.pop("MPLCONFIGDIR", None)
    return manifests


def report_section(folder_name, data):
    captions = [
        "R128 имеет минимальную ошибку в 826 из 997 задач; MC mean4 и PI выигрывают в 106 и 65 задачах. Это распределение по всему пригодному набору, а не только по Test.",
        "На Test модель правильно выбирает лучший метод в 88,67% случаев, постоянный R128 — в 90,67%. По Accuracy модель уступает этому простому baseline.",
        "Macro-F1 учитывает все три класса. Значения 0,3144 у ML и 0,3170 у постоянного R128 показывают, что улучшения распознавания трёх классов на Test не получено.",
        "Средняя абсолютная ошибка в x=0: ML — 1,658850, R128 — 1,658962, MC mean4 — 0,017275, PI — 0,054156. Средние ML и R128 почти совпадают; среди постоянных решений меньшее среднее у MC mean4. Крупные ошибки R128 существенно влияют на среднее.",
        "Строки показывают истинный лучший метод, столбцы — выбор ML. Модель распознала 133 победы R128, ни одной из 7 побед MC и ни одной из 7 побед PI; PI не выбран ни разу.",
        "Показаны только пять наибольших сохранённых importance выбранной модели; значения не перенормированы. b_rms — среднеквадратичное b; b_sample_24 — одно из сохранённых значений b; λ — параметр резольвентного уравнения; изменчивость a — сумма |Δa| по входной сетке; диффузионная длина — прежний входной признак √(2AT). Важность описывает модель и не доказывает причинную связь.",
    ]
    lines = ["## Простые графики результатов", "",
        "Оформление от 4 ноября 2026 года использует только сохранённые результаты. "
        "В графиках 2–5 показан прежний Test из 150 задач и модель, выбранная по latest Validation. "
        "Новых расчётов ОДУ, обучений, предсказаний моделей и подбора параметров не было. "
        "Все столбчатые диаграммы имеют линейную шкалу от нуля; округлены только подписи.", ""]
    for index, (graph, caption) in enumerate(zip(data["graphs"], captions), 1):
        lines.extend([f'### {index}. {graph["title"]}', "",
            f'![{graph["title"]}]({folder_name}/{graph["name"]}.png)', "", caption, "",
            f'[PDF]({folder_name}/{graph["name"]}.pdf)', ""])
    lines.extend([f"[Числа и источники графиков]({folder_name}/figure_data.json), "
                  f"[проверка оформления]({folder_name}/verification.json). "
                  f"Предыдущая версия отчёта сохранена в [REPORT.before.md]({folder_name}/REPORT.before.md). "
                  "Прежние графики и manifests сохранены как артефакты оформления от 3 ноября.", ""])
    return "\n".join(lines)


def build(result, folder_name):
    result = result.resolve()
    folder = result / folder_name
    if Path(folder_name).name != folder_name or folder.exists():
        raise ValueError("provide a new direct subdirectory name")
    data = saved_values(result)
    previous = {str(p.relative_to(result)): sha(p) for p in sorted(result.rglob("*")) if p.is_file()}
    folder.mkdir()
    shutil.copy2(result / "REPORT.md", folder / "REPORT.before.md")
    write_json(folder / "previous_artifacts_sha256.json", previous)
    inputs = ["labels.csv", "dataset.json", "test_metrics.json", "selection_lock.json", "feature_importance.csv", "feature_analysis.json"]
    write_json(folder / "source_sha256.json", {p: sha(result / p) for p in inputs})
    write_json(folder / "figure_data.json", data)
    manifests = draw_figures(folder, data)
    write_json(folder / "figure_manifest.json", dict(figures=manifests, figure_count=6,
        formats=["png", "pdf"], numerical_calls=0, fit_calls=0, inference_calls=0))
    brief = ("Ключевые выводы по графикам:\n\n"
        "- На всех 997 задачах чаще побеждает R128: 826 случаев против 106 у MC mean4 и 65 у PI.\n"
        "- На Test ML уступает постоянному R128 по Accuracy (88,67% против 90,67%) и Macro-F1 (0,3144 против 0,3170).\n"
        "- По средней ошибке среди постоянных решений лучше MC mean4: 0,017275; у ML средняя ошибка 1,658850.\n"
        "- Модель не распознала ни одной из 14 Test-побед MC/PI; полезное улучшение выбора трёх методов не подтверждено.\n"
        "- Наибольшая сохранённая importance у b_rms; это описание модели, а не доказательство причинности.\n\n")
    original = (folder / "REPORT.before.md").read_text()
    assert "## Простые графики результатов" not in original
    marker = "## Данные и численная цель"
    assert marker in original
    updated = original.replace(marker, brief + marker, 1)
    marker = "## Проверка, артефакты и воспроизведение"
    assert marker in updated
    updated = updated.replace(marker, report_section(folder_name, data) + "\n" + marker, 1)
    temporary = result / "REPORT.md.presentation.tmp"
    temporary.write_text(updated)
    os.replace(temporary, result / "REPORT.md")
    (folder / "README.md").write_text("# Простые графики ML-эксперимента\n\n" + brief +
        "Все шесть графиков сохранены в PNG и PDF; точные значения — figure_data.json, "
        "контрольные суммы изображений — figure_manifest.json. Графики вставлены в ../REPORT.md.\n")
    changed = [p for p, h in previous.items() if not (result / p).is_file() or sha(result / p) != h]
    assert changed == ["REPORT.md"], changed
    assert sha(folder / "REPORT.before.md") == previous["REPORT.md"]
    assert all(sha(result / p) == h for p, h in read(folder / "source_sha256.json").items())
    write_json(folder / "verification.json", dict(passed=True, saved_values_checked=True,
        selected_model_matches_lock=True, previous_files_checked=len(previous),
        changed_previous_files=changed, previous_report_restorable=True,
        figure_count=6, image_count=12, numerical_calls=0, fit_calls=0,
        inference_calls=0, hyperparameter_search_calls=0, source_data_unchanged=True))
    shutil.copy2(__file__, folder / Path(__file__).name)
    print(json.dumps(dict(folder=str(folder), figures=6, report_updated=True,
        numerical_calls=0, fit_calls=0, inference_calls=0), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=Path("results/ml_three_method_selector_2026-11-03"))
    parser.add_argument("--folder", default="simple_figures_2026-11-04")
    args = parser.parse_args()
    build(args.result, args.folder)
