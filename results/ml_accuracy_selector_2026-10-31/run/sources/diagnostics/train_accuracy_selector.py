"""One fixed CatBoost fit for saved R128 versus MC1024 errors at x=0.

Reads existing numerical outputs; never executes a numerical method. No search,
class reweighting, validation tuning, resampling, or early stopping is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ml.accuracy_selector import (AccuracyMethodSelector, CATBOOST_CONFIG,
                                  EPSILON, METHODS, SCHEMA, selection_metrics, sha)
from ml.remizov_features import VERSION, load_inputs

DATA = ROOT / "data/all_methods_v1_1000_2026-10-18"


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    names = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def summary(rows):
    return dict(n=len(rows), winners={m:sum(r["best_method"] == m for r in rows) for m in METHODS},
                exact_ties=sum(r["E_R"] == r["E_MC"] for r in rows),
                remizov_success_count=sum(r["remizov_success"] for r in rows),
                mc_success_count=sum(r["mc_success"] for r in rows),
                oracle_success_count=sum(r["chosen_success"] for r in rows))


def fmt(value):
    return "—" if value is None else f"{value:.8g}"


def pct(value):
    return "—" if value is None else f"{100*value:.3f}%"


def make_report(out, audit, metrics, registry, evidence):
    total = audit["paired_summary"]
    lines = ["# Выбор более точного метода в x=0, ε=0.05", "",
             "Основное сравнение: сохранённый Remizov **n=128** и Monte Carlo "
             "**M=1024, h=1/128**, без изменения параметров и новых решений ОДУ. "
             "Метка определяется меньшей абсолютной ошибкой в **вычисленном x=0**; "
             "точное равенство разрешается в пользу Remizov. Скорость не участвует.", "",
             f"Сопоставимы **{total['n']}/1000** уравнений. Remizov точнее "
             f"**{total['winners']['remizov']}**, Monte Carlo **{total['winners']['monte_carlo']}**, "
             f"равенств **{total['exact_ties']}**. Успехи ε=0.05: Remizov "
             f"**{total['remizov_success_count']}/{total['n']}**, MC "
             f"**{total['mc_success_count']}/{total['n']}**, oracle "
             f"**{total['oracle_success_count']}/{total['n']}**.", "",
             "У MC E_MC — **медиана четырёх абсолютных ошибок seed**, а не ошибка "
             "усреднённого решения. В таблицах сохранены непрерывные ошибки, f(0), "
             "exact(0), seed, число seed, разброс и ссылки на исходники. Сравнения "
             "с порогом выполняются до форматирования без округления.", "",
             "## Покрытие и обучение", "",
             "| Split | Исходно | Сопоставимо | Нет пар | R точнее | MC точнее | R успех | MC успех | Oracle успех |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("train", "validation", "test"):
        s = audit["by_split"][split]
        p = s["summary"]
        lines.append(f"| {split} | {s['original']} | {p['n']} | {s['original']-p['n']} | "
                     f"{p['winners']['remizov']} | {p['winners']['monte_carlo']} | "
                     f"{p['remizov_success_count']} | {p['mc_success_count']} | {p['oracle_success_count']} |")
    lines += ["", "Три исходных вычислительных отказа Remizov исключены из обучения и "
              "парных метрик; их ошибки и winner-метки отсутствуют. Они перечислены в "
              "`coverage.csv` и не превращены в проигрыши Remizov. Исходные семейства "
              "и топологии не пересекаются между train/validation/test.", "",
              "Ровно один CatBoostClassifier с двумя классами; параметры в `model/registry.json`. "
              "Fit использует только сопоставимый train. Параметры зафиксированы до fit; "
              "подбора, балансировки, early stopping и выбора по validation/test нет. "
              "253 прежних признака строятся только из a,b,c,g, аналитических сертификатов "
              "и AST коэффициентов. Exact, ошибки, ответы, success и время в X не входят. "
              "Прежние дескрипторы pre_* с n=128 сохранены без изменений.", "",
              "## Метрики выбора", "",
              "Accuracy — совпадение выбранного метода с фактическим winner. "
              "Balanced accuracy — средний recall классов с ненулевой поддержкой; "
              "macro-F1 усредняется по обоим фиксированным классам с zero_division=0. "
              "Regret = E_selected − min(E_R,E_MC). Успех выбранного метода отличается "
              "от oracle chosen_success в исходной разметке.", "",
              "| Split | Стратегия | Accuracy | Balanced accuracy | Macro-F1 | E≤0.05 | Средняя E | Медианная E | Средний regret | Медианный regret | Max regret |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, strategies in metrics.items():
        for strategy, m in strategies.items():
            lines.append(f"| {split} | {strategy} | {pct(m['accuracy'])} | {pct(m['balanced_accuracy'])} | "
                         f"{fmt(m['macro_f1'])} | {pct(m['success_rate'])} | {fmt(m['mean_error'])} | "
                         f"{fmt(m['median_error'])} | {fmt(m['mean_regret'])} | {fmt(m['median_regret'])} | {fmt(m['max_regret'])} |")
    lines += ["", "| Split | Выбрано R | Выбрано MC | Доля R | Доля MC |", "|---|---:|---:|---:|---:|"]
    for split in metrics:
        m = metrics[split]["ML"]
        lines.append(f"| {split} | {m['selection_counts']['remizov']} | {m['selection_counts']['monte_carlo']} | "
                     f"{pct(m['selection_fractions']['remizov'])} | {pct(m['selection_fractions']['monte_carlo'])} |")
    cm = metrics["test"]["ML"]["confusion_matrix"]
    lines += ["", "Confusion matrix на test: строки — фактический winner, столбцы — выбор ML.", "",
              "| Факт / ML | remizov | monte_carlo |", "|---|---:|---:|",
              f"| remizov | {cm[0][0]} | {cm[0][1]} |", f"| monte_carlo | {cm[1][0]} | {cm[1][1]} |", ""]
    all_predictions = Counter(row["selected_method"] for row in audit["prediction_rows"])
    if all_predictions.get("monte_carlo", 0) == 0:
        lines += ["**CatBoost всегда выбирает Remizov на всех сопоставимых уравнениях. "
                  "Различение полезных случаев Monte Carlo этой моделью не подтверждено.**", ""]
    test = metrics["test"]
    if test["ML"]["selection_counts"]["monte_carlo"] == 0 and test["ML"]["n"]:
        lines += [f"**На test CatBoost всегда выбирает Remizov ({test['ML']['n']}/{test['ML']['n']}). "
                  f"Все {test['ML']['true_class_counts']['monte_carlo']} случая, где MC точнее, пропущены. "
                  "Преимущества перед постоянным Remizov на test нет.**", ""]
    lines += [f"Test: ML правильно выбирает метод в **{test['ML']['accuracy']*100:.3f}%** случаев. "
              f"Против постоянного R: Δaccuracy **{(test['ML']['accuracy']-test['always_remizov']['accuracy'])*100:+.3f} п.п.**, "
              f"Δmean E **{test['ML']['mean_error']-test['always_remizov']['mean_error']:+.8g}**; "
              f"против постоянного MC: Δaccuracy **{(test['ML']['accuracy']-test['always_monte_carlo']['accuracy'])*100:+.3f} п.п.**, "
              f"Δmean E **{test['ML']['mean_error']-test['always_monte_carlo']['mean_error']:+.8g}**. "
              "Совпадение с большинством и средняя численная ошибка отвечают на разные вопросы; "
              "редкие крупные конечные ошибки не удалены и не заменены результатами другого профиля.", "",
              "## Математическая и пространственная область", "",
              "Оба метода аппроксимируют одно ограниченное на ℝ решение "
              "(λ−L)y=g_R, L=a∂²+b∂+c_R, c=c_R−λ, g_R=−g. В Chernoff входит c_R; "
              "в FK используется dX=b dt+√(2a)dW и exp(∫(c_R−λ)dt)g_R. "
              "Exact с компактным носителем и коэффициенты сверены по dataset SHA и expression_hash. "
              "Согласованные начальные данные не превращают задачу в произвольную задачу Коши. "
              "Исторические горизонты T и хвостовые параметры различны и сохранены: "
              "R — legacy tail_eps=1e−8 без G; MC — сертифицированный G и tail_eps=1e−4. "
              "Окно R [-13,13] с taper — численная аппроксимация глобальной задачи; "
              "MC траектории этим окном не ограничены. Равенство усечённых интегралов "
              "или вычислительных бюджетов не заявляется.", "",
              "**Основные выводы относятся только к x=0.** Отдельные таблицы `grid33.csv` "
              "используют исключительно реальные результаты обоих методов на 33 общих узлах; "
              "их покрытие, профили и число seed указаны в `evidence_audit.json`. Даже grid33 "
              "не доказывает точность между узлами на всём интервале.", "",
              "## Дополнительные данные и ограничения", "",
              "n=256 сохранён в `supplementary_n256.csv`; эти строки исключены из основного "
              "датасета, fit и всех приведённых ML-метрик. Прежние внешние 60 уравнений "
              "относятся к другому dataset SHA: MC4096 есть только на 20, на 40 пары отсутствуют. "
              "Их n256/MC4096 результаты вынесены в `external_n256.csv`, без приписывания "
              "к исходному test150 и без оценки основной модели на другом профиле.", "",
              "Это ретроспективная проверка на исходных раздельных семействах; старые результаты "
              "уже были раскрыты предыдущими исследованиями. Test не участвовал в fit или выборе "
              "параметров новой модели. Небольшое число MC-winner ограничивает оценку их распознавания; "
              "вероятности CatBoost не являются гарантией точности следующего решения. "
              "Время методов не входит ни в цель, ни в признаки, ни в выбор.", "",
              "## Воспроизведение", "", "```bash",
              "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/train_accuracy_selector.py \\",
              "  --out results/ml_accuracy_selector_reproduction",
              "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_accuracy_selector.py",
              "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/verify_accuracy_selector.py \\",
              f"  --out {out.relative_to(ROOT)}",
              "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/predict_accuracy_selector.py \\",
              f"  --model {out.relative_to(ROOT)}/model --input {out.relative_to(ROOT)}/example_input.json",
              "```", "", "Полные метрики: `metrics.json`; прогнозы: `predictions.csv`; сравнимые пары: "
              "`comparable_x0.csv`; покрытие: `coverage.csv`; 4000 MC seed: `mc_seeds.csv`; "
              "источники и SHA: `source_sha256.json`; модель: `model/model.cbm`. "
              "Новые численные вызовы: **0**; fit основного CatBoost: **1**."]
    lines += ["", "## Отдельное покрытие grid33", "",
              "| Сравнение | Пар / исходно | Отсутствует | R точнее | MC точнее | R успех | MC успех | Oracle успех |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    groups = evidence["audit"].get("grid_groups", evidence["audit"].get("grid", {}).get("grid_groups", []))
    for group in groups:
        w, s = group["winners"], group["success_counts"]
        lines.append(f"| {group['name']} | {group['pairs_n']}/{group['universe_n']} | "
                     f"{group['missing_pairs_n']} | {w['remizov']} | {w['monte_carlo']} | "
                     f"{s['remizov']} | {s['monte_carlo']} | {s['oracle']} |")
    lines += ["", "Основная сеточная таблица использует массовый R128 taper и один "
              "сохранённый primary MC1024 seed из joint100. Это отдельная оценка с "
              "другим seed-бюджетом, чем основная медиана четырёх x0-ошибок; "
              "запуски MC4096 не подмешиваются. Все 100 основных grid33-пар относятся "
              "к train: отсутствует 600 train + 150 validation + 150 test пар. "
              "Неравное число исторически накопленных MC4096 seed в дополнительных "
              "таблицах сохранено и отражено в `mc_seed_count`."]
    test_misses = sorted((r for r in audit["prediction_rows"] if r["split"] == "test"
                          and r["best_method"] != r["selected_method"]),
                         key=lambda r:float(r["regret"]), reverse=True)
    if test_misses:
        lines += ["", "## Ошибки выбора на test", "",
                  "| Уравнение | E_R | E_MC | Выбор ML | Winner | Regret |",
                  "|---|---:|---:|---|---|---:|"]
        for r in test_misses:
            lines.append(f"| {r['equation_id']} | {fmt(float(r['E_R']))} | {fmt(float(r['E_MC']))} | "
                         f"{r['selected_method']} | {r['best_method']} | {fmt(float(r['regret']))} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def run(out):
    from catboost import CatBoostClassifier, Pool
    from ml.accuracy_evidence import build_evidence

    started = time.perf_counter()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "plan.json", dict(schema=SCHEMA, epsilon=EPSILON,
               catboost=CATBOOST_CONFIG, fit_count=1, tuning=False, class_balancing=False,
               numerical_calls=0, original_split=[700, 150, 150],
               primary="saved mass n128 taper versus mass MC1024 median4 at x0",
               n256_role="supplementary only; user revision 2026-10-31"))
    evidence = build_evidence()
    inputs = load_inputs(DATA)
    names = inputs["feature_names"]
    if len(names) != 253:
        raise ValueError("expected existing 253 input-only features")
    counts = Counter(map(str, inputs["splits"]))
    if counts != dict(train=700, validation=150, test=150):
        raise ValueError("original split changed")
    lookup = {str(pid):i for i, pid in enumerate(inputs["ids"])}
    rows = evidence["main_rows"]
    for row in rows:
        metadata = inputs["metadata"][row["equation_id"]]
        if any(row[key] != metadata[key] for key in ("expression_hash", "family_id", "split", "structure_signature")):
            raise ValueError("evidence/input identity mismatch")
        if row["remizov_config"]["n"] != 128:
            raise ValueError("n256 cannot enter primary model")
    if len({r["equation_id"] for r in rows}) != len(rows):
        raise ValueError("equations duplicated in ML dataset")
    train_rows = [r for r in rows if r["split"] == "train"]
    train_indices = [lookup[r["equation_id"]] for r in train_rows]
    train_labels = [r["best_method"] for r in train_rows]
    model = CatBoostClassifier(**CATBOOST_CONFIG)
    fit_started = time.perf_counter()
    model.fit(Pool(inputs["X"][train_indices], label=train_labels, feature_names=names))
    fit_seconds = time.perf_counter() - fit_started
    model_dir = out / "model"
    model_dir.mkdir()
    model.save_model(str(model_dir / "model.cbm"))
    context = dict(remizov=train_rows[0]["remizov_config"],
                   monte_carlo=dict(M=1024, dt=1/128, seeds=[202610250101,202610250102,202610250103,202610250104],
                                    estimator="median(abs(f_seed(0)-exact(0)))", tail_eps=1e-4),
                   metric="absolute_error_at_x0", domain="bounded solution on R")
    registry = dict(schema=SCHEMA, epsilon=EPSILON, methods=list(METHODS),
                    feature_version=VERSION, feature_names=names,
                    feature_code_sha256=sha(ROOT / "ml/remizov_features.py"),
                    input_grid_sha256=inputs["grid_sha256"], model_sha256=sha(model_dir / "model.cbm"),
                    catboost=CATBOOST_CONFIG, fit_count=1,
                    observed_training_classes=dict(Counter(train_labels)),
                    single_class_training=len(set(train_labels)) == 1,
                    train_equations=[r["equation_id"] for r in train_rows],
                    train_families=sorted({r["family_id"] for r in train_rows}),
                    train_topologies=sorted({r["structure_signature"] for r in train_rows}),
                    comparison_context=context, class_balancing=False,
                    validation_test_used_for_fit_or_tuning=False, numerical_calls=0)
    write_json(model_dir / "registry.json", registry)
    restored = AccuracyMethodSelector(model_dir)
    metrics = {}
    prediction_rows = []
    by_split = {}
    for split in ("train", "validation", "test"):
        own = [r for r in rows if r["split"] == split]
        indices = [lookup[r["equation_id"]] for r in own]
        values = inputs["X"][indices]
        predictions, probabilities = restored.predict_features(values, names)
        if len(own):
            original = model.predict_proba(Pool(values, feature_names=names), thread_count=1)
            np.testing.assert_array_equal(probabilities, original)
        metrics[split] = {strategy:selection_metrics(own, choice) for strategy, choice in (
            ("ML", predictions), ("always_remizov", [METHODS[0]] * len(own)),
            ("always_monte_carlo", [METHODS[1]] * len(own)),
            ("oracle", [r["best_method"] for r in own]))}
        by_split[split] = dict(original=counts[split], summary=summary(own))
        for row, prediction, probability in zip(own, predictions, probabilities):
            error = row["E_R"] if prediction == METHODS[0] else row["E_MC"]
            prediction_rows.append(dict(equation_id=row["equation_id"], expression_hash=row["expression_hash"],
                                        family_id=row["family_id"], split=split, best_method=row["best_method"],
                                        E_R=row["E_R"], E_MC=row["E_MC"], selected_method=prediction,
                                        E_selected=error, selected_success=error <= EPSILON,
                                        regret=error-min(row["E_R"], row["E_MC"]),
                                        p_remizov=float(probability[0]), p_monte_carlo=float(probability[1])))
    feature_audit = dict(n_features=len(names), feature_version=VERSION, preprocessing="none; input-only deterministic feature extraction",
                         forbidden_inputs_used=False, family_and_topology_split_verified=True,
                         original_counts=dict(counts), fit_rows=len(train_rows), feature_names=names)
    audit = dict(paired_summary=summary(rows), by_split=by_split, numerical_calls=0,
                 fit_count=1, catboost_fit_seconds=fit_seconds, feature_audit=feature_audit,
                 prediction_rows=prediction_rows)
    write_json(out / "metrics.json", metrics)
    write_json(out / "data_audit.json", {k:v for k,v in audit.items() if k != "prediction_rows"})
    write_json(out / "evidence_audit.json", evidence["audit"])
    write_json(out / "evidence.json", evidence)
    for file, key in (("comparable_x0", "main_rows"), ("coverage", "coverage_rows"),
                      ("mc_seeds", "mc_seed_rows"), ("supplementary_n256", "supplementary_rows"),
                      ("external_n256", "external_rows"), ("grid33", "grid_rows")):
        write_csv(out / (file + ".csv"), evidence.get(key, []))
    write_csv(out / "grid33_seeds.csv", evidence.get("grid_seed_rows", []))
    write_csv(out / "predictions.csv", prediction_rows)
    write_json(out / "confusion_matrix_test.json", dict(class_order=list(METHODS),
               rows="true winner", columns="ML choice", n=metrics["test"]["ML"]["n"],
               matrix=metrics["test"]["ML"]["confusion_matrix"]))
    np.savez_compressed(out / "input_features.npz", X=inputs["X"], feature_names=names,
                        equation_id=inputs["ids"], family_id=inputs["families"], split=inputs["splits"])
    with np.load(DATA / "all_methods_equations_v1_ml.npz", allow_pickle=False) as archive:
        i = train_indices[0]
        metadata = inputs["metadata"][str(inputs["ids"][i])]
        from ml.remizov_features import CERT_FIELDS
        meta = {k:metadata[k] for k in CERT_FIELDS}
        meta["serialized_ast"] = {k:metadata["serialized_ast"][k] for k in ("a_raw", "b_raw", "c_raw")}
        payload = dict(coefficients=archive["X"][i].tolist(), x_grid=archive["x_grid"].tolist(), coefficient_metadata=meta)
    write_json(out / "example_input.json", payload)
    write_json(out / "example_prediction.json", restored.predict(**payload))
    source_files = ["ml/accuracy_selector.py", "ml/accuracy_evidence.py", "ml/accuracy_grid_evidence.py",
                    "ml/remizov_features.py", "diagnostics/train_accuracy_selector.py",
                    "diagnostics/predict_accuracy_selector.py", "diagnostics/verify_accuracy_selector.py",
                    "tests/test_accuracy_selector.py", "tests/test_accuracy_grid_evidence.py", "requirements-ml.txt"]
    pins = dict(evidence["source_sha256"])
    for name in source_files:
        path = ROOT / name
        if path.exists():
            destination = out / "sources" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            pins[name] = sha(path)
    write_json(out / "source_sha256.json", pins)
    make_report(out, audit, metrics, registry, evidence)
    write_json(out / "environment.json", dict(python=platform.python_version(), platform=platform.platform(),
               versions={name:importlib.metadata.version(name) for name in ("numpy", "scipy", "catboost", "scikit-learn")},
               threads={k:v for k,v in os.environ.items() if k.endswith("NUM_THREADS") or k == "VECLIB_MAXIMUM_THREADS"},
               fit_count=1, numerical_calls=0, wall_seconds=time.perf_counter()-started))
    write_json(out / "artifact_sha256.json", {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps(dict(out=str(out), pairs=len(rows), observed_training_classes=registry["observed_training_classes"],
                         test=metrics["test"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
