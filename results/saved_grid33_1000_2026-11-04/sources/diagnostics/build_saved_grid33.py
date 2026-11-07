#!/usr/bin/env python3
"""Compare saved answers on 33 nodes. No solver, exact evaluator, or ML imports.

Run from the project root with the local .venv; --out must be a NEW directory.
MC has only x=0: E0 is a lower bound for an extension retaining that same value.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID33 = np.linspace(-1, 1, 33)
GRID129 = np.linspace(-1, 1, 129)
EPSILON = 0.05
DATASET = Path("data/all_methods_v1_1000_2026-10-18")
IVP = Path("results/four_method_accuracy_1000_2026-11-01")
R_DIR = Path("results/chernov_solve1000_2026-10-18/solutions")
MC_DIR = Path("results/mc1000_x0_2026-10-25/attempts")
R_CONFIG = dict(T=None, boundary="taper", dx=.0125, interp="cubic",
                max_work_grid_points=100000, n=128, n_jobs=1, n_steps=32768,
                num_t=513, padding=12., start_method="spawn", t_grading="quadratic",
                tail_eps=1e-8, taper_width=None)
IVP_CONFIG = dict(x0=-1., x1=1., n_steps=128, h=1 / 64, a_min=.001,
                  target_x=0., target_index=64, epsilon=.05)
MC_CONFIG = dict(M=1024, T=None, chunk=1024, dt=1 / 128, max_batch_paths=1024,
                 max_path_steps=100000000, max_runtime_seconds=60., max_steps=100000,
                 stream_id=0, tail_eps=1e-4)
FIELDS = ("equation_id", "family_id", "R_status", "R_available", "R_E33",
          "R_accuracy_met", "RK4_E33", "RK4_accuracy_met", "PI_E33", "PI_accuracy_met",
          "MC_mean4_0", "MC_E0_lower_bound", "MC_E33", "MC_accuracy_met",
          "MC_threshold_evidence", "margin_R", "margin_PI", "margin_MC_lower_bound",
          "best_available_method", "winner_scope", "inferred_unique_four_method_winner")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_vector(value, length, label):
    raw = np.asarray(value)
    require(raw.dtype.kind in "fiu", f"{label}: expected numeric values")
    array = raw.astype(float)
    require(array.shape == (length,) and np.isfinite(array).all(),
            f"{label}: expected {length} finite values")
    return array


def grid_error(values, exact):
    """Discrete maximum, with strict shape/finite checks (no interpolation)."""
    y = finite_vector(values, 33, "grid answer")
    ref = finite_vector(exact, 33, "grid exact")
    result = float(np.max(np.abs(y - ref)))
    require(np.isfinite(result), "nonfinite grid error")
    return result


def check_identity(record, meta, key="equation_id"):
    require(record.get(key) == meta["equation_id"], "record equation ID mismatch")
    require(record.get("expression_hash") == meta["expression_hash"], "expression hash mismatch")


def ivp_answer(record, meta, method):
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    require(hashlib.sha256(encoded).hexdigest() == record.get("record_sha256"),
            "IVP record checksum mismatch")
    check_identity(record, meta)
    require(record.get("family_id") == meta["family_id"] and
            record.get("structure_signature") == meta["structure_signature"], "IVP family mismatch")
    require(record.get("schema") == "fixed_ivp_mass_record_v1" and
            record.get("method") == method and record.get("config") == IVP_CONFIG,
            "IVP method/config mismatch")
    require(record.get("inputs") == {k: meta[k] for k in ("x0", "x1", "y0", "v0")},
            "IVP initial conditions mismatch")
    q = record.get("result") or {}
    require(record.get("usable") is True and record.get("status") == "success" and
            q.get("success") is True and q.get("failure_reason") == "", "unusable IVP record")
    require(q.get("equation_id") == meta["equation_id"] and q.get("n_steps") == 128 and
            q.get("h") == 1 / 64, "IVP result identity/config mismatch")
    require(q.get("x") == GRID129.tolist(), "IVP grid mismatch")
    y = finite_vector(q.get("y"), 129, "IVP y")
    dy = finite_vector(q.get("dy"), 129, "IVP dy")
    require(y[0] == meta["y0"] and dy[0] == meta["v0"], "IVP trajectory initial state mismatch")
    exact = finite_vector(record.get("exact_grid33"), 33, "IVP exact")
    error = grid_error(y[::4], exact)
    e0 = abs(float(y[64]) - float(exact[16]))
    require(record.get("E_inf_grid33") == error and record.get("E0") == e0 and
            record.get("exact0") == exact[16] and record.get("accuracy_met") == (e0 <= EPSILON),
            "saved IVP error mismatch")
    return exact, error


def classify_row(meta, remizov, rk4, pi, mc_records):
    """Validate one saved bundle and return measurements, bounds, and coverage."""
    exact, rk4_error = ivp_answer(rk4, meta, "rk4")
    pi_exact, pi_error = ivp_answer(pi, meta, "product_integral")
    require(np.array_equal(exact, pi_exact), "RK4/PI exact arrays differ")
    pid = meta["equation_id"]
    require(remizov.get("problem_id") == pid and remizov.get("config") == R_CONFIG,
            "Remizov identity/config mismatch")
    require(remizov.get("target_x") == GRID33.tolist(), "Remizov target grid mismatch")
    r_status = remizov.get("computation_status")
    r_error = None
    if r_status == "success":
        check_identity(remizov.get("metadata") or {}, meta)
        q = remizov.get("result") or {}
        require(q.get("success") is True and q.get("failure_reason") == "" and
                q.get("problem_id") == pid and q.get("n") == 128, "unusable Remizov result")
        require(q.get("x") == GRID33.tolist(), "Remizov result grid mismatch")
        require(np.array_equal(finite_vector(remizov.get("y_exact"), 33, "Remizov exact"), exact),
                "Remizov/IVP exact arrays differ")
        r_error = grid_error(q.get("f"), exact)
        require(r_error == remizov.get("max_abs_error"), "saved Remizov error mismatch")
    else:
        require(r_status == "worker_exception" and remizov.get("result") is None and
                remizov.get("max_abs_error") is None, "unexpected Remizov failure contract")
    require(len(mc_records) == 4, "expected four MC seeds")
    values = []
    for index, record in enumerate(mc_records):
        check_identity(record, meta, "problem_id")
        require(record.get("seed_index") == index and record.get("config") ==
                dict(MC_CONFIG, seed=202610250101 + index), "MC seed/config mismatch")
        q = record.get("result") or {}
        require(record.get("status") == "completed" and q.get("success") is True and
                q.get("failure_reason") == "" and q.get("problem_id") == pid, "unusable MC record")
        require(record.get("target_x") == q.get("x") == [0.] and
                q.get("completed_paths") == [1024] and record.get("y_exact") == exact[16],
                "MC grid/exact/paths mismatch")
        value = float(finite_vector(q.get("f"), 1, "MC answer")[0])
        require(abs(value - float(exact[16])) == record.get("abs_error"), "saved MC error mismatch")
        values.append(value)
    mean = float(np.mean(values, dtype=np.float64))
    bound = abs(mean - float(exact[16]))
    require(np.isfinite(mean) and np.isfinite(bound), "nonfinite MC aggregate")
    margins = dict(R=None if r_error is None else r_error - rk4_error,
                   PI=pi_error - rk4_error, MC_lower_bound=bound - rk4_error)
    errors = dict(R=r_error, RK4=rk4_error, PI=pi_error)
    available = {m: e for m, e in errors.items() if e is not None}
    best = "|".join(m for m, e in available.items() if e == min(available.values()))
    strict_four = r_error is not None and all(v > 0 for v in margins.values())
    return dict(equation_id=pid, family_id=meta["family_id"], R_status=r_status,
                R_available=r_error is not None, R_E33=r_error,
                R_accuracy_met=None if r_error is None else r_error <= EPSILON,
                RK4_E33=rk4_error, RK4_accuracy_met=rk4_error <= EPSILON,
                PI_E33=pi_error, PI_accuracy_met=pi_error <= EPSILON,
                MC_mean4_0=mean, MC_E0_lower_bound=bound, MC_E33=None, MC_accuracy_met=None,
                MC_threshold_evidence="fails_by_lower_bound" if bound > EPSILON else "unknown",
                margin_R=margins["R"], margin_PI=margins["PI"],
                margin_MC_lower_bound=margins["MC_lower_bound"], best_available_method=best,
                winner_scope="all_four_inferred" if strict_four else "available_only",
                inferred_unique_four_method_winner="RK4" if strict_four else None)


def statistics(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), accurate=int(np.count_nonzero(a <= EPSILON)),
                mean=float(a.mean()), median=float(np.median(a)),
                q95=float(np.quantile(a, .95)), maximum=float(a.max())) if len(a) else dict(
                    n=0, accurate=0, mean=None, median=None, q95=None, maximum=None)


def summarize(rows):
    common = [r for r in rows if r["R_available"]]
    missing = [r["equation_id"] for r in rows if not r["R_available"]]
    margins = {}
    for method in ("R", "PI", "MC_lower_bound"):
        candidates = [(r["margin_" + method], r["equation_id"]) for r in common]
        value, pid = min(candidates) if candidates else (None, None)
        margins[method] = dict(margin=value, equation_id=pid)
    return dict(schema="saved_grid33_comparison_v1", total_equations=len(rows),
                common_equations=len(common), missing_remizov_ids=missing,
                metric="max_k abs(saved_y(x_k)-saved_exact(x_k)); discrete, not continuous supremum",
                target_x=GRID33.tolist(), epsilon=EPSILON,
                full_available={m: statistics([r[m + "_E33"] for r in rows if r[m + "_E33"] is not None])
                                for m in ("R", "RK4", "PI")},
                common997={m: statistics([r[m + "_E33"] for r in common]) for m in ("R", "RK4", "PI")},
                strict_dominance=dict(counts_all_available={m: sum(r["margin_" + m] is not None and
                    r["margin_" + m] > 0 for r in rows) for m in ("R", "PI", "MC_lower_bound")},
                    minimum_margins_common997=margins,
                    inferred_unique_four_method_winners=sum(r["inferred_unique_four_method_winner"] == "RK4" for r in rows)),
                mc_threshold_evidence=dict(Counter(r["MC_threshold_evidence"] for r in rows)),
                MC_full_grid33_calculated=False,
                MC_bound_condition="Extend the same saved arithmetic mean4 answer at x=0; no claim for a new MC run",
                exact_source="saved arrays, identical across RK4/PI/R where available; no exact evaluation",
                saved_configs=dict(R=R_CONFIG, RK4_PI=IVP_CONFIG, MC_common=MC_CONFIG,
                                   MC_seeds=list(range(202610250101, 202610250105))),
                raw_records_checked=dict(R=len(rows), R_success=len(common), RK4=len(rows), PI=len(rows), MC=4 * len(rows)),
                new_solver_calls=0, new_fit_calls=0, new_model_inference_calls=0)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def report(summary):
    lines = ["# Сравнение сохранённых ответов на 33 узлах", "",
             "Метрика: `E33 = max |y_numeric(x_k) − y_exact(x_k)|`, "
             "`x_k = −1 + k/16`, k=0…32; точность означает `E33 ≤ 0.05`. "
             "Это дискретная ошибка, а не сертификат непрерывной нормы.", "",
             "Все значения и точные массивы прочитаны из старых JSON. "
             "Для RK4/PI взят каждый четвёртый из 129 сохранённых узлов; "
             "у Ремизова уже сохранены нужные 33 узла. Интерполяция не используется.", "",
             "Источники: `data/all_methods_v1_1000_2026-10-18/` (ID/метаданные), "
             "`results/chernov_solve1000_2026-10-18/solutions/` (R), "
             "`results/four_method_accuracy_1000_2026-11-01/records/` (RK4/PI), "
             "`results/mc1000_x0_2026-10-25/attempts/` (MC, четыре seed на задачу). "
             "Сохранённая R-серия использует `chernov_solver.ChernovSolver.solve`, "
             "а не новый запуск адаптивного `remizov_solver.py`.", "",
             "| Метод | Доступно из 1000 | E33 ≤ 0.05 | Среднее E33 | Медиана | Максимум |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, name in (("R", "Ремизов"), ("RK4", "RK4"), ("PI", "Продукт-интеграл")):
        s = summary["full_available"][key]
        lines.append(f"| {name} | {s['n']} | {s['accurate']} | {s['mean']:.12g} | {s['median']:.12g} | {s['maximum']:.12g} |")
    common = summary["common997"]
    lines.extend(["", f"На общем пересечении {summary['common_equations']} задач точных: "
                  f"Ремизов {common['R']['accurate']}, RK4 {common['RK4']['accurate']}, "
                  f"PI {common['PI']['accurate']}.", "",
                  "Проверены строгие неокруглённые разности ошибок. Минимальные запасы "
                  "на общем пересечении (ошибка конкурента минус E33 RK4):", ""])
    for method, value in summary["strict_dominance"]["minimum_margins_common997"].items():
        lines.append(f"- {method}: **{value['margin']:.17g}**, `{value['equation_id']}`.")
    counts = summary["strict_dominance"]["counts_all_available"]
    wins = summary["strict_dominance"]["inferred_unique_four_method_winners"]
    lines.extend(["", f"RK4 строго точнее Ремизова на {counts['R']} доступных задачах, "
                  f"PI на {counts['PI']}, нижней границы MC на {counts['MC_lower_bound']}. "
                  f"Единственный четырёхметодный победитель RK4 установлен на {wins} задачах.", "",
                  "Для MC сохранены четыре ответа только в x=0. Используется "
                  "`E_MC,0 = abs(mean(f_seed0,…,f_seed3) − exact(0))`, "
                  "а не среднее/медиана seed-ошибок. При дополнении того же ответа "
                  "на сетку `E_MC,33 ≥ E_MC,0`. Поэтому `E_RK4,33 < E_MC,0` "
                  "доказывает строгую победу без расчёта полного MC33. Это не вывод "
                  "для новой случайной реализации, математического ожидания MC или другого бюджета.", "",
                  "MC_E33 и MC_accuracy_met оставлены пустыми. Если нижняя граница >0.05, "
                  "порог заведомо нарушен; иначе достижение порога неизвестно. Счётчики: "
                  + json.dumps(summary["mc_threshold_evidence"], ensure_ascii=False) + ".", "",
                  "Отдельные отсутствующие ответы Ремизова (`worker_exception`):", ""])
    lines.extend("- `" + pid + "`." for pid in summary["missing_remizov_ids"])
    lines.extend(["", "Их R_E33/R_accuracy_met/margin_R и четырёхметодный winner пусты; "
                  "они остаются в общей таблице и в `missing_remizov.csv`. "
                  "Для них известна лишь победа RK4 над доступным PI и нижней границей MC. "
                  "Отсутствие результата не считается численной ошибкой или поражением Ремизова.", "",
                  "Сравнение относится к данному датасету и сохранённым конфигурациям. "
                  "RK4/PI используют точные начальные условия; R/MC приближают ограниченное "
                  "решение на всей прямой. У изготовленных задач целевая функция общая, "
                  "но условия и бюджеты методов различаются. Универсальное превосходство RK4 не установлено.", "",
                  "Воспроизведение из корня (новые каталоги):", "", "```bash",
                  "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/build_saved_grid33.py \\",
                  "  --out results/saved_grid33_reproduction",
                  "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/verify_saved_grid33.py \\",
                  "  --out results/saved_grid33_reproduction",
                  "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \\",
                  "  tests/test_saved_grid33.py tests/test_four_method_accuracy.py", "```", "",
                  "Проверки: builder проверяет ID/hash/конфигурации, статусы, конечность/формы, "
                  "вложенность сеток, точные НУ IVP, равенство сохранённых exact-массивов, "
                  "сохранённые ошибки и seal RK4/PI; `input_sha256.json` фиксирует прочитанные "
                  "файлы до и после. Независимая сверка CSV/summary — в `verification.json`; "
                  "в поставляемом комплекте результаты тестов и повторного построения "
                  "зафиксированы в `checks.json`.", "",
                  "Новых solver/MC/fit/inference-вызовов: 0. Генератор, алгоритмы, параметры, "
                  "ML-модели и исторические отчёты не изменены. Найденное приложение "
                  "`82dee293…/Pasted text.txt` содержит задание на аудит, а не готовый "
                  "аудиторский отчёт; его требования сверены с реальными кодом и JSON.", "",
                  "Остаётся вне этой задачи: полная сеточная ошибка MC (для независимого "
                  "измерения и сравнения без RK4), три отсутствующих ответа R, проверка "
                  "непрерывной нормы и новая проверка ML по сеточной цели. Прежние ML-метрики "
                  "относятся к E0 и на E33 не переносятся. На нынешних 997 четырёхметодных "
                  "метках только один класс RK4, поэтому содержательное обучение такого "
                  "селектора не требуется. Эти работы автоматически не запускаются.", ""])
    return "\n".join(lines)


def build(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    require(not out.exists(), "output directory already exists; choose a new --out")
    sources = {}

    def load_bytes(relative):
        path = root / relative
        data = path.read_bytes()
        sources[str(relative)] = hashlib.sha256(data).hexdigest()
        return data

    def load(relative):
        return json.loads(load_bytes(relative))

    metadata = [json.loads(line) for line in load_bytes(
        DATASET / "all_methods_equations_v1_metadata.jsonl").decode().splitlines()]
    ids = [m["equation_id"] for m in metadata]
    require(len(ids) == len(set(ids)) == 1000, "expected 1000 unique dataset IDs")
    plan = load(IVP / "plan.json")
    require(ids == plan["equation_ids"], "dataset/plan IDs differ")
    load_bytes(DATASET / "all_methods_equations_v1.csv")
    require(sources[str(DATASET / "all_methods_equations_v1.csv")] == plan["dataset_sha256"],
            "dataset checksum differs from IVP plan")
    rows = []
    for meta in metadata:
        pid = meta["equation_id"]
        r = load(R_DIR / (pid + ".json"))
        rk4, pi = [load(IVP / "records" / method / (pid + ".json"))
                   for method in ("rk4", "product_integral")]
        for record in (rk4, pi):
            require(record.get("plan_sha256") == sources[str(IVP / "plan.json")] and
                    record.get("dataset_sha256") == plan["dataset_sha256"], "IVP provenance mismatch")
        mc = [load(MC_DIR / (pid + f"_seed{i}.json")) for i in range(4)]
        rows.append(classify_row(meta, r, rk4, pi, mc))
    summary = summarize(rows)
    require(all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                for name, digest in sources.items()), "an input changed while reading")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "grid33_all1000.csv", rows)
    write_csv(out / "missing_remizov.csv", [r for r in rows if not r["R_available"]])
    write_json(out / "summary.json", summary)
    write_json(out / "input_sha256.json", dict(sorted(sources.items())))
    write_json(out / "reproduce.json", dict(
        python=sys.version, numpy=np.__version__, platform=platform.platform(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_unchanged=True, input_files=len(sources),
        command="PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/build_saved_grid33.py --out <NEW_DIRECTORY>"))
    with (out / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write(report(summary))
    print(json.dumps(dict(output=str(out), equations=len(rows), common=summary["common_equations"],
                          accurate={m: v["accurate"] for m, v in summary["full_available"].items()},
                          strict_four_winners=summary["strict_dominance"]["inferred_unique_four_method_winners"],
                          new_solver_calls=0), ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.out)


if __name__ == "__main__":
    main()
