"""Publication figures from saved solve() outputs; never run an ODE solver.

Coefficients and the dense exact curve are evaluated by the existing safe AST
loader. Numerical values and errors use only the 33 saved target nodes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl_remizov_examples_20261020")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from all_methods_dataset_loader import load_all_methods_equations

IDS = ("AMV1_F0000_S00", "AMV1_F0141_S00")
BLUE, RED, GREEN = "#17648a", "#bd4933", "#398255"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_csv(path: Path, header, columns) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(zip(*columns))


def new_axes(pid: str, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    ax.set(title=f"{pid}: {title}", xlabel="$x$", ylabel=ylabel, xlim=(-1, 1))
    ax.set_xticks(np.linspace(-1, 1, 9))
    ax.grid(True, alpha=.22, linewidth=.6)
    ax.set_axisbelow(True)
    return fig, ax


def save_figure(fig, out: Path, stem: str) -> None:
    for extension in ("pdf", "png"):
        path = out / f"{stem}.{extension}"
        if path.exists():
            raise FileExistsError(path)
        metadata = {"Creator": "diagnostics/plot_remizov_examples.py"}
        fig.savefig(path, dpi=300, metadata=metadata)
    plt.close(fig)


def make_latex(out: Path, items) -> None:
    lines = [
        "% Add in the preamble: \\usepackage{graphicx}",
        "% Russian captions require an existing Russian-language setup.",
        "% Upload the seven PDFs to figures/remizov/ in Overleaf.",
        "% Insert this fragment with \\input{figures/remizov/figures.tex}.",
        "",
    ]
    for stem, caption, label in items:
        lines += [
            r"\begin{figure}[htbp]", r"  \centering",
            rf"  \includegraphics[width=0.92\linewidth]{{figures/remizov/{stem}.pdf}}",
            f"  \\caption{{{caption}}}", f"  \\label{{fig:{label}}}",
            r"\end{figure}", "",
        ]
    with (out / "figures.tex").open("x") as stream:
        stream.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/all_methods_v1_1000_2026-10-18")
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results/chernov_solve1000_2026-10-18")
    parser.add_argument("--dense-points", type=int, default=2001)
    args = parser.parse_args()
    if args.dense_points < 33:
        parser.error("--dense-points must be at least 33")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    csv_path = args.data_dir / "all_methods_equations_v1.csv"
    meta_path = args.data_dir / "all_methods_equations_v1_metadata.jsonl"
    plan_path = args.results_dir / "plan.json"
    solution_paths = {pid: args.results_dir / "solutions" / f"{pid}.json" for pid in IDS}
    protected = [*args.data_dir.glob("all_methods_equations_v1*"),
                 *solution_paths.values(), plan_path, ROOT / "chernov_solver.py",
                 ROOT / "remizov_solver.py", ROOT / "all_methods_dataset_loader.py",
                 ROOT / "src/all_methods_expressions.py"]
    before = {str(p.resolve()): sha(p) for p in protected if p.is_file()}
    write_json(out / "inputs_before_sha256.json", before)
    shutil.copy2(ROOT / "PROJECT_STATUS.md", out / "PROJECT_STATUS.before.md")
    equations = {e.equation_id: e for e in
                 load_all_methods_equations(csv_path, meta_path)}
    plan = json.loads(plan_path.read_text())
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
        "savefig.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False,
    })
    summaries, latex_items = [], []
    dense_x = np.linspace(-1, 1, args.dense_points)
    for pid in IDS:
        record = json.loads(solution_paths[pid].read_text())
        equation = equations[pid]
        result = record["result"]
        if not (record["implementation"] == "chernov_solver.ChernovSolver.solve"
                and result["scheme"] == "solve" and result["success"]
                and record["problem_id"] == equation.equation_id
                and record["config"] == plan["config"]
                and record["metadata"]["expression_hash"] == equation.metadata["expression_hash"]):
            raise ValueError(f"Source identity mismatch for {pid}")
        x = np.asarray(result["x"], dtype=float)
        numeric = np.asarray(result["f"], dtype=float)
        exact = np.asarray(record["y_exact"], dtype=float)
        if not (x.shape == numeric.shape == exact.shape == (33,)
                and np.array_equal(x, np.linspace(-1, 1, 33))
                and np.all(np.isfinite(numeric)) and np.all(np.isfinite(exact))):
            raise ValueError(f"Unexpected saved arrays for {pid}")
        np.testing.assert_allclose(equation.y_exact(x), exact, rtol=1e-13, atol=1e-14)
        error = np.abs(numeric - exact)
        peak = int(np.argmax(error))
        if float(error[peak]) != record["max_abs_error"]:
            raise ValueError(f"Saved error mismatch for {pid}")
        problem = equation.remizov_problem
        a = np.asarray(problem.a(dense_x))
        b = np.asarray(problem.b(dense_x))
        c = np.asarray(problem.c_remizov(dense_x)) - problem.lambda_value
        exact_dense = np.asarray(equation.y_exact(dense_x))
        if not all(z.shape == dense_x.shape and np.all(np.isfinite(z))
                   for z in (a, b, c, exact_dense)):
            raise ValueError(f"Invalid dense evaluation for {pid}")
        np.testing.assert_allclose(
            np.asarray([equation.cauchy_problem.c(float(t)) for t in x]),
            np.asarray(problem.c_remizov(x)) - problem.lambda_value,
            rtol=1e-13, atol=1e-14)
        write_csv(out / f"{pid}_coefficients_exact.csv",
                  ["x", "a", "b", "c", "y_exact"],
                  [dense_x, a, b, c, exact_dense])
        write_csv(out / f"{pid}_saved_samples.csv",
                  ["x", "y_exact", "y_num", "abs_error"],
                  [x, exact, numeric, error])
        latex_id = pid.replace("_", r"\_")

        fig, ax = new_axes(pid, "Коэффициенты", "Значение коэффициента")
        for values, color, style, label in (
                (a, BLUE, "-", "$a(x)$"), (b, GREEN, "--", "$b(x)$"),
                (c, RED, "-.", "$c(x)=c_R(x)-\\lambda$")):
            ax.plot(dense_x, values, color=color, ls=style, lw=1.8, label=label)
        ax.legend(loc="best")
        save_figure(fig, out, f"{pid}_coefficients")
        latex_items.append((f"{pid}_coefficients",
            f"Коэффициенты исходного уравнения {latex_id} на $[-1,1]$.",
            f"{pid.lower()}-coefficients"))

        fig, ax = new_axes(pid, "Точное и численное решения", "$y(x)$")
        ax.plot(dense_x, exact_dense, color=BLUE, lw=2, label="$y_{\\mathrm{exact}}(x)$")
        ax.plot(x, numeric, color=RED, lw=1.0, ls="--", marker="o", ms=4,
                markerfacecolor="white", markeredgewidth=1.1,
                label="$y_{\\mathrm{num}}(x)$: 33 сохранённых узла")
        ax.legend(loc="upper right")
        if pid == "AMV1_F0141_S00":
            ax.margins(y=.15)
            inset = ax.inset_axes([.30, .10, .55, .31])
            inset.plot(dense_x, exact_dense, color=BLUE, lw=1.4)
            inset.set(xlim=(-1, 1), xlabel="$x$", ylabel="$y_{\\mathrm{exact}}$")
            inset.set_title("Точное решение: отдельный масштаб", fontsize=8)
            inset.tick_params(labelsize=7)
            inset.xaxis.label.set_size(8)
            inset.yaxis.label.set_size(8)
            inset.grid(alpha=.2)
        save_figure(fig, out, f"{pid}_solutions")
        caption = (f"Точное и сохранённое численное решения {latex_id}. "
                   "Маркеры обозначают 33 расчётных узла; штрихи соединяют соседние узлы.")
        if pid == "AMV1_F0141_S00":
            caption += " На вставке точное решение показано в собственном масштабе."
        latex_items.append((f"{pid}_solutions", caption, f"{pid.lower()}-solutions"))

        for logarithmic in (False, True) if pid == "AMV1_F0141_S00" else (False,):
            suffix = "absolute_error_log" if logarithmic else "absolute_error"
            title = "Абсолютная ошибка: логарифмическая шкала" if logarithmic else "Абсолютная ошибка"
            fig, ax = new_axes(pid, title, "$|y_{\\mathrm{num}}-y_{\\mathrm{exact}}|$")
            shown = np.where(error > 0, error, np.nan) if logarithmic else error
            ax.plot(x, shown, color=RED, lw=1.1, marker="o", ms=4,
                    label="Ошибка в 33 сохранённых узлах")
            threshold_outside = not logarithmic and float(error[peak]) < .001
            if threshold_outside:
                # Keep the small error's structure visible; do not stretch the
                # ordinate to a threshold more than an order of magnitude away.
                ax.set_ylim(0, float(error[peak]) * 1.35)
                ax.text(.025, .95, "Порог 0.01 выше показанного масштаба",
                        transform=ax.transAxes, va="top", fontsize=8, color=GREEN)
            else:
                ax.axhline(.01, color=GREEN, ls="--", lw=1.2, label="Порог 0.01")
            if logarithmic:
                ax.set_yscale("log")
                ax.grid(True, which="minor", alpha=.12, linewidth=.5)
            else:
                ax.set_ylim(bottom=0)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
            ax.legend(loc="upper right")
            ax.annotate(f"$E_\\infty={error[peak]:.6g}$\n$x={x[peak]:g}$",
                        xy=(x[peak], error[peak]), xycoords="data",
                        xytext=(.38, .72), textcoords="axes fraction", fontsize=9,
                        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": .8},
                        bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#cccccc"})
            save_figure(fig, out, f"{pid}_{suffix}")
            latex_items.append((f"{pid}_{suffix}",
                f"Абсолютная ошибка {latex_id} в 33 сохранённых узлах"
                + ("; логарифмическая шкала по вертикали." if logarithmic else ".")
                + (" Все ошибки ниже $0.01$; порог вне показанного масштаба."
                   if threshold_outside else " Пунктир обозначает порог $0.01$."),
                f"{pid.lower()}-{suffix.replace('_', '-')}"))
        summaries.append({
            "equation_id": pid, "status": record["status"], "config": record["config"],
            "expression_hash": equation.metadata["expression_hash"],
            "saved_samples": int(x.size), "dense_analytic_samples": int(dense_x.size),
            "max_abs_error": float(error[peak]), "max_error_x": float(x[peak]),
            "exact_at_max_error": float(exact[peak]), "numeric_at_max_error": float(numeric[peak]),
            "zero_errors_not_shown_on_log_scale": int(np.count_nonzero(error == 0)),
            "exact_matches_loader": True, "original_c_matches_loader": True,
        })
    make_latex(out, latex_items)
    readme = """# Рисунки примеров метода Ремизова

Созданы только из production-датасета и сохранённых ответов ChernovSolver.solve.
Новых решений ОДУ нет. Для каждого примера: коэффициенты, решения и абсолютная
ошибка. Для F0141_S00 дополнительно дана ошибка с логарифмической осью Y;
на графике решений есть вставка точного решения в собственном масштабе.

- PDF: векторные рисунки со встроенным шрифтом; основной формат для Overleaf.
- PNG: 300 dpi, 2160 x 1380 пикселей.
- Коэффициенты a,b,c и точная кривая: безопасный AST-загрузчик, 2001 точка.
  Показан коэффициент c=c_R-lambda исходного ОДУ, а не c_R оператора Чернова.
- Численное решение и абсолютная ошибка: только 33 сохранённых узла [-1,1].
  Соединяющие отрезки нужны для чтения графика; новых значений решателя нет.
  E_inf является максимумом на этих 33 точках, не непрерывной оценкой ошибки.
- CSV рядом содержат все данные рисунков. manifest.json фиксирует конфигурации,
  источники, SHA-256 до/после и проверки совпадения с загрузчиком.

## Overleaf

Загрузите семь PDF и figures.tex в каталог figures/remizov проекта Overleaf.
В преамбуле нужен \\usepackage{graphicx}; подписи требуют настройки русского языка.
Вставка всех рисунков: \\input{figures/remizov/figures.tex}.
Можно выбрать отдельные figure из этого файла.

## Повторное построение

Из корня репозитория (выберите новый, ещё не существующий output-dir):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/plot_remizov_examples.py \\
  --output-dir results/remizov_example_figures_2026-10-20_repeat
```

Скрипт не перезаписывает существующий выходной каталог и не вызывает решатели.
"""
    with (out / "README.md").open("x") as stream:
        stream.write(readme.replace("2001 точка", f"{args.dense_points} точка"))
    after = {p: sha(Path(p)) for p in before}
    if before != after:
        raise RuntimeError("An input file changed while plotting")
    write_json(out / "manifest.json", {
        "scope": "saved results and analytic AST evaluation only", "solver_calls": 0,
        "python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "script_sha256": sha(Path(__file__)), "source_sha256_before": before,
        "source_sha256_after": after, "sources_unchanged": True,
        "figures": [q[0] for q in latex_items], "pdf_count": len(latex_items),
        "png_dpi": 300, "numerical_interpolation": "none; straight segments between saved samples",
        "examples": summaries,
    })
    print(json.dumps({"output_dir": str(out), "pdf_count": len(latex_items),
                      "png_count": len(latex_items), "solver_calls": 0,
                      "examples": [{k: q[k] for k in ("equation_id", "max_abs_error", "max_error_x")}
                                   for q in summaries]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
