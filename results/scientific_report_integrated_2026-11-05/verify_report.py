"""Verify this saved-results report without importing any project solver or ML.

Run with Python 3.10+ from any directory.  Only report_verification.json is
written.  The builder is replayed in an isolated temporary copy, never in the
delivered report directory.  --self-test exercises rejection of corrupt input.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, name):
    require(math.isfinite(actual) and math.isfinite(expected), name + ": nonfinite")
    require(math.isclose(actual, expected, rel_tol=2e-13, abs_tol=1e-18),
            f"{name}: {actual!r} != {expected!r}")


def uncomment(tex):
    # This report has no verbatim environments.  An escaped percent remains text.
    lines = []
    for line in tex.splitlines(keepends=True):
        for position, char in enumerate(line):
            if char == "%" and not escaped(line, position):
                line = line[:position] + ("\n" if line.endswith("\n") else "")
                break
        lines.append(line)
    return "".join(lines)


def escaped(text, position):
    backslashes = 0
    while position and text[position-1] == "\\":
        position -= 1
        backslashes += 1
    return bool(backslashes % 2)


def latex_checks(tex):
    text = uncomment(tex)
    require(not re.search(r"\\n(?![A-Za-z])", text), "literal \\n artifact")
    depth = 0
    for token in re.finditer(r"[{}]", text):
        if escaped(text, token.start()):
            continue
        depth += 1 if token.group() == "{" else -1
        require(depth >= 0, "closing brace without opening brace")
    require(depth == 0, "unclosed braces")
    stack = []
    for token in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        operation, environment = token.groups()
        if operation == "begin":
            stack.append(environment)
        else:
            require(stack and stack.pop() == environment,
                    "environment nesting mismatch: " + environment)
    require(not stack, "unclosed environment")
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    references = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text)
    bibliography = re.findall(r"\\bibitem\{([^}]+)\}", text)
    citations = [key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text)
                 for key in group.split(",")]
    require(len(labels) == len(set(labels)), "duplicate label")
    require(len(bibliography) == len(set(bibliography)), "duplicate bibliography key")
    require(set(references) <= set(labels), "unresolved reference")
    require(set(citations) <= set(bibliography), "unresolved citation")
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)
    return dict(labels=len(labels), references=len(references),
                citations=len(citations), bibliography_items=len(bibliography),
                figure_references=figures, unused_bibliography=sorted(set(bibliography)-set(citations)))


def block(text, start, end=None):
    require(text.count(start) == 1, "nonunique block start: " + start)
    first = text.index(start)
    if end is None:
        return text[first:]
    require(text.count(end) == 1, "nonunique block end: " + end)
    return text[first:text.index(end, first)]


def quantile95(values):
    # Linear empirical quantile: explicit order-statistic definition, no NumPy.
    values = sorted(values)
    position = .95 * (len(values)-1)
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (position-lo) * (values[hi]-values[lo])


def describe(values):
    require(bool(values), "empty distribution")
    require(all(math.isfinite(x) and x >= 0 for x in values), "invalid error")
    return dict(n=len(values), accurate=sum(x <= .05 for x in values),
                mean=math.fsum(values)/len(values), median=statistics.median(values),
                q95=quantile95(values), maximum=max(values))


def printed_number(token):
    """Return displayed numeric value and half a last-place rounding unit."""
    token = token.strip().strip("$")
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\\cdot10\^\{([+-]?\d+)\})?", token)
    require(match is not None, "unsupported numeric TeX cell: " + token)
    mantissa, exponent = match.groups()
    exponent = int(exponent or 0)
    places = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return float(mantissa)*10**exponent, .5*10**(exponent-places)


def compare_printed(token, actual, name):
    displayed, half_unit = printed_number(token)
    require(abs(displayed-actual) <= half_unit + abs(actual)*2e-15,
            f"{name}: displayed {token} does not round {actual!r}")


def table_rows(tex, label):
    location = tex.index("\\label{" + label + "}")
    start = tex.rfind(r"\begin{table}", 0, location)
    end = tex.index(r"\end{table}", location)
    table = tex[start:end]
    require(table.count(r"\midrule") == 1, "unexpected table body: " + label)
    body = table.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    return [[cell.strip() for cell in row.strip().split("&")]
            for row in body.split(r"\\") if "&" in row]


def verify_sources(directory):
    manifest = json.loads((directory/"source_manifest.json").read_text())
    require(len(manifest) == 62, "source manifest must contain 62 copied artifacts")
    for relative, record in manifest.items():
        require(sha256(directory/relative) == record["sha256"], "copied source hash: " + relative)
    roots = [parent for parent in directory.parents
             if (parent/manifest["base_report.tex"]["source"]).is_file()]
    if roots:
        root = roots[0]
        for record in manifest.values():
            require(sha256(root/record["source"]) == record["sha256"],
                    "historical source hash: " + record["source"])
        originals = len(manifest)
    else:
        originals = "unavailable in this standalone copy; copied hashes checked"
    return dict(copied_artifacts=len(manifest), historical_originals=originals)


def verify_preservation(base, report):
    stages = [
        ("methods_entire", r"\subsection{Ход работы}", r"\subsection{Результаты}"),
        ("E0_results_entire", r"\subsection{Результаты}",
         r"\subsubsection{Отдельная проверка максимальной ошибки на 33 узлах}\label{sec:grid33}"),
        ("ML3_results_entire", r"\subsubsection{Выбор из трёх методов}\label{sec:ml3-res}",
         r"\section{Заключение}"),
        ("appendix_to_EOF", r"\section*{Приложение. Признаки модели выбора метода}", None),
    ]
    result = {}
    citation_line = "Сохранённые метрики и их проверка приведены в источнике~\\cite{savedthreeML}.\n"
    for name, start, end in stages:
        old, new = block(base, start, end), block(report, start, end)
        if name == "ML3_results_entire":
            require(new.count(citation_line) == 1, "ML3 must have exactly the permitted citation line")
            new = new.replace(citation_line, "")
        require(old == new, "unnecessary change to protected base block: " + name)
        result[name] = dict(identical=True, sha256=hashlib.sha256(old.encode()).hexdigest(), characters=len(old))
    return result


def verify_grid33(directory, report):
    rows = read_csv(directory/"evidence/grid33/grid33_all1000.csv")
    comparison = read_csv(directory/"evidence/grid33_figures/comparison.csv")
    summary = json.loads((directory/"evidence/grid33/summary.json").read_text())
    require(len(rows) == len({r["equation_id"] for r in rows}) == 1000, "1000 unique grid rows")
    require(all(not r["MC_E33"] and not r["MC_accuracy_met"] for r in rows), "MC grid error must remain unavailable")
    common = [r for r in rows if r["R_E33"]]
    missing = [r for r in rows if not r["R_E33"]]
    require(len(common) == 997 and len(missing) == 3, "Remizov coverage")
    require(sorted(r["equation_id"] for r in missing) == sorted(summary["missing_remizov_ids"]), "missing IDs")
    for row in rows:
        for method in ("R", "RK4", "PI"):
            if row[method+"_E33"]:
                correct = float(row[method+"_E33"]) <= .05
                require(row[method+"_accuracy_met"] == str(correct), "accuracy flag: " + row["equation_id"])
            else:
                require(method == "R" and not row[method+"_accuracy_met"], "missing value mislabelled")
        require(row["R_available"] == str(bool(row["R_E33"])), "R availability flag")
    computed = {}
    statistic_cells = 0
    for scope, subset in (("all_available", rows), ("common997", common)):
        for method in ("R", "RK4", "PI", "MC"):
            field = "MC_E0_lower_bound" if method == "MC" else method+"_E33"
            values = [float(r[field]) for r in subset if r[field]]
            stats = describe(values)
            computed[scope, method] = stats
            matches = [r for r in comparison if r["scope"] == scope and r["method"] == method]
            require(len(matches) == 1, "comparison row uniqueness")
            saved = matches[0]
            require(int(saved["n_values"]) == len(values) and int(saved["n_scope"]) == len(subset), "comparison denominators")
            for key in ("mean", "median", "q95", "maximum"):
                close(stats[key], float(saved[key]), scope+"/"+method+"/"+key)
                statistic_cells += 1
            if method != "MC":
                require(saved["metric"] == "E33", "E33 metric name")
                require(int(saved["E33_pass"]) == stats["accurate"], "comparison pass count")
                require(int(saved["E33_fail"]) == len(values)-stats["accurate"], "comparison fail count")
                named = summary["full_available" if scope == "all_available" else scope][method]
                for key in stats:
                    close(float(stats[key]), float(named[key]), "summary/"+scope+"/"+method+"/"+key)
                    statistic_cells += 1
            else:
                require(saved["metric"] == "E0_lower_bound", "MC is a lower bound")
                failed = sum(x > .05 for x in values)
                require(int(saved["E33_fail_lower_bound"]) == failed == 62, "MC certified failures")
                require(int(saved["E33_unknown"]) == len(values)-failed, "MC unknown count")
                require(not saved["E33_pass"] and not saved["E33_fail"], "MC must not claim full-grid outcomes")
    require(len(comparison) == 8, "eight scope/method comparison rows")
    require(summary["mc_threshold_evidence"] == {"unknown": 938, "fails_by_lower_bound": 62}, "MC evidence summary")
    for method, field in (("R", "R_E33"), ("PI", "PI_E33"), ("MC_lower_bound", "MC_E0_lower_bound")):
        available = [r for r in rows if r[field]]
        require(all(float(r[field]) > float(r["RK4_E33"]) for r in available), "strict margin: " + method)
        expected_count = summary["strict_dominance"]["counts_all_available"][method]
        require(len(available) == expected_count, "strict dominance count")
        minimum = min(common, key=lambda r: float(r[field])-float(r["RK4_E33"]))
        saved = summary["strict_dominance"]["minimum_margins_common997"][method]
        close(float(minimum[field])-float(minimum["RK4_E33"]), saved["margin"], "minimum margin")
        require(minimum["equation_id"] == saved["equation_id"], "minimum margin ID")

    # Parse rendered table cells; verify rounding intervals rather than calling
    # the builder's formatting function or reproducing its table strings.
    full_rows = table_rows(report, "tab:grid33-full")
    require(len(full_rows) == 4, "full comparison has four rows")
    displayed_cells = 0
    for cells, method in zip(full_rows, ("R", "RK4", "PI", "MC")):
        require(len(cells) == 6, "full comparison width")
        expected_label = {"R": "Ремизов,", "RK4": "RK4,", "PI": "PI,", "MC": "MC,"}[method]
        require(cells[0].startswith(expected_label), "full comparison method label")
        stats = computed["all_available", method]
        require(int(cells[1]) == stats["n"], "printed full count")
        for cell, key in zip(cells[2:5], ("mean", "median", "maximum")):
            compare_printed(cell, stats[key], "printed full/"+method+"/"+key)
            displayed_cells += 1
        if method == "MC":
            require(re.search(r"\\ge\s*62", cells[5]) and "938" in cells[5], "printed MC unknown outcomes")
        else:
            require(cells[5] == f'{stats["accurate"]}/{stats["n"]}', "printed full pass count")
    common_rows = table_rows(report, "tab:grid33-common")
    require(len(common_rows) == 4, "common comparison has four rows")
    for cells, method in zip(common_rows[:3], ("R", "RK4", "PI")):
        expected_label = {"R": "Ремизова", "RK4": "RK4", "PI": "Мультипликативный интеграл"}[method]
        require(cells[0] == expected_label, "common comparison method label")
        stats = computed["common997", method]
        for cell, key in zip(cells[1:5], ("mean", "median", "q95", "maximum")):
            compare_printed(cell, stats[key], "printed common/"+method+"/"+key)
            displayed_cells += 1
        require(cells[5] == f'{stats["accurate"]}/{stats["n"]}', "printed common pass count")
    require("полная сеточная ошибка не вычислена" in " ".join(common_rows[3]), "printed MC grid limitation")
    return dict(unique_equations=len(rows), common_equations=len(common),
                missing_remizov=[r["equation_id"] for r in missing],
                independent_summary_cells=statistic_cells, printed_numeric_cells=displayed_cells,
                MC_certified_failures=62, MC_unknown=938,
                strict_four_method_winners=len(common))


def verify_rebuild(directory):
    targets = ("report.tex", "report_changes.diff", "editorial_changes.json")
    frozen = {name: sha256(directory/name) for name in targets}
    with tempfile.TemporaryDirectory(prefix="report2488-replay-") as temporary:
        replay = Path(temporary)
        for name in ("build_report.py", "base_report.tex"):
            shutil.copy2(directory/name, replay/name)
        for name in ("evidence", "fragments"):
            shutil.copytree(directory/name, replay/name)
        call = subprocess.run([sys.executable, "-B", str(replay/"build_report.py")],
                              cwd=replay, check=True, text=True, capture_output=True)
        for name in targets:
            require(sha256(replay/name) == frozen[name], "builder replay differs: " + name)
    require(all(sha256(directory/name) == frozen[name] for name in targets), "delivered outputs were changed during verification")
    summary = json.loads(call.stdout)
    summary["target"] = "<isolated temporary directory>/report.tex"
    return dict(byte_identical_outputs=list(targets), isolated_temporary_directory=True,
                builder_summary=summary)


def self_checks():
    """Mutation tests: reject errors a report consumer needs detected."""
    good = (r"\begin{document}\label{ok}\ref{ok}\cite{evidence}"
            r"\begin{table}\end{table}\bibitem{evidence}x\end{document}")
    latex_checks(good)
    latex_checks(good.replace("x", r"\paragraph{Text\\}\{escaped set\} 50\%"))
    corruptions = {
        "missing_reference": good.replace(r"\ref{ok}", r"\ref{absent}"),
        "duplicate_label": good.replace(r"\label{ok}", r"\label{ok}\label{ok}"),
        "missing_citation": good.replace(r"\cite{evidence}", r"\cite{absent}"),
        "crossed_environments": good.replace(r"\end{table}", r"\end{figure}"),
        "unclosed_brace": good + "{",
        "literal_newline_escape": good + r"\n\caption{x}",
    }
    results = {}
    for name, bad in corruptions.items():
        try:
            latex_checks(bad)
        except ValueError:
            results[name] = "rejected as expected"
        else:
            raise ValueError("self-check accepted corruption: " + name)
    compare_printed(r"$1.205\cdot10^{-6}$", 1.2049420163392674e-6, "correct rounding")
    try:
        compare_printed(r"$1.305\cdot10^{-6}$", 1.2049420163392674e-6, "changed table statistic")
    except ValueError:
        results["changed_table_statistic"] = "rejected as expected"
    else:
        raise ValueError("self-check accepted changed table statistic")
    require(quantile95([0., 10.]) == 9.5, "linear quantile test")
    results["linear_quantile_order_statistic"] = "passed"
    results["escaped_braces_percent_and_linebreak"] = "passed"
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run only mutation/self checks; write no files")
    args = parser.parse_args()
    tests = self_checks()
    if args.self_test:
        print(json.dumps(dict(status="passed", self_checks=tests), ensure_ascii=False, indent=2))
        return 0
    checks = {}
    result = dict(schema="integrated_report_saved_only_verification_v1", status="failed",
                  checks=checks, new_solver_calls=0, new_MC_calls=0,
                  new_ML_fits=0, new_model_inference_calls=0)
    try:
        checks["mutation_self_checks"] = tests
        checks["62_source_hashes"] = verify_sources(HERE)
        base, report = (HERE/"base_report.tex").read_text(), (HERE/"report.tex").read_text()
        checks["latex_static_structure"] = latex_checks(report)
        figures = checks["latex_static_structure"]["figure_references"]
        require(len(figures) == len(set(figures)) == 17, "17 unique report figures")
        for relative in figures:
            require((HERE/"figures"/relative).is_file(), "missing figure: " + relative)
        checks["17_referenced_figures"] = dict(present=17, ready_E33_PDFs=sum(x.startswith("grid33_") and x.endswith(".pdf") for x in figures))
        require(checks["17_referenced_figures"]["ready_E33_PDFs"] == 4, "four ready E33 PDFs")
        checks["protected_base_blocks"] = verify_preservation(base, report)
        checks["grid33_independent_CSV_and_TeX"] = verify_grid33(HERE, report)
        checks["isolated_builder_replay"] = verify_rebuild(HERE)
        result["status"] = "passed"
        result["report_sha256"] = sha256(HERE/"report.tex")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    (HERE/"report_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
