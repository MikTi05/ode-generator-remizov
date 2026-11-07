"""Independently verify a saved-only grid33 release, without importing its builder.

Only JSON/CSV files are read.  The arithmetic uses Python's scalar ``abs`` and
``max``; no ODE solver, exact-function evaluator, feature code or ML is loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "equation_id", "family_id", "R_status", "R_available", "R_E33",
    "R_accuracy_met", "RK4_E33", "RK4_accuracy_met", "PI_E33",
    "PI_accuracy_met", "MC_mean4_0", "MC_E0_lower_bound", "MC_E33",
    "MC_accuracy_met", "MC_threshold_evidence", "margin_R", "margin_PI",
    "margin_MC_lower_bound", "best_available_method", "winner_scope",
    "inferred_unique_four_method_winner",
)
GRID33 = [-1 + i / 16 for i in range(33)]
GRID129 = [-1 + i / 64 for i in range(129)]
R_CONFIG = dict(T=None, boundary="taper", dx=.0125, interp="cubic",
                max_work_grid_points=100000, n=128, n_jobs=1, n_steps=32768,
                num_t=513, padding=12.0, start_method="spawn",
                t_grading="quadratic", tail_eps=1e-8, taper_width=None)
IVP_CONFIG = dict(x0=-1., x1=1., n_steps=128, h=1 / 64, a_min=.001,
                  target_x=0., target_index=64, epsilon=.05)
MC_CONFIG = dict(M=1024, T=None, chunk=1024, dt=1 / 128, max_batch_paths=1024,
                 max_path_steps=100000000, max_runtime_seconds=60.,
                 max_steps=100000, stream_id=0, tail_eps=.0001)
MISSING_R = ["AMV1_F0134_S03", "AMV1_F0136_S01", "AMV1_F0139_S00"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_vector(values, size, label):
    require(isinstance(values, list) and len(values) == size, label + ": shape")
    require(all(type(v) in (float, int) and math.isfinite(v) for v in values),
            label + ": nonfinite/non-numeric")
    return values


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def stats(values):
    ordered = sorted(values)
    position = .95 * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    q95 = ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return dict(n=len(values), accurate=sum(v <= .05 for v in values),
                mean=statistics.mean(values), median=statistics.median(values),
                q95=q95, maximum=max(values))


def check_stats(observed, expected, label):
    for key, value in expected.items():
        if key in ("n", "accurate"):
            require(observed[key] == value, label + ": " + key)
        else:
            require(math.isclose(observed[key], value, rel_tol=2e-13, abs_tol=1e-15),
                    label + ": " + key)


def verify(out, root=ROOT):
    """Read all raw records, compare every release cell, then write verification."""
    root, out = Path(root).resolve(), Path(out).resolve()
    before = {}

    def read(path):
        path = root / path
        data = path.read_bytes()
        before[path] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    metadata_path = root / "data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1_metadata.jsonl"
    before[metadata_path] = sha(metadata_path)
    metadata_records = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    metadata = {record["equation_id"]: record for record in metadata_records}
    require(len(metadata_records) == len(metadata) == 1000, "canonical IDs/count")
    require(len({m["family_id"] for m in metadata.values()}) == 200, "family count")
    require(all(m["master_seed"] == 2488 and m["generation_version"] == "all_methods_v1"
                for m in metadata.values()), "dataset version/seed")
    plan_path = Path("results/four_method_accuracy_1000_2026-11-01/plan.json")
    plan = read(plan_path)
    require(plan["equation_ids"] == [m["equation_id"] for m in metadata_records],
            "IVP plan IDs/order")
    dataset_path = root / "data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1.csv"
    before[dataset_path] = sha(dataset_path)
    require(before[dataset_path] == plan["dataset_sha256"], "IVP plan dataset SHA")
    r_dir = Path("results/chernov_solve1000_2026-10-18/solutions")
    ivp_dir = Path("results/four_method_accuracy_1000_2026-11-01/records")
    mc_dir = Path("results/mc1000_x0_2026-10-25/attempts")
    for directory in (r_dir, ivp_dir / "rk4", ivp_dir / "product_integral"):
        require({p.stem for p in (root / directory).glob("*.json")} == set(metadata),
                "raw coverage: " + str(directory))
    require({p.stem for p in (root / mc_dir).glob("*.json")} ==
            {pid + "_seed" + str(seed) for pid in metadata for seed in range(4)},
            "raw MC coverage")
    csv_path = out / "grid33_all1000.csv"
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames) == FIELDS, "CSV schema")
        published_rows = list(reader)
    published = {row["equation_id"]: row for row in published_rows}
    require(len(published) == len(published_rows) == 1000 and set(published) == set(metadata),
            "CSV IDs/count/duplicates")
    rows = []
    raw_counts = dict(R=0, RK4=0, PI=0, MC_seeds=0)
    for pid in sorted(metadata):
        m = metadata[pid]
        row = dict(equation_id=pid, family_id=m["family_id"])
        exact = None
        for method, label in (("rk4", "RK4"), ("product_integral", "PI")):
            raw = read(ivp_dir / method / (pid + ".json"))
            body = {k: v for k, v in raw.items() if k != "record_sha256"}
            require(hashlib.sha256(canonical(body)).hexdigest() == raw["record_sha256"],
                    pid + ": IVP checksum")
            require(raw["method"] == method and raw["schema"] == "fixed_ivp_mass_record_v1"
                    and raw["config"] == IVP_CONFIG, pid + ": IVP method/config")
            require(raw["plan_sha256"] == before[root / plan_path]
                    and raw["dataset_sha256"] == before[dataset_path], pid + ": IVP provenance")
            for key in ("equation_id", "expression_hash", "family_id", "structure_signature"):
                require(raw[key] == m[key], pid + ": IVP identity " + key)
            require(raw["inputs"] == {k: m[k] for k in ("x0", "x1", "y0", "v0")},
                    pid + ": IVP inputs")
            q = raw["result"]
            require(raw["usable"] is True and raw["status"] == "success"
                    and q["success"] is True and q["failure_reason"] == "", pid + ": IVP status")
            require(q["equation_id"] == pid and q["x"] == GRID129
                    and q["n_steps"] == 128 and q["h"] == 1 / 64, pid + ": IVP grid")
            ys = finite_vector(q["y"], 129, pid + ": IVP y")
            ds = finite_vector(q["dy"], 129, pid + ": IVP dy")
            require(ys[0] == m["y0"] and ds[0] == m["v0"], pid + ": IVP initial state")
            local_exact = finite_vector(raw["exact_grid33"], 33, pid + ": exact")
            require(exact is None or local_exact == exact, pid + ": exact across IVP methods")
            exact = local_exact
            error = max(abs(ys[4 * i] - exact[i]) for i in range(33))
            require(error == raw["E_inf_grid33"] and raw["E0"] == abs(ys[64] - exact[16])
                    and raw["exact0"] == exact[16], pid + ": IVP saved error")
            row[label + "_E33"] = error
            row[label + "_accuracy_met"] = error <= .05
            raw_counts[label] += 1
        raw = read(r_dir / (pid + ".json"))
        raw_counts["R"] += 1
        require(raw["problem_id"] == pid and raw["config"] == R_CONFIG
                and raw["target_x"] == GRID33, pid + ": R identity/config/grid")
        status = raw["computation_status"]
        row.update(R_status=status, R_available=status == "success", R_E33=None,
                   R_accuracy_met=None, margin_R=None)
        if status == "success":
            q = raw["result"]
            require(q["success"] is True and q["failure_reason"] == ""
                    and q["x"] == GRID33 and q["problem_id"] == pid and q["n"] == 128,
                    pid + ": R result")
            require(raw["y_exact"] == exact, pid + ": R exact")
            require(raw["metadata"]["expression_hash"] == m["expression_hash"], pid + ": R expression")
            ys = finite_vector(q["f"], 33, pid + ": R f")
            error = max(abs(ys[i] - exact[i]) for i in range(33))
            require(error == raw["max_abs_error"], pid + ": R saved error")
            row.update(R_E33=error, R_accuracy_met=error <= .05,
                       margin_R=error - row["RK4_E33"])
        else:
            require(status == "worker_exception" and raw["result"] is None
                    and raw["max_abs_error"] is None, pid + ": R missing contract")
        fs = []
        for seed in range(4):
            raw = read(mc_dir / (pid + "_seed" + str(seed) + ".json"))
            raw_counts["MC_seeds"] += 1
            q, c = raw["result"], raw["config"]
            require(raw["problem_id"] == pid and raw["expression_hash"] == m["expression_hash"]
                    and raw["seed_index"] == seed, pid + ": MC identity")
            require(raw["status"] == "completed" and q["success"] is True
                    and q["failure_reason"] == "" and q["x"] == [0.] and raw["target_x"] == [0.]
                    and q["problem_id"] == pid and q["completed_paths"] == [1024],
                    pid + ": MC status/grid")
            require(raw["y_exact"] == exact[16], pid + ": MC exact")
            require(c == dict(MC_CONFIG, seed=202610250101 + seed), pid + ": MC config")
            fs.append(finite_vector(q["f"], 1, pid + ": MC f")[0])
            require(abs(fs[-1] - exact[16]) == raw["abs_error"], pid + ": MC saved error")
        mean = sum(fs) / 4
        lower = abs(mean - exact[16])
        row.update(MC_mean4_0=mean, MC_E0_lower_bound=lower, MC_E33=None,
                   MC_accuracy_met=None,
                   MC_threshold_evidence="fails_by_lower_bound" if lower > .05 else "unknown",
                   margin_PI=row["PI_E33"] - row["RK4_E33"],
                   margin_MC_lower_bound=lower - row["RK4_E33"])
        available = {label: row[label + "_E33"] for label in ("R", "RK4", "PI")
                     if row[label + "_E33"] is not None}
        row["best_available_method"] = min(available, key=available.get)
        proven = (row["R_available"] and row["margin_R"] > 0
                  and row["margin_PI"] > 0 and row["margin_MC_lower_bound"] > 0)
        row["winner_scope"] = "all_four_inferred" if proven else "available_only"
        row["inferred_unique_four_method_winner"] = "RK4" if proven else None
        for key in FIELDS:
            expected, observed = row[key], published[pid][key]
            if expected is None:
                require(observed == "", pid + ": missing cell " + key)
            elif type(expected) in (int, float):
                require(float(observed) == expected, pid + ": numeric cell " + key)
            else:
                require(observed == str(expected), pid + ": cell " + key)
        rows.append(row)
    common = [r for r in rows if r["R_available"]]
    missing = [r["equation_id"] for r in rows if not r["R_available"]]
    require(missing == MISSING_R and len(common) == 997, "missing R IDs")
    with (out / "missing_remizov.csv").open(newline="") as handle:
        missing_reader = csv.DictReader(handle)
        require(tuple(missing_reader.fieldnames) == FIELDS, "missing R CSV schema")
        missing_rows = list(missing_reader)
    require(missing_rows == [published[pid] for pid in missing], "missing R CSV full rows")
    full_stats = {method: stats([r[method + "_E33"] for r in rows if r[method + "_E33"] is not None])
                  for method in ("R", "RK4", "PI")}
    common_stats = {method: stats([r[method + "_E33"] for r in common]) for method in ("R", "RK4", "PI")}
    margins = {method: dict(zip(("margin", "equation_id"),
                               min((r[field], r["equation_id"]) for r in common)))
               for method, field in (("R", "margin_R"), ("PI", "margin_PI"),
                                     ("MC_lower_bound", "margin_MC_lower_bound"))}
    dominance = {method: sum(r[field] is not None and r[field] > 0 for r in rows)
                 for method, field in (("R", "margin_R"), ("PI", "margin_PI"),
                                       ("MC_lower_bound", "margin_MC_lower_bound"))}
    require(dominance == dict(R=997, PI=1000, MC_lower_bound=1000), "strict dominance")
    require([full_stats[m]["accurate"] for m in ("R", "RK4", "PI")] == [970, 1000, 404],
            "full pass counts")
    require([common_stats[m]["accurate"] for m in ("R", "RK4", "PI")] == [970, 997, 402],
            "common pass counts")
    summary = json.loads((out / "summary.json").read_text())
    for key, value in dict(total_equations=1000, common_equations=997,
                           missing_remizov_ids=missing, MC_full_grid33_calculated=False,
                           target_x=GRID33, epsilon=.05, new_solver_calls=0,
                           new_fit_calls=0, new_model_inference_calls=0).items():
        require(summary[key] == value, "summary: " + key)
    require(summary["raw_records_checked"] == dict(R=1000, R_success=997, RK4=1000,
                                                   PI=1000, MC=4000), "summary: raw counts")
    require(summary["mc_threshold_evidence"] == dict(unknown=938, fails_by_lower_bound=62),
            "summary: MC threshold counts")
    require(summary["saved_configs"] == dict(R=R_CONFIG, RK4_PI=IVP_CONFIG,
                                            MC_common=MC_CONFIG,
                                            MC_seeds=[202610250101 + s for s in range(4)]),
            "summary: saved configs")
    for label, expected in (("full_available", full_stats), ("common997", common_stats)):
        for method, values in expected.items():
            check_stats(summary[label][method], values, "summary: " + label + "/" + method)
    d = summary["strict_dominance"]
    require(d["counts_all_available"] == dominance, "summary: dominance counts")
    require(d["inferred_unique_four_method_winners"] == 997, "summary: winner count")
    for method, expected in margins.items():
        observed = d["minimum_margins_common997"][method]
        require(observed == expected, "summary: minimum margin " + method)
    require(all(sha(path) == digest for path, digest in before.items()), "raw inputs changed during verification")
    expected_hashes = {str(p.relative_to(root)): digest for p, digest in before.items()}
    require(json.loads((out / "input_sha256.json").read_text()) == expected_hashes,
            "input manifest: full consumed-source hash mapping")
    source_digest = hashlib.sha256(canonical({str(p.relative_to(root)): digest
                                              for p, digest in before.items()})).hexdigest()
    result = dict(passed=True, independent_implementation="stdlib scalar max/abs/sum; no builder import",
                  csv_rows_checked=1000, csv_cells_checked=1000 * len(FIELDS),
                  raw_records_checked=raw_counts, source_files_checked=len(before),
                  source_sha256_aggregate=source_digest, source_unchanged=True,
                  artifacts_sha256={name: sha(out / name) for name in
                                    ("grid33_all1000.csv", "missing_remizov.csv", "summary.json",
                                     "input_sha256.json", "REPORT.md", "reproduce.json")},
                  missing_remizov_rows_checked=3, missing_remizov_cells_checked=3 * len(FIELDS),
                  full_input_sha256_mapping_verified=True,
                  full_available=full_stats, common997=common_stats,
                  missing_remizov_ids=missing,
                  strict_dominance=dict(counts_all_available=dominance,
                                        minimum_margins_common997=margins,
                                        inferred_unique_four_method_winners=997),
                  MC_full_grid33_calculated=False, new_solver_calls=0,
                  condition="MC grid extension must preserve the same saved mean4 value at x=0")
    (out / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                                     allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.out, args.root)
    print(json.dumps({"passed": result["passed"], "csv_rows_checked": result["csv_rows_checked"],
                      "raw_records_checked": result["raw_records_checked"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
