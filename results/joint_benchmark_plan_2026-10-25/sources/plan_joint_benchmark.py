"""Read saved artifacts and prepare a proposal; never import or run a solver.

Resource arithmetic is not a new numerical experiment. Output is deliberately
not an executable compare_chernov request/plan. Dates follow the project task.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SMALL = Path("data/generated")
LARGE = Path("data/all_methods_v1_1000_2026-10-18")
GRID_IDS = tuple(f"AMV1_F{i:04d}_S00" for i in (0, 1, 2, 3, 14, 18))
CHERNOV = dict(n=128, num_t=513, n_steps=32768, dx=.0125, padding=12.,
               tail_eps=1e-4, T=None, t_grading="quadratic", interp="cubic",
               boundary="zero", taper_width=None, max_work_grid_points=100000,
               n_jobs=1, start_method="spawn")
MC = dict(M=None, dt=1/128, T=None, tail_eps=1e-4, seed=None, stream_id=0,
          chunk=1024, max_steps=100000, max_path_steps=300000000,
          max_batch_paths=1024, max_runtime_seconds=None)


def read(path: Path):
    return json.loads((ROOT / path).read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rounded_T(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("a positive saved certified horizon is required")
    return math.ceil(value * 8) / 8


def mc_cost(T: float, M: int, K: int) -> dict:
    if T * 128 != int(T * 128) or M < 2 or K < 1:
        raise ValueError("dyadic horizon, M>=2 and K>=1 required")
    N = int(T * 128)
    return dict(steps=N, path_steps=K*M*N, normal_draws=K*M*N,
                batches=K*math.ceil(M/1024),
                coefficient_calls=K*math.ceil(M/1024)*(4*N+2),
                coefficient_values=K*M*(4*N+2))


def all_success_lower(n: int, alpha: float = .05) -> float:
    """One-sided Clopper-Pearson lower bound when all n trials pass."""
    if n < 1 or not 0 < alpha < 1:
        raise ValueError("invalid binomial design")
    return alpha ** (1/n)


def required_all_success(p: float = .95, alpha: float = .05) -> int:
    if not 0 < p < 1 or not 0 < alpha < 1:
        raise ValueError("probabilities must lie in (0,1)")
    return math.ceil(math.log(alpha)/math.log(p))


def metadata(folder: Path):
    path = ROOT / folder / "all_methods_equations_v1_metadata.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def dataset_digest(folder: Path) -> str:
    files = sorted((ROOT / folder).glob("all_methods_equations_v1*"))
    return hashlib.sha256("".join(p.name+":"+digest(p)+"\n" for p in files).encode()).hexdigest()


def family_audit():
    items, summaries = [], {}
    topology_splits = defaultdict(set)
    source_rows = {}
    for namespace, folder in (("small20", SMALL), ("production1000", LARGE)):
        rows, fingerprint = metadata(folder), dataset_digest(folder)
        source_rows[namespace] = rows
        family_splits = defaultdict(set)
        for r in rows:
            family_splits[r["family_id"]].add(r["split"])
            topology_splits[r["structure_signature"]].add(r["split"])
            items.append(dict(dataset=namespace, dataset_sha256=fingerprint,
                              equation_id=r["equation_id"], expression_hash=r["expression_hash"],
                              family_id=r["family_id"], structure_signature=r["structure_signature"],
                              split=r["split"], complexity=r["complexity_level"], regime=r["regime"],
                              solver_policy_development_exposed=True,
                              pilot_training_eligible=False))
        if any(len(v) != 1 for v in family_splits.values()):
            raise ValueError("family crosses splits")
        summaries[namespace] = dict(rows=len(rows), families=len(family_splits),
            splits=dict(Counter(r["split"] for r in rows)),
            family_splits=dict(Counter(next(iter(s)) for s in family_splits.values())),
            dataset_sha256=fingerprint)
    a, b = source_rows.values()
    summaries["cross_dataset"] = dict(
        shared_ids=len({r["equation_id"] for r in a} & {r["equation_id"] for r in b}),
        shared_expression_hashes=len({r["expression_hash"] for r in a} & {r["expression_hash"] for r in b}),
        shared_structure_signatures=len({r["structure_signature"] for r in a} & {r["structure_signature"] for r in b}),
        topology_split_conflicts=sum(len(s) > 1 for s in topology_splits.values()))
    return items, summaries


def history():
    """Independently recompute errors from all stored 33-component answers."""
    statuses, by_cell, by_split, by_regime = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    times, errors, checked = [], [], 0
    by_id = {r["equation_id"]: r for r in metadata(LARGE)}
    for file in sorted((ROOT/"results/chernov_solve1000_2026-10-18/solutions").glob("*.json")):
        r = json.loads(file.read_text())
        statuses[r["status"]] += 1
        # The three original worker exits contain no returned metadata/result.
        m = r.get("metadata") or by_id[r["problem_id"]]
        by_cell[m["complexity_level"]+"/"+m["regime"]][r["status"]] += 1
        by_regime[m["regime"]][r["status"]] += 1
        by_split[m["split"]][r["status"]] += 1
        if r.get("result") and r["result"]["success"]:
            f, exact = r["result"]["f"], r["y_exact"]
            if len(f) != 33 or len(exact) != 33:
                raise ValueError("historical grid is not 33 points")
            err = max(abs(a-b) for a,b in zip(f,exact))
            if not math.isfinite(err) or err != r["max_abs_error"]:
                raise ValueError("saved exact error mismatch")
            if (err <= .01) != (r["status"] == "accuracy_pass"):
                raise ValueError("saved accuracy status mismatch")
            checked += 1
            times.append(r["solver_seconds"])
            errors.append(err)
    saved = read(Path("results/chernov_managed_completion_2026-10-18/final/statistics.json"))
    if dict(statuses) != saved["outcomes"] or checked != saved["finite_results"]:
        raise ValueError("historical outcomes mismatch")
    with (ROOT/"results/chernov_dataset20_2026-10-18/attempts.csv").open(newline="") as f:
        small = [r for r in csv.DictReader(f) if r["method"] == "solve"]
    return dict(raw_records=sum(statuses.values()), raw_exact_errors_rechecked=checked,
                outcomes=dict(statuses), by_cell=dict(by_cell), by_regime=dict(by_regime),
                by_split=dict(by_split), historical_statistics=saved,
                recomputed_solver_seconds_sum=math.fsum(times),
                recomputed_error_median=statistics.median(errors),
                small20=small)


def make_proposal(ids: list[str], fingerprint: str):
    def seeds(mode, count):
        return [int(hashlib.sha256(f"joint-pilot-v1|{fingerprint}|{mode}|{i}".encode()).hexdigest()[:8],16)
                for i in range(count)]
    point_seeds, grid_seeds = seeds("point", 8), seeds("grid33", 4)
    if len(set(point_seeds+grid_seeds)) != 12:
        raise ValueError("seed collision")
    return dict(schema_version="joint_benchmark_proposal_v1", design_date="2026-10-25",
        execution_authorized=False, executable_plan=False, new_solver_runs=0,
        accuracy=dict(epsilon=.01, tail_epsilon=1e-4, statistical_planning_halfwidth=.005,
                      reliability_target=.95, confidence=.95, practical_time_margin=.10),
        primary_implementation=dict(remizov="chernov_solver.ChernovSolver.solve",
                                    monte_carlo="mc_solver.solve_monte_carlo"),
        base_configs=dict(remizov=CHERNOV, monte_carlo=MC),
        T_policy="shared ceil(saved certified T(tail_eps=1e-4)*8)/8; revalidate certificate before future execution",
        phases=[dict(name="point", ids=ids, target_x=[0.], M=2048, mc_repeats=8,
                     root_seeds=point_seeds, remizov_repeats=3, phase_wall_seconds=2700),
                dict(name="grid33", ids=list(GRID_IDS), target_x=[-1+j/16 for j in range(33)],
                     M=1024, mc_repeats=4, root_seeds=grid_seeds,
                     remizov_repeats=3, phase_wall_seconds=2700)],
        budgets=dict(worker_wall_seconds=60, worker_rss_mib=384, total_session_seconds=5400,
                     serial_workers=1, blas_threads=1, automatic_retries=0,
                     on_first_resource_limit="same equation/mode/method: remaining repeats not_run_after_resource_limit",
                     on_numerical_failure="retain failure and continue predetermined independent seed list",
                     on_phase_limit="retain partial series and mark unstarted trials not_run_phase_budget"),
        interval_power=dict(point_best_possible_lower=all_success_lower(8),
                            grid_best_possible_lower=all_success_lower(4),
                            n_all_success_needed_for_p95=required_all_success(),
                            confidence_scope="one profile and one target mode; not familywise across equations"),
        results_status="design_only_no_new_comparative_results",
        pilot_labels_training_eligible=False)


def resources(saved_history):
    agg = read(Path("results/mc_f0002_confirmation_2026-10-24/aggregates.json"))
    finest = agg["refinement128"]["levels"][-1]
    if finest["h"] != 1/128:
        raise ValueError("wrong timing anchor")
    per_update = finest["median_runtime_seconds"] / (1024*8144)
    old = {r["problem_id"]:r for r in saved_history["small20"]}
    rows = []
    for c in read(Path("results/forcing_bounds_2026-10-24/survey/certificates.json")):
        T = rounded_T(c["T"])
        p, g = mc_cost(T,2048,1), mc_cost(T,1024,33)
        row = dict(problem_id=c["problem_id"], complexity=c["complexity"], regime=c["regime"],
                   G=c["G"], gap=c["gap"], T_certified_saved=c["T"], T_proposed=T,
                   tail_bound_formula=c["G"]*math.exp(-c["gap"]*T)/c["gap"],
                   grid_in_recommended_pilot=c["problem_id"] in GRID_IDS,
                   h=1/128, steps=p["steps"], M_point=2048, M_grid=1024,
                   point_path_steps=p["path_steps"], grid33_path_steps=g["path_steps"],
                   point_coefficient_calls=p["coefficient_calls"], grid33_coefficient_calls=g["coefficient_calls"],
                   point_coefficient_values=p["coefficient_values"], grid33_coefficient_values=g["coefficient_values"],
                   point_f0002_rate_proxy_seconds=p["path_steps"]*per_update,
                   grid33_f0002_rate_proxy_seconds=g["path_steps"]*per_update,
                   remizov_operator_applications=128*512, remizov_work_grid_points=2081,
                   remizov_legacy_seconds=float(old[c["problem_id"]]["solver_seconds"]),
                   remizov_legacy_error=float(old[c["problem_id"]]["max_abs_error"]),
                   remizov_legacy_T=float(old[c["problem_id"]]["T"]))
        rows.append(row)
    selected = [r for r in rows if r["grid_in_recommended_pilot"]]
    sum_column = lambda key, rr: math.fsum(r[key] for r in rr)
    costs = dict(
        forecast_is_not_measurement_or_bound=True,
        mc_anchor=dict(problem="AMV1_F0002_S00", h=1/128,T=63.625,M=1024,chunk=1024,
                       seconds=finest["median_runtime_seconds"], seconds_per_update=per_update),
        point20_mc_updates_8seeds=sum(r["point_path_steps"]*8 for r in rows),
        grid6_mc_updates_4seeds=sum(r["grid33_path_steps"]*4 for r in selected),
        grid20_mc_updates_4seeds=sum(r["grid33_path_steps"]*4 for r in rows),
        point20_mc_proxy_seconds=8*sum_column("point_f0002_rate_proxy_seconds",rows),
        grid6_mc_proxy_seconds=4*sum_column("grid33_f0002_rate_proxy_seconds",selected),
        grid20_mc_proxy_seconds=4*sum_column("grid33_f0002_rate_proxy_seconds",rows),
        point20_remizov_legacy_time_proxy_seconds=3*sum_column("remizov_legacy_seconds",rows),
        grid6_remizov_legacy_time_proxy_seconds=3*sum_column("remizov_legacy_seconds",selected),
        grid20_remizov_legacy_time_proxy_seconds=3*sum_column("remizov_legacy_seconds",rows),
        point20_max_attempts=20*(8+3), grid6_max_attempts=6*(4+3),
        grid20_max_attempts=20*(4+3),
        caveats=["F0002 throughput is a scenario only; L2-L4 AST costs and other x are unmeasured",
                 "Chernov timings use historical taper and different T; zero/common T timings unknown",
                 "Timeouts/early resource stops reduce completed work; partial work still consumes budget",
                 "Point and grid profiles differ in M and repeat count; accuracy labels do not transfer"])
    return rows, costs


def protected_snapshot(out: Path):
    paths = set(ROOT.glob("*.py"))
    paths.update(ROOT / s for s in ("MIGRATION_MANIFEST.json", "AGENTS.md"))
    for folder in ("src", "diagnostics", "tests", "docs", "data", "results", "comparison_inputs", ".claude"):
        paths.update(p for p in (ROOT/folder).rglob("*") if p.is_file()
                     and not p.is_relative_to(out) and "__pycache__" not in p.parts
                     and p.name != ".DS_Store")
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths) if p.is_file()}


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n")


def save_csv(path: Path, rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, type=Path)
    args=ap.parse_args()
    out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/"results"):
        ap.error("new directory must be under results")
    # Validate inputs before producing files. No evaluator/solver is imported.
    families, audit=family_audit()
    historical=history()
    rows, costs=resources(historical)
    proposal=make_proposal([r["problem_id"] for r in rows], audit["small20"]["dataset_sha256"])
    if len(rows)!=20 or len(families)!=1020:
        raise ValueError("unexpected dataset size")
    out.mkdir(parents=True, exist_ok=False)
    (out/"PROJECT_STATUS.before.md").write_bytes((ROOT/"PROJECT_STATUS.md").read_bytes())
    save_json(out/"protected_before.json", protected_snapshot(out))
    for name,value in (("proposal.json",proposal), ("historical_evidence.json",historical),
                       ("family_audit.json",audit), ("resource_summary.json",costs)):
        save_json(out/name,value)
    save_csv(out/"resources20.csv",rows)
    save_csv(out/"family_manifest.csv",families)
    print(json.dumps(dict(design_only=True, raw_history_rechecked=historical["raw_records"],
                         resources20=len(rows), output=str(out)),ensure_ascii=False))


if __name__ == "__main__":
    main()
