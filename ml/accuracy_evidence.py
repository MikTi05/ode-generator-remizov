"""Rebuild two-method accuracy evidence from existing numerical answers only.

The primary cohort is the user's frozen mass R128/taper and MC1024/x0 pair.
R256 and grid33 cohorts remain separate. This module never calls a numerical
solver, generates equations, fits models, or writes artifacts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from ml.accuracy_selector import EPSILON, accuracy_target

ROOT = Path(__file__).resolve().parents[1]
DATA = "data/all_methods_v1_1000_2026-10-18"
EXTERNAL_DATA = "results/ml_selector_final_2026-10-27/final_experiment/test_inputs"
GRID = [-1.0 + i / 16 for i in range(33)]
MASS_R = "results/chernov_solve1000_2026-10-18"
MASS_MC = "results/mc1000_x0_2026-10-25"
EXTERNAL = "results/ml_selector_final140_protocol_2026-10-27"
R256_SOURCES = (
    "results/final_grid5_2026-10-25",
    "results/remizov_r0_2026-10-26",
    "results/ml_probability_validation_2026-10-26",
    "results/ml_switch5_r_timing_2026-10-26",
    "results/f0047_observed_timing_2026-10-26",
    "results/f0047_three_profile_2026-10-26",
)


def _check(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pin(root, path, pins):
    path = Path(path)
    name = str(path.relative_to(root))
    digest = _sha(path)
    _check(name not in pins or pins[name] == digest, "source changed while reading: " + name)
    pins[name] = digest
    return name


def _read(root, path, pins):
    _pin(root, path, pins)
    return json.loads(Path(path).read_text())


def _plan(root, folder, pins):
    path = root / folder / "plan.json"
    plan = _read(root, path, pins)
    # Historical source-code versions are not current implementations. Dataset
    # bytes, however, must still match the experiment's original data pins.
    for name, expected in plan.get("protected_sha256", {}).items():
        if name.startswith("data/"):
            _check(_sha(root / name) == expected, "historical dataset changed: " + name)
            _pin(root, root / name, pins)
    return plan, _sha(path)


def _dataset(root, relative, pins):
    from all_methods_dataset_loader import load_all_methods_equations

    folder = root / relative
    csv_path = folder / "all_methods_equations_v1.csv"
    metadata_path = folder / "all_methods_equations_v1_metadata.jsonl"
    _pin(root, csv_path, pins)
    _pin(root, metadata_path, pins)
    metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    by_id = {m["equation_id"]: m for m in metadata}
    _check(len(by_id) == len(metadata), "duplicate canonical equation identity")
    family_splits, topology_splits = {}, {}
    for m in metadata:
        for groups, key in ((family_splits, m["family_id"]),
                            (topology_splits, m["structure_signature"])):
            _check(groups.setdefault(key, m["split"]) == m["split"], "family split leakage")
    # Only reconstruct the safe existing exact evaluator; never run a solver.
    equations = load_all_methods_equations(csv_path, metadata_path)
    exact = {e.equation_id: np.asarray(e.y_exact(np.asarray(GRID)), dtype=float)
             for e in equations}
    _check(set(exact) == set(by_id), "safe-loader dataset identity mismatch")
    _check(all(v.shape == (33,) and np.isfinite(v).all() for v in exact.values()),
           "invalid canonical exact values")
    return by_id, exact, _sha(csv_path)


def _identity(metadata, dataset_sha):
    return {"dataset_sha256": dataset_sha, **{key: metadata[key] for key in (
        "equation_id", "expression_hash", "family_id", "split", "structure_signature")}}


def _validate_identity(record, metadata, nested=False):
    own = record.get("metadata") if nested else record
    _check(isinstance(own, dict), "missing result identity")
    for key in ("expression_hash", "family_id", "split"):
        _check(own.get(key) == metadata[key], "result identity mismatch: " + key)
    if nested:
        _check(own.get("structure_signature") == metadata["structure_signature"],
               "result topology mismatch")
        _check(own.get("serialized_ast") == metadata["serialized_ast"], "result AST mismatch")


def _arrays(record, metadata, exact, *, mass=False):
    result = record.get("result") or {}
    _check(result.get("success") is True and result.get("failure_reason") == "",
           "nonfinite or failed numerical answer")
    _check(result.get("problem_id") == metadata["equation_id"], "result problem mismatch")
    _check(result.get("x") == GRID, "grid33 coordinates differ")
    values = np.asarray(result.get("f"), dtype=float)
    saved_exact = np.asarray(record.get("y_exact"), dtype=float)
    _check(values.shape == saved_exact.shape == (33,) and np.isfinite(values).all(),
           "missing finite grid33 answer")
    _check(np.array_equal(saved_exact, exact), "noncanonical saved exact solution")
    _check(math.isfinite(result.get("T", math.nan)) and result["T"] > 0, "invalid horizon")
    error = float(np.max(np.abs(values - saved_exact)))
    saved_error = record.get("max_abs_error") if mass else record.get("E_grid", record.get("E_inf"))
    _check(error == saved_error, "saved grid error disagrees with raw arrays")
    return values, saved_exact, result


def _remizov_answer(record, metadata, exact, config, sources, *, mass=False):
    _validate_identity(record, metadata, nested=mass)
    values, saved_exact, result = _arrays(record, metadata, exact, mass=mass)
    n = 128 if mass else 256
    expected = dict(n=n, boundary="taper" if mass else "zero", num_t=513,
                    dx=.0125, padding=12., interp="cubic", n_jobs=1,
                    n_steps=32768, max_work_grid_points=100000,
                    start_method="spawn", taper_width=None,
                    t_grading="quadratic", tail_eps=1e-8 if mass else 1e-4)
    _check(set(config) == set(expected) | {"T"} and
           all(config[key] == value for key, value in expected.items()),
           "unsupported Remizov physical profile")
    _check(config["T"] is None if mass else config["T"] == result["T"],
           "Remizov horizon/config mismatch")
    _check(result.get("n") == n and result.get("scheme") == "solve" and
           result.get("time_nodes") == 513 and result.get("actual_dx") == .0125 and
           result.get("work_grid_points") == 2081 and result.get("work_domain") == [-13., 13.] and
           result.get("n_operator_applications") == n * 512 and
           result.get("n_spline_builds") == n * 512 + 1,
           "Remizov result profile mismatch")
    return dict(f=values.tolist(), exact=saved_exact.tolist(), config=config,
                T=result["T"], sources=sources, f0=float(values[16]),
                y_exact0=float(saved_exact[16]),
                seconds=record.get("method_wall_seconds", result.get("runtime_seconds")))


def _mass_remizov(root, metadata, exact, pins):
    plan, plan_sha = _plan(root, MASS_R, pins)
    safety = _read(root, root / MASS_R / "safety_policy_v2.json", pins)
    # The saved plan was later reserialized. Terminal records keep the original
    # execution hash, independently retained in the runtime safety amendment.
    execution_plan_sha = safety["plan_sha256"]
    _check(safety["numerical_config_unchanged"] == plan["config"],
           "historical mass profile amendment mismatch")
    _check(plan["target_x"] == GRID and set(plan["equation_ids"]) == set(metadata),
           "mass Remizov plan coverage mismatch")
    answers, missing = {}, {}
    paths = sorted((root / MASS_R / "solutions").glob("*.json"))
    _check({p.stem for p in paths} == set(metadata), "mass Remizov terminal coverage mismatch")
    for path in paths:
        record = _read(root, path, pins)
        pid = path.stem
        _check(record.get("problem_id") == pid and record.get("plan_sha256") == execution_plan_sha and
               record.get("config") == plan["config"] and record.get("target_x") == GRID,
               "mass Remizov request/plan mismatch")
        source = str(path.relative_to(root))
        if record.get("computation_status") != "success":
            _check(not (record.get("result") or {}).get("f") and record.get("max_abs_error") is None,
                   "failed Remizov record contains a usable answer")
            missing[pid] = dict(reason="remizov_" + str(record.get("computation_status")),
                                source=source, config=record["config"])
            continue
        answers[pid] = _remizov_answer(record, metadata[pid], exact[pid], record["config"],
                                       [source], mass=True)
    return answers, missing


def _spawn_key(pid, x, stream):
    raw = b"mc_point_v1\0" + pid.encode() + struct.pack("<d", 0. if x == 0 else x)
    return [*struct.unpack("<8I", hashlib.sha256(raw).digest()), stream]


def _validate_mc_result(record, metadata, exact, config, source, dataset_sha, *, grid=False):
    result = record.get("result") or {}
    pid = metadata["equation_id"]
    _check(record.get("status") == "completed" and result.get("success") is True and
           result.get("failure_reason") == "" and result.get("problem_id") == pid,
           "missing complete Monte Carlo answer")
    own = record.get("metadata_identity", record)
    _check(all(own.get(key) == metadata[key] for key in ("expression_hash", "family_id", "split")),
           "Monte Carlo identity mismatch")
    x = GRID if grid else [0.]
    f = np.asarray(result.get("f"), dtype=float)
    saved_exact = np.asarray(record.get("y_exact"), dtype=float)
    expected_exact = exact if grid else np.asarray(exact[16])
    _check(result.get("x") == x and f.shape == (len(x),) and np.isfinite(f).all() and
           np.array_equal(saved_exact, expected_exact), "MC coordinates/output/exact mismatch")
    M = 4096 if grid else 1024
    _check(config.get("M") == M and config.get("dt") == 1/128 and config.get("tail_eps") == 1e-4 and
           config.get("T") is None and config.get("chunk") == 1024 and
           config.get("max_batch_paths") == 1024 and config.get("max_steps") == 100000 and
           config.get("stream_id") == 0, "unsupported Monte Carlo physical profile")
    _check(result.get("M") == M and result.get("completed_paths") == [M] * len(x) and
           result.get("seed_entropy") == config["seed"] and result.get("stream_id") == 0 and
           result.get("backend") == "numba" and result.get("block_steps") == 64 and
           result.get("dt_requested") == 1/128 and 0 < result.get("dt_actual", 0) <= 1/128 and
           result.get("rng_scheme") == "PCG64/SeedSequence/point_address_v1" and
           result.get("point_spawn_keys") == [_spawn_key(pid, point, 0) for point in x],
           "Monte Carlo result/config/RNG mismatch")
    T, steps = result["T"], result["n_steps"]
    _check(math.isfinite(T) and T > 0 and steps == math.ceil(T / config["dt"]) and
           result["dt_actual"] == T / steps and
           result.get("n_path_steps") == result.get("n_normal_draws") == M * steps * len(x) and
           M * steps * len(x) <= config["max_path_steps"] and result.get("tail_tolerance_met") is True,
           "Monte Carlo horizon/work mismatch")
    index = 16 if grid else 0
    f0 = float(f[index])
    exact0 = float(exact[16])
    error = abs(f0 - exact0)
    if not grid:
        _check(error == record.get("abs_error"), "saved MC x0 error mismatch")
    return dict(_identity(metadata, dataset_sha), seed=int(config["seed"]),
                E_MC_seed=error, error=error, f0=f0, y_exact0=exact0,
                success=bool(error <= EPSILON), epsilon=EPSILON, config=config,
                T=float(T), n_steps=int(steps), source=source,
                metric="absolute_error_at_x0", rng_scheme=result["rng_scheme"],
                point_spawn_key=result["point_spawn_keys"][index],
                seconds=record.get("method_wall_seconds", result.get("runtime_seconds")))


def _mass_mc(root, metadata, exact, dataset_sha, pins):
    plan, plan_sha = _plan(root, MASS_MC, pins)
    _check(plan["target_x"] == [0.] and plan["seeds"] ==
           [202610250101, 202610250102, 202610250103, 202610250104] and
           set(plan["equation_ids"]) == set(metadata), "mass Monte Carlo plan mismatch")
    grouped, rows = defaultdict(list), []
    paths = sorted((root / MASS_MC / "attempts").glob("*.json"))
    _check(len(paths) == 4 * len(metadata), "missing mass Monte Carlo seed records")
    keys = set()
    for path in paths:
        record = _read(root, path, pins)
        pid = record.get("problem_id")
        _check(pid in metadata and record.get("plan_sha256") == plan_sha, "mass MC plan/ID mismatch")
        index = record.get("seed_index")
        _check(isinstance(index, int) and 0 <= index < 4 and (pid, index) not in keys,
               "duplicate or invalid MC seed key")
        keys.add((pid, index))
        config = dict(plan["config"], seed=plan["seeds"][index])
        _check(record.get("config") == config and path.stem == pid + "_seed" + str(index),
               "mass Monte Carlo frozen config mismatch")
        row = _validate_mc_result(record, metadata[pid], exact[pid], config,
                                  str(path.relative_to(root)), dataset_sha)
        row.update(seed_index=index, cohort="original_mass_x0", context="MC1024_x0")
        grouped[pid].append(row)
        rows.append(row)
    for pid in metadata:
        grouped[pid].sort(key=lambda row: row["seed_index"])
        _check([r["seed_index"] for r in grouped[pid]] == [0, 1, 2, 3], "incomplete MC seed set")
    return dict(grouped), rows


def _pair(metadata, dataset_sha, remizov, seeds, cohort, context, *, missing=None):
    _check(bool(seeds), "a comparable pair needs real Monte Carlo seed observations")
    errors = [seed["E_MC_seed"] for seed in seeds]
    exact0 = seeds[0]["y_exact0"]
    _check(all(seed["y_exact0"] == exact0 for seed in seeds), "MC seeds disagree on exact")
    em = float(np.median(errors))
    row = dict(_identity(metadata, dataset_sha), cohort=cohort, context=context,
               metric="absolute_error_at_x0", epsilon=EPSILON,
               E_R=None, E_MC=em, best_method=None, remizov_success=None,
               mc_success=bool(em <= EPSILON), chosen_success=None,
               remizov_f0=None, y_exact0=exact0,
               mc_seed_count=len(seeds), mc_seeds=[s["seed"] for s in seeds],
               mc_seed_errors=errors, mc_seed_f0=[s["f0"] for s in seeds],
               mc_error_min=float(np.min(errors)), mc_error_max=float(np.max(errors)),
               mc_error_std=float(np.std(errors)), mc_error_iqr=float(np.percentile(errors, 75)-np.percentile(errors, 25)),
               mc_configs=[s["config"] for s in seeds], mc_sources=[s["source"] for s in seeds],
               remizov_config=None, remizov_sources=[], remizov_T=None,
               pair_available=False, missing_reason=None)
    if remizov is None:
        _check(missing is not None, "missing Remizov answer requires an explicit reason")
        row.update(remizov_config=missing["config"], remizov_sources=[missing["source"]],
                   missing_reason=missing["reason"])
        return row
    _check(remizov["y_exact0"] == exact0, "methods have different exact x0 values")
    er = abs(remizov["f0"] - exact0)
    row.update(accuracy_target(er, errors))
    row.update(remizov_f0=remizov["f0"], remizov_config=remizov["config"],
               remizov_sources=remizov["sources"], remizov_T=remizov["T"], pair_available=True)
    return row


def _n256(root, metadata, exact, pins):
    answers, duplicates = {}, []
    for folder in R256_SOURCES:
        plan, plan_sha = _plan(root, folder, pins)
        for path in sorted((root / folder / "records").glob("*.json")):
            record = json.loads(path.read_text())
            job = record.get("job") or {}
            config = record.get("config", job.get("config", {}))
            if config.get("n") != 256:
                continue
            _pin(root, path, pins)
            pid = record.get("equation_id", job.get("equation_id", job.get("problem_id")))
            _check(pid in metadata and record.get("status") == "completed" and
                   record.get("plan_sha256") == plan_sha and job in plan["jobs"],
                   "supplementary R256 plan/identity mismatch")
            _check(record.get("config", config) == job["config"], "R256 config disagreement")
            source = str(path.relative_to(root))
            answer = _remizov_answer(record, metadata[pid], exact[pid], config, [source])
            if pid in answers:
                old = answers[pid]
                _check(old["f"] == answer["f"] and old["config"] == answer["config"] and
                       old["exact"] == answer["exact"], "conflicting deterministic R256 repeats")
                old["sources"].append(source)
                duplicates.append(dict(equation_id=pid, kept=old["sources"][0], duplicate=source,
                                       reason="deterministic timing replay, not a new equation"))
            else:
                answers[pid] = answer
    return answers, duplicates


def _external(root, metadata, exact, dataset_sha, pins):
    plan, plan_sha = _plan(root, EXTERNAL, pins)
    _check(plan["data_dir"] == EXTERNAL_DATA and len(plan["cases"]) == 60,
           "external fixed dataset mismatch")
    cases = {c["equation_id"]: c for c in plan["cases"]}
    _check(set(cases) == set(metadata), "external case coverage mismatch")
    answers, mc = {}, defaultdict(list)
    for path in sorted((root / EXTERNAL / "records").glob("*.json")):
        record = json.loads(path.read_text())
        profile = record.get("profile")
        if profile not in ("R256", "MC1"):
            continue
        _pin(root, path, pins)
        job = record["job"]
        pid = record["equation_id"]
        _check(pid in metadata and record.get("plan_sha256") == plan_sha and job in plan["jobs"] and
               record.get("config") == job["config"] and record.get("status") == "completed",
               "external result/plan mismatch")
        _check(cases[pid]["dataset_sha256"] == dataset_sha and
               cases[pid]["expression_hash"] == metadata[pid]["expression_hash"] and
               job["identity"] == dataset_sha + "|" + pid + "|" + metadata[pid]["expression_hash"],
               "external composite identity mismatch")
        source = str(path.relative_to(root))
        if profile == "R256":
            _check(pid not in answers, "duplicate external deterministic answer")
            answers[pid] = _remizov_answer(record, metadata[pid], exact[pid], job["config"], [source])
        else:
            seed = _validate_mc_result(record, metadata[pid], exact[pid], job["config"],
                                       source, dataset_sha, grid=True)
            _check(seed["seed"] == plan["MC_seed"] and record["seed"] == seed["seed"],
                   "external seed schedule mismatch")
            mc[pid].append(seed)
    _check(len(answers) == 60 and set(mc) == set(plan["MC_equation_ids"]),
           "external R60/MC20 coverage mismatch")
    rows = [_pair(metadata[pid], dataset_sha, answers[pid], mc[pid],
                  "external_final20_x0", "other_context_R256_zero_vs_MC4096_single_seed_x0")
            for pid in sorted(mc)]
    return rows, answers, mc


def _summary(rows):
    return dict(pairs=len(rows), winners=dict(Counter(r["best_method"] for r in rows)),
                remizov_success=sum(r["remizov_success"] for r in rows),
                mc_success=sum(r["mc_success"] for r in rows),
                oracle_success=sum(r["chosen_success"] for r in rows))


def build_evidence(root=ROOT):
    """Return JSON-serializable primary, coverage and separate research tables."""
    root = Path(root).resolve()
    pins = {}
    metadata, exact, dataset_sha = _dataset(root, DATA, pins)
    _check(len(metadata) == 1000 and Counter(m["split"] for m in metadata.values()) ==
           dict(train=700, validation=150, test=150), "canonical split changed")
    remizov, missing = _mass_remizov(root, metadata, exact, pins)
    mc, seed_rows = _mass_mc(root, metadata, exact, dataset_sha, pins)
    coverage = [_pair(metadata[pid], dataset_sha, remizov.get(pid), mc[pid],
                      "original_mass_x0", "mass_R128_taper_vs_MC1024_median_seed_error_x0",
                      missing=missing.get(pid)) for pid in sorted(metadata)]
    main = [r for r in coverage if r["pair_available"]]
    n256, repeats = _n256(root, metadata, exact, pins)
    supplementary = [_pair(metadata[pid], dataset_sha, n256[pid], mc[pid],
                           "original_supplementary_n256_x0", "other_context_R256_zero_vs_MC1024_x0")
                     for pid in sorted(n256)]
    ext_metadata, ext_exact, ext_subset_sha = _dataset(root, EXTERNAL_DATA, pins)
    # Final jobs identify the parent pool, while test_inputs is its fixed subset.
    ext_pool = root / "results/ml_selector_final_2026-10-27/final_experiment/input_pool/all_methods_equations_v1.csv"
    _pin(root, ext_pool, pins)
    ext_sha = _sha(ext_pool)
    external, ext_r, ext_mc = _external(root, ext_metadata, ext_exact, ext_sha, pins)
    from ml.accuracy_grid_evidence import build_grid_evidence
    grid = build_grid_evidence(root, original_metadata=metadata, external_metadata=ext_metadata)
    for name, digest in grid["source_sha256"].items():
        _check(name not in pins or pins[name] == digest, "grid/source fingerprint mismatch")
        pins[name] = digest
    audit = dict(
        schema="saved_two_method_accuracy_evidence_v1", epsilon=EPSILON,
        numerical_solver_calls=0, numerical_parameters_changed=False,
        primary_context="mass R128 cubic/taper legacy horizon versus mass MC1024 x0, median4 absolute seed errors",
        supplementary_context="R256 cubic/zero forcing-aware horizon; excluded from primary fit/evaluation",
        continuous_errors_preserved=True, comparison_without_rounding=True,
        exact_values_verified_by_safe_loader=True,
        dataset_identity="SHA256 of canonical CSV bytes + equation_id + expression_hash",
        canonical_dataset_sha256=dataset_sha, external_dataset_sha256=ext_sha,
        external_subset_csv_sha256=ext_subset_sha,
        original_split=dict(train=700, validation=150, test=150),
        original_family_counts={s:len({m["family_id"] for m in metadata.values() if m["split"] == s})
                                for s in ("train", "validation", "test")},
        paired_summary=_summary(main),
        by_split={s:_summary([r for r in main if r["split"] == s]) for s in ("train", "validation", "test")},
        original_x0_missing_pair_count=len(missing), missing_remizov=missing,
        standalone_mc_equations=len(mc), standalone_mc_seed_records=len(seed_rows),
        standalone_mc_success=sum(float(np.median([s["E_MC_seed"] for s in seeds])) <= EPSILON
                                  for seeds in mc.values()),
        supplementary_n256_pairs=len(supplementary),
        supplementary_n256_by_split=dict(Counter(r["split"] for r in supplementary)),
        supplementary_n256_deterministic_duplicates=repeats,
        external_R256_equations=len(ext_r), external_MC4096_equations=len(ext_mc),
        external_comparable_x0_pairs=len(external), external_missing_MC_pairs=len(ext_r)-len(ext_mc),
        external_seed_count_per_equation=1,
        external_used_for_primary_fit_or_metrics=False,
        main_MC1024_and_supplementary_MC4096_seeds_combined=False,
        spatial_scope="primary x=0 only; grid33 summaries are separately scoped; neither is a continuous interval certificate",
        grid=grid["audit"], grid_groups=grid["audit"]["grid_groups"], source_files=len(pins),
        no_censoring_or_replacement_of_finite_outliers=True,
    )
    return dict(main_rows=main, coverage_rows=coverage, mc_seed_rows=seed_rows,
                supplementary_rows=supplementary, external_rows=external,
                grid_rows=grid["rows"], grid_seed_rows=grid.get("seed_rows", []),
                source_sha256=pins, audit=audit)
