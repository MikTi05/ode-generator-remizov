"""Planning arithmetic/data contracts only; these tests run no ODE solver."""
import ast
import math
from pathlib import Path

import pytest

from diagnostics.plan_joint_benchmark import (
    ROOT, all_success_lower, family_audit, make_proposal, mc_cost,
    required_all_success, rounded_T,
)


def test_shared_horizon_preserves_tail_certificate_and_dyadic_steps():
    # Saved F0002 certificate; only scalar tail arithmetic, no trajectories.
    T = rounded_T(63.55987155784591)
    assert T == 63.625
    assert 380.9995260880843 * math.exp(-.2596238700079207*T) / .2596238700079207 < 1e-4
    assert mc_cost(T, 1024, 1)["steps"] == 8144


def test_mc_accounting_distinguishes_points_from_paths():
    one = mc_cost(63.625, 1024, 1)
    grid = mc_cost(63.625, 1024, 33)
    assert one["path_steps"] == 8_339_456
    assert grid["path_steps"] == 275_202_048
    assert grid["coefficient_values"] == 4*grid["path_steps"] + 2*33*1024
    assert grid["coefficient_calls"] == 33*(4*8144+2)


def test_eight_successes_do_not_establish_95_percent_reliability():
    assert .687 < all_success_lower(8) < .689
    assert .472 < all_success_lower(4) < .474
    assert required_all_success(.95) == 59
    assert all_success_lower(58) < .95 <= all_success_lower(59)
    assert required_all_success(.90) == 29


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_saved_horizon_rejected(bad):
    with pytest.raises(ValueError):
        rounded_T(bad)


def test_family_identity_and_exposure_are_retained():
    rows, audit = family_audit()
    assert len(rows) == 1020
    assert audit["small20"]["splits"] == {"validation": 10, "test": 10}
    assert audit["production1000"]["family_splits"] == {"train": 140, "validation": 30, "test": 30}
    assert audit["cross_dataset"] == dict(shared_ids=20, shared_expression_hashes=0,
                                        shared_structure_signatures=1, topology_split_conflicts=0)
    assert not any(r["pilot_training_eligible"] for r in rows)
    assert len({(r["dataset_sha256"], r["equation_id"], r["expression_hash"]) for r in rows}) == 1020


def test_proposal_cannot_authorize_execution_or_reuse_point_seeds():
    proposal=make_proposal([f"AMV1_F{i:04d}_S00" for i in range(20)], "test-fingerprint")
    assert proposal["execution_authorized"] is False
    assert proposal["executable_plan"] is False
    assert proposal["new_solver_runs"] == 0
    point, grid = proposal["phases"]
    assert point["target_x"] == [0.]
    assert grid["target_x"] == [-1+i/16 for i in range(33)]
    assert set(point["root_seeds"]).isdisjoint(grid["root_seeds"])
    assert proposal["pilot_labels_training_eligible"] is False


def test_planner_does_not_import_numerical_execution_infrastructure():
    tree=ast.parse((ROOT/"diagnostics/plan_joint_benchmark.py").read_text())
    imported=set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert imported <= {"__future__", "argparse", "collections", "csv", "hashlib", "json", "math", "pathlib", "statistics"}
