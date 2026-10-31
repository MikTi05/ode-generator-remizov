"""Deterministic generator for the ``all_methods_equations_v1`` dataset.

The construction in this module deliberately differs from the older general
IVP generator.  Every generated equation has a known compactly-supported exact
solution and satisfies the coefficient hypotheses needed by all three methods
currently in scope (RK4, the left-exponential product integral, and Remizov).

This module does not run any numerical ODE solver and does not create method
labels.  Its public entry points are :func:`generate_dataset`,
:func:`validate_record`, :func:`write_dataset`, and
:func:`validate_dataset_artifacts`.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence
import zipfile

import numpy as np

from .all_methods_expressions import (
    ALLOWED_BINARY_OPS,
    ALLOWED_PRIMITIVE_OPS,
    ALLOWED_UNARY_OPS,
    SafeExpressionNode,
    canonical_topology,
    evaluate_expression,
    evaluate_solution,
    formula_strings,
    serialize_expression,
    serialize_solution_expressions,
    topology_signature,
    tree_metrics,
)


GENERATION_VERSION = "all_methods_v1"
METADATA_SCHEMA_VERSION = "all_methods_metadata_v1"
COMPLEXITY_LEVELS = ("L1", "L2", "L3", "L4")
REGIMES = (
    "constant",
    "slow_varying",
    "moderate_varying",
    "oscillatory",
    "high_contrast",
)
CHANNEL_ORDER = ("a", "b", "c", "g")
CSV_FIELDS = (
    "equation_id",
    "a_expr",
    "b_expr",
    "c_expr",
    "g_expr",
    "x0",
    "x1",
    "y0",
    "v0",
)
OUTPUT_FILENAMES = {
    "csv": "all_methods_equations_v1.csv",
    "metadata": "all_methods_equations_v1_metadata.jsonl",
    "ml": "all_methods_equations_v1_ml.npz",
    "report": "all_methods_equations_v1_report.json",
}

_COMPLEXITY_SPECS: Mapping[str, tuple[int, int, int, int, float]] = {
    # minimum depth, maximum depth, minimum nodes, maximum nodes, max frequency
    "L1": (1, 2, 3, 7, 1.5),
    "L2": (2, 3, 6, 12, 3.0),
    "L3": (3, 4, 10, 18, 5.0),
    "L4": (4, 5, 14, 24, 8.0),
}


class GenerationError(RuntimeError):
    """Raised when a requested deterministic dataset cannot be constructed."""


class RecordValidationError(ValueError):
    """Raised when one generated sample violates a mathematical invariant."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class AllMethodsConfig:
    """Configuration for a balanced all-methods dataset.

    ``n_families`` must be divisible by the 20 complexity/regime cells.  The
    production defaults therefore create ten families per cell and five
    numerical parameterizations per family (1000 equations total).
    """

    seed: int = 2488
    n_families: int = 200
    samples_per_family: int = 5
    ml_grid_size: int = 256
    validation_grid_size: int = 2049
    max_attempts: int = 100
    max_abs_safety: float = 1.0e6
    a_min: float = 1.0e-3
    x0: float = -1.0
    x1: float = 1.0
    residual_rtol: float = 1.0e-10

    @property
    def master_seed(self) -> int:
        return self.seed

    @property
    def total_equations(self) -> int:
        return self.n_families * self.samples_per_family

    @property
    def families_per_cell(self) -> int:
        return self.n_families // (len(COMPLEXITY_LEVELS) * len(REGIMES))

    def validate(self) -> None:
        if not isinstance(self.seed, (int, np.integer)) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.n_families, int) or self.n_families <= 0:
            raise ValueError("n_families must be a positive integer")
        n_cells = len(COMPLEXITY_LEVELS) * len(REGIMES)
        if self.n_families % n_cells:
            raise ValueError(f"n_families must be divisible by {n_cells}")
        if not isinstance(self.samples_per_family, int) or self.samples_per_family <= 0:
            raise ValueError("samples_per_family must be a positive integer")
        if not isinstance(self.ml_grid_size, int) or self.ml_grid_size < 2:
            raise ValueError("ml_grid_size must be an integer >= 2")
        if not isinstance(self.validation_grid_size, int) or self.validation_grid_size < 17:
            raise ValueError("validation_grid_size must be an integer >= 17")
        if not isinstance(self.max_attempts, int) or self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        for name, value in (
            ("max_abs_safety", self.max_abs_safety),
            ("a_min", self.a_min),
            ("residual_rtol", self.residual_rtol),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not all(math.isfinite(float(v)) for v in (self.x0, self.x1)):
            raise ValueError("x0 and x1 must be finite")
        if (self.x0, self.x1) != (-1.0, 1.0):
            raise ValueError("all_methods_v1 has the fixed working interval [-1, 1]")
        if self.a_min > 0.5:
            raise ValueError("a_min is incompatible with the certified A_min range")


@dataclass(frozen=True)
class FamilyTopology:
    family_id: str
    family_index: int
    family_seed: int
    complexity_level: str
    regime: str
    split: str
    a_raw: SafeExpressionNode
    b_raw: SafeExpressionNode
    c_raw: SafeExpressionNode
    y_shape: SafeExpressionNode
    serialized_topology: Mapping[str, Any]
    structure_signature: str
    metrics: Mapping[str, Any]


@dataclass
class GeneratedEquation:
    """One accepted equation plus non-serialized generation state."""

    csv_row: Dict[str, Any]
    metadata: Dict[str, Any]
    concrete_nodes: Mapping[str, SafeExpressionNode]
    construction: Mapping[str, float]
    ml_channels: np.ndarray
    ml_y_exact: np.ndarray
    ml_dy_exact: np.ndarray
    validation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AllMethodsDataset:
    config: AllMethodsConfig
    records: list[GeneratedEquation]
    rejections: list[Dict[str, Any]]
    arrays: Dict[str, np.ndarray]
    report: Dict[str, Any]

    @property
    def equations(self) -> list[Dict[str, Any]]:
        return [record.csv_row for record in self.records]

    @property
    def metadata(self) -> list[Dict[str, Any]]:
        return [record.metadata for record in self.records]


def _stable_seed(master_seed: int, *path: int) -> int:
    """Derive a platform-stable uint64 seed from a NumPy SeedSequence path."""

    sequence = np.random.SeedSequence(int(master_seed), spawn_key=tuple(int(p) for p in path))
    state = sequence.generate_state(1, dtype=np.uint64)
    return int(state[0])


def _rng(master_seed: int, *path: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(int(master_seed), spawn_key=tuple(int(p) for p in path))
    return np.random.Generator(np.random.PCG64(sequence))


def _float_text(value: float) -> str:
    return format(float(value), ".17g")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _make_node(
    op: str,
    children: Sequence[SafeExpressionNode] = (),
    **params: float,
) -> SafeExpressionNode:
    return SafeExpressionNode(op=op, children=tuple(children), params=dict(params))


def _random_leaf(rng: np.random.Generator, *, allow_constant: bool = True) -> SafeExpressionNode:
    choices = sorted(
        name for name in ALLOWED_PRIMITIVE_OPS if allow_constant or name != "constant"
    )
    op = str(rng.choice(choices))
    if op == "constant":
        return _make_node(op, value=0.0)
    if op in {"sin_affine", "cos_affine", "tanh_affine"}:
        return _make_node(op, omega=1.0, phi=0.0)
    if op in {"gaussian", "lorentz"}:
        return _make_node(op, alpha=1.0, mu=0.0)
    if op == "saturating":
        return _make_node(op, mu=0.0)
    raise AssertionError(f"unhandled safe leaf operator {op!r}")


def _random_tree_at_depth(
    rng: np.random.Generator,
    depth: int,
    *,
    force_nonconstant: bool,
) -> SafeExpressionNode:
    if depth <= 1:
        return _random_leaf(rng, allow_constant=not force_nonconstant)
    choose_binary = bool(rng.random() < (0.45 if depth <= 2 else 0.58))
    if choose_binary:
        op = str(rng.choice(sorted(ALLOWED_BINARY_OPS)))
        deep_on_left = bool(rng.integers(0, 2))
        side_depth = int(rng.integers(1, depth))
        deep = _random_tree_at_depth(rng, depth - 1, force_nonconstant=force_nonconstant)
        side = _random_tree_at_depth(rng, side_depth, force_nonconstant=False)
        children = (deep, side) if deep_on_left else (side, deep)
        if op == "lincomb":
            return _make_node(op, children, alpha=1.0, beta=1.0)
        return _make_node(op, children)
    op = str(rng.choice(sorted(ALLOWED_UNARY_OPS)))
    child = _random_tree_at_depth(rng, depth - 1, force_nonconstant=force_nonconstant)
    return _make_node(op, (child,))


def _generate_tree_topology(
    rng: np.random.Generator,
    complexity_level: str,
    *,
    constant: bool = False,
) -> SafeExpressionNode:
    if constant:
        return _make_node("constant", value=0.0)
    min_depth, max_depth, min_nodes, max_nodes, _ = _COMPLEXITY_SPECS[complexity_level]
    for _ in range(2000):
        depth = int(rng.integers(min_depth, max_depth + 1))
        node = _random_tree_at_depth(rng, depth, force_nonconstant=True)
        metrics = tree_metrics(node)
        node_count = metrics.nodes
        actual_depth = metrics.depth
        if min_nodes <= node_count <= max_nodes and min_depth <= actual_depth <= max_depth:
            return node
    raise GenerationError(
        f"could not generate a {complexity_level} topology within depth/node limits"
    )


def _sample_signed_scale(
    rng: np.random.Generator,
    low: float,
    high: float,
) -> float:
    magnitude = float(rng.uniform(low, high))
    return magnitude if bool(rng.integers(0, 2)) else -magnitude


def _frequency_range(regime: str, level_cap: float) -> tuple[float, float]:
    if regime == "slow_varying":
        return 0.12, min(1.0, level_cap)
    if regime == "moderate_varying":
        return 0.45, min(3.0, level_cap)
    if regime == "oscillatory":
        return min(1.25, level_cap), level_cap
    if regime == "high_contrast":
        return 0.3, min(4.0, level_cap)
    return 0.2, level_cap


def _parameterize_tree(
    topology: SafeExpressionNode,
    rng: np.random.Generator,
    complexity_level: str,
    regime: str,
) -> SafeExpressionNode:
    _, _, _, _, level_cap = _COMPLEXITY_SPECS[complexity_level]
    op = topology.op
    children = tuple(
        _parameterize_tree(child, rng, complexity_level, regime)
        for child in topology.children
    )
    if op == "constant":
        return _make_node(op, value=float(rng.uniform(-1.5, 1.5)))
    if op in {"sin_affine", "cos_affine", "tanh_affine"}:
        low, high = _frequency_range(regime, level_cap)
        omega = _sample_signed_scale(rng, low, max(low, high))
        return _make_node(op, omega=omega, phi=float(rng.uniform(-math.pi, math.pi)))
    if op in {"gaussian", "lorentz"}:
        if regime == "slow_varying":
            alpha_range = (0.08, 0.7)
        elif regime == "oscillatory":
            alpha_range = (0.5, min(8.0, 1.0 + level_cap))
        elif regime == "high_contrast":
            alpha_range = (0.2, 5.0)
        else:
            alpha_range = (0.15, min(4.0, 0.7 + level_cap))
        return _make_node(
            op,
            alpha=float(rng.uniform(*alpha_range)),
            mu=float(rng.uniform(-1.5, 1.5)),
        )
    if op == "saturating":
        return _make_node(op, mu=float(rng.uniform(-1.5, 1.5)))
    if op == "lincomb":
        return _make_node(
            op,
            children,
            alpha=_sample_signed_scale(rng, 0.25, 1.6),
            beta=_sample_signed_scale(rng, 0.25, 1.6),
        )
    if op in set(ALLOWED_UNARY_OPS) | {"mul"}:
        return _make_node(op, children)
    raise AssertionError(f"unhandled safe operator {op!r}")


def _cell_split_assignments(
    count: int,
    cell_index: int,
    rng: np.random.Generator,
) -> list[str]:
    """Return deterministic, cell-stratified family split assignments.

    Production cells contain ten families.  Even cells use 7/2/1 and odd
    cells 7/1/2 for train/validation/test, giving exactly 70/15/15 overall.
    Smaller test configurations use the closest integer allocation and
    alternate the validation/test rounding remainder between cells.
    """

    if count == 10:
        split_counts = (7, 2, 1) if cell_index % 2 == 0 else (7, 1, 2)
    else:
        train = int(math.floor(0.70 * count))
        remaining = count - train
        if cell_index % 2 == 0:
            validation = (remaining + 1) // 2
        else:
            validation = remaining // 2
        split_counts = (train, validation, remaining - validation)
    labels = (
        ["train"] * split_counts[0]
        + ["validation"] * split_counts[1]
        + ["test"] * split_counts[2]
    )
    permutation = rng.permutation(count)
    result = [""] * count
    for label, position in zip(labels, permutation):
        result[int(position)] = label
    return result


def _topology_payload(trees: Mapping[str, SafeExpressionNode]) -> Dict[str, Any]:
    return {name: json.loads(canonical_topology(trees[name])) for name in sorted(trees)}


def _metrics_payload(trees: Mapping[str, SafeExpressionNode]) -> Dict[str, Any]:
    per_tree = {name: tree_metrics(trees[name]).to_dict() for name in sorted(trees)}
    return {
        "per_tree": per_tree,
        "total_nodes": sum(int(item["nodes"]) for item in per_tree.values()),
        "max_depth": max(int(item["depth"]) for item in per_tree.values()),
        "total_nonlinear_nodes": sum(
            int(item["nonlinear_nodes"]) for item in per_tree.values()
        ),
    }


def _generate_families(config: AllMethodsConfig) -> list[FamilyTopology]:
    families: list[FamilyTopology] = []
    signatures: set[str] = set()
    family_index = 0
    per_cell = config.families_per_cell
    for level_index, complexity in enumerate(COMPLEXITY_LEVELS):
        for regime_index, regime in enumerate(REGIMES):
            cell_index = level_index * len(REGIMES) + regime_index
            split_rng = _rng(config.seed, 2, cell_index)
            splits = _cell_split_assignments(per_cell, cell_index, split_rng)
            for local_index in range(per_cell):
                family_id = f"F{family_index:04d}"
                family_seed = _stable_seed(config.seed, 10, family_index)
                selected: tuple[
                    Mapping[str, SafeExpressionNode], Dict[str, Any], str, Dict[str, Any]
                ] | None = None
                for topology_attempt in range(config.max_attempts):
                    family_rng = _rng(config.seed, 11, family_index, topology_attempt)
                    coefficients_constant = regime == "constant"
                    trees = {
                        "a_raw": _generate_tree_topology(
                            family_rng, complexity, constant=coefficients_constant
                        ),
                        "b_raw": _generate_tree_topology(
                            family_rng, complexity, constant=coefficients_constant
                        ),
                        "c_raw": _generate_tree_topology(
                            family_rng, complexity, constant=coefficients_constant
                        ),
                        # The exact solution remains structurally rich even in
                        # the constant-coefficient regime.
                        "y_shape": _generate_tree_topology(family_rng, complexity),
                    }
                    signature = topology_signature(trees)
                    if signature in signatures:
                        continue
                    selected = (
                        trees,
                        _topology_payload(trees),
                        signature,
                        _metrics_payload(trees),
                    )
                    break
                if selected is None:
                    raise GenerationError(
                        f"{family_id}: could not create a unique structure signature "
                        f"after {config.max_attempts} attempts"
                    )
                trees, serialized_topology, signature, metrics = selected
                signatures.add(signature)
                families.append(
                    FamilyTopology(
                        family_id=family_id,
                        family_index=family_index,
                        family_seed=family_seed,
                        complexity_level=complexity,
                        regime=regime,
                        split=splits[local_index],
                        a_raw=trees["a_raw"],
                        b_raw=trees["b_raw"],
                        c_raw=trees["c_raw"],
                        y_shape=trees["y_shape"],
                        serialized_topology=serialized_topology,
                        structure_signature=signature,
                        metrics=metrics,
                    )
                )
                family_index += 1
    return families


def _sample_construction_parameters(
    rng: np.random.Generator,
    regime: str,
) -> Dict[str, float]:
    if regime == "high_contrast":
        a_lower = float(rng.uniform(0.5, 0.9))
        a_upper = float(rng.uniform(max(a_lower + 0.8, 2.0), 3.0))
        b_scale = float(rng.uniform(1.5, 5.0))
        c_scale = float(rng.uniform(1.5, 4.0))
        gap = float(rng.uniform(0.12, 1.8))
    elif regime == "slow_varying":
        a_lower = float(rng.uniform(0.8, 1.5))
        a_upper = float(rng.uniform(a_lower + 0.3, min(3.0, a_lower + 1.0)))
        b_scale = float(rng.uniform(0.15, 1.2))
        c_scale = float(rng.uniform(0.15, 1.3))
        gap = float(rng.uniform(0.25, 1.4))
    elif regime == "oscillatory":
        a_lower = float(rng.uniform(0.6, 1.4))
        a_upper = float(rng.uniform(a_lower + 0.4, min(3.0, a_lower + 1.5)))
        b_scale = float(rng.uniform(0.5, 2.8))
        c_scale = float(rng.uniform(0.4, 2.8))
        gap = float(rng.uniform(0.18, 1.5))
    elif regime == "moderate_varying":
        a_lower = float(rng.uniform(0.7, 1.5))
        a_upper = float(rng.uniform(a_lower + 0.3, min(3.0, a_lower + 1.3)))
        b_scale = float(rng.uniform(0.3, 2.0))
        c_scale = float(rng.uniform(0.25, 2.0))
        gap = float(rng.uniform(0.2, 1.5))
    elif regime == "constant":
        a_lower = float(rng.uniform(0.7, 1.5))
        a_upper = float(rng.uniform(a_lower + 0.3, min(3.0, a_lower + 1.2)))
        b_scale = float(rng.uniform(0.15, 1.8))
        c_scale = float(rng.uniform(0.15, 1.8))
        gap = float(rng.uniform(0.2, 1.5))
    else:  # pragma: no cover - regimes are checked by construction.
        raise AssertionError(regime)

    a_mid = 0.5 * (a_lower + a_upper)
    a_half = 0.5 * (a_upper - a_lower)
    bump_mu = float(rng.uniform(-0.25, 0.25))
    bump_radius = float(rng.uniform(1.5, 3.0))
    target_amplitude = float(rng.uniform(0.5, 1.5))
    return {
        "a_lower": a_lower,
        "a_upper": a_upper,
        "a_mid": a_mid,
        "a_half": a_half,
        "b_scale": b_scale,
        "c_scale": c_scale,
        "gap": gap,
        "lambda": c_scale + gap,
        "bump_mu": bump_mu,
        "bump_radius": bump_radius,
        "target_amplitude": target_amplitude,
    }


def _coefficient_values(
    nodes: Mapping[str, SafeExpressionNode],
    construction: Mapping[str, float],
    x: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_a = np.asarray(evaluate_expression(nodes["a_raw"], x), dtype=np.float64)
    raw_b = np.asarray(evaluate_expression(nodes["b_raw"], x), dtype=np.float64)
    raw_c = np.asarray(evaluate_expression(nodes["c_raw"], x), dtype=np.float64)
    with np.errstate(all="ignore"):
        a = construction["a_mid"] + construction["a_half"] * np.tanh(raw_a)
        b = construction["b_scale"] * np.tanh(raw_b)
        c_r = construction["c_scale"] * np.tanh(raw_c)
        c = c_r - construction["lambda"]
    return a, b, c_r, c


def _all_values(
    nodes: Mapping[str, SafeExpressionNode],
    construction: Mapping[str, float],
    x: np.ndarray | float,
) -> Dict[str, np.ndarray]:
    points = np.asarray(x, dtype=np.float64)
    a, b, c_r, c = _coefficient_values(nodes, construction, points)
    jet = evaluate_solution(
        nodes["y_shape"],
        construction["bump_mu"],
        construction["bump_radius"],
        construction["solution_scale"],
        points,
    )
    y = np.asarray(jet.value, dtype=np.float64)
    dy = np.asarray(jet.d1, dtype=np.float64)
    d2y = np.asarray(jet.d2, dtype=np.float64)
    with np.errstate(all="ignore"):
        g = a * d2y + b * dy + c * y
    return {
        "a": a,
        "b": b,
        "c_r": c_r,
        "c": c,
        "g": g,
        "g_r": -g,
        "y": y,
        "dy": dy,
        "d2y": d2y,
    }


def build_problem_formula_strings(
    nodes: Mapping[str, SafeExpressionNode],
    construction: Mapping[str, float],
) -> Dict[str, str]:
    """Derive the canonical persisted formulas from a concrete safe AST.

    Formula strings are descriptive interchange data; numerical evaluation
    uses the AST.  The loader and artifact validator call this same builder so
    a persisted formula can never silently describe a different equation.
    """

    raw_a = formula_strings(nodes["a_raw"])["value"]
    raw_b = formula_strings(nodes["b_raw"])["value"]
    raw_c = formula_strings(nodes["c_raw"])["value"]
    a = (
        f"(({_float_text(construction['a_mid'])})+"
        f"({_float_text(construction['a_half'])})*tanh({raw_a}))"
    )
    b = f"(({_float_text(construction['b_scale'])})*tanh({raw_b}))"
    c_r = f"(({_float_text(construction['c_scale'])})*tanh({raw_c}))"
    c = f"(({c_r})-({_float_text(construction['lambda'])}))"
    solution = serialize_solution_expressions(
        nodes["y_shape"],
        construction["bump_mu"],
        construction["bump_radius"],
        construction["solution_scale"],
    )
    g = (
        f"(({a})*({solution['d2y_exact_expr']})+"
        f"({b})*({solution['dy_exact_expr']})+"
        f"({c})*({solution['y_exact_expr']}))"
    )
    return {
        "a_expr": a,
        "b_expr": b,
        "c_expr": c,
        "g_expr": g,
        "remizov_c_expr": c_r,
        "remizov_g_expr": f"(-({g}))",
        **solution,
    }


def _finite_max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values, dtype=np.float64))))


def _range(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(np.min(array)), "max": float(np.max(array))}


def validate_record(
    record: GeneratedEquation,
    config: AllMethodsConfig | None = None,
    *,
    validation_grid: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Validate one in-memory record and return detailed diagnostics.

    Certified inequalities are checked from the stored construction first.
    Grid checks are deliberately treated only as implementation smoke checks,
    never as proofs of the global bounds.
    """

    cfg = config or AllMethodsConfig()
    cfg.validate()
    metadata = record.metadata
    construction = record.construction
    nodes = record.concrete_nodes
    if validation_grid is None:
        grid = np.linspace(cfg.x0, cfg.x1, cfg.validation_grid_size, dtype=np.float64)
    else:
        grid = np.asarray(validation_grid, dtype=np.float64)
        if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
            raise RecordValidationError("invalid_validation_grid", "grid must be finite and 1-D")

    a_lower = float(metadata["a_lower_cert"])
    a_upper = float(metadata["a_upper_cert"])
    b_abs = float(metadata["b_abs_cert"])
    c_abs = float(metadata["remizov_c_abs_cert"])
    gap = float(metadata["remizov_gap_cert"])
    lam = float(metadata["remizov_lambda"])
    if not (a_lower > 0.0 and a_lower >= cfg.a_min and a_upper >= a_lower):
        raise RecordValidationError("certificate_failure", "invalid certified a interval")
    if not (c_abs > 0.0 and gap > 0.0 and lam > c_abs):
        raise RecordValidationError("certificate_failure", "invalid Remizov certificate")
    if not math.isclose(lam - c_abs, gap, rel_tol=2.0e-15, abs_tol=2.0e-15):
        raise RecordValidationError("certificate_failure", "lambda-c_abs differs from gap")

    construction_fields = (
        "a_mid",
        "a_half",
        "b_scale",
        "c_scale",
        "lambda",
        "gap",
        "bump_mu",
        "bump_radius",
        "solution_scale",
        "target_amplitude",
    )
    try:
        parameters = {name: float(construction[name]) for name in construction_fields}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RecordValidationError(
            "certificate_failure", "missing or invalid construction parameter"
        ) from exc
    if not all(math.isfinite(value) for value in parameters.values()):
        raise RecordValidationError("certificate_failure", "non-finite construction parameter")

    def require_close(label: str, actual: float, expected: float) -> None:
        if not math.isclose(actual, expected, rel_tol=2.0e-13, abs_tol=2.0e-13):
            raise RecordValidationError(
                "certificate_failure",
                f"{label}: {actual:.17g} != {expected:.17g}",
            )

    require_close("a_lower", a_lower, parameters["a_mid"] - parameters["a_half"])
    require_close("a_upper", a_upper, parameters["a_mid"] + parameters["a_half"])
    require_close("b_abs", b_abs, parameters["b_scale"])
    require_close("c_R_abs", c_abs, parameters["c_scale"])
    require_close("lambda", lam, parameters["lambda"])
    require_close("gap", gap, parameters["gap"])
    require_close(
        "lambda construction",
        parameters["lambda"],
        parameters["c_scale"] + parameters["gap"],
    )
    range_tol = 2.0e-13
    if not (
        0.5 - range_tol <= a_lower <= 1.5 + range_tol
        and a_lower + 0.3 - range_tol <= a_upper <= 3.0 + range_tol
        and b_abs > 0.0
        and c_abs > 0.0
        and -0.25 - range_tol <= parameters["bump_mu"] <= 0.25 + range_tol
        and 1.5 - range_tol <= parameters["bump_radius"] <= 3.0 + range_tol
        and 0.5 - range_tol
        <= parameters["target_amplitude"]
        <= 1.5 + range_tol
        and parameters["a_half"] > 0.0
        and parameters["solution_scale"] > 0.0
    ):
        raise RecordValidationError(
            "certificate_failure", "construction parameters violate all_methods_v1 ranges"
        )

    try:
        values = _all_values(nodes, construction, grid)
    except Exception as exc:
        raise RecordValidationError("derivative_failure", str(exc)) from exc
    for name, array in values.items():
        array = np.asarray(array)
        if np.iscomplexobj(array) or not np.all(np.isfinite(array)):
            raise RecordValidationError("nonfinite_expression", f"{name} is non-finite/non-real")

    guarded_names = ("a", "b", "c", "g", "y", "dy", "d2y")
    maxima = {name: _finite_max_abs(values[name]) for name in guarded_names}
    safety_max = max(maxima.values())
    if safety_max > cfg.max_abs_safety:
        worst = max(maxima, key=maxima.__getitem__)
        raise RecordValidationError(
            "safety_bound_exceeded",
            f"max_abs({worst})={maxima[worst]:.17g} > {cfg.max_abs_safety:.17g}",
        )

    bound_tol = 5.0e-13 * max(1.0, a_upper, b_abs, c_abs, lam)
    if float(np.min(values["a"])) < a_lower - bound_tol:
        raise RecordValidationError("certificate_smoke_failure", "sampled a below lower certificate")
    if float(np.max(values["a"])) > a_upper + bound_tol:
        raise RecordValidationError("certificate_smoke_failure", "sampled a above upper certificate")
    if _finite_max_abs(values["b"]) > b_abs + bound_tol:
        raise RecordValidationError("certificate_smoke_failure", "sampled |b| exceeds certificate")
    if _finite_max_abs(values["c_r"]) > c_abs + bound_tol:
        raise RecordValidationError("certificate_smoke_failure", "sampled |c_R| exceeds certificate")
    if float(np.max(values["c"])) > -gap + bound_tol:
        raise RecordValidationError("certificate_smoke_failure", "sampled c exceeds -gap")

    c_identity = values["c"] - (values["c_r"] - lam)
    gr_identity = values["g_r"] + values["g"]
    if _finite_max_abs(c_identity) > 2.0e-14 * max(1.0, c_abs, lam):
        raise RecordValidationError("residual_failure", "c != c_R-lambda")
    if _finite_max_abs(gr_identity) > 0.0:
        raise RecordValidationError("residual_failure", "g_R != -g")

    residual = (
        values["a"] * values["d2y"]
        + values["b"] * values["dy"]
        + values["c"] * values["y"]
        - values["g"]
    )
    residual_scale = max(
        1.0,
        _finite_max_abs(values["a"] * values["d2y"]),
        _finite_max_abs(values["b"] * values["dy"]),
        _finite_max_abs(values["c"] * values["y"]),
        _finite_max_abs(values["g"]),
    )
    max_residual = _finite_max_abs(residual)
    if max_residual > cfg.residual_rtol * residual_scale:
        raise RecordValidationError(
            "residual_failure",
            f"max residual {max_residual:.17g}, scale {residual_scale:.17g}",
        )

    x0_values = _all_values(nodes, construction, np.asarray([cfg.x0]))
    y0_actual = float(x0_values["y"][0])
    v0_actual = float(x0_values["dy"][0])
    y0_stored = float(record.csv_row["y0"])
    v0_stored = float(record.csv_row["v0"])
    if not math.isclose(y0_stored, y0_actual, rel_tol=2.0e-13, abs_tol=2.0e-13):
        raise RecordValidationError("initial_condition_failure", "y0 != y_exact(x0)")
    if not math.isclose(v0_stored, v0_actual, rel_tol=2.0e-13, abs_tol=2.0e-13):
        raise RecordValidationError("initial_condition_failure", "v0 != dy_exact(x0)")
    if math.hypot(y0_actual, v0_actual) < 1.0e-8:
        raise RecordValidationError("initial_state_degenerate", "hypot(y0,v0) < 1e-8")

    support_left = float(metadata["support_left"])
    support_right = float(metadata["support_right"])
    require_close(
        "support_left",
        support_left,
        parameters["bump_mu"] - parameters["bump_radius"],
    )
    require_close(
        "support_right",
        support_right,
        parameters["bump_mu"] + parameters["bump_radius"],
    )
    if not (support_left < cfg.x0 and support_right > cfg.x1):
        raise RecordValidationError("support_failure", "support does not cover [-1,1]")
    outside = np.asarray(
        [support_left - 1.0, support_left, support_right, support_right + 1.0],
        dtype=np.float64,
    )
    outside_values = _all_values(nodes, construction, outside)
    for name in ("y", "dy", "d2y", "g", "g_r"):
        if _finite_max_abs(outside_values[name]) > 1.0e-14:
            raise RecordValidationError("support_failure", f"{name} nonzero outside support")
    epsilon = min(1.0e-6, 1.0e-6 * construction["bump_radius"])
    near_boundary = np.asarray(
        [support_left + epsilon, support_left, support_right, support_right - epsilon]
    )
    boundary_values = _all_values(nodes, construction, near_boundary)
    boundary_max = max(
        _finite_max_abs(boundary_values[name]) for name in ("y", "dy", "d2y", "g")
    )
    if boundary_max > 1.0e-8:
        raise RecordValidationError(
            "support_failure", f"boundary continuity smoke value {boundary_max:.17g}"
        )

    observed_amplitude = _finite_max_abs(values["y"])
    target_amplitude = float(metadata["target_amplitude"])
    require_close(
        "target_amplitude metadata",
        target_amplitude,
        parameters["target_amplitude"],
    )
    if not math.isclose(
        observed_amplitude, target_amplitude, rel_tol=5.0e-13, abs_tol=5.0e-13
    ):
        raise RecordValidationError(
            "normalization_failure",
            f"observed amplitude {observed_amplitude:.17g} != target {target_amplitude:.17g}",
        )

    diagnostics: Dict[str, Any] = {
        "max_residual": max_residual,
        "residual_scale": residual_scale,
        "max_abs": maxima,
        "sampled_ranges": {
            name: _range(values[name]) for name in ("a", "b", "c_r", "c", "g")
        },
        "observed_amplitude": observed_amplitude,
        "support_boundary_smoke_max": boundary_max,
    }
    record.validation = diagnostics
    return diagnostics


def _candidate_record(
    config: AllMethodsConfig,
    family: FamilyTopology,
    sample_index: int,
    attempt: int,
) -> GeneratedEquation:
    sample_seed = _stable_seed(config.seed, 20, family.family_index, sample_index)
    attempt_seed = _stable_seed(config.seed, 21, family.family_index, sample_index, attempt)
    rng = _rng(config.seed, 21, family.family_index, sample_index, attempt)
    parameter_regime = "moderate_varying" if family.regime == "constant" else family.regime
    nodes = {
        "a_raw": _parameterize_tree(
            family.a_raw, rng, family.complexity_level, family.regime
        ),
        "b_raw": _parameterize_tree(
            family.b_raw, rng, family.complexity_level, family.regime
        ),
        "c_raw": _parameterize_tree(
            family.c_raw, rng, family.complexity_level, family.regime
        ),
        "y_shape": _parameterize_tree(
            family.y_shape, rng, family.complexity_level, parameter_regime
        ),
    }
    construction = _sample_construction_parameters(rng, family.regime)
    normalization_grid = np.linspace(
        config.x0, config.x1, config.validation_grid_size, dtype=np.float64
    )
    try:
        raw_solution = evaluate_solution(
            nodes["y_shape"],
            construction["bump_mu"],
            construction["bump_radius"],
            1.0,
            normalization_grid,
        )
    except Exception as exc:
        raise RecordValidationError("derivative_failure", str(exc)) from exc
    raw_max = _finite_max_abs(np.asarray(raw_solution.value))
    if not math.isfinite(raw_max):
        raise RecordValidationError("nonfinite_expression", "raw solution is non-finite")
    if raw_max < 1.0e-3:
        raise RecordValidationError("near_zero_solution", f"raw max={raw_max:.17g}")
    construction["solution_scale"] = construction["target_amplitude"] / raw_max

    equation_id = f"AMV1_{family.family_id}_S{sample_index:02d}"
    try:
        formulas = build_problem_formula_strings(nodes, construction)
        serialized_ast = {
            "a_raw": serialize_expression(nodes["a_raw"]),
            "b_raw": serialize_expression(nodes["b_raw"]),
            "c_raw": serialize_expression(nodes["c_raw"]),
            "y_shape": serialize_expression(nodes["y_shape"]),
            "construction": {
                "a_mid": construction["a_mid"],
                "a_half": construction["a_half"],
                "b_scale": construction["b_scale"],
                "c_scale": construction["c_scale"],
                "lambda": construction["lambda"],
                "gap": construction["gap"],
                "bump_mu": construction["bump_mu"],
                "bump_radius": construction["bump_radius"],
                "solution_scale": construction["solution_scale"],
            },
        }
        # Exercise deterministic JSON serialization before accepting a sample.
        _canonical_json(serialized_ast)
    except Exception as exc:
        raise RecordValidationError("serialization_failure", str(exc)) from exc

    initial = evaluate_solution(
        nodes["y_shape"],
        construction["bump_mu"],
        construction["bump_radius"],
        construction["solution_scale"],
        np.asarray([config.x0]),
    )
    y0 = float(np.asarray(initial.value)[0])
    v0 = float(np.asarray(initial.d1)[0])
    csv_row: Dict[str, Any] = {
        "equation_id": equation_id,
        "a_expr": formulas["a_expr"],
        "b_expr": formulas["b_expr"],
        "c_expr": formulas["c_expr"],
        "g_expr": formulas["g_expr"],
        "x0": _float_text(config.x0),
        "x1": _float_text(config.x1),
        "y0": _float_text(y0),
        "v0": _float_text(v0),
    }
    support_left = construction["bump_mu"] - construction["bump_radius"]
    support_right = construction["bump_mu"] + construction["bump_radius"]
    metadata: Dict[str, Any] = {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "generation_version": GENERATION_VERSION,
        "equation_id": equation_id,
        "master_seed": config.seed,
        "family_id": family.family_id,
        "family_seed": family.family_seed,
        "sample_seed": sample_seed,
        "attempt_seed": attempt_seed,
        "sample_index": sample_index,
        "accepted_attempt": attempt + 1,
        "complexity_level": family.complexity_level,
        "regime": family.regime,
        "split": family.split,
        "structure_signature": family.structure_signature,
        "serialized_topology": family.serialized_topology,
        "tree_metrics": family.metrics,
        "serialized_ast": serialized_ast,
        "a_expr": formulas["a_expr"],
        "b_expr": formulas["b_expr"],
        "c_expr": formulas["c_expr"],
        "g_expr": formulas["g_expr"],
        "y_exact_expr": formulas["y_exact_expr"],
        "dy_exact_expr": formulas["dy_exact_expr"],
        "d2y_exact_expr": formulas["d2y_exact_expr"],
        "remizov_c_expr": formulas["remizov_c_expr"],
        "remizov_g_expr": formulas["remizov_g_expr"],
        "remizov_lambda": construction["lambda"],
        "a_lower_cert": construction["a_lower"],
        "a_upper_cert": construction["a_upper"],
        "b_abs_cert": construction["b_scale"],
        "remizov_c_abs_cert": construction["c_scale"],
        "remizov_gap_cert": construction["gap"],
        "support_left": support_left,
        "support_right": support_right,
        "target_amplitude": construction["target_amplitude"],
        "x0": config.x0,
        "x1": config.x1,
        "y0": y0,
        "v0": v0,
        "channel_order": list(CHANNEL_ORDER),
    }
    ml_grid = np.linspace(config.x0, config.x1, config.ml_grid_size, dtype=np.float64)
    ml_values = _all_values(nodes, construction, ml_grid)
    record = GeneratedEquation(
        csv_row=csv_row,
        metadata=metadata,
        concrete_nodes=nodes,
        construction=construction,
        ml_channels=np.stack([ml_values[name] for name in CHANNEL_ORDER]).astype(
            np.float64, copy=False
        ),
        ml_y_exact=np.asarray(ml_values["y"], dtype=np.float64),
        ml_dy_exact=np.asarray(ml_values["dy"], dtype=np.float64),
    )
    validate_record(record, config)
    record.metadata["validation"] = record.validation
    return record


def _unicode_array(values: Iterable[object]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max(1, *(len(value) for value in strings))
    return np.asarray(strings, dtype=f"<U{width}")


def _distribution(values: Sequence[float]) -> Dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: math.nan for key in ("min", "max", "mean", "std", "q05", "median", "q95")} | {"count": 0}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _build_arrays(records: Sequence[GeneratedEquation], config: AllMethodsConfig) -> Dict[str, np.ndarray]:
    return {
        "x_grid": np.linspace(config.x0, config.x1, config.ml_grid_size, dtype=np.float64),
        "X": np.stack([record.ml_channels for record in records]).astype(np.float64, copy=False),
        "y0": np.asarray([float(record.csv_row["y0"]) for record in records], dtype=np.float64),
        "v0": np.asarray([float(record.csv_row["v0"]) for record in records], dtype=np.float64),
        "y_exact": np.stack([record.ml_y_exact for record in records]).astype(
            np.float64, copy=False
        ),
        "dy_exact": np.stack([record.ml_dy_exact for record in records]).astype(
            np.float64, copy=False
        ),
        "equation_id": _unicode_array(record.metadata["equation_id"] for record in records),
        "family_id": _unicode_array(record.metadata["family_id"] for record in records),
        "complexity_level": _unicode_array(
            record.metadata["complexity_level"] for record in records
        ),
        "regime": _unicode_array(record.metadata["regime"] for record in records),
        "split": _unicode_array(record.metadata["split"] for record in records),
        "channel_order": _unicode_array(CHANNEL_ORDER),
    }


def _aggregate_sampled_range(
    records: Sequence[GeneratedEquation],
    name: str,
) -> Dict[str, float]:
    return {
        "min": min(float(record.validation["sampled_ranges"][name]["min"]) for record in records),
        "max": max(float(record.validation["sampled_ranges"][name]["max"]) for record in records),
    }


def summarize_dataset(
    records: Sequence[GeneratedEquation],
    rejections: Sequence[Mapping[str, Any]],
    config: AllMethodsConfig,
    arrays: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, Any]:
    """Create the deterministic report content, excluding output file hashes."""

    if not records:
        raise GenerationError("cannot summarize an empty dataset")
    metadata = [record.metadata for record in records]
    complexity_counts = Counter(str(item["complexity_level"]) for item in metadata)
    regime_counts = Counter(str(item["regime"]) for item in metadata)
    split_counts = Counter(str(item["split"]) for item in metadata)
    cell_counts = Counter(
        (str(item["complexity_level"]), str(item["regime"])) for item in metadata
    )
    rejection_counts = Counter(str(item["reason"]) for item in rejections)
    y0 = [float(item["y0"]) for item in metadata]
    v0 = [float(item["v0"]) for item in metadata]
    worst = max(records, key=lambda record: float(record.validation["max_residual"]))
    lambda_values = [float(item["remizov_lambda"]) for item in metadata]
    gap_values = [float(item["remizov_gap_cert"]) for item in metadata]
    a_lower_values = [float(item["a_lower_cert"]) for item in metadata]
    a_upper_values = [float(item["a_upper_cert"]) for item in metadata]
    b_cert_values = [float(item["b_abs_cert"]) for item in metadata]
    c_cert_values = [float(item["remizov_c_abs_cert"]) for item in metadata]
    array_map = arrays or _build_arrays(records, config)
    report: Dict[str, Any] = {
        "generation_version": GENERATION_VERSION,
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "master_seed": config.seed,
        "configuration": {
            "seed": config.seed,
            "n_families": config.n_families,
            "samples_per_family": config.samples_per_family,
            "ml_grid_size": config.ml_grid_size,
            "validation_grid_size": config.validation_grid_size,
            "max_attempts": config.max_attempts,
            "max_abs_safety": config.max_abs_safety,
            "a_min": config.a_min,
            "x0": config.x0,
            "x1": config.x1,
            "residual_rtol": config.residual_rtol,
        },
        "total_equations": len(records),
        "total_families": len({str(item["family_id"]) for item in metadata}),
        "samples_per_family": config.samples_per_family,
        "counts_by_complexity": {
            level: complexity_counts.get(level, 0) for level in COMPLEXITY_LEVELS
        },
        "counts_by_regime": {regime: regime_counts.get(regime, 0) for regime in REGIMES},
        "counts_by_split": {
            split: split_counts.get(split, 0) for split in ("train", "validation", "test")
        },
        "counts_by_complexity_regime": {
            f"{level}|{regime}": cell_counts.get((level, regime), 0)
            for level in COMPLEXITY_LEVELS
            for regime in REGIMES
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "total_rejections": len(rejections),
        "max_generation_attempts": max(int(item["accepted_attempt"]) for item in metadata),
        "y0_distribution": _distribution(y0),
        "v0_distribution": _distribution(v0),
        "certified_ranges": {
            "a_lower_cert": {"min": min(a_lower_values), "max": max(a_lower_values)},
            "a_upper_cert": {"min": min(a_upper_values), "max": max(a_upper_values)},
            "b_abs_cert": {"min": min(b_cert_values), "max": max(b_cert_values)},
            "remizov_c_abs_cert": {
                "min": min(c_cert_values),
                "max": max(c_cert_values),
            },
            "remizov_lambda": {"min": min(lambda_values), "max": max(lambda_values)},
            "remizov_gap_cert": {"min": min(gap_values), "max": max(gap_values)},
        },
        "coefficient_sampled_ranges": {
            name: _aggregate_sampled_range(records, name)
            for name in ("a", "b", "c_r", "c", "g")
        },
        "global_max_residual": float(worst.validation["max_residual"]),
        "worst_residual_equation_id": str(worst.metadata["equation_id"]),
        "max_ml_tensor_absolute_value": float(np.max(np.abs(array_map["X"]))),
        "exact_symbolic_duplicates": 0,
        "exact_feature_duplicates": 0,
        "structure_signature_count": len(
            {str(item["structure_signature"]) for item in metadata}
        ),
        "channel_order": list(CHANNEL_ORDER),
        "npz_shapes": {name: list(value.shape) for name, value in array_map.items()},
        "npz_dtypes": {name: str(value.dtype) for name, value in array_map.items()},
        "family_split_leakage": False,
        "output_sha256": {},
    }
    return report


def _validate_collection_invariants(
    records: Sequence[GeneratedEquation],
    families: Sequence[FamilyTopology],
    config: AllMethodsConfig,
) -> None:
    if len(records) != config.total_equations:
        raise GenerationError(
            f"expected {config.total_equations} records, generated {len(records)}"
        )
    ids = [str(record.metadata["equation_id"]) for record in records]
    sample_seeds = [int(record.metadata["sample_seed"]) for record in records]
    if len(ids) != len(set(ids)):
        raise GenerationError("duplicate equation_id after generation")
    if len(sample_seeds) != len(set(sample_seeds)):
        raise GenerationError("duplicate sample_seed after generation")
    family_counts = Counter(str(record.metadata["family_id"]) for record in records)
    if set(family_counts.values()) != {config.samples_per_family}:
        raise GenerationError("incorrect number of accepted samples in at least one family")
    if len(family_counts) != config.n_families:
        raise GenerationError("incorrect family count")

    family_splits: Dict[str, set[str]] = {}
    for record in records:
        family_splits.setdefault(str(record.metadata["family_id"]), set()).add(
            str(record.metadata["split"])
        )
    leaking = {family_id: splits for family_id, splits in family_splits.items() if len(splits) != 1}
    if leaking:
        raise GenerationError(f"family split leakage: {leaking}")

    expected_per_cell = config.families_per_cell * config.samples_per_family
    cell_counts = Counter(
        (str(record.metadata["complexity_level"]), str(record.metadata["regime"]))
        for record in records
    )
    for level in COMPLEXITY_LEVELS:
        for regime in REGIMES:
            if cell_counts[(level, regime)] != expected_per_cell:
                raise GenerationError(f"unbalanced cell {level}/{regime}")
    signatures = [family.structure_signature for family in families]
    if len(signatures) != len(set(signatures)):
        raise GenerationError("duplicate family structure signature")

    nontrivial_y0 = sum(abs(float(record.csv_row["y0"])) > 1.0e-12 for record in records)
    nontrivial_v0 = sum(abs(float(record.csv_row["v0"])) > 1.0e-12 for record in records)
    threshold = max(1, int(math.ceil(0.05 * len(records))))
    if nontrivial_y0 < threshold or nontrivial_v0 < threshold:
        raise GenerationError("initial-condition distribution is conceptually degenerate")


def generate_dataset(config: AllMethodsConfig | None = None) -> AllMethodsDataset:
    """Generate a complete balanced dataset entirely in memory.

    Retries are stateless: the RNG for one attempt is derived only from the
    master seed and ``(family_index, sample_index, attempt)``.  Rejecting one
    sample therefore cannot perturb any other family or sample stream.
    """

    cfg = config or AllMethodsConfig()
    cfg.validate()
    families = _generate_families(cfg)
    records: list[GeneratedEquation] = []
    rejections: list[Dict[str, Any]] = []
    expression_hashes: set[str] = set()
    feature_hashes: set[str] = set()

    for family in families:
        for sample_index in range(cfg.samples_per_family):
            accepted: GeneratedEquation | None = None
            last_detail = ""
            for attempt in range(cfg.max_attempts):
                attempt_seed = _stable_seed(
                    cfg.seed, 21, family.family_index, sample_index, attempt
                )
                try:
                    candidate = _candidate_record(cfg, family, sample_index, attempt)
                    expression_payload = [candidate.csv_row[field] for field in CSV_FIELDS[1:5]]
                    expression_hash = hashlib.sha256(
                        _canonical_json(expression_payload).encode("utf-8")
                    ).hexdigest()
                    feature_hash = hashlib.sha256(
                        np.ascontiguousarray(candidate.ml_channels).tobytes()
                    ).hexdigest()
                    if expression_hash in expression_hashes:
                        raise RecordValidationError(
                            "duplicate_expression", "exact symbolic duplicate"
                        )
                    if feature_hash in feature_hashes:
                        raise RecordValidationError(
                            "duplicate_features", "exact numerical feature duplicate"
                        )
                    expression_hashes.add(expression_hash)
                    feature_hashes.add(feature_hash)
                    candidate.metadata["expression_hash"] = expression_hash
                    candidate.metadata["feature_hash"] = feature_hash
                    accepted = candidate
                    break
                except RecordValidationError as exc:
                    reason, detail = exc.reason, exc.detail
                except Exception as exc:
                    reason, detail = "serialization_failure", f"{type(exc).__name__}: {exc}"
                last_detail = detail
                rejections.append(
                    {
                        "generation_version": GENERATION_VERSION,
                        "master_seed": cfg.seed,
                        "family_id": family.family_id,
                        "family_seed": family.family_seed,
                        "sample_index": sample_index,
                        "sample_seed": _stable_seed(
                            cfg.seed, 20, family.family_index, sample_index
                        ),
                        "attempt": attempt + 1,
                        "attempt_seed": attempt_seed,
                        "complexity_level": family.complexity_level,
                        "regime": family.regime,
                        "reason": reason,
                        "detail": detail,
                    }
                )
            if accepted is None:
                raise GenerationError(
                    f"{family.family_id}/sample {sample_index} exhausted "
                    f"{cfg.max_attempts} attempts; last detail: {last_detail}"
                )
            records.append(accepted)

    _validate_collection_invariants(records, families, cfg)
    arrays = _build_arrays(records, cfg)
    report = summarize_dataset(records, rejections, cfg, arrays)
    return AllMethodsDataset(
        config=cfg,
        records=records,
        rejections=rejections,
        arrays=arrays,
        report=report,
    )


# Verbose aliases are useful to callers that keep multiple generator versions.
generate_all_methods_dataset = generate_dataset
generate_all_methods_dataset_v1 = generate_dataset


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(CSV_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(CSV_FIELDS):
            raise GenerationError("CSV row does not have the exact nine-column schema")
        writer.writerow({field: row[field] for field in CSV_FIELDS})
    return stream.getvalue().encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (_canonical_json(row) + "\n").encode("utf-8") for row in rows
    )


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Build an NPZ whose member ordering and timestamps are reproducible."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise GenerationError(f"NPZ array {name!r} has forbidden object dtype")
            member = io.BytesIO()
            np.save(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return output.getvalue()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset(
    dataset: AllMethodsDataset,
    output_dir: str | Path = "data",
) -> Dict[str, Any]:
    """Atomically write CSV, metadata JSONL, deterministic NPZ, and report."""

    output = Path(output_dir)
    paths = {name: output / filename for name, filename in OUTPUT_FILENAMES.items()}
    _atomic_write_bytes(paths["csv"], _csv_bytes(dataset.equations))
    _atomic_write_bytes(paths["metadata"], _jsonl_bytes(dataset.metadata))
    _atomic_write_bytes(paths["ml"], _deterministic_npz_bytes(dataset.arrays))
    data_hashes = {
        paths[key].name: _sha256(paths[key]) for key in ("csv", "metadata", "ml")
    }
    dataset.report["output_sha256"] = data_hashes
    report_payload = (
        json.dumps(_jsonable(dataset.report), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(paths["report"], report_payload)
    all_hashes = dict(data_hashes)
    all_hashes[paths["report"].name] = _sha256(paths["report"])
    return {
        "paths": {key: str(path) for key, path in paths.items()},
        "sha256": all_hashes,
    }


write_all_methods_dataset = write_dataset
write_all_methods_dataset_v1 = write_dataset


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    def strict_object(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
        value: Dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise GenerationError(f"{path}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise GenerationError(f"{path}: non-finite JSON number {value}")

    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise GenerationError(f"{path}: blank JSONL line {line_number}")
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=strict_object,
                    parse_constant=reject_constant,
                )
            except GenerationError:
                raise
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise GenerationError(
                    f"{path}: invalid JSONL line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise GenerationError(f"{path}: JSONL line {line_number} is not an object")
            rows.append(value)
    return rows


_ARTIFACT_CONFIG_FIELDS = (
    "seed",
    "n_families",
    "samples_per_family",
    "ml_grid_size",
    "validation_grid_size",
    "max_attempts",
    "max_abs_safety",
    "a_min",
    "x0",
    "x1",
    "residual_rtol",
)
_ARTIFACT_AST_FIELDS = frozenset(
    {"a_raw", "b_raw", "c_raw", "y_shape", "construction"}
)
_ARTIFACT_CONSTRUCTION_FIELDS = frozenset(
    {
        "a_mid",
        "a_half",
        "b_scale",
        "c_scale",
        "lambda",
        "gap",
        "bump_mu",
        "bump_radius",
        "solution_scale",
    }
)
_ARTIFACT_METADATA_FIELDS = frozenset(
    {
        "metadata_schema_version",
        "generation_version",
        "equation_id",
        "master_seed",
        "family_id",
        "family_seed",
        "sample_seed",
        "attempt_seed",
        "sample_index",
        "accepted_attempt",
        "complexity_level",
        "regime",
        "split",
        "structure_signature",
        "serialized_topology",
        "tree_metrics",
        "serialized_ast",
        "a_expr",
        "b_expr",
        "c_expr",
        "g_expr",
        "y_exact_expr",
        "dy_exact_expr",
        "d2y_exact_expr",
        "remizov_c_expr",
        "remizov_g_expr",
        "remizov_lambda",
        "a_lower_cert",
        "a_upper_cert",
        "b_abs_cert",
        "remizov_c_abs_cert",
        "remizov_gap_cert",
        "support_left",
        "support_right",
        "target_amplitude",
        "x0",
        "x1",
        "y0",
        "v0",
        "channel_order",
        "validation",
        "expression_hash",
        "feature_hash",
    }
)


def _artifact_number(source: Mapping[str, Any], field: str, label: str) -> float:
    value = source.get(field)
    if isinstance(value, (bool, np.bool_)):
        raise GenerationError(f"{label}: {field} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GenerationError(f"{label}: {field} must be a finite real number") from exc
    if not math.isfinite(result):
        raise GenerationError(f"{label}: {field} must be finite")
    return result


def _artifact_integer(
    source: Mapping[str, Any], field: str, label: str, *, minimum: int = 0
) -> int:
    value = source.get(field)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise GenerationError(f"{label}: {field} must be an integer")
    result = int(value)
    if result < minimum:
        raise GenerationError(f"{label}: {field} must be >= {minimum}")
    return result


def _artifact_close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=2.0e-13, abs_tol=2.0e-13)


def _require_artifact_close(label: str, actual: float, expected: float) -> None:
    if not _artifact_close(actual, expected):
        raise GenerationError(
            f"{label}: found {actual:.17g}, expected {expected:.17g}"
        )


def _require_same_json(label: str, actual: Any, expected: Any) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise GenerationError(f"{label} mismatch")


def _decode_artifact_record(
    row: Mapping[str, str],
    metadata: Mapping[str, Any],
    config: AllMethodsConfig,
    ml_grid: np.ndarray,
) -> GeneratedEquation:
    """Reconstruct one record solely from persisted safe AST data."""

    equation_id = str(metadata.get("equation_id", ""))
    label = equation_id or "metadata row"
    missing = sorted(_ARTIFACT_METADATA_FIELDS - set(metadata))
    if missing:
        raise GenerationError(f"{label}: missing metadata fields {missing}")
    if not equation_id or equation_id != equation_id.strip():
        raise GenerationError("metadata equation_id must be a nonempty trimmed string")
    if row.get("equation_id") != equation_id:
        raise GenerationError(f"{equation_id}: CSV/metadata equation_id mismatch")
    if metadata.get("generation_version") != GENERATION_VERSION:
        raise GenerationError(f"{equation_id}: generation_version mismatch")
    if metadata.get("metadata_schema_version") != METADATA_SCHEMA_VERSION:
        raise GenerationError(f"{equation_id}: metadata schema version mismatch")
    if metadata.get("channel_order") != list(CHANNEL_ORDER):
        raise GenerationError(f"{equation_id}: metadata channel order mismatch")
    if _artifact_integer(metadata, "master_seed", label) != int(config.seed):
        raise GenerationError(f"{equation_id}: master_seed mismatch")

    level = metadata.get("complexity_level")
    regime = metadata.get("regime")
    split = metadata.get("split")
    if level not in COMPLEXITY_LEVELS:
        raise GenerationError(f"{equation_id}: invalid complexity_level {level!r}")
    if regime not in REGIMES:
        raise GenerationError(f"{equation_id}: invalid regime {regime!r}")
    if split not in {"train", "validation", "test"}:
        raise GenerationError(f"{equation_id}: invalid split {split!r}")

    ast_data = metadata.get("serialized_ast")
    if not isinstance(ast_data, Mapping) or set(ast_data) != _ARTIFACT_AST_FIELDS:
        raise GenerationError(f"{equation_id}: serialized_ast schema mismatch")
    try:
        nodes = {
            name: SafeExpressionNode.from_dict(ast_data[name])
            for name in ("a_raw", "b_raw", "c_raw", "y_shape")
        }
    except Exception as exc:
        raise GenerationError(f"{equation_id}: unsafe or invalid serialized AST: {exc}") from exc
    for name, node in nodes.items():
        _require_same_json(
            f"{equation_id}: serialized_ast.{name} round trip",
            ast_data[name],
            serialize_expression(node),
        )

    raw_construction = ast_data.get("construction")
    if not isinstance(raw_construction, Mapping) or set(raw_construction) != _ARTIFACT_CONSTRUCTION_FIELDS:
        raise GenerationError(f"{equation_id}: serialized construction schema mismatch")
    construction = {
        field: _artifact_number(raw_construction, field, label)
        for field in _ARTIFACT_CONSTRUCTION_FIELDS
    }
    construction.update(
        {
            "a_lower": _artifact_number(metadata, "a_lower_cert", label),
            "a_upper": _artifact_number(metadata, "a_upper_cert", label),
            "target_amplitude": _artifact_number(metadata, "target_amplitude", label),
        }
    )
    if construction["a_half"] <= 0.0:
        raise GenerationError(f"{equation_id}: a_half must be positive")
    if construction["b_scale"] <= 0.0 or construction["c_scale"] <= 0.0:
        raise GenerationError(f"{equation_id}: b_scale and c_scale must be positive")
    if construction["gap"] <= 0.0 or construction["bump_radius"] <= 0.0:
        raise GenerationError(f"{equation_id}: gap and bump_radius must be positive")
    if construction["solution_scale"] <= 0.0:
        raise GenerationError(f"{equation_id}: solution_scale must be positive")

    a_lower = construction["a_lower"]
    a_upper = construction["a_upper"]
    _require_artifact_close(
        f"{equation_id}: a_lower construction",
        a_lower,
        construction["a_mid"] - construction["a_half"],
    )
    _require_artifact_close(
        f"{equation_id}: a_upper construction",
        a_upper,
        construction["a_mid"] + construction["a_half"],
    )
    _require_artifact_close(
        f"{equation_id}: b_abs certificate",
        _artifact_number(metadata, "b_abs_cert", label),
        construction["b_scale"],
    )
    _require_artifact_close(
        f"{equation_id}: c_R certificate",
        _artifact_number(metadata, "remizov_c_abs_cert", label),
        construction["c_scale"],
    )
    _require_artifact_close(
        f"{equation_id}: lambda metadata",
        _artifact_number(metadata, "remizov_lambda", label),
        construction["lambda"],
    )
    _require_artifact_close(
        f"{equation_id}: gap metadata",
        _artifact_number(metadata, "remizov_gap_cert", label),
        construction["gap"],
    )
    _require_artifact_close(
        f"{equation_id}: lambda construction",
        construction["lambda"],
        construction["c_scale"] + construction["gap"],
    )
    _require_artifact_close(
        f"{equation_id}: support_left construction",
        _artifact_number(metadata, "support_left", label),
        construction["bump_mu"] - construction["bump_radius"],
    )
    _require_artifact_close(
        f"{equation_id}: support_right construction",
        _artifact_number(metadata, "support_right", label),
        construction["bump_mu"] + construction["bump_radius"],
    )
    range_tol = 2.0e-13
    if not (0.5 - range_tol <= a_lower <= 1.5 + range_tol):
        raise GenerationError(f"{equation_id}: a_lower_cert outside v1 range")
    if not (a_lower + 0.3 - range_tol <= a_upper <= 3.0 + range_tol):
        raise GenerationError(f"{equation_id}: a_upper_cert outside v1 range")
    if not (-0.25 - range_tol <= construction["bump_mu"] <= 0.25 + range_tol):
        raise GenerationError(f"{equation_id}: bump_mu outside v1 range")
    if not (1.5 - range_tol <= construction["bump_radius"] <= 3.0 + range_tol):
        raise GenerationError(f"{equation_id}: bump_radius outside v1 range")
    if not (0.5 - range_tol <= construction["target_amplitude"] <= 1.5 + range_tol):
        raise GenerationError(f"{equation_id}: target_amplitude outside v1 range")

    expected_topology = _topology_payload(nodes)
    expected_metrics = _metrics_payload(nodes)
    expected_signature = topology_signature(nodes)
    _require_same_json(
        f"{equation_id}: serialized_topology",
        metadata.get("serialized_topology"),
        expected_topology,
    )
    _require_same_json(
        f"{equation_id}: tree_metrics",
        metadata.get("tree_metrics"),
        expected_metrics,
    )
    if metadata.get("structure_signature") != expected_signature:
        raise GenerationError(f"{equation_id}: structure_signature mismatch")

    min_depth, max_depth, min_nodes, max_nodes, frequency_cap = _COMPLEXITY_SPECS[str(level)]
    for name, node in nodes.items():
        metrics = tree_metrics(node)
        constant_coefficient = regime == "constant" and name != "y_shape"
        if constant_coefficient:
            if node.op != "constant" or metrics.nodes != 1 or metrics.depth != 1:
                raise GenerationError(
                    f"{equation_id}: constant regime has nonconstant {name} topology"
                )
        elif not (
            min_depth <= metrics.depth <= max_depth
            and min_nodes <= metrics.nodes <= max_nodes
        ):
            raise GenerationError(
                f"{equation_id}: {name} depth/node metrics violate {level}"
            )
        parameter_regime = "moderate_varying" if regime == "constant" and name == "y_shape" else str(regime)
        for current in _walk_artifact_tree(node):
            if current.op in {"sin_affine", "cos_affine", "tanh_affine"}:
                omega = abs(float(current.params["omega"]))
                low, high = _frequency_range(parameter_regime, frequency_cap)
                if not (low - range_tol <= omega <= max(low, high) + range_tol):
                    raise GenerationError(
                        f"{equation_id}: {name} frequency outside {level}/{parameter_regime} range"
                    )
            elif current.op == "constant":
                if not (-1.5 - range_tol <= current.params["value"] <= 1.5 + range_tol):
                    raise GenerationError(f"{equation_id}: constant leaf outside range")
            elif current.op in {"gaussian", "lorentz"}:
                if not (-1.5 - range_tol <= current.params["mu"] <= 1.5 + range_tol):
                    raise GenerationError(f"{equation_id}: localized center outside range")
                if parameter_regime == "slow_varying":
                    alpha_low, alpha_high = 0.08, 0.7
                elif parameter_regime == "oscillatory":
                    alpha_low, alpha_high = 0.5, min(8.0, 1.0 + frequency_cap)
                elif parameter_regime == "high_contrast":
                    alpha_low, alpha_high = 0.2, 5.0
                else:
                    alpha_low, alpha_high = 0.15, min(4.0, 0.7 + frequency_cap)
                if not (
                    alpha_low - range_tol
                    <= current.params["alpha"]
                    <= alpha_high + range_tol
                ):
                    raise GenerationError(
                        f"{equation_id}: localized width outside {parameter_regime} range"
                    )
            elif current.op == "saturating":
                if not (-1.5 - range_tol <= current.params["mu"] <= 1.5 + range_tol):
                    raise GenerationError(f"{equation_id}: saturating center outside range")
            elif current.op == "lincomb":
                for parameter in ("alpha", "beta"):
                    magnitude = abs(float(current.params[parameter]))
                    if not (0.25 - range_tol <= magnitude <= 1.6 + range_tol):
                        raise GenerationError(f"{equation_id}: lincomb scale outside range")

    formulas = build_problem_formula_strings(nodes, construction)
    for field, expected in formulas.items():
        if metadata.get(field) != expected:
            raise GenerationError(f"{equation_id}: metadata {field} is not AST-derived")
        if field in {"a_expr", "b_expr", "c_expr", "g_expr"} and row.get(field) != expected:
            raise GenerationError(f"{equation_id}: CSV {field} is not AST-derived")
    for field in ("x0", "x1", "y0", "v0"):
        csv_value = _artifact_number(row, field, equation_id)
        metadata_value = _artifact_number(metadata, field, equation_id)
        _require_artifact_close(
            f"{equation_id}: CSV/metadata {field}", csv_value, metadata_value
        )
    _require_artifact_close(
        f"{equation_id}: x0", _artifact_number(metadata, "x0", label), config.x0
    )
    _require_artifact_close(
        f"{equation_id}: x1", _artifact_number(metadata, "x1", label), config.x1
    )

    normalization_grid = np.linspace(
        config.x0, config.x1, config.validation_grid_size, dtype=np.float64
    )
    raw_solution = evaluate_solution(
        nodes["y_shape"],
        construction["bump_mu"],
        construction["bump_radius"],
        1.0,
        normalization_grid,
    )
    raw_max = _finite_max_abs(np.asarray(raw_solution.value, dtype=np.float64))
    if raw_max < 1.0e-3:
        raise GenerationError(f"{equation_id}: raw manufactured solution is near zero")
    _require_artifact_close(
        f"{equation_id}: solution normalization scale",
        construction["solution_scale"],
        construction["target_amplitude"] / raw_max,
    )

    ml_values = _all_values(nodes, construction, ml_grid)
    record = GeneratedEquation(
        csv_row=dict(row),
        metadata=dict(metadata),
        concrete_nodes=nodes,
        construction=construction,
        ml_channels=np.stack([ml_values[name] for name in CHANNEL_ORDER]).astype(
            np.float64, copy=False
        ),
        ml_y_exact=np.asarray(ml_values["y"], dtype=np.float64),
        ml_dy_exact=np.asarray(ml_values["dy"], dtype=np.float64),
    )
    diagnostics = validate_record(record, config)
    _require_same_json(
        f"{equation_id}: stored validation diagnostics",
        metadata.get("validation"),
        diagnostics,
    )
    expression_payload = [row[field] for field in CSV_FIELDS[1:5]]
    expression_hash = hashlib.sha256(
        _canonical_json(expression_payload).encode("utf-8")
    ).hexdigest()
    feature_hash = hashlib.sha256(
        np.ascontiguousarray(record.ml_channels).tobytes()
    ).hexdigest()
    if metadata.get("expression_hash") != expression_hash:
        raise GenerationError(f"{equation_id}: expression_hash mismatch")
    if metadata.get("feature_hash") != feature_hash:
        raise GenerationError(f"{equation_id}: feature_hash mismatch")
    return record


def _walk_artifact_tree(node: SafeExpressionNode) -> Iterable[SafeExpressionNode]:
    yield node
    for child in node.children:
        yield from _walk_artifact_tree(child)


def _validate_wide_grid(record: GeneratedEquation, config: AllMethodsConfig) -> Dict[str, Any]:
    """Auxiliary global-behaviour smoke check; certificates remain constructive."""

    equation_id = str(record.metadata["equation_id"])
    grid = np.linspace(-10.0, 10.0, 1025, dtype=np.float64)
    values = _all_values(record.concrete_nodes, record.construction, grid)
    for name, value in values.items():
        array = np.asarray(value)
        if np.iscomplexobj(array) or not np.all(np.isfinite(array)):
            raise GenerationError(f"{equation_id}: wide-grid {name} is non-finite/non-real")
    guarded = ("a", "b", "c", "g", "y", "dy", "d2y")
    maxima = {name: _finite_max_abs(values[name]) for name in guarded}
    if max(maxima.values()) > config.max_abs_safety:
        raise GenerationError(f"{equation_id}: wide-grid emergency safety bound exceeded")
    a_lower = float(record.metadata["a_lower_cert"])
    a_upper = float(record.metadata["a_upper_cert"])
    b_abs = float(record.metadata["b_abs_cert"])
    c_abs = float(record.metadata["remizov_c_abs_cert"])
    lam = float(record.metadata["remizov_lambda"])
    tolerance = 5.0e-12 * max(1.0, a_upper, b_abs, c_abs, lam)
    if (
        np.any(values["a"] < a_lower - tolerance)
        or np.any(values["a"] > a_upper + tolerance)
        or np.any(np.abs(values["b"]) > b_abs + tolerance)
        or np.any(np.abs(values["c_r"]) > c_abs + tolerance)
    ):
        raise GenerationError(f"{equation_id}: wide-grid certificate smoke failure")
    if not np.array_equal(values["g_r"], -values["g"]):
        raise GenerationError(f"{equation_id}: wide-grid g_R=-g identity failure")
    if not np.allclose(values["c"], values["c_r"] - lam, rtol=0.0, atol=tolerance):
        raise GenerationError(f"{equation_id}: wide-grid c=c_R-lambda identity failure")
    support_left = float(record.metadata["support_left"])
    support_right = float(record.metadata["support_right"])
    outside = (grid <= support_left) | (grid >= support_right)
    for name in ("y", "dy", "d2y", "g", "g_r"):
        if not np.array_equal(np.asarray(values[name])[outside], np.zeros(np.count_nonzero(outside))):
            raise GenerationError(f"{equation_id}: wide-grid {name} is nonzero outside support")
    residual = (
        values["a"] * values["d2y"]
        + values["b"] * values["dy"]
        + values["c"] * values["y"]
        - values["g"]
    )
    scale = max(
        1.0,
        _finite_max_abs(values["a"] * values["d2y"]),
        _finite_max_abs(values["b"] * values["dy"]),
        _finite_max_abs(values["c"] * values["y"]),
        _finite_max_abs(values["g"]),
    )
    max_residual = _finite_max_abs(residual)
    if max_residual > config.residual_rtol * scale:
        raise GenerationError(f"{equation_id}: wide-grid residual failure")
    return {"grid_size": int(grid.size), "max_residual": max_residual, "max_abs": maxima}


def validate_dataset_artifacts(
    output_dir: str | Path = "data",
    *,
    expected_config: AllMethodsConfig | None = None,
) -> Dict[str, Any]:
    """Independently validate every written v1 artifact, without solver runs.

    ``expected_config`` closes the cardinality trust boundary for callers that
    require a particular build (the production caller passes the default
    200-family, five-sample configuration).  Small test builds may omit it;
    their complete configuration is then authenticated by deterministic
    regeneration and byte-level comparison.
    """

    output = Path(output_dir)
    paths = {name: output / filename for name, filename in OUTPUT_FILENAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise GenerationError(f"missing dataset artifacts: {', '.join(missing)}")

    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise GenerationError("dataset CSV does not have the exact nine-column schema")
        csv_rows = []
        for line_number, row in enumerate(reader, start=2):
            if None in row or set(row) != set(CSV_FIELDS):
                raise GenerationError(f"dataset CSV line {line_number} has extra cells")
            if any(row.get(field) in (None, "") for field in CSV_FIELDS):
                raise GenerationError(f"dataset CSV line {line_number} has an empty field")
            csv_rows.append(dict(row))

    metadata = _load_jsonl(paths["metadata"])

    def strict_report_object(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
        value: Dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise GenerationError(f"report contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_report_constant(value: str) -> None:
        raise GenerationError(f"report contains non-finite JSON number {value}")

    try:
        with paths["report"].open("r", encoding="utf-8") as handle:
            report = json.load(
                handle,
                object_pairs_hook=strict_report_object,
                parse_constant=reject_report_constant,
            )
    except GenerationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GenerationError(f"invalid report JSON: {exc}") from exc
    if not isinstance(report, Mapping):
        raise GenerationError("report must contain one JSON object")
    if report.get("generation_version") != GENERATION_VERSION:
        raise GenerationError("report generation_version mismatch")
    if report.get("metadata_schema_version") != METADATA_SCHEMA_VERSION:
        raise GenerationError("report metadata_schema_version mismatch")
    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping) or set(configuration) != set(_ARTIFACT_CONFIG_FIELDS):
        raise GenerationError("report configuration schema mismatch")
    for name in (
        "seed",
        "n_families",
        "samples_per_family",
        "ml_grid_size",
        "validation_grid_size",
        "max_attempts",
    ):
        _artifact_integer(configuration, name, "report configuration", minimum=0)
    for name in ("max_abs_safety", "a_min", "x0", "x1", "residual_rtol"):
        _artifact_number(configuration, name, "report configuration")
    try:
        config = AllMethodsConfig(
            **{name: configuration[name] for name in _ARTIFACT_CONFIG_FIELDS}
        )
        config.validate()
    except (TypeError, ValueError) as exc:
        raise GenerationError(f"invalid report configuration: {exc}") from exc
    if expected_config is not None:
        expected_config.validate()
        if config != expected_config:
            raise GenerationError(
                "artifact configuration does not match the caller's expected configuration"
            )

    expected_count = config.total_equations
    if len(csv_rows) != expected_count or len(metadata) != expected_count:
        raise GenerationError("artifact row count does not match authenticated configuration")
    csv_ids = [row["equation_id"] for row in csv_rows]
    metadata_ids = [str(item.get("equation_id", "")) for item in metadata]
    if csv_ids != metadata_ids or len(csv_ids) != len(set(csv_ids)):
        raise GenerationError("CSV/metadata equation IDs differ or are not unique")

    try:
        with np.load(paths["ml"], allow_pickle=False) as archive:
            member_names = list(archive.files)
            if len(member_names) != len(set(member_names)):
                raise GenerationError("NPZ contains duplicate member names")
            arrays = {name: np.array(archive[name], copy=True) for name in member_names}
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationError(f"cannot safely load NPZ: {exc}") from exc
    required_arrays = {
        "x_grid",
        "X",
        "y0",
        "v0",
        "y_exact",
        "dy_exact",
        "equation_id",
        "family_id",
        "complexity_level",
        "regime",
        "split",
        "channel_order",
    }
    if set(arrays) != required_arrays:
        raise GenerationError("NPZ member schema mismatch")
    expected_shapes = {
        "x_grid": (config.ml_grid_size,),
        "X": (expected_count, len(CHANNEL_ORDER), config.ml_grid_size),
        "y0": (expected_count,),
        "v0": (expected_count,),
        "y_exact": (expected_count, config.ml_grid_size),
        "dy_exact": (expected_count, config.ml_grid_size),
        "equation_id": (expected_count,),
        "family_id": (expected_count,),
        "complexity_level": (expected_count,),
        "regime": (expected_count,),
        "split": (expected_count,),
        "channel_order": (len(CHANNEL_ORDER),),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise GenerationError(f"NPZ {name} shape mismatch")
    for name in ("x_grid", "X", "y0", "v0", "y_exact", "dy_exact"):
        if arrays[name].dtype != np.dtype(np.float64):
            raise GenerationError(f"NPZ {name} must have exact float64 dtype")
        if not np.all(np.isfinite(arrays[name])):
            raise GenerationError(f"NPZ {name} contains a non-finite value")
    for name in ("equation_id", "family_id", "complexity_level", "regime", "split", "channel_order"):
        if arrays[name].dtype.kind != "U":
            raise GenerationError(f"NPZ {name} must use a Unicode dtype")
    expected_grid = np.linspace(config.x0, config.x1, config.ml_grid_size, dtype=np.float64)
    if not np.array_equal(arrays["x_grid"], expected_grid):
        raise GenerationError("NPZ x_grid is not the canonical v1 grid")
    if arrays["channel_order"].tolist() != list(CHANNEL_ORDER):
        raise GenerationError("NPZ channel order mismatch")

    records: list[GeneratedEquation] = []
    wide_diagnostics: list[Dict[str, Any]] = []
    expression_hashes: set[str] = set()
    feature_hashes: set[str] = set()
    family_records: Dict[str, list[GeneratedEquation]] = {}
    sample_seeds: set[int] = set()
    attempt_seeds: set[int] = set()
    sample_pairs: set[tuple[str, int]] = set()
    forbidden_label_fragments = ("best_method", "method_label", "solver_label", "target_method")

    for index, (row, item) in enumerate(zip(csv_rows, metadata)):
        if any(
            fragment in str(field).lower()
            for field in item
            for fragment in forbidden_label_fragments
        ):
            raise GenerationError(f"{item.get('equation_id', index)}: forbidden method-label field")
        record = _decode_artifact_record(row, item, config, expected_grid)
        wide_diagnostics.append(_validate_wide_grid(record, config))
        records.append(record)
        equation_id = str(item["equation_id"])
        family_id = str(item.get("family_id", ""))
        if not family_id.startswith("F") or not family_id[1:].isdigit():
            raise GenerationError(f"{equation_id}: invalid family_id")
        family_index = int(family_id[1:])
        if family_id != f"F{family_index:04d}":
            raise GenerationError(f"{equation_id}: noncanonical family_id")
        sample_index = _artifact_integer(item, "sample_index", equation_id)
        accepted_attempt = _artifact_integer(
            item, "accepted_attempt", equation_id, minimum=1
        )
        if sample_index >= config.samples_per_family:
            raise GenerationError(f"{equation_id}: sample_index out of range")
        if accepted_attempt > config.max_attempts:
            raise GenerationError(f"{equation_id}: accepted_attempt exceeds max_attempts")
        expected_id = f"AMV1_{family_id}_S{sample_index:02d}"
        if equation_id != expected_id:
            raise GenerationError(f"{equation_id}: ID is inconsistent with family/sample")
        family_seed = _artifact_integer(item, "family_seed", equation_id)
        sample_seed = _artifact_integer(item, "sample_seed", equation_id)
        attempt_seed = _artifact_integer(item, "attempt_seed", equation_id)
        if family_seed != _stable_seed(config.seed, 10, family_index):
            raise GenerationError(f"{equation_id}: family_seed mismatch")
        if sample_seed != _stable_seed(config.seed, 20, family_index, sample_index):
            raise GenerationError(f"{equation_id}: sample_seed mismatch")
        if attempt_seed != _stable_seed(
            config.seed, 21, family_index, sample_index, accepted_attempt - 1
        ):
            raise GenerationError(f"{equation_id}: attempt_seed mismatch")
        if sample_seed in sample_seeds or attempt_seed in attempt_seeds:
            raise GenerationError(f"{equation_id}: duplicate deterministic seed")
        sample_seeds.add(sample_seed)
        attempt_seeds.add(attempt_seed)
        pair = (family_id, sample_index)
        if pair in sample_pairs:
            raise GenerationError(f"{equation_id}: duplicate family/sample pair")
        sample_pairs.add(pair)
        family_records.setdefault(family_id, []).append(record)
        expression_hash = str(item["expression_hash"])
        feature_hash = str(item["feature_hash"])
        if expression_hash in expression_hashes:
            raise GenerationError(f"{equation_id}: exact symbolic duplicate")
        if feature_hash in feature_hashes:
            raise GenerationError(f"{equation_id}: exact numerical feature duplicate")
        expression_hashes.add(expression_hash)
        feature_hashes.add(feature_hash)

        reconstructed = {
            "X": record.ml_channels,
            "y0": np.asarray(float(record.csv_row["y0"])),
            "v0": np.asarray(float(record.csv_row["v0"])),
            "y_exact": record.ml_y_exact,
            "dy_exact": record.ml_dy_exact,
        }
        for name, expected in reconstructed.items():
            actual = arrays[name][index]
            if not np.array_equal(actual, expected):
                raise GenerationError(f"{equation_id}: NPZ {name} reconstruction mismatch")
        for name in ("equation_id", "family_id", "complexity_level", "regime", "split"):
            if str(arrays[name][index]) != str(item[name]):
                raise GenerationError(f"{equation_id}: NPZ {name} metadata mismatch")

    if set(int(family_id[1:]) for family_id in family_records) != set(range(config.n_families)):
        raise GenerationError("family IDs are not the canonical contiguous range")
    canonical_ids = [
        f"AMV1_F{family_index:04d}_S{sample_index:02d}"
        for family_index in range(config.n_families)
        for sample_index in range(config.samples_per_family)
    ]
    if csv_ids != canonical_ids:
        raise GenerationError("artifact rows are not in canonical family/sample order")

    family_signatures: set[str] = set()
    family_splits: Dict[str, str] = {}
    cell_family_counts: Counter[tuple[str, str]] = Counter()
    cell_equation_counts: Counter[tuple[str, str]] = Counter()
    split_family_counts: Counter[str] = Counter()
    split_equation_counts: Counter[str] = Counter()
    for family_id, grouped in family_records.items():
        if len(grouped) != config.samples_per_family:
            raise GenerationError(f"{family_id}: incorrect sample count")
        if {int(record.metadata["sample_index"]) for record in grouped} != set(
            range(config.samples_per_family)
        ):
            raise GenerationError(f"{family_id}: incomplete sample-index set")
        contracts = {
            (
                str(record.metadata["complexity_level"]),
                str(record.metadata["regime"]),
                str(record.metadata["split"]),
                str(record.metadata["structure_signature"]),
                _canonical_json(record.metadata["serialized_topology"]),
                _canonical_json(record.metadata["tree_metrics"]),
            )
            for record in grouped
        }
        if len(contracts) != 1:
            raise GenerationError(f"{family_id}: topology or family metadata varies across samples")
        level, regime, split, signature, _, _ = next(iter(contracts))
        if signature in family_signatures:
            raise GenerationError(f"{family_id}: structure signature duplicates another family")
        family_signatures.add(signature)
        family_splits[family_id] = split
        cell_family_counts[(level, regime)] += 1
        cell_equation_counts[(level, regime)] += len(grouped)
        split_family_counts[split] += 1
        split_equation_counts[split] += len(grouped)

        family_index = int(family_id[1:])
        cell_index = family_index // config.families_per_cell
        local_index = family_index % config.families_per_cell
        expected_level = COMPLEXITY_LEVELS[cell_index // len(REGIMES)]
        expected_regime = REGIMES[cell_index % len(REGIMES)]
        if (level, regime) != (expected_level, expected_regime):
            raise GenerationError(f"{family_id}: noncanonical complexity/regime cell")
        expected_splits = _cell_split_assignments(
            config.families_per_cell,
            cell_index,
            _rng(config.seed, 2, cell_index),
        )
        if split != expected_splits[local_index]:
            raise GenerationError(f"{family_id}: split does not match stateless seed plan")

    expected_per_cell_families = config.families_per_cell
    expected_per_cell_equations = expected_per_cell_families * config.samples_per_family
    for level in COMPLEXITY_LEVELS:
        for regime in REGIMES:
            if cell_family_counts[(level, regime)] != expected_per_cell_families:
                raise GenerationError(f"unbalanced family cell {level}/{regime}")
            if cell_equation_counts[(level, regime)] != expected_per_cell_equations:
                raise GenerationError(f"unbalanced equation cell {level}/{regime}")
    if any(len({record.metadata["split"] for record in grouped}) != 1 for grouped in family_records.values()):
        raise GenerationError("family leakage across splits")
    nontrivial_threshold = max(1, int(math.ceil(0.05 * expected_count)))
    if sum(abs(float(record.csv_row["y0"])) > 1.0e-12 for record in records) < nontrivial_threshold:
        raise GenerationError("y0 distribution is conceptually degenerate")
    if sum(abs(float(record.csv_row["v0"])) > 1.0e-12 for record in records) < nontrivial_threshold:
        raise GenerationError("v0 distribution is conceptually degenerate")

    reconstructed_arrays = _build_arrays(records, config)
    for name, expected in reconstructed_arrays.items():
        if not np.array_equal(arrays[name], expected):
            raise GenerationError(f"full NPZ {name} reconstruction mismatch")

    rejection_counts = report.get("rejection_counts")
    if not isinstance(rejection_counts, Mapping):
        raise GenerationError("report rejection_counts must be an object")
    normalized_rejection_counts: Dict[str, int] = {}
    for reason, count in rejection_counts.items():
        if not isinstance(reason, str) or not reason:
            raise GenerationError("report has an invalid rejection reason")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise GenerationError(f"report rejection count for {reason!r} is invalid")
        normalized_rejection_counts[reason] = count
    inferred_rejections = sum(int(item["accepted_attempt"]) - 1 for item in metadata)
    if sum(normalized_rejection_counts.values()) != inferred_rejections:
        raise GenerationError("report rejection counts disagree with accepted attempts")
    if report.get("total_rejections") != inferred_rejections:
        raise GenerationError("report total_rejections mismatch")

    actual_data_hashes = {
        paths[key].name: _sha256(paths[key]) for key in ("csv", "metadata", "ml")
    }
    if report.get("output_sha256") != actual_data_hashes:
        raise GenerationError("report output SHA-256 mapping mismatch")

    # Recompute every reported count/statistic from reconstructed functions.
    # Rejection reasons themselves are authenticated by the deterministic
    # regeneration below; here only their persisted counts are used.
    synthetic_rejections = [
        {"reason": reason}
        for reason, count in sorted(normalized_rejection_counts.items())
        for _ in range(count)
    ]
    expected_report = summarize_dataset(
        records, synthetic_rejections, config, reconstructed_arrays
    )
    expected_report["output_sha256"] = actual_data_hashes
    _require_same_json("dataset report", report, expected_report)

    # Stateless regeneration authenticates report cardinality, rejection
    # reasons, accepted attempt numbers, all formulas, and all numerical bytes
    # against the master seed instead of trusting self-reported metadata.
    regenerated = generate_dataset(config)
    regenerated_payloads = {
        "csv": _csv_bytes(regenerated.equations),
        "metadata": _jsonl_bytes(regenerated.metadata),
        "ml": _deterministic_npz_bytes(regenerated.arrays),
    }
    for key, payload in regenerated_payloads.items():
        expected_digest = hashlib.sha256(payload).hexdigest()
        if actual_data_hashes[paths[key].name] != expected_digest:
            raise GenerationError(
                f"{paths[key].name} is not reproducible from the authenticated seed/config"
            )
    regenerated_report = dict(regenerated.report)
    regenerated_report["output_sha256"] = actual_data_hashes
    _require_same_json("deterministically regenerated report", report, regenerated_report)
    canonical_report_payload = (
        json.dumps(
            _jsonable(regenerated_report),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if hashlib.sha256(canonical_report_payload).hexdigest() != _sha256(paths["report"]):
        raise GenerationError("report bytes are not in canonical reproducible form")

    # Exercise the public safe loader on the real pair.  Loading all rows is
    # intentional: its dataset-level checks run once, then compatibility is
    # explicitly sampled at the first, middle, and last records.
    try:
        from all_methods_dataset_loader import load_all_methods_equations

        loaded = load_all_methods_equations(paths["csv"], paths["metadata"])
    except Exception as exc:
        raise GenerationError(f"all-methods loader compatibility failure: {exc}") from exc
    if len(loaded) != expected_count:
        raise GenerationError("all-methods loader returned the wrong record count")
    probe = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    for index in sorted({0, expected_count // 2, expected_count - 1}):
        equation = loaded[index]
        record = records[index]
        expected = _all_values(record.concrete_nodes, record.construction, probe)
        if equation.equation_id != record.metadata["equation_id"]:
            raise GenerationError("loader changed equation order")
        ordinary_functions = {
            "a": equation.cauchy_problem.a,
            "b": equation.cauchy_problem.b,
            "c": equation.cauchy_problem.c,
            "g": equation.cauchy_problem.g,
        }
        for name, function in ordinary_functions.items():
            actual = np.asarray([function(float(x)) for x in probe], dtype=np.float64)
            if not np.allclose(actual, expected[name], rtol=2.0e-13, atol=2.0e-13):
                raise GenerationError(f"{equation.equation_id}: loader {name} mismatch")
        if not np.allclose(
            equation.remizov_problem.c_remizov(probe),
            expected["c_r"],
            rtol=2.0e-13,
            atol=2.0e-13,
        ):
            raise GenerationError(f"{equation.equation_id}: loader c_R mismatch")
        if not np.array_equal(equation.remizov_problem.g_remizov(probe), expected["g_r"]):
            raise GenerationError(f"{equation.equation_id}: loader g_R mismatch")
        for name, function in (
            ("y", equation.y_exact),
            ("dy", equation.dy_exact),
            ("d2y", equation.d2y_exact),
        ):
            if not np.allclose(function(probe), expected[name], rtol=2.0e-13, atol=2.0e-13):
                raise GenerationError(f"{equation.equation_id}: loader {name} mismatch")

    all_hashes = dict(actual_data_hashes)
    all_hashes[paths["report"].name] = _sha256(paths["report"])
    return {
        "valid": True,
        "generation_version": GENERATION_VERSION,
        "master_seed": config.seed,
        "total_equations": expected_count,
        "total_families": len(family_records),
        "samples_per_family": config.samples_per_family,
        "counts_by_split": dict(sorted(split_equation_counts.items())),
        "family_split_leakage": False,
        "structure_signature_count": len(family_signatures),
        "wide_grid_size": 1025,
        "wide_grid_max_residual": max(
            float(item["max_residual"]) for item in wide_diagnostics
        ),
        "loader_compatibility_checked_ids": [
            csv_ids[index] for index in sorted({0, expected_count // 2, expected_count - 1})
        ],
        "npz_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "npz_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "reproducible_from_seed": True,
        "sha256": all_hashes,
    }


__all__ = [
    "AllMethodsConfig",
    "AllMethodsDataset",
    "CHANNEL_ORDER",
    "COMPLEXITY_LEVELS",
    "CSV_FIELDS",
    "FamilyTopology",
    "GENERATION_VERSION",
    "GeneratedEquation",
    "GenerationError",
    "METADATA_SCHEMA_VERSION",
    "OUTPUT_FILENAMES",
    "REGIMES",
    "RecordValidationError",
    "build_problem_formula_strings",
    "generate_all_methods_dataset",
    "generate_all_methods_dataset_v1",
    "generate_dataset",
    "summarize_dataset",
    "validate_dataset_artifacts",
    "validate_record",
    "write_all_methods_dataset",
    "write_all_methods_dataset_v1",
    "write_dataset",
]
