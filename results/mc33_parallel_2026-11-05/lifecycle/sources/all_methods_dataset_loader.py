"""Load the versioned dataset shared by all three implemented methods.

The CSV expressions are human-readable records.  Executable functions are
reconstructed exclusively from the whitelisted serialized AST stored in the
matching JSONL metadata file; expression strings are never passed to ``eval``
or ``sympy.sympify``.  A single evaluator bundle supplies both the ordinary
IVP representation and the Remizov representation, so the identities
``c = c_R - lambda`` and ``g_R = -g`` hold by construction after loading.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.problem_types import CauchyProblem
from remizov_solver import RemizovProblem
from src.all_methods_expressions import (
    SafeExpressionNode,
    canonical_topology,
    evaluate_expression,
    evaluate_solution,
    serialize_expression,
    topology_signature,
)
from src.all_methods_dataset_v1 import build_problem_formula_strings


DEFAULT_CSV_PATH = Path("data/all_methods_equations_v1.csv")
DEFAULT_METADATA_PATH = Path("data/all_methods_equations_v1_metadata.jsonl")
GENERATION_VERSION = "all_methods_v1"
METADATA_SCHEMA_VERSION = "all_methods_metadata_v1"

CSV_FIELDS: Tuple[str, ...] = (
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

_EXPRESSION_FIELDS: Tuple[str, ...] = (
    "a_expr",
    "b_expr",
    "c_expr",
    "g_expr",
)

_METADATA_REQUIRED_FIELDS: Tuple[str, ...] = (
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
    "generation_version",
    "metadata_schema_version",
    "serialized_ast",
)

_SERIALIZED_AST_FIELDS = frozenset(
    {"a_raw", "b_raw", "c_raw", "y_shape", "construction"}
)
_CONSTRUCTION_FIELDS = frozenset(
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
_COMPLEXITY_LEVELS = frozenset({"L1", "L2", "L3", "L4"})
_REGIMES = frozenset(
    {"constant", "slow_varying", "moderate_varying", "oscillatory", "high_contrast"}
)
_SPLITS = frozenset({"train", "validation", "test"})
_CHANNEL_ORDER = ("a", "b", "c", "g")
_WORKING_X0 = -1.0
_WORKING_X1 = 1.0
_A_MIN = 1.0e-3
_TARGET_AMPLITUDE_MIN = 0.5
_TARGET_AMPLITUDE_MAX = 1.5
_FAMILY_ID_PATTERN = re.compile(r"F(?P<family_index>[0-9]{4})")
_EQUATION_ID_PATTERN = re.compile(
    r"AMV1_(?P<family_id>F[0-9]{4})_S(?P<sample_index>[0-9]{2})"
)


class AllMethodsDatasetFormatError(ValueError):
    """Raised when the v1 CSV/JSONL pair is unsafe or internally inconsistent."""


class AllMethodsEvaluationError(ValueError):
    """Raised when a reconstructed expression is evaluated on invalid data."""


@dataclass(frozen=True)
class AllMethodsEquation:
    """The two solver representations and exact manufactured solution."""

    cauchy_problem: CauchyProblem
    remizov_problem: RemizovProblem
    y_exact: Callable[[Any], Any]
    dy_exact: Callable[[Any], Any]
    d2y_exact: Callable[[Any], Any]
    metadata: Mapping[str, Any]

    @property
    def equation_id(self) -> str:
        """Return the shared identifier of both solver representations."""

        return self.cauchy_problem.equation_id


def _format_error(equation_id: str, message: str) -> AllMethodsDatasetFormatError:
    return AllMethodsDatasetFormatError(f"{equation_id}: {message}")


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_object(text: str, path: Path, line_number: int) -> Dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise AllMethodsDatasetFormatError(
            f"{path}: JSONL line {line_number}: invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AllMethodsDatasetFormatError(
            f"{path}: JSONL line {line_number}: expected a JSON object"
        )
    return value


def _required_string(
    source: Mapping[str, Any], field: str, equation_id: str
) -> str:
    value = source.get(field)
    if not isinstance(value, str) or not value or value != value.strip():
        raise _format_error(equation_id, f"field {field!r} must be a nonempty trimmed string")
    return value


def _finite_number(
    source: Mapping[str, Any], field: str, equation_id: str
) -> float:
    value = source.get(field)
    if isinstance(value, bool):
        raise _format_error(equation_id, f"field {field!r} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _format_error(
            equation_id, f"field {field!r} must be a finite real number"
        ) from exc
    if not math.isfinite(result):
        raise _format_error(equation_id, f"field {field!r} is non-finite")
    return result


def _integer(
    source: Mapping[str, Any], field: str, equation_id: str, *, minimum: int = 0
) -> int:
    value = source.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _format_error(
            equation_id, f"field {field!r} must be an integer >= {minimum}"
        )
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=2.0e-13, abs_tol=2.0e-13)


def _canonical_json(value: Any) -> str:
    """Serialize schema data canonically and reject non-standard JSON numbers."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AllMethodsDatasetFormatError(
            f"metadata value is not canonical finite JSON: {exc}"
        ) from exc


def _stable_seed(master_seed: int, *path: int) -> int:
    """Mirror the documented stateless SeedSequence address calculation."""

    sequence = np.random.SeedSequence(
        int(master_seed), spawn_key=tuple(int(item) for item in path)
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _require_close(
    equation_id: str, label: str, actual: float, expected: float
) -> None:
    if not _close(actual, expected):
        raise _format_error(
            equation_id,
            f"{label} mismatch: found {actual!r}, expected {expected!r}",
        )


def _real_coordinates(points: Any, label: str) -> Tuple[np.ndarray, bool]:
    raw = np.asarray(points)
    if np.iscomplexobj(raw):
        raise AllMethodsEvaluationError(f"{label}: coordinates must be real")
    try:
        coordinates = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AllMethodsEvaluationError(f"{label}: coordinates must be real numeric values") from exc
    if not np.all(np.isfinite(coordinates)):
        raise AllMethodsEvaluationError(f"{label}: coordinates must be finite")
    return coordinates, coordinates.ndim == 0


def _real_values(raw: Any, shape: Tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(raw)
    if np.iscomplexobj(array):
        # Exact zero imaginary parts are harmless representations of real
        # numbers; any actual non-real value is rejected rather than cast.
        if not np.all(np.imag(array) == 0.0):
            raise AllMethodsEvaluationError(f"{label}: expression returned complex values")
        array = np.real(array)
    try:
        numeric = np.asarray(array, dtype=np.float64)
        if numeric.ndim == 0:
            values = np.full(shape, float(numeric), dtype=np.float64)
        else:
            values = np.broadcast_to(numeric, shape).astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AllMethodsEvaluationError(
            f"{label}: expression result is not real and shape-compatible"
        ) from exc
    if not np.all(np.isfinite(values)):
        raise AllMethodsEvaluationError(f"{label}: expression returned non-finite values")
    return values


def _shape_preserving(
    function: Callable[[np.ndarray], Any], equation_id: str, role: str
) -> Callable[[Any], Any]:
    label = f"{equation_id}:{role}"

    def evaluate(points: Any) -> Any:
        coordinates, scalar = _real_coordinates(points, label)
        try:
            raw = function(coordinates)
        except AllMethodsEvaluationError:
            raise
        except Exception as exc:
            raise AllMethodsEvaluationError(f"{label}: evaluation failed: {exc}") from exc
        values = _real_values(raw, coordinates.shape, label)
        if scalar:
            return float(values)
        return values

    return evaluate


def _scalar_finite(
    function: Callable[[Any], Any], equation_id: str, role: str
) -> Callable[[float], float]:
    label = f"{equation_id}:{role}"

    def evaluate(x: float) -> float:
        coordinates, scalar = _real_coordinates(x, label)
        if not scalar:
            raise AllMethodsEvaluationError(f"{label}: expected one scalar coordinate")
        try:
            raw = function(coordinates)
        except AllMethodsEvaluationError:
            raise
        except Exception as exc:
            raise AllMethodsEvaluationError(f"{label}: evaluation failed: {exc}") from exc
        return float(_real_values(raw, (), label))

    return evaluate


@dataclass(frozen=True)
class _Construction:
    a_mid: float
    a_half: float
    b_scale: float
    c_scale: float
    lambda_value: float
    gap: float
    bump_mu: float
    bump_radius: float
    solution_scale: float


class _EvaluatorBundle:
    """Authoritative evaluators reconstructed from safe AST data."""

    def __init__(
        self,
        a_raw: SafeExpressionNode,
        b_raw: SafeExpressionNode,
        c_raw: SafeExpressionNode,
        y_shape: SafeExpressionNode,
        construction: _Construction,
    ) -> None:
        self.a_raw = a_raw
        self.b_raw = b_raw
        self.c_raw = c_raw
        self.y_shape = y_shape
        self.parameters = construction

    def a(self, x: np.ndarray) -> Any:
        p = self.parameters
        return p.a_mid + p.a_half * np.tanh(evaluate_expression(self.a_raw, x))

    def b(self, x: np.ndarray) -> Any:
        p = self.parameters
        return p.b_scale * np.tanh(evaluate_expression(self.b_raw, x))

    def c_remizov(self, x: np.ndarray) -> Any:
        p = self.parameters
        return p.c_scale * np.tanh(evaluate_expression(self.c_raw, x))

    def c(self, x: np.ndarray) -> Any:
        return self.c_remizov(x) - self.parameters.lambda_value

    def solution_jet(self, x: np.ndarray) -> Tuple[Any, Any, Any]:
        p = self.parameters
        result = evaluate_solution(
            self.y_shape,
            p.bump_mu,
            p.bump_radius,
            p.solution_scale,
            x,
        )
        try:
            return result.value, result.d1, result.d2
        except AttributeError as exc:
            raise AllMethodsEvaluationError(
                "safe expression engine returned an invalid solution jet"
            ) from exc

    def y(self, x: np.ndarray) -> Any:
        return self.solution_jet(x)[0]

    def dy(self, x: np.ndarray) -> Any:
        return self.solution_jet(x)[1]

    def d2y(self, x: np.ndarray) -> Any:
        return self.solution_jet(x)[2]

    def g(self, x: np.ndarray) -> Any:
        y, dy, d2y = self.solution_jet(x)
        return self.a(x) * d2y + self.b(x) * dy + self.c(x) * y

    def g_remizov(self, x: np.ndarray) -> Any:
        return -self.g(x)


def _decode_node(serialized: Any, equation_id: str, role: str) -> SafeExpressionNode:
    if not isinstance(serialized, dict):
        raise _format_error(equation_id, f"serialized_ast.{role} must be an object")
    try:
        node = SafeExpressionNode.from_dict(serialized)
    except Exception as exc:
        raise _format_error(
            equation_id, f"invalid safe AST in serialized_ast.{role}: {exc}"
        ) from exc
    if _canonical_json(serialized) != _canonical_json(serialize_expression(node)):
        raise _format_error(
            equation_id,
            f"serialized_ast.{role} is not the canonical safe-AST representation",
        )
    return node


def _decode_bundle(metadata: Mapping[str, Any], equation_id: str) -> _EvaluatorBundle:
    serialized = metadata.get("serialized_ast")
    if not isinstance(serialized, dict):
        raise _format_error(equation_id, "serialized_ast must be an object")
    keys = frozenset(serialized)
    if keys != _SERIALIZED_AST_FIELDS:
        missing = sorted(_SERIALIZED_AST_FIELDS - keys)
        extra = sorted(keys - _SERIALIZED_AST_FIELDS)
        raise _format_error(
            equation_id,
            f"serialized_ast schema mismatch (missing={missing}, extra={extra})",
        )

    raw_construction = serialized["construction"]
    if not isinstance(raw_construction, dict):
        raise _format_error(equation_id, "serialized_ast.construction must be an object")
    construction_keys = frozenset(raw_construction)
    if construction_keys != _CONSTRUCTION_FIELDS:
        missing = sorted(_CONSTRUCTION_FIELDS - construction_keys)
        extra = sorted(construction_keys - _CONSTRUCTION_FIELDS)
        raise _format_error(
            equation_id,
            f"serialized_ast.construction schema mismatch (missing={missing}, extra={extra})",
        )

    parameters = _Construction(
        a_mid=_finite_number(raw_construction, "a_mid", equation_id),
        a_half=_finite_number(raw_construction, "a_half", equation_id),
        b_scale=_finite_number(raw_construction, "b_scale", equation_id),
        c_scale=_finite_number(raw_construction, "c_scale", equation_id),
        lambda_value=_finite_number(raw_construction, "lambda", equation_id),
        gap=_finite_number(raw_construction, "gap", equation_id),
        bump_mu=_finite_number(raw_construction, "bump_mu", equation_id),
        bump_radius=_finite_number(raw_construction, "bump_radius", equation_id),
        solution_scale=_finite_number(raw_construction, "solution_scale", equation_id),
    )
    if parameters.a_half <= 0.0:
        raise _format_error(equation_id, "construction a_half must be positive")
    if parameters.b_scale <= 0.0 or parameters.c_scale <= 0.0:
        raise _format_error(equation_id, "construction b_scale and c_scale must be positive")
    if parameters.gap <= 0.0 or parameters.bump_radius <= 0.0:
        raise _format_error(equation_id, "construction gap and bump_radius must be positive")
    if parameters.solution_scale == 0.0:
        raise _format_error(equation_id, "construction solution_scale must be nonzero")

    return _EvaluatorBundle(
        _decode_node(serialized["a_raw"], equation_id, "a_raw"),
        _decode_node(serialized["b_raw"], equation_id, "b_raw"),
        _decode_node(serialized["c_raw"], equation_id, "c_raw"),
        _decode_node(serialized["y_shape"], equation_id, "y_shape"),
        parameters,
    )


def _validate_metadata_shape(metadata: Mapping[str, Any], equation_id: str) -> None:
    missing = [field for field in _METADATA_REQUIRED_FIELDS if field not in metadata]
    if missing:
        raise _format_error(equation_id, f"missing metadata fields: {', '.join(missing)}")

    if _required_string(metadata, "generation_version", equation_id) != GENERATION_VERSION:
        raise _format_error(equation_id, f"unsupported generation_version; expected {GENERATION_VERSION!r}")
    if _required_string(metadata, "metadata_schema_version", equation_id) != METADATA_SCHEMA_VERSION:
        raise _format_error(
            equation_id,
            f"unsupported metadata_schema_version; expected {METADATA_SCHEMA_VERSION!r}",
        )
    if _required_string(metadata, "complexity_level", equation_id) not in _COMPLEXITY_LEVELS:
        raise _format_error(equation_id, "invalid complexity_level")
    if _required_string(metadata, "regime", equation_id) not in _REGIMES:
        raise _format_error(equation_id, "invalid regime")
    if _required_string(metadata, "split", equation_id) not in _SPLITS:
        raise _format_error(equation_id, "invalid split")
    for field in (
        "family_id",
        "structure_signature",
        "a_expr",
        "b_expr",
        "c_expr",
        "g_expr",
        "y_exact_expr",
        "dy_exact_expr",
        "d2y_exact_expr",
        "remizov_c_expr",
        "remizov_g_expr",
    ):
        _required_string(metadata, field, equation_id)
    for field in (
        "master_seed",
        "family_seed",
        "sample_seed",
        "attempt_seed",
        "sample_index",
    ):
        _integer(metadata, field, equation_id, minimum=0)
    _integer(metadata, "accepted_attempt", equation_id, minimum=1)
    channel_order = metadata.get("channel_order")
    if (
        not isinstance(channel_order, list)
        or channel_order != list(_CHANNEL_ORDER)
        or any(not isinstance(item, str) for item in channel_order)
    ):
        raise _format_error(
            equation_id,
            f"channel_order must be exactly {list(_CHANNEL_ORDER)!r}",
        )
    serialized_topology = metadata.get("serialized_topology")
    if not isinstance(serialized_topology, dict):
        raise _format_error(equation_id, "serialized_topology must be an object")
    for field in (
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
    ):
        _finite_number(metadata, field, equation_id)


def _validate_construction(
    metadata: Mapping[str, Any], bundle: _EvaluatorBundle, equation_id: str
) -> None:
    p = bundle.parameters
    a_lower = _finite_number(metadata, "a_lower_cert", equation_id)
    a_upper = _finite_number(metadata, "a_upper_cert", equation_id)
    b_abs = _finite_number(metadata, "b_abs_cert", equation_id)
    c_abs = _finite_number(metadata, "remizov_c_abs_cert", equation_id)
    lambda_value = _finite_number(metadata, "remizov_lambda", equation_id)
    gap = _finite_number(metadata, "remizov_gap_cert", equation_id)
    support_left = _finite_number(metadata, "support_left", equation_id)
    support_right = _finite_number(metadata, "support_right", equation_id)
    target_amplitude = _finite_number(metadata, "target_amplitude", equation_id)

    if a_lower < _A_MIN or a_upper < a_lower:
        raise _format_error(equation_id, "invalid analytic a certificate")
    if b_abs < 0.0 or c_abs < 0.0:
        raise _format_error(equation_id, "absolute certificates must be nonnegative")
    if lambda_value <= c_abs or gap <= 0.0:
        raise _format_error(equation_id, "invalid analytic Remizov lambda/gap certificate")
    if support_right <= support_left:
        raise _format_error(equation_id, "invalid compact-support interval")
    if not (support_left < _WORKING_X0 and support_right > _WORKING_X1):
        raise _format_error(
            equation_id,
            "certified support must strictly cover the fixed interval [-1, 1]",
        )
    if not (_TARGET_AMPLITUDE_MIN <= target_amplitude <= _TARGET_AMPLITUDE_MAX):
        raise _format_error(
            equation_id,
            "target_amplitude must be in the closed interval [0.5, 1.5]",
        )

    _require_close(equation_id, "a_lower_cert", a_lower, p.a_mid - p.a_half)
    _require_close(equation_id, "a_upper_cert", a_upper, p.a_mid + p.a_half)
    _require_close(equation_id, "b_abs_cert", b_abs, p.b_scale)
    _require_close(equation_id, "remizov_c_abs_cert", c_abs, p.c_scale)
    _require_close(equation_id, "remizov_lambda", lambda_value, p.lambda_value)
    _require_close(equation_id, "remizov_gap_cert", gap, p.gap)
    _require_close(equation_id, "lambda minus c certificate", lambda_value - c_abs, gap)
    _require_close(equation_id, "support_left", support_left, p.bump_mu - p.bump_radius)
    _require_close(equation_id, "support_right", support_right, p.bump_mu + p.bump_radius)


def _topology_payload(bundle: _EvaluatorBundle) -> Dict[str, Any]:
    trees = {
        "a_raw": bundle.a_raw,
        "b_raw": bundle.b_raw,
        "c_raw": bundle.c_raw,
        "y_shape": bundle.y_shape,
    }
    return {
        name: json.loads(canonical_topology(trees[name])) for name in sorted(trees)
    }


def _validate_ast_metadata(
    row: Mapping[str, str],
    metadata: Mapping[str, Any],
    bundle: _EvaluatorBundle,
    equation_id: str,
) -> None:
    """Bind all persisted formulas and structural claims to the safe AST."""

    nodes = {
        "a_raw": bundle.a_raw,
        "b_raw": bundle.b_raw,
        "c_raw": bundle.c_raw,
        "y_shape": bundle.y_shape,
    }
    p = bundle.parameters
    construction = {
        "a_mid": p.a_mid,
        "a_half": p.a_half,
        "b_scale": p.b_scale,
        "c_scale": p.c_scale,
        "lambda": p.lambda_value,
        "gap": p.gap,
        "bump_mu": p.bump_mu,
        "bump_radius": p.bump_radius,
        "solution_scale": p.solution_scale,
    }
    try:
        expected_formulas = build_problem_formula_strings(nodes, construction)
    except Exception as exc:
        raise _format_error(
            equation_id, f"could not derive canonical formulas from safe AST: {exc}"
        ) from exc
    for field, expected in expected_formulas.items():
        if metadata.get(field) != expected:
            raise _format_error(
                equation_id, f"metadata field {field!r} is not AST-derived"
            )
        if field in _EXPRESSION_FIELDS and row.get(field) != expected:
            raise _format_error(
                equation_id, f"CSV field {field!r} is not AST-derived"
            )

    expected_topology = _topology_payload(bundle)
    if _canonical_json(metadata.get("serialized_topology")) != _canonical_json(
        expected_topology
    ):
        raise _format_error(
            equation_id, "serialized_topology does not match the concrete AST topology"
        )
    expected_signature = topology_signature(nodes)
    if metadata.get("structure_signature") != expected_signature:
        raise _format_error(
            equation_id, "structure_signature does not match the concrete AST topology"
        )


def _validate_row_metadata(
    row: Mapping[str, str], metadata: Mapping[str, Any], equation_id: str
) -> None:
    for field in _EXPRESSION_FIELDS:
        csv_expression = row.get(field, "")
        metadata_expression = metadata.get(field)
        if csv_expression != metadata_expression:
            raise _format_error(equation_id, f"CSV/metadata mismatch for {field}")
    for field in ("x0", "x1", "y0", "v0"):
        _require_close(
            equation_id,
            f"CSV/metadata {field}",
            _finite_number(row, field, equation_id),
            _finite_number(metadata, field, equation_id),
        )


def _validate_numeric_reconstruction(
    metadata: Mapping[str, Any], bundle: _EvaluatorBundle, equation_id: str
) -> None:
    x0 = _finite_number(metadata, "x0", equation_id)
    x1 = _finite_number(metadata, "x1", equation_id)
    if not _close(x0, _WORKING_X0) or not _close(x1, _WORKING_X1):
        raise _format_error(equation_id, "all_methods_v1 requires x0=-1 and x1=1")
    if x1 <= x0:
        raise _format_error(equation_id, "expected x1 > x0")

    functions = {
        "a": _shape_preserving(bundle.a, equation_id, "a"),
        "b": _shape_preserving(bundle.b, equation_id, "b"),
        "c_R": _shape_preserving(bundle.c_remizov, equation_id, "c_R"),
        "c": _shape_preserving(bundle.c, equation_id, "c"),
        "g": _shape_preserving(bundle.g, equation_id, "g"),
        "g_R": _shape_preserving(bundle.g_remizov, equation_id, "g_R"),
        "y": _shape_preserving(bundle.y, equation_id, "y_exact"),
        "dy": _shape_preserving(bundle.dy, equation_id, "dy_exact"),
        "d2y": _shape_preserving(bundle.d2y, equation_id, "d2y_exact"),
    }
    probes = np.linspace(x0, x1, 17, dtype=np.float64)
    values = {name: function(probes) for name, function in functions.items()}

    scale = max(
        1.0,
        float(np.max(np.abs(values["a"] * values["d2y"]))),
        float(np.max(np.abs(values["b"] * values["dy"]))),
        float(np.max(np.abs(values["c"] * values["y"]))),
        float(np.max(np.abs(values["g"]))),
    )
    residual = (
        values["a"] * values["d2y"]
        + values["b"] * values["dy"]
        + values["c"] * values["y"]
        - values["g"]
    )
    if float(np.max(np.abs(residual))) > 1.0e-12 * scale:
        raise _format_error(equation_id, "ordinary manufactured residual is inconsistent")
    if not np.allclose(
        values["c"],
        values["c_R"] - bundle.parameters.lambda_value,
        rtol=2.0e-13,
        atol=2.0e-13,
    ):
        raise _format_error(equation_id, "numeric identity c = c_R - lambda failed")
    if not np.allclose(values["g_R"], -values["g"], rtol=0.0, atol=0.0):
        raise _format_error(equation_id, "numeric identity g_R = -g failed")

    a_lower = _finite_number(metadata, "a_lower_cert", equation_id)
    a_upper = _finite_number(metadata, "a_upper_cert", equation_id)
    b_abs = _finite_number(metadata, "b_abs_cert", equation_id)
    c_abs = _finite_number(metadata, "remizov_c_abs_cert", equation_id)
    tolerance = 5.0e-12 * max(1.0, abs(a_lower), abs(a_upper), b_abs, c_abs)
    if (
        np.any(values["a"] < a_lower - tolerance)
        or np.any(values["a"] > a_upper + tolerance)
        or np.any(np.abs(values["b"]) > b_abs + tolerance)
        or np.any(np.abs(values["c_R"]) > c_abs + tolerance)
    ):
        raise _format_error(equation_id, "reconstructed functions violate analytic certificates")

    expected_y0 = _finite_number(metadata, "y0", equation_id)
    expected_v0 = _finite_number(metadata, "v0", equation_id)
    actual_y0 = float(functions["y"](x0))
    actual_v0 = float(functions["dy"](x0))
    _require_close(equation_id, "y0 vs y_exact(x0)", expected_y0, actual_y0)
    _require_close(equation_id, "v0 vs dy_exact(x0)", expected_v0, actual_v0)
    if math.hypot(actual_y0, actual_v0) < 1.0e-8:
        raise _format_error(equation_id, "initial state has hypot(y0, v0) < 1e-8")

    support_left = _finite_number(metadata, "support_left", equation_id)
    support_right = _finite_number(metadata, "support_right", equation_id)
    radius = support_right - support_left
    outside = np.array(
        [support_left - max(1.0, radius), support_right + max(1.0, radius)],
        dtype=np.float64,
    )
    for name in ("y", "dy", "d2y", "g", "g_R"):
        if not np.allclose(functions[name](outside), 0.0, rtol=0.0, atol=1.0e-14):
            raise _format_error(equation_id, f"{name} is nonzero outside certified support")


def _build_equation(
    row: Mapping[str, str], metadata: Dict[str, Any]
) -> AllMethodsEquation:
    equation_id = _required_string(metadata, "equation_id", "metadata row")
    csv_id = row.get("equation_id", "")
    if csv_id != equation_id:
        raise _format_error(equation_id, "CSV/metadata equation_id mismatch")
    _validate_metadata_shape(metadata, equation_id)
    _validate_row_metadata(row, metadata, equation_id)
    bundle = _decode_bundle(metadata, equation_id)
    _validate_construction(metadata, bundle, equation_id)
    _validate_ast_metadata(row, metadata, bundle, equation_id)
    _validate_numeric_reconstruction(metadata, bundle, equation_id)

    x0 = _finite_number(metadata, "x0", equation_id)
    x1 = _finite_number(metadata, "x1", equation_id)
    y0 = _finite_number(metadata, "y0", equation_id)
    v0 = _finite_number(metadata, "v0", equation_id)

    a_scalar = _scalar_finite(bundle.a, equation_id, "a")
    b_scalar = _scalar_finite(bundle.b, equation_id, "b")
    c_scalar = _scalar_finite(bundle.c, equation_id, "c")
    g_scalar = _scalar_finite(bundle.g, equation_id, "g")
    cauchy = CauchyProblem(
        equation_id=equation_id,
        a=a_scalar,
        b=b_scalar,
        c=c_scalar,
        g=g_scalar,
        x0=x0,
        x1=x1,
        y0=y0,
        v0=v0,
    )

    a_vector = _shape_preserving(bundle.a, equation_id, "a")
    b_vector = _shape_preserving(bundle.b, equation_id, "b")
    c_remizov_vector = _shape_preserving(bundle.c_remizov, equation_id, "c_R")
    g_remizov_vector = _shape_preserving(bundle.g_remizov, equation_id, "g_R")
    remizov = RemizovProblem(
        problem_id=equation_id,
        a=a_vector,
        b=b_vector,
        c_remizov=c_remizov_vector,
        g_remizov=g_remizov_vector,
        lambda_value=_finite_number(metadata, "remizov_lambda", equation_id),
        a_lower_bound=_finite_number(metadata, "a_lower_cert", equation_id),
        a_upper_bound=_finite_number(metadata, "a_upper_cert", equation_id),
        b_abs_upper_bound=_finite_number(metadata, "b_abs_cert", equation_id),
        c_abs_upper_bound=_finite_number(metadata, "remizov_c_abs_cert", equation_id),
        bounds_certification="analytic",
        bounds_domain="all_real_x",
        x_target_min=x0,
        x_target_max=x1,
    )

    return AllMethodsEquation(
        cauchy_problem=cauchy,
        remizov_problem=remizov,
        y_exact=_shape_preserving(bundle.y, equation_id, "y_exact"),
        dy_exact=_shape_preserving(bundle.dy, equation_id, "dy_exact"),
        d2y_exact=_shape_preserving(bundle.d2y, equation_id, "d2y_exact"),
        metadata=dict(metadata),
    )


def _read_csv(path: Path) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    order: List[str] = []
    rows: Dict[str, Dict[str, str]] = {}
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise AllMethodsDatasetFormatError(f"cannot open CSV {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AllMethodsDatasetFormatError(f"{path}: missing CSV header")
        if tuple(reader.fieldnames) != CSV_FIELDS:
            raise AllMethodsDatasetFormatError(
                f"{path}: unexpected CSV schema; expected {CSV_FIELDS!r}, got {tuple(reader.fieldnames)!r}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise AllMethodsDatasetFormatError(
                    f"{path}: CSV line {line_number}: row has cells beyond the nine-column schema"
                )
            equation_id = row.get("equation_id", "")
            if not equation_id or equation_id != equation_id.strip():
                raise AllMethodsDatasetFormatError(
                    f"{path}: CSV line {line_number}: invalid equation_id"
                )
            if equation_id in rows:
                raise AllMethodsDatasetFormatError(
                    f"{path}: CSV line {line_number}: duplicate equation_id {equation_id!r}"
                )
            for field in CSV_FIELDS:
                if row.get(field) in (None, ""):
                    raise _format_error(equation_id, f"CSV field {field!r} is empty")
            order.append(equation_id)
            rows[equation_id] = dict(row)
    return order, rows


def _read_metadata(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    order: List[str] = []
    records: Dict[str, Dict[str, Any]] = {}
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise AllMethodsDatasetFormatError(f"cannot open metadata JSONL {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise AllMethodsDatasetFormatError(
                    f"{path}: JSONL line {line_number}: blank lines are not allowed"
                )
            metadata = _strict_json_object(line, path, line_number)
            raw_id = metadata.get("equation_id")
            if not isinstance(raw_id, str) or not raw_id or raw_id != raw_id.strip():
                raise AllMethodsDatasetFormatError(
                    f"{path}: JSONL line {line_number}: invalid equation_id"
                )
            if raw_id in records:
                raise AllMethodsDatasetFormatError(
                    f"{path}: JSONL line {line_number}: duplicate equation_id {raw_id!r}"
                )
            order.append(raw_id)
            records[raw_id] = metadata
    return order, records


def _validate_dataset_level_metadata(
    metadata_order: Sequence[str], records: Mapping[str, Mapping[str, Any]]
) -> None:
    sample_seeds: Dict[int, str] = {}
    attempt_seeds: Dict[int, str] = {}
    family_samples: Dict[Tuple[str, int], str] = {}
    family_contract: Dict[str, Tuple[Any, ...]] = {}
    family_seed_owner: Dict[int, str] = {}
    signature_owner: Dict[str, str] = {}
    dataset_master_seed: int | None = None
    for equation_id in metadata_order:
        metadata = records[equation_id]
        _validate_metadata_shape(metadata, equation_id)
        master_seed = _integer(metadata, "master_seed", equation_id, minimum=0)
        if dataset_master_seed is None:
            dataset_master_seed = master_seed
        elif master_seed != dataset_master_seed:
            raise _format_error(
                equation_id,
                f"master_seed differs from dataset seed {dataset_master_seed}",
            )

        family_id = _required_string(metadata, "family_id", equation_id)
        family_match = _FAMILY_ID_PATTERN.fullmatch(family_id)
        equation_match = _EQUATION_ID_PATTERN.fullmatch(equation_id)
        if family_match is None:
            raise _format_error(equation_id, "family_id must have v1 form Fdddd")
        if equation_match is None:
            raise _format_error(
                equation_id, "equation_id must have v1 form AMV1_Fdddd_Sdd"
            )
        if equation_match.group("family_id") != family_id:
            raise _format_error(equation_id, "equation_id/family_id mismatch")
        family_index = int(family_match.group("family_index"))

        family_seed = _integer(metadata, "family_seed", equation_id, minimum=0)
        expected_family_seed = _stable_seed(master_seed, 10, family_index)
        if family_seed != expected_family_seed:
            raise _format_error(equation_id, "family_seed does not match its SeedSequence path")
        previous_family = family_seed_owner.get(family_seed)
        if previous_family is not None and previous_family != family_id:
            raise _format_error(
                equation_id,
                f"family_seed is also assigned to family {previous_family!r}",
            )
        family_seed_owner[family_seed] = family_id

        sample_index = _integer(metadata, "sample_index", equation_id, minimum=0)
        if int(equation_match.group("sample_index")) != sample_index:
            raise _format_error(equation_id, "equation_id/sample_index mismatch")
        sample_seed = _integer(metadata, "sample_seed", equation_id, minimum=0)
        expected_sample_seed = _stable_seed(master_seed, 20, family_index, sample_index)
        if sample_seed != expected_sample_seed:
            raise _format_error(equation_id, "sample_seed does not match its SeedSequence path")
        previous = sample_seeds.get(sample_seed)
        if previous is not None:
            raise _format_error(
                equation_id, f"sample_seed duplicates equation {previous!r}"
            )
        sample_seeds[sample_seed] = equation_id

        accepted_attempt = _integer(
            metadata, "accepted_attempt", equation_id, minimum=1
        )
        attempt_seed = _integer(metadata, "attempt_seed", equation_id, minimum=0)
        expected_attempt_seed = _stable_seed(
            master_seed,
            21,
            family_index,
            sample_index,
            accepted_attempt - 1,
        )
        if attempt_seed != expected_attempt_seed:
            raise _format_error(equation_id, "attempt_seed does not match its SeedSequence path")
        previous = attempt_seeds.get(attempt_seed)
        if previous is not None:
            raise _format_error(
                equation_id, f"attempt_seed duplicates equation {previous!r}"
            )
        attempt_seeds[attempt_seed] = equation_id

        sample_key = (family_id, sample_index)
        previous = family_samples.get(sample_key)
        if previous is not None:
            raise _format_error(
                equation_id,
                f"family/sample pair duplicates equation {previous!r}",
            )
        family_samples[sample_key] = equation_id

        serialized_topology = metadata.get("serialized_topology")
        topology_json = _canonical_json(serialized_topology)
        signature = _required_string(metadata, "structure_signature", equation_id)
        contract = (
            master_seed,
            family_seed,
            _required_string(metadata, "complexity_level", equation_id),
            _required_string(metadata, "regime", equation_id),
            _required_string(metadata, "split", equation_id),
            signature,
            topology_json,
        )
        previous_contract = family_contract.get(family_id)
        if previous_contract is not None and contract != previous_contract:
            raise _format_error(
                equation_id,
                "family members disagree on seed, complexity, regime, split, or topology",
            )
        family_contract[family_id] = contract

        signature_family = signature_owner.get(signature)
        if signature_family is not None and signature_family != family_id:
            raise _format_error(
                equation_id,
                f"structure_signature is also assigned to family {signature_family!r}",
            )
        signature_owner[signature] = family_id


def load_all_methods_equations(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> List[AllMethodsEquation]:
    """Load and strictly validate the paired all-methods v1 dataset.

    Returned records follow CSV order.  Both files must contain exactly the
    same unique equation IDs.  No solver is run by this function.
    """

    csv_file = Path(csv_path)
    metadata_file = Path(metadata_path)
    csv_order, rows = _read_csv(csv_file)
    metadata_order, metadata = _read_metadata(metadata_file)
    csv_ids = set(rows)
    metadata_ids = set(metadata)
    if csv_ids != metadata_ids:
        missing_metadata = sorted(csv_ids - metadata_ids)
        missing_csv = sorted(metadata_ids - csv_ids)
        raise AllMethodsDatasetFormatError(
            "CSV/metadata equation ID sets differ "
            f"(missing_metadata={missing_metadata}, missing_csv={missing_csv})"
        )
    if not csv_order:
        raise AllMethodsDatasetFormatError("all-methods dataset is empty")
    _validate_dataset_level_metadata(metadata_order, metadata)
    return [_build_equation(rows[equation_id], metadata[equation_id]) for equation_id in csv_order]


def load_all_methods_equation(
    equation_id: str,
    csv_path: Path | str = DEFAULT_CSV_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> AllMethodsEquation:
    """Load one equation by ID while still validating the complete file pair."""

    if not isinstance(equation_id, str) or not equation_id:
        raise ValueError("equation_id must be a nonempty string")
    equations = load_all_methods_equations(csv_path=csv_path, metadata_path=metadata_path)
    for equation in equations:
        if equation.equation_id == equation_id:
            return equation
    raise KeyError(f"All-methods equation not found: {equation_id}")


__all__ = [
    "AllMethodsDatasetFormatError",
    "AllMethodsEquation",
    "AllMethodsEvaluationError",
    "CSV_FIELDS",
    "DEFAULT_CSV_PATH",
    "DEFAULT_METADATA_PATH",
    "GENERATION_VERSION",
    "METADATA_SCHEMA_VERSION",
    "load_all_methods_equation",
    "load_all_methods_equations",
]
