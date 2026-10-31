"""Safe expression trees and analytic derivatives for ``all_methods_v1``.

The grammar in this module is deliberately smaller than the historical
``grammar-v1.1`` grammar.  Every admitted expression is globally defined,
bounded and :math:`C^\infty` on the real line.  Trees are data -- they are
never evaluated with :func:`eval` -- and deserialization validates an exact
operation/arity/parameter whitelist before a node can be constructed.

The derivative engine propagates a second-order jet ``(value, d1, d2)``
recursively.  It is used both for coefficient expressions and for the
manufactured compact-support solution.  No finite differences or symbolic
``Piecewise`` expressions are involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


ScalarOrArray = Union[float, NDArray[np.float64]]


class ExpressionValidationError(ValueError):
    """Raised when a serialized tree is outside the safe grammar."""


class ExpressionEvaluationError(FloatingPointError):
    """Raised when otherwise-safe arithmetic cannot be represented finitely."""


# Each specification is ``(arity, exact parameter names)``.  Adding an entry
# here is a deliberate schema/version change; unknown operations are rejected.
_NODE_SPECS: Mapping[str, Tuple[int, frozenset[str]]] = MappingProxyType(
    {
        # Globally bounded primitive leaves.
        "constant": (0, frozenset({"value"})),
        "sin_affine": (0, frozenset({"omega", "phi"})),
        "cos_affine": (0, frozenset({"omega", "phi"})),
        "tanh_affine": (0, frozenset({"omega", "phi"})),
        "gaussian": (0, frozenset({"alpha", "mu"})),
        "lorentz": (0, frozenset({"alpha", "mu"})),
        "saturating": (0, frozenset({"mu"})),
        # Globally safe unary compositions.
        "sin": (1, frozenset()),
        "cos": (1, frozenset()),
        "tanh": (1, frozenset()),
        "exp_neg_square": (1, frozenset()),
        "rational_saturate": (1, frozenset()),
        # Safe binary compositions.
        "lincomb": (2, frozenset({"alpha", "beta"})),
        "mul": (2, frozenset()),
    }
)

PRIMITIVE_OPS = frozenset(
    {
        "constant",
        "sin_affine",
        "cos_affine",
        "tanh_affine",
        "gaussian",
        "lorentz",
        "saturating",
    }
)
UNARY_OPS = frozenset({"sin", "cos", "tanh", "exp_neg_square", "rational_saturate"})
BINARY_OPS = frozenset({"lincomb", "mul"})
ALLOWED_OPS = frozenset(_NODE_SPECS)

# Exported both as documentation and for tests which assert that generated
# trees contain none of the unsafe historical grammar nodes.  Validation is
# whitelist-based, so *every* name outside ALLOWED_OPS is forbidden too.
FORBIDDEN_OPS = frozenset(
    {
        "log",
        "sqrt",
        "div",
        "divide",
        "reciprocal",
        "exp",
        "tan",
        "polynomial",
        "pow",
        "var",
        "abs",
        "sign",
        "min",
        "max",
        "piecewise",
        "Piecewise",
    }
)

# Verbose aliases make the whitelist self-documenting at call sites.
ALLOWED_PRIMITIVE_OPS = PRIMITIVE_OPS
ALLOWED_UNARY_OPS = UNARY_OPS
ALLOWED_BINARY_OPS = BINARY_OPS


def _coerce_finite_parameter(op: str, name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ExpressionValidationError(
            f"{op}: parameter {name!r} must be a real JSON number, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ExpressionValidationError(f"{op}: parameter {name!r} must be finite")
    return result


@dataclass(frozen=True)
class SafeExpressionNode:
    """One strictly validated node in the ``all_methods_v1`` safe AST.

    ``params`` is copied into an immutable, key-sorted mapping.  Therefore a
    caller cannot mutate a validated node after construction.
    """

    op: str
    children: Tuple["SafeExpressionNode", ...] = ()
    params: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.op, str) or self.op not in _NODE_SPECS:
            raise ExpressionValidationError(f"unknown or forbidden operation: {self.op!r}")

        try:
            children = tuple(self.children)
        except TypeError as exc:
            raise ExpressionValidationError(f"{self.op}: children must be a sequence") from exc
        if any(not isinstance(child, SafeExpressionNode) for child in children):
            raise ExpressionValidationError(
                f"{self.op}: every child must be a SafeExpressionNode"
            )

        expected_arity, expected_params = _NODE_SPECS[self.op]
        if len(children) != expected_arity:
            raise ExpressionValidationError(
                f"{self.op}: expected arity {expected_arity}, got {len(children)}"
            )
        if not isinstance(self.params, Mapping):
            raise ExpressionValidationError(f"{self.op}: params must be a mapping")
        actual_params = set(self.params)
        if actual_params != set(expected_params):
            missing = sorted(expected_params - actual_params)
            extra = sorted(actual_params - expected_params)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ExpressionValidationError(f"{self.op}: invalid parameters ({', '.join(details)})")

        normalized: Dict[str, float] = {}
        for name in sorted(expected_params):
            if not isinstance(name, str):
                raise ExpressionValidationError(f"{self.op}: parameter keys must be strings")
            normalized[name] = _coerce_finite_parameter(self.op, name, self.params[name])

        # Gaussian and Lorentz leaves only have the stated global shape when
        # alpha is strictly positive (alpha == 0 would be a disguised constant).
        if self.op in {"gaussian", "lorentz"} and normalized["alpha"] <= 0.0:
            raise ExpressionValidationError(f"{self.op}: alpha must be > 0")

        object.__setattr__(self, "children", children)
        object.__setattr__(self, "params", MappingProxyType(normalized))

    def to_dict(self) -> Dict[str, object]:
        """Return the deterministic, JSON-safe concrete-tree representation."""

        return {
            "op": self.op,
            "params": {name: self.params[name] for name in sorted(self.params)},
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SafeExpressionNode":
        """Reconstruct and validate a tree without executing serialized data."""

        if not isinstance(data, Mapping):
            raise ExpressionValidationError("serialized node must be a mapping")
        allowed_fields = {"op", "params", "children"}
        extra_fields = set(data) - allowed_fields
        if extra_fields:
            raise ExpressionValidationError(
                f"serialized node has unknown fields: {sorted(str(item) for item in extra_fields)}"
            )
        if "op" not in data:
            raise ExpressionValidationError("serialized node is missing 'op'")
        op = data["op"]
        if not isinstance(op, str) or op not in _NODE_SPECS:
            raise ExpressionValidationError(f"unknown or forbidden operation: {op!r}")

        raw_params = data.get("params", {})
        raw_children = data.get("children", [])
        if not isinstance(raw_params, Mapping):
            raise ExpressionValidationError(f"{op}: serialized params must be a mapping")
        if not isinstance(raw_children, (list, tuple)):
            raise ExpressionValidationError(f"{op}: serialized children must be a list")
        if any(not isinstance(name, str) for name in raw_params):
            raise ExpressionValidationError(f"{op}: serialized parameter keys must be strings")

        children = tuple(cls.from_dict(child) for child in raw_children)
        return cls(op=op, children=children, params=dict(raw_params))


def serialize_expression(node: SafeExpressionNode) -> Dict[str, object]:
    """Serialize a validated concrete AST to JSON-safe data."""

    validate_expression(node)
    return node.to_dict()


def deserialize_expression(data: Mapping[str, object]) -> SafeExpressionNode:
    """Deserialize a concrete AST through the strict grammar whitelist."""

    return SafeExpressionNode.from_dict(data)


def expression_json(node: SafeExpressionNode) -> str:
    """Return a byte-deterministic JSON serialization of a concrete tree."""

    return json.dumps(
        serialize_expression(node),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def expression_from_json(payload: str) -> SafeExpressionNode:
    """Decode JSON and strictly reconstruct a :class:`SafeExpressionNode`."""

    if not isinstance(payload, str):
        raise ExpressionValidationError("serialized expression JSON must be a string")
    try:
        data = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExpressionValidationError("invalid expression JSON") from exc
    return deserialize_expression(data)


def iter_nodes(node: SafeExpressionNode) -> Iterable[SafeExpressionNode]:
    """Yield nodes in deterministic pre-order."""

    validate_expression(node)
    yield node
    for child in node.children:
        yield from iter_nodes(child)


def validate_expression(node: SafeExpressionNode) -> None:
    """Assert that ``node`` and its complete subtree belong to the whitelist."""

    if not isinstance(node, SafeExpressionNode):
        raise ExpressionValidationError("expression root must be a SafeExpressionNode")
    # Construction already validates nodes.  Walking is intentional: it also
    # guards against exotic object-level mutation/corruption before loading.
    for current in _iter_nodes_unchecked(node):
        if current.op not in _NODE_SPECS:
            raise ExpressionValidationError(f"unknown or forbidden operation: {current.op!r}")
        arity, parameter_names = _NODE_SPECS[current.op]
        if len(current.children) != arity or set(current.params) != set(parameter_names):
            raise ExpressionValidationError(f"corrupt node schema for operation {current.op!r}")
        for name, value in current.params.items():
            _coerce_finite_parameter(current.op, name, value)


def _iter_nodes_unchecked(node: SafeExpressionNode) -> Iterable[SafeExpressionNode]:
    yield node
    for child in node.children:
        if not isinstance(child, SafeExpressionNode):
            raise ExpressionValidationError("corrupt expression child")
        yield from _iter_nodes_unchecked(child)


def forbidden_operations(node: SafeExpressionNode) -> frozenset[str]:
    """Return forbidden operation names found in a tree (normally empty)."""

    # Valid SafeExpressionNode instances can only contain ALLOWED_OPS.  Keeping
    # this helper makes the generator invariant explicit and easy to test.
    validate_expression(node)
    return frozenset(current.op for current in _iter_nodes_unchecked(node) if current.op not in ALLOWED_OPS)


@dataclass(frozen=True)
class TreeMetrics:
    """Pure structural metrics; primitive leaves have depth one."""

    depth: int
    nodes: int
    leaves: int
    unary_nodes: int
    binary_nodes: int
    nonlinear_nodes: int
    max_nonlinear_nesting: int
    operator_counts: Mapping[str, int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "depth": self.depth,
            "nodes": self.nodes,
            "leaves": self.leaves,
            "unary_nodes": self.unary_nodes,
            "binary_nodes": self.binary_nodes,
            "nonlinear_nodes": self.nonlinear_nodes,
            "max_nonlinear_nesting": self.max_nonlinear_nesting,
            "operator_counts": dict(sorted(self.operator_counts.items())),
        }


def tree_metrics(node: SafeExpressionNode) -> TreeMetrics:
    """Compute deterministic topology metrics for one safe tree."""

    validate_expression(node)
    counts: Dict[str, int] = {}

    def visit(current: SafeExpressionNode, nonlinear_ancestors: int) -> Tuple[int, int, int, int, int, int]:
        counts[current.op] = counts.get(current.op, 0) + 1
        nonlinear_here = 1 if current.op in (UNARY_OPS | {"mul"}) else 0
        chain = nonlinear_ancestors + nonlinear_here if nonlinear_here else 0
        if not current.children:
            return 1, 1, 1, 0, 0, chain

        child_stats = [visit(child, chain) for child in current.children]
        return (
            1 + max(item[0] for item in child_stats),
            1 + sum(item[1] for item in child_stats),
            sum(item[2] for item in child_stats),
            (1 if len(current.children) == 1 else 0) + sum(item[3] for item in child_stats),
            (1 if len(current.children) == 2 else 0) + sum(item[4] for item in child_stats),
            max([chain] + [item[5] for item in child_stats]),
        )

    depth, nodes, leaves, unary, binary, nesting = visit(node, 0)
    nonlinear = sum(counts.get(op, 0) for op in (UNARY_OPS | {"mul"}))
    return TreeMetrics(
        depth=depth,
        nodes=nodes,
        leaves=leaves,
        unary_nodes=unary,
        binary_nodes=binary,
        nonlinear_nodes=nonlinear,
        max_nonlinear_nesting=nesting,
        operator_counts=MappingProxyType(dict(sorted(counts.items()))),
    )


def topology_dict(node: SafeExpressionNode) -> Dict[str, object]:
    """Return canonical topology data with all numerical parameters removed."""

    validate_expression(node)

    def build(current: SafeExpressionNode) -> Dict[str, object]:
        children = [build(child) for child in current.children]
        # Both admitted binary operators are algebraically commutative once
        # lincomb's numerical alpha/beta parameters are removed.
        if current.op in {"lincomb", "mul"}:
            children.sort(
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
            )
        return {"op": current.op, "children": children}

    return build(node)


def canonical_topology(node: SafeExpressionNode) -> str:
    """Return a stable parameter-free canonical representation."""

    return json.dumps(topology_dict(node), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def topology_signature(
    trees: Union[SafeExpressionNode, Mapping[str, SafeExpressionNode], Sequence[SafeExpressionNode]],
) -> str:
    """SHA-256 signature of one topology or an ordered/named collection.

    For the four-tree family signature, passing ``{"a": ..., "b": ...,
    "c": ..., "y": ...}`` is recommended.  Mapping keys are sorted and are
    part of the canonical payload; numerical node parameters are not.
    """

    if isinstance(trees, SafeExpressionNode):
        payload: object = topology_dict(trees)
    elif isinstance(trees, Mapping):
        if any(not isinstance(key, str) for key in trees):
            raise ExpressionValidationError("topology mapping keys must be strings")
        payload = {key: topology_dict(trees[key]) for key in sorted(trees)}
    elif isinstance(trees, Sequence) and not isinstance(trees, (str, bytes, bytearray)):
        payload = [topology_dict(tree) for tree in trees]
    else:
        raise ExpressionValidationError("topology_signature expects a node, mapping, or sequence")
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Clear aliases for callers using structural rather than topological wording.
structural_signature = topology_signature
structure_signature = topology_signature
analyze_expression = tree_metrics


@dataclass(frozen=True)
class Jet2:
    """Function value and its first two analytic derivatives."""

    value: ScalarOrArray
    d1: ScalarOrArray
    d2: ScalarOrArray

    @property
    def first(self) -> ScalarOrArray:
        return self.d1

    @property
    def second(self) -> ScalarOrArray:
        return self.d2


def _prepare_real_points(x: ArrayLike) -> Tuple[NDArray[np.float64], bool]:
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise ExpressionEvaluationError("x must be real; complex values are not accepted")
    if raw.dtype.kind in {"b", "O", "S", "U", "V"}:
        try:
            points = raw.astype(np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ExpressionEvaluationError("x must contain real numeric values") from exc
    else:
        points = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise ExpressionEvaluationError("x contains NaN or infinity")
    return points, points.ndim == 0


def _finite_jet(jet: Jet2, op: str) -> Jet2:
    for label, component in (("value", jet.value), ("d1", jet.d1), ("d2", jet.d2)):
        values = np.asarray(component)
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ExpressionEvaluationError(f"{op}: non-finite or non-real {label}")
    return jet


def _evaluate_array(node: SafeExpressionNode, x: NDArray[np.float64]) -> Jet2:
    op = node.op
    p = node.params

    with np.errstate(all="ignore"):
        if op == "constant":
            value = np.full_like(x, p["value"], dtype=np.float64)
            zero = np.zeros_like(x, dtype=np.float64)
            return Jet2(value, zero, zero)

        if op in {"sin_affine", "cos_affine", "tanh_affine"}:
            omega = p["omega"]
            theta = omega * x + p["phi"]
            if op == "sin_affine":
                value = np.sin(theta)
                d1 = omega * np.cos(theta)
                d2 = -(omega * omega) * value
            elif op == "cos_affine":
                value = np.cos(theta)
                d1 = -omega * np.sin(theta)
                d2 = -(omega * omega) * value
            else:
                value = np.tanh(theta)
                sech2 = 1.0 - value * value
                d1 = omega * sech2
                d2 = -2.0 * omega * omega * value * sech2
            return _finite_jet(Jet2(value, d1, d2), op)

        if op == "gaussian":
            alpha = p["alpha"]
            delta = x - p["mu"]
            value = np.exp(-alpha * delta * delta)
            d1 = -2.0 * alpha * delta * value
            d2 = (4.0 * alpha * alpha * delta * delta - 2.0 * alpha) * value
            return _finite_jet(Jet2(value, d1, d2), op)

        if op == "lorentz":
            alpha = p["alpha"]
            delta = x - p["mu"]
            denominator = 1.0 + alpha * delta * delta
            value = 1.0 / denominator
            d1 = -2.0 * alpha * delta / (denominator * denominator)
            d2 = 2.0 * alpha * (3.0 * alpha * delta * delta - 1.0) / (
                denominator * denominator * denominator
            )
            return _finite_jet(Jet2(value, d1, d2), op)

        if op == "saturating":
            delta = x - p["mu"]
            denominator = 1.0 + delta * delta
            value = delta / np.sqrt(denominator)
            d1 = denominator ** -1.5
            d2 = -3.0 * delta * denominator ** -2.5
            return _finite_jet(Jet2(value, d1, d2), op)

        children = [_evaluate_array(child, x) for child in node.children]
        if op == "lincomb":
            left, right = children
            return _finite_jet(
                Jet2(
                    p["alpha"] * left.value + p["beta"] * right.value,
                    p["alpha"] * left.d1 + p["beta"] * right.d1,
                    p["alpha"] * left.d2 + p["beta"] * right.d2,
                ),
                op,
            )
        if op == "mul":
            left, right = children
            return _finite_jet(
                Jet2(
                    left.value * right.value,
                    left.d1 * right.value + left.value * right.d1,
                    left.d2 * right.value
                    + 2.0 * left.d1 * right.d1
                    + left.value * right.d2,
                ),
                op,
            )

        child = children[0]
        u, du, d2u = child.value, child.d1, child.d2
        if op == "sin":
            value = np.sin(u)
            outer_d1 = np.cos(u)
            outer_d2 = -value
        elif op == "cos":
            value = np.cos(u)
            outer_d1 = -np.sin(u)
            outer_d2 = -value
        elif op == "tanh":
            value = np.tanh(u)
            outer_d1 = 1.0 - value * value
            outer_d2 = -2.0 * value * outer_d1
        elif op == "exp_neg_square":
            value = np.exp(-(u * u))
            outer_d1 = -2.0 * u * value
            outer_d2 = (4.0 * u * u - 2.0) * value
        elif op == "rational_saturate":
            denominator = 1.0 + u * u
            value = u / denominator
            outer_d1 = (1.0 - u * u) / (denominator * denominator)
            outer_d2 = 2.0 * u * (u * u - 3.0) / (
                denominator * denominator * denominator
            )
        else:  # Defensive even though validation is whitelist-based.
            raise ExpressionEvaluationError(f"unsupported operation {op!r}")
        return _finite_jet(
            Jet2(value, outer_d1 * du, outer_d2 * du * du + outer_d1 * d2u),
            op,
        )


def _restore_shape(jet: Jet2, scalar: bool) -> Jet2:
    if scalar:
        return Jet2(float(np.asarray(jet.value)), float(np.asarray(jet.d1)), float(np.asarray(jet.d2)))
    return Jet2(
        np.asarray(jet.value, dtype=np.float64),
        np.asarray(jet.d1, dtype=np.float64),
        np.asarray(jet.d2, dtype=np.float64),
    )


def evaluate_jet2(node: SafeExpressionNode, x: ArrayLike) -> Jet2:
    """Evaluate value, first derivative and second derivative analytically."""

    validate_expression(node)
    points, scalar = _prepare_real_points(x)
    return _restore_shape(_evaluate_array(node, points), scalar)


def evaluate_expression(node: SafeExpressionNode, x: ArrayLike) -> ScalarOrArray:
    """Evaluate only the value while retaining scalar/array input shape."""

    return evaluate_jet2(node, x).value


def evaluate_derivatives(node: SafeExpressionNode, x: ArrayLike) -> Jet2:
    """Explicit alias for callers interested in the analytic derivatives."""

    return evaluate_jet2(node, x)


def normalized_bump_jet2(x: ArrayLike, mu: float, R: float) -> Jet2:
    r"""Evaluate the normalized compact-support :math:`C^\infty` bump.

    With ``z=(x-mu)/R`` the value is

    ``exp(1 - 1/(1-z**2))`` for ``abs(z) < 1`` and exactly zero outside.
    The mask is applied before divisions, and both analytic derivatives are
    exactly zero on and outside the support boundary.
    """

    mu_value = _coerce_finite_parameter("bump", "mu", mu)
    radius_value = _coerce_finite_parameter("bump", "R", R)
    if radius_value <= 0.0:
        raise ExpressionValidationError("bump: radius must be > 0")
    points, scalar = _prepare_real_points(x)
    value = np.zeros_like(points, dtype=np.float64)
    d1 = np.zeros_like(points, dtype=np.float64)
    d2 = np.zeros_like(points, dtype=np.float64)
    z = (points - mu_value) / radius_value
    inside = np.abs(z) < 1.0

    if np.any(inside):
        zi = z[inside] if z.ndim else np.asarray([float(z)])
        q = 1.0 - zi * zi
        with np.errstate(all="ignore"):
            phi = np.exp(1.0 - 1.0 / q)
        local_d1 = np.zeros_like(phi)
        local_d2 = np.zeros_like(phi)
        # Once the exponential underflows to zero, the true derivatives are
        # also far below float64 resolution.  Avoid forming q**-4 there.
        represented = phi != 0.0
        if np.any(represented):
            zr = zi[represented]
            qr = q[represented]
            phir = phi[represented]
            log_d1_z = -2.0 * zr / (qr * qr)
            log_d2_z = -2.0 / (qr * qr) - 8.0 * zr * zr / (qr * qr * qr)
            local_d1[represented] = phir * log_d1_z / radius_value
            local_d2[represented] = (
                phir * (log_d2_z + log_d1_z * log_d1_z) / (radius_value * radius_value)
            )

        if z.ndim:
            value[inside] = phi
            d1[inside] = local_d1
            d2[inside] = local_d2
        else:
            value[...] = phi[0]
            d1[...] = local_d1[0]
            d2[...] = local_d2[0]

    return _restore_shape(_finite_jet(Jet2(value, d1, d2), "bump"), scalar)


def normalized_bump(x: ArrayLike, mu: float, R: float) -> ScalarOrArray:
    """Evaluate only the normalized bump value."""

    return normalized_bump_jet2(x, mu, R).value


def evaluate_solution(
    shape: SafeExpressionNode,
    mu: float,
    R: float,
    scale: float,
    x: ArrayLike,
) -> Jet2:
    """Evaluate ``scale * bump(mu,R) * shape`` and two derivatives."""

    scale_value = _coerce_finite_parameter("solution", "scale", scale)
    shape_jet = evaluate_jet2(shape, x)
    bump_jet = normalized_bump_jet2(x, mu, R)
    with np.errstate(all="ignore"):
        result = Jet2(
            scale_value * bump_jet.value * shape_jet.value,
            scale_value * (bump_jet.d1 * shape_jet.value + bump_jet.value * shape_jet.d1),
            scale_value
            * (
                bump_jet.d2 * shape_jet.value
                + 2.0 * bump_jet.d1 * shape_jet.d1
                + bump_jet.value * shape_jet.d2
            ),
        )
    return _finite_jet(result, "solution")


@dataclass(frozen=True)
class _FormulaJet:
    value: str
    d1: str
    d2: str


def _format_number(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    return format(value, ".17g")


def _sq(text: str) -> str:
    return f"(({text})**2)"


def _cube(text: str) -> str:
    return f"(({text})**3)"


def _formula_jet(node: SafeExpressionNode, variable: str) -> _FormulaJet:
    op = node.op
    p = node.params
    if op == "constant":
        return _FormulaJet(_format_number(p["value"]), "0", "0")
    if op in {"sin_affine", "cos_affine", "tanh_affine"}:
        omega = _format_number(p["omega"])
        phi = _format_number(p["phi"])
        theta = f"(({omega})*({variable})+({phi}))"
        if op == "sin_affine":
            return _FormulaJet(
                f"sin({theta})",
                f"(({omega})*cos({theta}))",
                f"(-(({omega})**2)*sin({theta}))",
            )
        if op == "cos_affine":
            return _FormulaJet(
                f"cos({theta})",
                f"(-({omega})*sin({theta}))",
                f"(-(({omega})**2)*cos({theta}))",
            )
        value = f"tanh({theta})"
        sech2 = f"(1-{_sq(value)})"
        return _FormulaJet(
            value,
            f"(({omega})*({sech2}))",
            f"(-2*(({omega})**2)*({value})*({sech2}))",
        )
    if op == "gaussian":
        alpha = _format_number(p["alpha"])
        mu = _format_number(p["mu"])
        delta = f"(({variable})-({mu}))"
        value = f"exp(-({alpha})*{_sq(delta)})"
        return _FormulaJet(
            value,
            f"(-2*({alpha})*({delta})*({value}))",
            f"((4*(({alpha})**2)*{_sq(delta)}-2*({alpha}))*({value}))",
        )
    if op == "lorentz":
        alpha = _format_number(p["alpha"])
        mu = _format_number(p["mu"])
        delta = f"(({variable})-({mu}))"
        denominator = f"(1+({alpha})*{_sq(delta)})"
        return _FormulaJet(
            f"(1/({denominator}))",
            f"(-2*({alpha})*({delta})/{_sq(denominator)})",
            f"(2*({alpha})*(3*({alpha})*{_sq(delta)}-1)/{_cube(denominator)})",
        )
    if op == "saturating":
        mu = _format_number(p["mu"])
        delta = f"(({variable})-({mu}))"
        denominator = f"(1+{_sq(delta)})"
        return _FormulaJet(
            f"(({delta})/sqrt({denominator}))",
            f"(1/(({denominator})**1.5))",
            f"(-3*({delta})/(({denominator})**2.5))",
        )

    children = [_formula_jet(child, variable) for child in node.children]
    if op == "lincomb":
        left, right = children
        alpha, beta = _format_number(p["alpha"]), _format_number(p["beta"])
        return _FormulaJet(
            f"(({alpha})*({left.value})+({beta})*({right.value}))",
            f"(({alpha})*({left.d1})+({beta})*({right.d1}))",
            f"(({alpha})*({left.d2})+({beta})*({right.d2}))",
        )
    if op == "mul":
        left, right = children
        return _FormulaJet(
            f"(({left.value})*({right.value}))",
            f"(({left.d1})*({right.value})+({left.value})*({right.d1}))",
            "(({}*{})+2*({})*({})+({})*({}))".format(
                f"({left.d2})",
                f"({right.value})",
                left.d1,
                right.d1,
                left.value,
                right.d2,
            ),
        )

    child = children[0]
    u, du, d2u = child.value, child.d1, child.d2
    if op == "sin":
        value, outer_d1, outer_d2 = f"sin({u})", f"cos({u})", f"(-sin({u}))"
    elif op == "cos":
        value, outer_d1, outer_d2 = f"cos({u})", f"(-sin({u}))", f"(-cos({u}))"
    elif op == "tanh":
        value = f"tanh({u})"
        outer_d1 = f"(1-{_sq(value)})"
        outer_d2 = f"(-2*({value})*({outer_d1}))"
    elif op == "exp_neg_square":
        value = f"exp(-{_sq(u)})"
        outer_d1 = f"(-2*({u})*({value}))"
        outer_d2 = f"((4*{_sq(u)}-2)*({value}))"
    elif op == "rational_saturate":
        denominator = f"(1+{_sq(u)})"
        value = f"(({u})/({denominator}))"
        outer_d1 = f"((1-{_sq(u)})/{_sq(denominator)})"
        outer_d2 = f"(2*({u})*({_sq(u)}-3)/{_cube(denominator)})"
    else:  # pragma: no cover - protected by SafeExpressionNode validation.
        raise ExpressionValidationError(f"cannot serialize operation {op!r}")
    return _FormulaJet(
        value,
        f"(({outer_d1})*({du}))",
        f"(({outer_d2})*{_sq(du)}+({outer_d1})*({d2u}))",
    )


def formula_strings(node: SafeExpressionNode, variable: str = "x") -> Dict[str, str]:
    """Return deterministic analytic ``value``, ``d1`` and ``d2`` formulas."""

    validate_expression(node)
    if not isinstance(variable, str) or not variable.isidentifier():
        raise ExpressionValidationError("formula variable must be a valid identifier")
    jet = _formula_jet(node, variable)
    return {"value": jet.value, "d1": jet.d1, "d2": jet.d2}


def serialize_formula(
    node: SafeExpressionNode,
    derivative_order: int = 0,
    variable: str = "x",
) -> str:
    """Serialize one analytic formula (order 0, 1, or 2)."""

    if derivative_order not in (0, 1, 2):
        raise ExpressionValidationError("derivative_order must be 0, 1, or 2")
    key = ("value", "d1", "d2")[derivative_order]
    return formula_strings(node, variable)[key]


def serialize_value(node: SafeExpressionNode, variable: str = "x") -> str:
    return serialize_formula(node, 0, variable)


def serialize_d1(node: SafeExpressionNode, variable: str = "x") -> str:
    return serialize_formula(node, 1, variable)


def serialize_d2(node: SafeExpressionNode, variable: str = "x") -> str:
    return serialize_formula(node, 2, variable)


def bump_formula_strings(mu: float, R: float, variable: str = "x") -> Dict[str, str]:
    """Return deterministic declarative formulas for the three bump jets.

    ``bump``, ``bump_d1`` and ``bump_d2`` are intentionally explicit safe
    vocabulary tokens, not Python/SymPy ``Piecewise`` code.  Numerical loading
    reconstructs the AST plus ``mu``/``R`` and calls
    :func:`normalized_bump_jet2`; it never evaluates these strings.
    """

    mu_value = _coerce_finite_parameter("bump", "mu", mu)
    radius_value = _coerce_finite_parameter("bump", "R", R)
    if radius_value <= 0.0:
        raise ExpressionValidationError("bump: radius must be > 0")
    if not isinstance(variable, str) or not variable.isidentifier():
        raise ExpressionValidationError("formula variable must be a valid identifier")
    args = f"{variable},{_format_number(mu_value)},{_format_number(radius_value)}"
    return {
        "value": f"bump({args})",
        "d1": f"bump_d1({args})",
        "d2": f"bump_d2({args})",
    }


def serialize_solution_expressions(
    shape: SafeExpressionNode,
    mu: float,
    R: float,
    scale: float,
    variable: str = "x",
) -> Dict[str, str]:
    """Serialize deterministic human-readable manufactured-solution formulas.

    The formulas mirror exactly the analytic product rules used by
    :func:`evaluate_solution`.  They are descriptive metadata; reconstruction
    uses the concrete ``shape`` AST and the finite scalar parameters.
    """

    scale_text = _format_number(_coerce_finite_parameter("solution", "scale", scale))
    bump = bump_formula_strings(mu, R, variable)
    shape_formula = formula_strings(shape, variable)
    y = f"(({scale_text})*({bump['value']})*({shape_formula['value']}))"
    dy = (
        f"(({scale_text})*((({bump['d1']})*({shape_formula['value']}))"
        f"+(({bump['value']})*({shape_formula['d1']}))))"
    )
    d2y = (
        f"(({scale_text})*((({bump['d2']})*({shape_formula['value']}))"
        f"+(2*({bump['d1']})*({shape_formula['d1']}))"
        f"+(({bump['value']})*({shape_formula['d2']}))))"
    )
    return {"y_exact_expr": y, "dy_exact_expr": dy, "d2y_exact_expr": d2y}


# Radius terminology is mathematically clearer internally, but the dataset
# specification denotes it by R.  These aliases ease metadata-driven callers.
evaluate_bump = normalized_bump
evaluate_bump_jet2 = normalized_bump_jet2


__all__ = [
    "ALLOWED_BINARY_OPS",
    "ALLOWED_OPS",
    "ALLOWED_PRIMITIVE_OPS",
    "ALLOWED_UNARY_OPS",
    "BINARY_OPS",
    "ExpressionEvaluationError",
    "ExpressionValidationError",
    "FORBIDDEN_OPS",
    "Jet2",
    "PRIMITIVE_OPS",
    "SafeExpressionNode",
    "ScalarOrArray",
    "TreeMetrics",
    "UNARY_OPS",
    "analyze_expression",
    "bump_formula_strings",
    "canonical_topology",
    "deserialize_expression",
    "evaluate_bump",
    "evaluate_bump_jet2",
    "evaluate_derivatives",
    "evaluate_expression",
    "evaluate_jet2",
    "evaluate_solution",
    "expression_from_json",
    "expression_json",
    "forbidden_operations",
    "formula_strings",
    "iter_nodes",
    "normalized_bump",
    "normalized_bump_jet2",
    "serialize_d1",
    "serialize_d2",
    "serialize_expression",
    "serialize_formula",
    "serialize_solution_expressions",
    "serialize_value",
    "structural_signature",
    "structure_signature",
    "topology_dict",
    "topology_signature",
    "tree_metrics",
    "validate_expression",
]
