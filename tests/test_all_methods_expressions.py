"""Tests for the safe analytic-expression layer used by all-methods v1.

The derivative oracle in this file is intentionally independent of the
project's recursive jet implementation: every representative AST is rebuilt
as a SymPy expression and differentiated by SymPy.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
import unittest

import numpy as np
import sympy as sp

from src.all_methods_expressions import (
    ALLOWED_OPS,
    FORBIDDEN_OPS,
    ExpressionEvaluationError,
    ExpressionValidationError,
    SafeExpressionNode,
    canonical_topology,
    deserialize_expression,
    evaluate_expression,
    evaluate_jet2,
    evaluate_solution,
    expression_from_json,
    expression_json,
    forbidden_operations,
    iter_nodes,
    normalized_bump_jet2,
    serialize_expression,
    topology_signature,
)


def _node(op: str, *children: SafeExpressionNode, **params: float) -> SafeExpressionNode:
    return SafeExpressionNode(op=op, children=children, params=params)


def _sympy_expression(node: SafeExpressionNode, x: sp.Symbol) -> sp.Expr:
    """Independent, test-local interpretation of the public grammar."""

    op = node.op
    p = node.params
    if op == "constant":
        return sp.Float(p["value"])
    if op in {"sin_affine", "cos_affine", "tanh_affine"}:
        theta = sp.Float(p["omega"]) * x + sp.Float(p["phi"])
        return {
            "sin_affine": sp.sin,
            "cos_affine": sp.cos,
            "tanh_affine": sp.tanh,
        }[op](theta)
    if op == "gaussian":
        return sp.exp(-sp.Float(p["alpha"]) * (x - sp.Float(p["mu"])) ** 2)
    if op == "lorentz":
        return 1 / (1 + sp.Float(p["alpha"]) * (x - sp.Float(p["mu"])) ** 2)
    if op == "saturating":
        delta = x - sp.Float(p["mu"])
        return delta / sp.sqrt(1 + delta**2)

    children = [_sympy_expression(child, x) for child in node.children]
    if op == "lincomb":
        return sp.Float(p["alpha"]) * children[0] + sp.Float(p["beta"]) * children[1]
    if op == "mul":
        return children[0] * children[1]

    u = children[0]
    if op == "sin":
        return sp.sin(u)
    if op == "cos":
        return sp.cos(u)
    if op == "tanh":
        return sp.tanh(u)
    if op == "exp_neg_square":
        return sp.exp(-(u**2))
    if op == "rational_saturate":
        return u / (1 + u**2)
    raise AssertionError(f"test oracle has no mapping for {op!r}")


def _as_vector(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=np.float64)
    return np.broadcast_to(array, shape).astype(np.float64, copy=False)


def _representative_roots() -> list[SafeExpressionNode]:
    """One non-degenerate representative with every operation as a root."""

    sine = _node("sin_affine", omega=1.7, phi=-0.31)
    cosine = _node("cos_affine", omega=-0.8, phi=0.47)
    base = _node("lincomb", sine, cosine, alpha=0.73, beta=-1.12)
    return [
        _node("constant", value=-0.37),
        sine,
        cosine,
        _node("tanh_affine", omega=1.2, phi=-0.2),
        _node("gaussian", alpha=0.85, mu=-0.15),
        _node("lorentz", alpha=1.3, mu=0.22),
        _node("saturating", mu=-0.4),
        _node("sin", base),
        _node("cos", base),
        _node("tanh", base),
        _node("exp_neg_square", base),
        _node("rational_saturate", base),
        base,
        _node("mul", base, _node("gaussian", alpha=0.6, mu=0.1)),
    ]


class SafeAstValidationTests(unittest.TestCase):
    def test_representatives_cover_exactly_the_whitelisted_operations(self) -> None:
        self.assertEqual({tree.op for tree in _representative_roots()}, set(ALLOWED_OPS))

    def test_round_trip_is_deterministic_and_preserves_values(self) -> None:
        tree = _node(
            "tanh",
            _node(
                "mul",
                _node("sin_affine", omega=2.25, phi=-0.75),
                _node("lorentz", alpha=0.4, mu=0.2),
            ),
        )
        payload = expression_json(tree)
        rebuilt = expression_from_json(payload)

        self.assertEqual(payload, expression_json(rebuilt))
        self.assertEqual(serialize_expression(tree), serialize_expression(rebuilt))
        np.testing.assert_array_equal(
            evaluate_expression(tree, np.linspace(-2.0, 2.0, 21)),
            evaluate_expression(rebuilt, np.linspace(-2.0, 2.0, 21)),
        )
        self.assertEqual(forbidden_operations(rebuilt), frozenset())

    def test_parameters_and_children_are_immutable_after_validation(self) -> None:
        tree = _node("constant", value=1.0)
        with self.assertRaises(TypeError):
            tree.params["value"] = 2.0  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            tree.op = "cos"  # type: ignore[misc]

    def test_every_explicitly_forbidden_operation_is_rejected(self) -> None:
        for operation in sorted(FORBIDDEN_OPS):
            with self.subTest(operation=operation), self.assertRaises(
                ExpressionValidationError
            ):
                deserialize_expression(
                    {"op": operation, "params": {}, "children": []}
                )

    def test_unknown_and_malformed_serialized_nodes_are_rejected(self) -> None:
        malformed = [
            {},
            {"op": "definitely_unknown", "params": {}, "children": []},
            {"op": "constant", "params": {"value": 1.0}, "children": [], "extra": 1},
            {"op": "constant", "params": {}, "children": []},
            {"op": "constant", "params": {"value": 1.0, "other": 2.0}, "children": []},
            {"op": "constant", "params": {"value": True}, "children": []},
            {"op": "constant", "params": {"value": float("nan")}, "children": []},
            {"op": "constant", "params": {"value": float("inf")}, "children": []},
            {"op": "constant", "params": {"value": 1.0}, "children": "not-a-list"},
            {"op": "sin", "params": {}, "children": []},
            {
                "op": "sin",
                "params": {},
                "children": [
                    {"op": "constant", "params": {"value": 1.0}, "children": []},
                    {"op": "constant", "params": {"value": 2.0}, "children": []},
                ],
            },
            {"op": "lincomb", "params": {"alpha": 1.0}, "children": []},
            {"op": "gaussian", "params": {"alpha": 0.0, "mu": 0.0}, "children": []},
            {"op": "lorentz", "params": {"alpha": -1.0, "mu": 0.0}, "children": []},
        ]
        for index, raw in enumerate(malformed):
            with self.subTest(index=index, raw=raw), self.assertRaises(
                ExpressionValidationError
            ):
                deserialize_expression(raw)

    def test_malformed_json_is_rejected_without_execution(self) -> None:
        for payload in (
            "",
            "not json",
            "[]",
            json.dumps({"op": "log", "children": [], "params": {}}),
        ):
            with self.subTest(payload=payload), self.assertRaises(
                ExpressionValidationError
            ):
                expression_from_json(payload)
        with self.assertRaises(ExpressionValidationError):
            expression_from_json(123)  # type: ignore[arg-type]

    def test_topology_removes_numbers_but_retains_tree_structure(self) -> None:
        first = _node("sin_affine", omega=0.5, phi=0.1)
        second = _node("sin_affine", omega=7.5, phi=-2.0)
        different = _node("cos_affine", omega=0.5, phi=0.1)

        self.assertEqual(canonical_topology(first), canonical_topology(second))
        self.assertEqual(topology_signature(first), topology_signature(second))
        self.assertNotEqual(topology_signature(first), topology_signature(different))


class AnalyticJetTests(unittest.TestCase):
    def test_every_operation_matches_independent_sympy_value_and_derivatives(self) -> None:
        symbol = sp.Symbol("x", real=True)
        points = np.array([-1.31, -0.72, -0.11, 0.0, 0.39, 0.88, 1.42])

        for tree in _representative_roots():
            with self.subTest(root_operation=tree.op):
                expression = _sympy_expression(tree, symbol)
                expected_components = []
                for order in (0, 1, 2):
                    derivative = sp.diff(expression, symbol, order)
                    function = sp.lambdify(symbol, derivative, modules="numpy")
                    expected_components.append(_as_vector(function(points), points.shape))

                actual = evaluate_jet2(tree, points)
                np.testing.assert_allclose(
                    actual.value, expected_components[0], rtol=2.0e-12, atol=2.0e-12
                )
                np.testing.assert_allclose(
                    actual.d1, expected_components[1], rtol=4.0e-12, atol=4.0e-12
                )
                np.testing.assert_allclose(
                    actual.d2, expected_components[2], rtol=8.0e-12, atol=8.0e-12
                )

    def test_deep_composition_matches_sympy(self) -> None:
        tree = _node(
            "rational_saturate",
            _node(
                "mul",
                _node(
                    "exp_neg_square",
                    _node(
                        "lincomb",
                        _node("sin", _node("saturating", mu=-0.2)),
                        _node("cos", _node("gaussian", alpha=0.7, mu=0.4)),
                        alpha=-0.8,
                        beta=1.3,
                    ),
                ),
                _node("tanh", _node("lorentz", alpha=1.1, mu=-0.5)),
            ),
        )
        symbol = sp.Symbol("x", real=True)
        expression = _sympy_expression(tree, symbol)
        points = np.linspace(-1.5, 1.5, 31)
        actual = evaluate_jet2(tree, points)

        for actual_component, order in ((actual.value, 0), (actual.d1, 1), (actual.d2, 2)):
            expected_function = sp.lambdify(
                symbol, sp.diff(expression, symbol, order), modules="numpy"
            )
            expected = _as_vector(expected_function(points), points.shape)
            np.testing.assert_allclose(actual_component, expected, rtol=2.0e-11, atol=2.0e-11)

    def test_scalar_and_vector_evaluation_contract(self) -> None:
        tree = _node("sin", _node("tanh_affine", omega=1.2, phi=0.3))
        scalar = evaluate_jet2(tree, 0.25)
        vector = evaluate_jet2(tree, np.array([0.25, -0.4]))

        self.assertIsInstance(scalar.value, float)
        self.assertIsInstance(scalar.d1, float)
        self.assertIsInstance(scalar.d2, float)
        self.assertEqual(np.asarray(vector.value).shape, (2,))
        self.assertEqual(np.asarray(vector.d1).shape, (2,))
        self.assertEqual(np.asarray(vector.d2).shape, (2,))
        self.assertAlmostEqual(scalar.value, float(vector.value[0]), places=15)
        self.assertAlmostEqual(scalar.d1, float(vector.d1[0]), places=15)
        self.assertAlmostEqual(scalar.d2, float(vector.d2[0]), places=15)

    def test_invalid_evaluation_coordinates_are_rejected(self) -> None:
        tree = _node("constant", value=1.0)
        for coordinates in (float("nan"), float("inf"), 1.0 + 1.0j, [0.0, "x"]):
            with self.subTest(coordinates=coordinates), self.assertRaises(
                ExpressionEvaluationError
            ):
                evaluate_jet2(tree, coordinates)


class CompactSupportBumpTests(unittest.TestCase):
    def test_bump_has_true_compact_support_and_normalized_center(self) -> None:
        mu = 0.23
        radius = 1.7
        left, right = mu - radius, mu + radius
        points = np.array(
            [left - 2.0, left, left + 1.0e-6, mu, right - 1.0e-6, right, right + 2.0]
        )
        jet = normalized_bump_jet2(points, mu, radius)

        for component in (jet.value, jet.d1, jet.d2):
            self.assertEqual(component[0], 0.0)
            self.assertEqual(component[1], 0.0)
            self.assertEqual(component[-2], 0.0)
            self.assertEqual(component[-1], 0.0)
            self.assertTrue(np.all(np.isfinite(component)))
        self.assertEqual(jet.value[3], 1.0)
        self.assertEqual(jet.d1[3], 0.0)
        self.assertAlmostEqual(jet.d2[3], -2.0 / radius**2, places=14)

        # A Gaussian surrogate could never satisfy this exact outside-support
        # identity at finite points.
        self.assertTrue(np.array_equal(jet.value[[0, 1, 5, 6]], np.zeros(4)))

    def test_bump_derivatives_match_symbolic_interior_formula(self) -> None:
        symbol = sp.Symbol("x", real=True)
        mu = -0.17
        radius = 2.1
        z = (symbol - mu) / radius
        interior = sp.exp(1 - 1 / (1 - z**2))
        points = np.array([mu - 0.8 * radius, mu - 0.2, mu, mu + 0.35, mu + 0.8 * radius])
        actual = normalized_bump_jet2(points, mu, radius)

        for actual_component, order in ((actual.value, 0), (actual.d1, 1), (actual.d2, 2)):
            function = sp.lambdify(symbol, sp.diff(interior, symbol, order), modules="numpy")
            expected = _as_vector(function(points), points.shape)
            np.testing.assert_allclose(actual_component, expected, rtol=3.0e-12, atol=3.0e-12)

    def test_bump_is_numerically_continuous_at_both_boundaries(self) -> None:
        mu = 0.0
        radius = 1.6
        epsilon = 1.0e-5
        points = np.array(
            [
                mu - radius - epsilon,
                mu - radius,
                mu - radius + epsilon,
                mu + radius - epsilon,
                mu + radius,
                mu + radius + epsilon,
            ]
        )
        jet = normalized_bump_jet2(points, mu, radius)
        for component in (jet.value, jet.d1, jet.d2):
            self.assertLessEqual(float(np.max(np.abs(component))), 1.0e-13)

    def test_solution_and_derivatives_vanish_outside_same_support(self) -> None:
        shape = _node(
            "lincomb",
            _node("sin_affine", omega=1.1, phi=0.2),
            _node("tanh", _node("lorentz", alpha=0.8, mu=-0.3)),
            alpha=0.7,
            beta=-1.2,
        )
        mu, radius, scale = 0.1, 1.8, 2.4
        points = np.array([mu - radius - 1.0, mu - radius, mu + radius, mu + radius + 1.0])
        jet = evaluate_solution(shape, mu, radius, scale, points)

        np.testing.assert_array_equal(jet.value, np.zeros_like(points))
        np.testing.assert_array_equal(jet.d1, np.zeros_like(points))
        np.testing.assert_array_equal(jet.d2, np.zeros_like(points))

    def test_bump_scalar_and_vector_results_agree(self) -> None:
        scalar = normalized_bump_jet2(0.3, 0.1, 1.5)
        vector = normalized_bump_jet2(np.array([0.3]), 0.1, 1.5)
        for name in ("value", "d1", "d2"):
            scalar_value = getattr(scalar, name)
            vector_value = getattr(vector, name)
            self.assertIsInstance(scalar_value, float)
            self.assertAlmostEqual(scalar_value, float(vector_value[0]), places=15)

    def test_invalid_bump_parameters_are_rejected(self) -> None:
        for radius in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(radius=radius), self.assertRaises(ExpressionValidationError):
                normalized_bump_jet2(0.0, 0.0, radius)
        with self.assertRaises(ExpressionValidationError):
            normalized_bump_jet2(0.0, float("nan"), 1.0)


if __name__ == "__main__":
    unittest.main()
