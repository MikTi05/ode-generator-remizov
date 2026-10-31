"""Dataset-level tests for the isolated ``all_methods_equations_v1`` pipeline.

The suite deliberately uses a small but complete 20-cell configuration.  It
checks generation, serialization and safe loading only; no numerical solver is
invoked here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from all_methods_dataset_loader import load_all_methods_equations
from src.all_methods_dataset_v1 import (
    AllMethodsConfig,
    CHANNEL_ORDER,
    COMPLEXITY_LEVELS,
    CSV_FIELDS,
    OUTPUT_FILENAMES,
    REGIMES,
    generate_dataset,
    validate_dataset_artifacts,
    write_dataset,
)
from src.all_methods_expressions import (
    ALLOWED_OPS,
    FORBIDDEN_OPS,
    SafeExpressionNode,
    canonical_topology,
    evaluate_expression,
    evaluate_solution,
    iter_nodes,
    topology_signature,
)


_NPZ_MEMBERS = {
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

_COMPLEXITY_LIMITS = {
    "L1": (1, 2, 3, 7),
    "L2": (2, 3, 6, 12),
    "L3": (3, 4, 10, 18),
    "L4": (4, 5, 14, 24),
}


def _nodes_from_metadata(metadata: dict[str, object]) -> dict[str, SafeExpressionNode]:
    serialized = metadata["serialized_ast"]
    assert isinstance(serialized, dict)
    return {
        role: SafeExpressionNode.from_dict(serialized[role])
        for role in ("a_raw", "b_raw", "c_raw", "y_shape")
    }


def _evaluate_record(record: object, points: np.ndarray) -> dict[str, np.ndarray]:
    """Independent assembly from a generated record's public AST data."""

    nodes = record.concrete_nodes
    construction = record.construction
    a_raw = np.asarray(evaluate_expression(nodes["a_raw"], points), dtype=np.float64)
    b_raw = np.asarray(evaluate_expression(nodes["b_raw"], points), dtype=np.float64)
    c_raw = np.asarray(evaluate_expression(nodes["c_raw"], points), dtype=np.float64)
    a = construction["a_mid"] + construction["a_half"] * np.tanh(a_raw)
    b = construction["b_scale"] * np.tanh(b_raw)
    c_remizov = construction["c_scale"] * np.tanh(c_raw)
    c = c_remizov - construction["lambda"]
    solution = evaluate_solution(
        nodes["y_shape"],
        construction["bump_mu"],
        construction["bump_radius"],
        construction["solution_scale"],
        points,
    )
    y = np.asarray(solution.value, dtype=np.float64)
    dy = np.asarray(solution.d1, dtype=np.float64)
    d2y = np.asarray(solution.d2, dtype=np.float64)
    g = a * d2y + b * dy + c * y
    return {
        "a": a,
        "b": b,
        "c_remizov": c_remizov,
        "c": c,
        "g": g,
        "g_remizov": -g,
        "y": y,
        "dy": dy,
        "d2y": d2y,
    }


class AllMethodsDatasetV1Tests(unittest.TestCase):
    """Exercise one family in every complexity/regime cell."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = AllMethodsConfig(
            seed=2488,
            n_families=20,
            samples_per_family=2,
            ml_grid_size=32,
            validation_grid_size=257,
            max_attempts=100,
        )
        cls.dataset = generate_dataset(cls.config)
        cls._temporary = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls._temporary.name)
        cls.write_result = write_dataset(cls.dataset, cls.output_dir)
        cls.csv_path = cls.output_dir / OUTPUT_FILENAMES["csv"]
        cls.metadata_path = cls.output_dir / OUTPUT_FILENAMES["metadata"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_same_seed_reproduces_every_record_and_tensor_bit_for_bit(self) -> None:
        repeated = generate_dataset(self.config)
        self.assertEqual(repeated.equations, self.dataset.equations)
        self.assertEqual(repeated.metadata, self.dataset.metadata)
        self.assertEqual(repeated.rejections, self.dataset.rejections)
        self.assertEqual(set(repeated.arrays), set(self.dataset.arrays))
        for name in repeated.arrays:
            np.testing.assert_array_equal(repeated.arrays[name], self.dataset.arrays[name])

    def test_different_master_seed_changes_parameters_and_structures(self) -> None:
        changed = generate_dataset(replace(self.config, seed=self.config.seed + 1))
        self.assertNotEqual(
            [row["a_expr"] for row in changed.equations],
            [row["a_expr"] for row in self.dataset.equations],
        )
        self.assertNotEqual(
            [item["sample_seed"] for item in changed.metadata],
            [item["sample_seed"] for item in self.dataset.metadata],
        )
        self.assertNotEqual(
            [item["structure_signature"] for item in changed.metadata],
            [item["structure_signature"] for item in self.dataset.metadata],
        )

    def test_counts_ids_families_and_exact_csv_schema(self) -> None:
        records = self.dataset.records
        metadata = self.dataset.metadata
        self.assertEqual(len(records), 40)
        self.assertEqual(self.config.total_equations, 40)
        self.assertTrue(all(tuple(record.csv_row) == CSV_FIELDS for record in records))

        equation_ids = [item["equation_id"] for item in metadata]
        sample_seeds = [item["sample_seed"] for item in metadata]
        self.assertEqual(len(equation_ids), len(set(equation_ids)))
        self.assertEqual(len(sample_seeds), len(set(sample_seeds)))
        family_counts = Counter(item["family_id"] for item in metadata)
        self.assertEqual(len(family_counts), 20)
        self.assertEqual(set(family_counts.values()), {2})
        for record in records:
            self.assertEqual(float(record.csv_row["x0"]), -1.0)
            self.assertEqual(float(record.csv_row["x1"]), 1.0)

    def test_family_topology_is_fixed_but_numerical_parameters_vary(self) -> None:
        grouped: dict[str, list[object]] = defaultdict(list)
        for record in self.dataset.records:
            grouped[record.metadata["family_id"]].append(record)

        signatures: list[str] = []
        for family_id, records in grouped.items():
            with self.subTest(family_id=family_id):
                self.assertEqual(len({r.metadata["structure_signature"] for r in records}), 1)
                self.assertEqual(len({str(r.metadata["serialized_topology"]) for r in records}), 1)
                signatures.append(records[0].metadata["structure_signature"])
                formula_sets = {
                    tuple(r.csv_row[field] for field in ("a_expr", "b_expr", "c_expr", "g_expr"))
                    for r in records
                }
                self.assertEqual(len(formula_sets), len(records))

                for record in records:
                    rebuilt = _nodes_from_metadata(record.metadata)
                    self.assertEqual(topology_signature(rebuilt), record.metadata["structure_signature"])
                    for role, node in rebuilt.items():
                        self.assertEqual(
                            canonical_topology(node),
                            canonical_topology(records[0].concrete_nodes[role]),
                        )
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_twenty_complexity_regime_cells_are_exactly_balanced(self) -> None:
        counts = Counter(
            (item["complexity_level"], item["regime"])
            for item in self.dataset.metadata
        )
        self.assertEqual(set(counts), set((level, regime) for level in COMPLEXITY_LEVELS for regime in REGIMES))
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(
            Counter(item["complexity_level"] for item in self.dataset.metadata),
            Counter({level: 10 for level in COMPLEXITY_LEVELS}),
        )
        self.assertEqual(
            Counter(item["regime"] for item in self.dataset.metadata),
            Counter({regime: 8 for regime in REGIMES}),
        )

    def test_family_based_split_has_no_leakage_and_production_plan_is_exact(self) -> None:
        family_splits: dict[str, set[str]] = defaultdict(set)
        for item in self.dataset.metadata:
            family_splits[item["family_id"]].add(item["split"])
        self.assertTrue(all(len(splits) == 1 for splits in family_splits.values()))

        production_family_plan = generate_dataset(
            AllMethodsConfig(
                seed=2488,
                n_families=200,
                samples_per_family=1,
                ml_grid_size=8,
                validation_grid_size=33,
            )
        )
        family_counts = Counter(item["split"] for item in production_family_plan.metadata)
        self.assertEqual(family_counts, Counter(train=140, validation=30, test=30))
        cell_counts = Counter(
            (item["complexity_level"], item["regime"], item["split"])
            for item in production_family_plan.metadata
        )
        for cell_index, (level, regime) in enumerate(
            (level, regime) for level in COMPLEXITY_LEVELS for regime in REGIMES
        ):
            with self.subTest(level=level, regime=regime):
                self.assertEqual(cell_counts[(level, regime, "train")], 7)
                expected = (2, 1) if cell_index % 2 == 0 else (1, 2)
                self.assertEqual(cell_counts[(level, regime, "validation")], expected[0])
                self.assertEqual(cell_counts[(level, regime, "test")], expected[1])

    def test_declared_complexity_limits_hold_for_every_nonconstant_tree(self) -> None:
        for item in self.dataset.metadata:
            minimum_depth, maximum_depth, minimum_nodes, maximum_nodes = _COMPLEXITY_LIMITS[
                item["complexity_level"]
            ]
            per_tree = item["tree_metrics"]["per_tree"]
            for role, metrics in per_tree.items():
                with self.subTest(equation_id=item["equation_id"], role=role):
                    if item["regime"] == "constant" and role in {"a_raw", "b_raw", "c_raw"}:
                        self.assertEqual((metrics["depth"], metrics["nodes"]), (1, 1))
                    else:
                        self.assertGreaterEqual(metrics["depth"], minimum_depth)
                        self.assertLessEqual(metrics["depth"], maximum_depth)
                        self.assertGreaterEqual(metrics["nodes"], minimum_nodes)
                        self.assertLessEqual(metrics["nodes"], maximum_nodes)

    def test_constructive_certificates_and_parameter_ranges_are_exact(self) -> None:
        for record in self.dataset.records:
            item = record.metadata
            p = record.construction
            with self.subTest(equation_id=item["equation_id"]):
                self.assertGreaterEqual(item["a_lower_cert"], 0.5)
                self.assertLessEqual(item["a_lower_cert"], 1.5)
                self.assertLessEqual(item["a_upper_cert"], 3.0)
                self.assertGreaterEqual(
                    item["a_upper_cert"] - item["a_lower_cert"], 0.3 - 2.0e-15
                )
                self.assertAlmostEqual(item["a_lower_cert"], p["a_mid"] - p["a_half"], places=14)
                self.assertAlmostEqual(item["a_upper_cert"], p["a_mid"] + p["a_half"], places=14)
                self.assertAlmostEqual(item["b_abs_cert"], p["b_scale"], places=14)
                self.assertAlmostEqual(item["remizov_c_abs_cert"], p["c_scale"], places=14)
                self.assertAlmostEqual(item["remizov_lambda"], p["c_scale"] + p["gap"], places=14)
                self.assertAlmostEqual(
                    item["remizov_lambda"] - item["remizov_c_abs_cert"],
                    item["remizov_gap_cert"],
                    places=14,
                )
                self.assertGreater(item["remizov_gap_cert"], 0.0)
                self.assertGreater(item["remizov_lambda"], item["remizov_c_abs_cert"])
                self.assertGreaterEqual(p["bump_mu"], -0.25)
                self.assertLessEqual(p["bump_mu"], 0.25)
                self.assertGreaterEqual(p["bump_radius"], 1.5)
                self.assertLessEqual(p["bump_radius"], 3.0)
                self.assertGreaterEqual(item["target_amplitude"], 0.5)
                self.assertLessEqual(item["target_amplitude"], 1.5)
                self.assertAlmostEqual(item["support_left"], p["bump_mu"] - p["bump_radius"])
                self.assertAlmostEqual(item["support_right"], p["bump_mu"] + p["bump_radius"])

    def test_c_and_remizov_right_hand_side_identities_and_residual(self) -> None:
        grid = np.linspace(-1.0, 1.0, 513)
        for record in self.dataset.records:
            values = _evaluate_record(record, grid)
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
            with self.subTest(equation_id=record.metadata["equation_id"]):
                np.testing.assert_allclose(
                    values["c"],
                    values["c_remizov"] - record.metadata["remizov_lambda"],
                    rtol=0.0,
                    atol=2.0e-14,
                )
                np.testing.assert_array_equal(values["g_remizov"], -values["g"])
                self.assertLessEqual(float(np.max(np.abs(residual))), 1.0e-10 * scale)

    def test_finiteness_safety_and_dense_certificate_smoke_checks(self) -> None:
        wide_grid = np.linspace(-10.0, 10.0, 1025)
        for record in self.dataset.records:
            values = _evaluate_record(record, wide_grid)
            item = record.metadata
            with self.subTest(equation_id=item["equation_id"]):
                for name, values_for_name in values.items():
                    self.assertTrue(np.all(np.isfinite(values_for_name)), name)
                    if name in {"a", "b", "c", "g", "y", "dy", "d2y"}:
                        self.assertLessEqual(
                            float(np.max(np.abs(values_for_name))),
                            self.config.max_abs_safety,
                        )
                tolerance = 1.0e-12
                self.assertGreaterEqual(float(np.min(values["a"])), item["a_lower_cert"] - tolerance)
                self.assertLessEqual(float(np.max(values["a"])), item["a_upper_cert"] + tolerance)
                self.assertLessEqual(float(np.max(np.abs(values["b"]))), item["b_abs_cert"] + tolerance)
                self.assertLessEqual(
                    float(np.max(np.abs(values["c_remizov"]))),
                    item["remizov_c_abs_cert"] + tolerance,
                )
                if item["regime"] == "constant":
                    self.assertEqual(float(np.ptp(values["a"])), 0.0)
                    self.assertEqual(float(np.ptp(values["b"])), 0.0)
                    self.assertEqual(float(np.ptp(values["c_remizov"])), 0.0)

    def test_initial_conditions_compact_support_and_normalization(self) -> None:
        normalization_grid = np.linspace(-1.0, 1.0, self.config.validation_grid_size)
        nonzero_y0 = 0
        nonzero_v0 = 0
        for record in self.dataset.records:
            item = record.metadata
            initial = _evaluate_record(record, np.asarray([-1.0]))
            with self.subTest(equation_id=item["equation_id"]):
                self.assertAlmostEqual(float(record.csv_row["y0"]), initial["y"][0], places=13)
                self.assertAlmostEqual(float(record.csv_row["v0"]), initial["dy"][0], places=13)
                self.assertGreaterEqual(np.hypot(initial["y"][0], initial["dy"][0]), 1.0e-8)
                self.assertLess(item["support_left"], -1.0)
                self.assertGreater(item["support_right"], 1.0)
                support_probes = np.asarray(
                    [
                        item["support_left"] - 1.0,
                        item["support_left"],
                        item["support_right"],
                        item["support_right"] + 1.0,
                    ]
                )
                supported = _evaluate_record(record, support_probes)
                for name in ("y", "dy", "d2y", "g", "g_remizov"):
                    np.testing.assert_array_equal(supported[name], np.zeros(4))
                normalized = _evaluate_record(record, normalization_grid)
                self.assertAlmostEqual(
                    float(np.max(np.abs(normalized["y"]))),
                    item["target_amplitude"],
                    places=12,
                )
            nonzero_y0 += abs(initial["y"][0]) > 1.0e-12
            nonzero_v0 += abs(initial["dy"][0]) > 1.0e-12
        self.assertGreaterEqual(nonzero_y0, 2)
        self.assertGreaterEqual(nonzero_v0, 2)

    def test_every_serialized_tree_uses_only_the_global_safe_whitelist(self) -> None:
        for item in self.dataset.metadata:
            for role, root in _nodes_from_metadata(item).items():
                operations = {node.op for node in iter_nodes(root)}
                with self.subTest(equation_id=item["equation_id"], role=role):
                    self.assertLessEqual(operations, ALLOWED_OPS)
                    self.assertTrue(operations.isdisjoint(FORBIDDEN_OPS))

    def test_npz_schema_channel_order_values_and_absence_of_method_labels(self) -> None:
        arrays = self.dataset.arrays
        n = self.config.total_equations
        m = self.config.ml_grid_size
        self.assertEqual(set(arrays), _NPZ_MEMBERS)
        self.assertEqual(arrays["X"].shape, (n, 4, m))
        self.assertEqual(arrays["x_grid"].shape, (m,))
        self.assertEqual(arrays["y_exact"].shape, (n, m))
        self.assertEqual(arrays["dy_exact"].shape, (n, m))
        self.assertEqual(arrays["y0"].shape, (n,))
        self.assertEqual(arrays["v0"].shape, (n,))
        self.assertEqual(arrays["channel_order"].tolist(), list(CHANNEL_ORDER))
        self.assertTrue(all(not value.dtype.hasobject for value in arrays.values()))
        for name in ("x_grid", "X", "y0", "v0", "y_exact", "dy_exact"):
            self.assertTrue(np.all(np.isfinite(arrays[name])))

        for index in (0, n // 2, n - 1):
            values = _evaluate_record(self.dataset.records[index], arrays["x_grid"])
            expected_x = np.stack([values[name] for name in CHANNEL_ORDER])
            np.testing.assert_array_equal(arrays["X"][index], expected_x)
            np.testing.assert_array_equal(arrays["y_exact"][index], values["y"])
            np.testing.assert_array_equal(arrays["dy_exact"][index], values["dy"])

        forbidden_label_fragments = ("best_method", "method_label", "solver_label", "target_method")
        all_field_names = set(CSV_FIELDS) | set(arrays)
        for item in self.dataset.metadata:
            all_field_names.update(item)
        self.assertFalse(
            any(fragment in field.lower() for field in all_field_names for fragment in forbidden_label_fragments)
        )
        expression_keys = [
            tuple(record.csv_row[field] for field in ("a_expr", "b_expr", "c_expr", "g_expr"))
            for record in self.dataset.records
        ]
        feature_keys = [record.ml_channels.tobytes() for record in self.dataset.records]
        self.assertEqual(len(expression_keys), len(set(expression_keys)))
        self.assertEqual(len(feature_keys), len(set(feature_keys)))

    def test_metadata_ast_reconstructs_the_authoritative_functions(self) -> None:
        probes = np.linspace(-1.0, 1.0, 37)
        for record in self.dataset.records:
            rebuilt = _nodes_from_metadata(record.metadata)
            with self.subTest(equation_id=record.metadata["equation_id"]):
                for role in rebuilt:
                    np.testing.assert_array_equal(
                        evaluate_expression(rebuilt[role], probes),
                        evaluate_expression(record.concrete_nodes[role], probes),
                    )
                construction = record.metadata["serialized_ast"]["construction"]
                for key in (
                    "a_mid",
                    "a_half",
                    "b_scale",
                    "c_scale",
                    "lambda",
                    "gap",
                    "bump_mu",
                    "bump_radius",
                    "solution_scale",
                ):
                    self.assertEqual(construction[key], record.construction[key])

    def test_written_artifacts_pass_full_non_solver_validator(self) -> None:
        result = validate_dataset_artifacts(self.output_dir)
        self.assertTrue(result["valid"])
        self.assertEqual(result["total_equations"], self.config.total_equations)
        self.assertEqual(result["total_families"], self.config.n_families)
        self.assertFalse(result["family_split_leakage"])
        self.assertEqual(
            set(result["sha256"]),
            {filename for filename in OUTPUT_FILENAMES.values()},
        )

    def test_safe_loader_is_compatible_for_first_middle_and_last_records(self) -> None:
        loaded = load_all_methods_equations(self.csv_path, self.metadata_path)
        self.assertEqual(len(loaded), self.config.total_equations)
        indices = (0, len(loaded) // 2, len(loaded) - 1)
        probes = np.linspace(-1.0, 1.0, 19)
        for index in indices:
            equation = loaded[index]
            source = self.dataset.records[index]
            expected = _evaluate_record(source, probes)
            cauchy = equation.cauchy_problem
            remizov = equation.remizov_problem
            with self.subTest(index=index, equation_id=equation.equation_id):
                self.assertEqual(equation.equation_id, source.metadata["equation_id"])
                self.assertAlmostEqual(cauchy.x0, -1.0)
                self.assertAlmostEqual(cauchy.x1, 1.0)
                self.assertAlmostEqual(cauchy.y0, float(source.csv_row["y0"]), places=13)
                self.assertAlmostEqual(cauchy.v0, float(source.csv_row["v0"]), places=13)
                np.testing.assert_allclose(
                    [cauchy.a(float(x)) for x in probes], expected["a"], rtol=2.0e-14, atol=2.0e-14
                )
                np.testing.assert_allclose(
                    [cauchy.b(float(x)) for x in probes], expected["b"], rtol=2.0e-14, atol=2.0e-14
                )
                np.testing.assert_allclose(
                    [cauchy.c(float(x)) for x in probes], expected["c"], rtol=2.0e-14, atol=2.0e-14
                )
                np.testing.assert_allclose(
                    [cauchy.g(float(x)) for x in probes], expected["g"], rtol=2.0e-14, atol=2.0e-14
                )
                np.testing.assert_allclose(remizov.c_remizov(probes), expected["c_remizov"])
                np.testing.assert_array_equal(remizov.g_remizov(probes), -expected["g"])
                np.testing.assert_allclose(equation.y_exact(probes), expected["y"])
                np.testing.assert_allclose(equation.dy_exact(probes), expected["dy"])
                np.testing.assert_allclose(equation.d2y_exact(probes), expected["d2y"])
                np.testing.assert_allclose(
                    np.asarray([cauchy.c(float(x)) for x in probes]),
                    remizov.c_remizov(probes) - remizov.lambda_value,
                )


if __name__ == "__main__":
    unittest.main()
