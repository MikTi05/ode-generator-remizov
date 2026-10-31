import csv
from pathlib import Path
import unittest

import sympy as sp

from create_remizov_benchmark import FIELDS, build_benchmark_rows
from validate_remizov_benchmark import X, load_benchmark, validate_benchmark


BENCHMARK_PATH = Path("data/remizov_benchmark.csv")


class RemizovBenchmarkTests(unittest.TestCase):
    def test_csv_is_deterministically_generated_and_does_not_duplicate_q_r(self) -> None:
        with BENCHMARK_PATH.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(tuple(reader.fieldnames or ()), FIELDS)
            self.assertNotIn("q_expr", reader.fieldnames or ())
            self.assertNotIn("r_expr", reader.fieldnames or ())
            rows = list(reader)
        self.assertEqual([row["benchmark_id"] for row in rows], ["R1", "R2", "R3", "R4", "R5"])
        self.assertEqual(rows, build_benchmark_rows())

    def test_all_symbolic_and_numerical_validations_pass(self) -> None:
        results = validate_benchmark(BENCHMARK_PATH, grid_size=20001)
        self.assertEqual(len(results), 5)
        for result in results:
            with self.subTest(benchmark_id=result.benchmark_id):
                self.assertTrue(result.symbolic_residual_zero)
                self.assertTrue(result.exact_solution_matches)
                self.assertTrue(result.initial_conditions_match)
                self.assertTrue(result.a_lower_positive)
                self.assertTrue(result.lambda_dominates_c)
                self.assertTrue(result.certified_metadata_valid)
                self.assertTrue(result.numerical.finite_coefficients)
                self.assertTrue(result.numerical.finite_g)
                self.assertLessEqual(result.numerical.residual_max_abs, 1.0e-10)
                self.assertLessEqual(result.numerical.maximum_bound_violation, 1.0e-12)
                self.assertTrue(result.valid)

    def test_ordinary_equation_identity_matches_exact_solution(self) -> None:
        tasks = load_benchmark(BENCHMARK_PATH)
        for task in tasks:
            with self.subTest(benchmark_id=task.benchmark_id):
                exact_d1 = sp.diff(task.exact_solution_expr, X)
                exact_d2 = sp.diff(exact_d1, X)
                cross_residual = sp.simplify(
                    task.a_expr * exact_d2
                    + task.b_expr * exact_d1
                    + task.q_expr * task.exact_solution_expr
                    - task.r_expr
                )
                self.assertEqual(cross_residual, 0)

if __name__ == "__main__":
    unittest.main()
