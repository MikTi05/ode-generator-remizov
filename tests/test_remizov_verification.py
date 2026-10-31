"""End-to-end verification tests for the serial Remizov implementation.

These tests deliberately exercise the public CSV -> loader -> problem -> solver
path.  They complement the low-level operator tests without turning the unit
suite into the full (and more expensive) verification report run.
"""

from pathlib import Path
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import verify_remizov_solver as verification
from remizov_benchmark_loader import (
    load_remizov_benchmark,
    make_verification_target_x,
)
from remizov_solver import RemizovConfig, solve_remizov


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "data" / "remizov_benchmark.csv"


class RemizovBenchmarkPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problems = load_remizov_benchmark(BENCHMARK_PATH)

    def test_csv_loader_problem_solver_pipeline_for_r1_through_r5(self) -> None:
        self.assertEqual(
            [problem.problem_id for problem in self.problems],
            ["R1", "R2", "R3", "R4", "R5"],
        )

        for problem in self.problems:
            with self.subTest(problem_id=problem.problem_id):
                # These are the analytically certified benchmark conditions,
                # not conclusions drawn from a finite numerical grid.
                self.assertEqual(problem.bounds_certification, "analytic")
                self.assertEqual(problem.bounds_domain, "all_real_x")
                self.assertGreater(problem.a_lower_bound, 0.0)
                self.assertGreater(
                    problem.lambda_value, problem.c_abs_upper_bound
                )

                target_x = make_verification_target_x(problem)
                result = solve_remizov(
                    problem,
                    target_x,
                    RemizovConfig(
                        n=1,
                        max_quadrature_refinements=8,
                        max_spatial_refinements=5,
                    ),
                )

                self.assertTrue(result.success, result.failure_reason)
                self.assertEqual(result.failure_reason, "")
                np.testing.assert_array_equal(result.x, target_x)
                self.assertEqual(result.f.shape, target_x.shape)
                self.assertTrue(np.all(np.isfinite(result.f)))
                self.assertTrue(math.isfinite(result.T))
                self.assertGreater(result.T, 0.0)
                self._assert_success_diagnostic_invariants(result)

    def test_strict_r1_manual_oracles(self) -> None:
        r1 = self.problems[0]
        strict_common = {
            "tail_eps": 1.0e-10,
            "quadrature_rtol": 1.0e-9,
            "quadrature_atol": 1.0e-11,
            "spatial_rtol": 1.0e-9,
            "spatial_atol": 1.0e-11,
            # This is the single, predeclared manual-verification budget.  A
            # failed run must remain failed rather than being retried with a
            # problem-dependent larger budget.
            "max_quadrature_refinements": 8,
            "max_spatial_refinements": 5,
        }
        cases = (
            (1, 1.48),
            (2, 1.1728395061728395),
        )

        for n, expected in cases:
            with self.subTest(n=n):
                result = solve_remizov(
                    r1,
                    np.array([0.0]),
                    RemizovConfig(n=n, **strict_common),
                )
                self.assertTrue(result.success, result.failure_reason)
                self.assertLess(abs(result.f[0] - expected), 1.0e-7)
                self._assert_success_diagnostic_invariants(result)

    def test_r1_finite_chernoff_approximations_converge_toward_exact(self) -> None:
        r1 = self.problems[0]
        target_x = make_verification_target_x(r1)
        exact = np.exp(-(target_x**2))
        errors = []

        # The full verification script also covers n=16 and n=32.  Stopping at
        # n=8 retains the established decreasing-error regression while
        # keeping this integration test suitable for every unit-test run.
        for n in (1, 2, 4, 8):
            with self.subTest(n=n):
                result = solve_remizov(
                    r1,
                    target_x,
                    RemizovConfig(
                        n=n,
                        max_quadrature_refinements=8,
                        max_spatial_refinements=5,
                    ),
                )
                self.assertTrue(result.success, result.failure_reason)
                error = float(np.max(np.abs(result.f - exact)))
                self.assertTrue(math.isfinite(error))
                errors.append(error)

        self.assertTrue(
            all(finer < coarser for coarser, finer in zip(errors, errors[1:])),
            f"expected decreasing R1 errors, got {errors}",
        )
        observed_orders = [
            math.log2(coarser / finer)
            for coarser, finer in zip(errors, errors[1:])
        ]
        self.assertTrue(all(order > 0.0 for order in observed_orders))

    def _assert_success_diagnostic_invariants(self, result) -> None:
        """Pin the documented cumulative-counter and paired-history semantics."""

        self.assertTrue(result.success, result.failure_reason)
        self.assertEqual(
            len(result.spatial_error_history),
            len(result.spatial_convergence_thresholds),
        )
        self.assertEqual(
            result.spatial_refinements,
            len(result.spatial_error_history),
        )
        self.assertEqual(
            len(result.spatial_dx_levels),
            len(result.spatial_error_history) + 1,
        )
        self.assertTrue(
            all(
                math.isfinite(error)
                and math.isfinite(threshold)
                and threshold > 0.0
                for error, threshold in zip(
                    result.spatial_error_history,
                    result.spatial_convergence_thresholds,
                )
            )
        )

        self.assertEqual(
            result.n_quadrature_node_evaluations,
            sum(
                sum(history.nt_levels)
                for history in result.quadrature_history
            ),
        )
        self.assertEqual(
            result.quadrature_refinements,
            sum(
                len(history.error_estimates)
                for history in result.quadrature_history
            ),
        )
        self.assertEqual(
            result.n_operator_applications,
            result.n * result.n_quadrature_node_evaluations,
        )
        self.assertEqual(result.n_spline_builds, result.n_operator_applications)
        self.assertEqual(
            result.n_interpolation_calls,
            4 * result.n_operator_applications,
        )
        self.assertEqual(
            result.n_interpolated_values,
            4 * result.spatial_grid_points_total_processed,
        )


class RemizovVerificationStudyContractTests(unittest.TestCase):
    def test_library_defaults_remain_distinct_from_verification_budget(self) -> None:
        defaults = RemizovConfig(n=1)
        self.assertEqual(defaults.max_quadrature_refinements, 4)
        self.assertEqual(defaults.max_spatial_refinements, 4)

        configured = verification.make_verification_config(n=7)
        self.assertEqual(configured.n, 7)
        expected = {
            "tail_eps": 1.0e-8,
            "nt_initial": 8,
            "quadrature_rtol": 1.0e-7,
            "quadrature_atol": 1.0e-9,
            "max_quadrature_refinements": 8,
            "dx_initial": 0.05,
            "spatial_rtol": 1.0e-7,
            "spatial_atol": 1.0e-9,
            "max_spatial_refinements": 5,
            "max_work_grid_points": 100000,
        }
        for name, value in expected.items():
            with self.subTest(parameter=name):
                self.assertEqual(getattr(configured, name), value)

    def test_manual_configuration_is_strict_and_has_one_fixed_budget(self) -> None:
        configured = verification.make_manual_oracle_config(n=2)
        self.assertEqual(configured.n, 2)
        self.assertEqual(configured.tail_eps, 1.0e-10)
        self.assertEqual(configured.nt_initial, 8)
        self.assertEqual(configured.quadrature_rtol, 1.0e-9)
        self.assertEqual(configured.quadrature_atol, 1.0e-11)
        self.assertEqual(configured.max_quadrature_refinements, 8)
        self.assertEqual(configured.dx_initial, 0.05)
        self.assertEqual(configured.spatial_rtol, 1.0e-9)
        self.assertEqual(configured.spatial_atol, 1.0e-11)
        self.assertEqual(configured.max_spatial_refinements, 5)
        self.assertEqual(configured.max_work_grid_points, 100000)

    def test_study_uses_one_uniform_budget_no_retry_and_includes_n32(self) -> None:
        calls = []

        def fake_solve(problem, target_x, config):
            points = np.asarray(target_x, dtype=float).copy()
            calls.append((problem.problem_id, points, config))
            # A deliberately nonconverged reportable case proves that the
            # driver records one failure and does not invoke a hidden retry.
            failed = (
                problem.problem_id == "R1"
                and config.n == 4
                and points.size == 257
                and config.tail_eps == 1.0e-8
            )
            return SimpleNamespace(
                n=config.n,
                success=not failed,
                failure_reason="quadrature_not_converged" if failed else "",
                x=points,
                f=np.array([], dtype=float) if failed else np.exp(-(points**2)),
                final_dx=config.dx_initial,
                final_nt=config.nt_initial,
                runtime_seconds=0.0,
            )

        with patch.object(verification, "solve_remizov", side_effect=fake_solve):
            studies, elapsed = verification.run_verification(BENCHMARK_PATH)

        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(len(studies), 16)  # 2 manual + 6 Stage A + 8 Stage B
        self.assertEqual(len(calls), 16)
        self.assertEqual(
            [run.result.n for run in studies if run.phase == "A"],
            [1, 2, 4, 8, 16, 32],
        )
        self.assertEqual(
            sum(
                problem_id == "R1"
                and config.n == 4
                and points.size == 257
                and config.tail_eps == 1.0e-8
                for problem_id, points, config in calls
            ),
            1,
        )
        failed = [run for run in studies if not run.result.success]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].result.failure_reason, "quadrature_not_converged")

        for run in studies:
            with self.subTest(label=run.label):
                config = run.config
                self.assertEqual(config.max_quadrature_refinements, 8)
                self.assertEqual(config.max_spatial_refinements, 5)
                if run.phase == "manual":
                    self.assertEqual(config.tail_eps, 1.0e-10)
                    self.assertEqual(config.quadrature_rtol, 1.0e-9)
                    self.assertEqual(config.quadrature_atol, 1.0e-11)
                    self.assertEqual(config.spatial_rtol, 1.0e-9)
                    self.assertEqual(config.spatial_atol, 1.0e-11)
                else:
                    self.assertIn(run.phase, {"A", "B"})
                    self.assertEqual(config.tail_eps, 1.0e-8)
                    self.assertEqual(config.quadrature_rtol, 1.0e-7)
                    self.assertEqual(config.quadrature_atol, 1.0e-9)
                    self.assertEqual(config.spatial_rtol, 1.0e-7)
                    self.assertEqual(config.spatial_atol, 1.0e-9)

    def test_control_indicator_helpers_are_diagnostics_not_retry_controls(self) -> None:
        run = SimpleNamespace(
            result=SimpleNamespace(
                success=True,
                tail_eps=1.0e-8,
                quadrature_error_estimate=2.0e-8,
                spatial_error_estimate=3.0e-8,
            ),
            max_error=1.5e-2,
        )
        indicator = verification.compute_control_indicator(run)
        ratio = verification.compute_control_to_method_error_ratio(run)
        self.assertEqual(indicator, 3.0e-8)
        self.assertAlmostEqual(ratio, 3.0e-8 / 1.5e-2)

        run.result.spatial_error_estimate = math.nan
        self.assertTrue(math.isnan(verification.compute_control_indicator(run)))
        self.assertTrue(
            math.isnan(verification.compute_control_to_method_error_ratio(run))
        )


if __name__ == "__main__":
    unittest.main()
