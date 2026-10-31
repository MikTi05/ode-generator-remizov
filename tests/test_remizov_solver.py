import math
from dataclasses import replace
import unittest

import numpy as np

from remizov_benchmark_loader import load_remizov_problem
from remizov_solver import (
    RemizovConfig,
    RemizovProblem,
    _RemizovFailure,
    _WorkCounters,
    _apply_operator_on_grid,
    compute_truncation_T,
    solve_remizov,
)


def constant_function(value):
    def evaluate(points):
        return np.full(np.asarray(points).shape, value, dtype=float)

    return evaluate


def toy_problem(**overrides):
    values = {
        "problem_id": "toy",
        "a": constant_function(1.0),
        "b": constant_function(0.0),
        "c_remizov": constant_function(0.0),
        "g_remizov": lambda x: np.exp(-(np.asarray(x) ** 2)),
        "lambda_value": 1.0,
        "a_lower_bound": 1.0,
        "a_upper_bound": 1.0,
        "b_abs_upper_bound": 0.0,
        "c_abs_upper_bound": 0.0,
        "bounds_certification": "analytic",
        "bounds_domain": "all_real_x",
        "x_target_min": -1.0,
        "x_target_max": 1.0,
    }
    values.update(overrides)
    return RemizovProblem(**values)


class RemizovSolverTests(unittest.TestCase):
    def test_r1_truncation_formula(self) -> None:
        problem = load_remizov_problem("R1")
        self.assertAlmostEqual(compute_truncation_T(problem, 1.0e-8), math.log(2.0e8), places=13)

    def test_constant_function_is_fixed_by_r1_operator(self) -> None:
        problem = load_remizov_problem("R1")
        source_x = np.linspace(-3.0, 3.0, 601)
        destination_x = np.linspace(-1.0, 1.0, 17)
        counters = _WorkCounters()
        result = _apply_operator_on_grid(
            problem,
            source_x,
            np.ones_like(source_x),
            destination_x,
            tau=0.25,
            counters=counters,
        )
        np.testing.assert_allclose(result, 1.0, rtol=0.0, atol=1.0e-14)
        self.assertEqual(counters.n_operator_applications, 1)
        self.assertEqual(counters.n_spline_builds, 1)
        self.assertEqual(counters.n_interpolation_calls, 4)
        self.assertEqual(counters.n_interpolated_values, 4 * destination_x.size)
        self.assertEqual(counters.spatial_grid_points_total_processed, destination_x.size)

    def test_operator_weights_drift_sign_and_absence_of_lambda(self) -> None:
        base = toy_problem(
            a=constant_function(1.2),
            b=constant_function(-0.3),
            c_remizov=constant_function(0.4),
            lambda_value=1.0,
            a_lower_bound=1.2,
            a_upper_bound=1.2,
            b_abs_upper_bound=0.3,
            c_abs_upper_bound=0.4,
        )
        source_x = np.linspace(-3.0, 3.0, 601)
        source_u = source_x**2
        destination_x = np.linspace(-1.0, 1.0, 11)
        tau = 0.04
        shift = 2.0 * math.sqrt(1.2 * tau)
        expected = (
            0.25 * (destination_x + shift) ** 2
            + 0.25 * (destination_x - shift) ** 2
            + 0.50 * (destination_x + 2.0 * (-0.3) * tau) ** 2
            + tau * 0.4 * destination_x**2
        )
        first = _apply_operator_on_grid(
            base, source_x, source_u, destination_x, tau, _WorkCounters()
        )
        second = _apply_operator_on_grid(
            replace(base, lambda_value=2.0),
            source_x,
            source_u,
            destination_x,
            tau,
            _WorkCounters(),
        )
        np.testing.assert_allclose(first, expected, rtol=0.0, atol=2.0e-14)
        np.testing.assert_array_equal(first, second)

    def test_t_zero_returns_exact_zero_without_numerical_work(self) -> None:
        problem = toy_problem(lambda_value=3.0)
        result = solve_remizov(
            problem,
            np.array([-1.0, 0.0, 1.0]),
            RemizovConfig(n=1, tail_eps=1.0),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.T, 0.0)
        np.testing.assert_array_equal(result.f, np.zeros(3))
        self.assertEqual(result.final_nt, 0)
        self.assertEqual(result.final_dx, 0.0)
        self.assertEqual(result.n_quadrature_node_evaluations, 0)
        self.assertEqual(result.n_operator_applications, 0)
        self.assertEqual(result.spatial_error_history, [])
        self.assertEqual(result.spatial_convergence_thresholds, [])

    def test_api_errors_raise_value_error(self) -> None:
        problem = toy_problem()
        for config in (RemizovConfig(n=0), RemizovConfig(n=True), RemizovConfig(n=1, tail_eps=0.0)):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    solve_remizov(problem, np.array([0.0]), config)
        for target in (np.array([]), np.array([0.0, 0.0]), np.array([math.nan])):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    solve_remizov(problem, target, RemizovConfig(n=1))

    def test_failure_statuses_preserve_empty_final_values(self) -> None:
        target = np.array([0.0])
        cases = (
            (
                "invalid_remizov_conditions",
                replace(toy_problem(), lambda_value=0.0),
                RemizovConfig(n=1),
            ),
            (
                "nonfinite_state",
                replace(toy_problem(), g_remizov=constant_function(math.inf)),
                RemizovConfig(n=1),
            ),
            (
                "work_domain_too_large",
                toy_problem(),
                RemizovConfig(n=1, max_work_grid_points=4),
            ),
            (
                "quadrature_not_converged",
                toy_problem(),
                RemizovConfig(n=1, max_quadrature_refinements=0),
            ),
            (
                "spatial_not_converged",
                toy_problem(),
                RemizovConfig(
                    n=1,
                    quadrature_rtol=1.0,
                    quadrature_atol=1.0,
                    max_quadrature_refinements=1,
                    max_spatial_refinements=0,
                ),
            ),
        )
        for reason, problem, config in cases:
            with self.subTest(reason=reason):
                result = solve_remizov(problem, target, config)
                self.assertFalse(result.success)
                self.assertEqual(result.failure_reason, reason)
                self.assertEqual(result.f.size, 0)
                np.testing.assert_array_equal(result.x, target)
                self.assertEqual(
                    len(result.spatial_error_history),
                    len(result.spatial_convergence_thresholds),
                )
        cap_result = solve_remizov(
            toy_problem(), target, RemizovConfig(n=1, max_work_grid_points=4)
        )
        self.assertGreater(cap_result.work_grid_points_required_max, 4)
        self.assertGreater(cap_result.work_domain_width_max, 0.0)

    def test_low_level_failure_statuses(self) -> None:
        source_x = np.linspace(-1.0, 1.0, 101)
        source_u = np.ones_like(source_x)
        destination_x = np.array([0.0])
        cases = (
            (
                "nonfinite_coefficient",
                replace(toy_problem(), a=constant_function(math.nan)),
                source_x,
                source_u,
                destination_x,
                0.01,
            ),
            (
                "certified_bound_violation",
                replace(toy_problem(), b=constant_function(0.1)),
                source_x,
                source_u,
                destination_x,
                0.01,
            ),
            (
                "interpolation_out_of_bounds",
                toy_problem(),
                source_x,
                source_u,
                np.array([0.9]),
                1.0,
            ),
            (
                "interpolation_failure",
                toy_problem(),
                np.array([-1.0, 0.0, 1.0]),
                np.ones(3),
                destination_x,
                0.01,
            ),
        )
        for reason, problem, sx, su, dx, tau in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(_RemizovFailure) as caught:
                    _apply_operator_on_grid(problem, sx, su, dx, tau, _WorkCounters())
                self.assertEqual(caught.exception.reason, reason)

    def test_manual_oracles_r1_n1_and_n2(self) -> None:
        problem = load_remizov_problem("R1")
        # Both oracle cases use the same predeclared scientific-verification
        # budget; there is no problem-dependent retry.
        for n, oracle in ((1, 1.48), (2, 1.1728395061728395)):
            with self.subTest(n=n):
                config = RemizovConfig(
                    n=n,
                    tail_eps=1.0e-10,
                    quadrature_rtol=1.0e-9,
                    quadrature_atol=1.0e-11,
                    spatial_rtol=1.0e-9,
                    spatial_atol=1.0e-11,
                    max_quadrature_refinements=8,
                    max_spatial_refinements=5,
                )
                result = solve_remizov(problem, np.array([0.0]), config)
                self.assertTrue(result.success, result.failure_reason)
                self.assertLess(abs(result.f[0] - oracle), 1.0e-7)
                self.assertEqual(
                    result.n_operator_applications,
                    n * result.n_quadrature_node_evaluations,
                )
                self.assertEqual(result.n_spline_builds, result.n_operator_applications)
                self.assertEqual(result.n_interpolation_calls, 4 * result.n_operator_applications)
                self.assertEqual(
                    result.n_interpolated_values,
                    4 * result.spatial_grid_points_total_processed,
                )
                self.assertEqual(
                    len(result.spatial_error_history),
                    len(result.spatial_convergence_thresholds),
                )
                self.assertGreater(len(result.spatial_error_history), 0)
                self.assertLessEqual(
                    result.spatial_error_history[-1],
                    result.spatial_convergence_thresholds[-1],
                )

    def test_spatial_failure_preserves_paired_error_and_threshold(self) -> None:
        result = solve_remizov(
            toy_problem(),
            np.array([0.0]),
            RemizovConfig(
                n=1,
                quadrature_rtol=1.0,
                quadrature_atol=1.0,
                max_quadrature_refinements=1,
                spatial_rtol=1.0e-30,
                spatial_atol=1.0e-30,
                max_spatial_refinements=1,
            ),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "spatial_not_converged")
        self.assertEqual(len(result.spatial_error_history), 1)
        self.assertEqual(len(result.spatial_convergence_thresholds), 1)
        self.assertGreater(
            result.spatial_error_history[0],
            result.spatial_convergence_thresholds[0],
        )


if __name__ == "__main__":
    unittest.main()
