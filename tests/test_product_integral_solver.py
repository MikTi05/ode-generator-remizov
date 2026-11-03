import math
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm as oracle_expm

from src.numerical_methods.product_integral_solver import (
    INVARIANT_FAIL_TOL,
    METHOD_ID,
    solve_product_integral_left_exponential,
)


def _solve(**overrides):
    arguments = {
        "equation_id": "unit-test",
        "a": lambda _x: 1.0,
        "b": lambda _x: 0.0,
        "c": lambda _x: 0.0,
        "g": lambda _x: 0.0,
        "x0": 0.0,
        "x1": 1.0,
        "y0": 0.0,
        "v0": 0.0,
        "n_steps": 10,
    }
    arguments.update(overrides)
    return solve_product_integral_left_exponential(**arguments)


class ProductIntegralConstantCoefficientTests(unittest.TestCase):
    def test_constant_forcing_is_exact_on_the_whole_trajectory(self) -> None:
        x0 = 0.0
        y0 = 0.0
        v0 = 0.0

        for n_steps in (1, 10, 100):
            with self.subTest(n_steps=n_steps):
                result = _solve(
                    equation_id="constant-forcing",
                    g=lambda _x: 1.0,
                    x0=x0,
                    x1=1.0,
                    y0=y0,
                    v0=v0,
                    n_steps=n_steps,
                )

                self.assertTrue(result.success, result.message)
                self.assertEqual(len(result.x), n_steps + 1)
                errors = []
                for x, y, dy in zip(result.x, result.y, result.dy):
                    t = x - x0
                    errors.extend(
                        (
                            abs(y - (y0 + v0 * t + 0.5 * t * t)),
                            abs(dy - (v0 + t)),
                        )
                    )
                self.assertLess(max(errors), 2.0e-12)

    def test_harmonic_oscillator_is_exact_on_the_whole_trajectory(self) -> None:
        x0 = 0.0
        y0 = 0.0
        v0 = 1.0

        for n_steps in (1, 10, 100):
            with self.subTest(n_steps=n_steps):
                result = _solve(
                    equation_id="harmonic",
                    c=lambda _x: 1.0,
                    x0=x0,
                    x1=1.0,
                    y0=y0,
                    v0=v0,
                    n_steps=n_steps,
                )

                self.assertTrue(result.success, result.message)
                errors = []
                for x, y, dy in zip(result.x, result.y, result.dy):
                    t = x - x0
                    expected_y = math.sin(t)
                    expected_dy = math.cos(t)
                    errors.extend((abs(y - expected_y), abs(dy - expected_dy)))
                self.assertLess(max(errors), 3.0e-12)

    def test_general_constant_coefficients_match_independent_expm_oracle(self) -> None:
        a_value = 1.7
        b_value = -0.4
        c_value = 2.3
        g_value = 0.8
        x0 = -0.7
        x1 = 0.9
        y0 = 0.35
        v0 = -1.1
        n_steps = 7

        result = _solve(
            equation_id="general-constant",
            a=lambda _x: a_value,
            b=lambda _x: b_value,
            c=lambda _x: c_value,
            g=lambda _x: g_value,
            x0=x0,
            x1=x1,
            y0=y0,
            v0=v0,
            n_steps=n_steps,
        )

        self.assertTrue(result.success, result.message)
        matrix = np.array(
            [
                [0.0, 1.0, 0.0],
                [-c_value / a_value, -b_value / a_value, g_value / a_value],
                [0.0, 0.0, 0.0],
            ]
        )
        initial = np.array([y0, v0, 1.0])
        for x, y, dy in zip(result.x, result.y, result.dy):
            expected = oracle_expm((x - x0) * matrix) @ initial
            np.testing.assert_allclose([y, dy], expected[:2], rtol=2.0e-13, atol=2.0e-13)


class ProductIntegralDiscretizationTests(unittest.TestCase):
    def test_manufactured_exponential_solution_has_first_order_convergence(self) -> None:
        # y=e^x solves y'' + x*y' + y = (x+2)e^x.
        errors = []
        for n_steps in (40, 80, 160, 320):
            result = _solve(
                equation_id="variable-manufactured",
                b=lambda x: x,
                c=lambda _x: 1.0,
                g=lambda x: (x + 2.0) * math.exp(x),
                y0=1.0,
                v0=1.0,
                n_steps=n_steps,
            )
            self.assertTrue(result.success, result.message)
            trajectory_error = max(
                abs(y - math.exp(x)) for x, y in zip(result.x, result.y)
            )
            errors.append(trajectory_error)

        self.assertTrue(
            all(fine < coarse for coarse, fine in zip(errors, errors[1:])),
            errors,
        )
        observed_orders = [
            math.log(coarse / fine, 2.0)
            for coarse, fine in zip(errors, errors[1:])
        ]
        self.assertTrue(
            all(0.90 <= order <= 1.10 for order in observed_orders),
            {"errors": errors, "orders": observed_orders},
        )

    def test_one_step_y_double_prime_equals_x_uses_left_endpoint(self) -> None:
        sampled_x = []

        def forcing(x):
            sampled_x.append(x)
            return x

        result = _solve(
            equation_id="left-endpoint",
            g=forcing,
            n_steps=1,
        )

        self.assertTrue(result.success, result.message)
        self.assertEqual(sampled_x, [0.0])
        # The frozen matrix sees g(0)=0.  Midpoint/end-point variants would
        # produce a nonzero result for this deliberately discriminating case.
        self.assertEqual(result.y[-1], 0.0)
        self.assertEqual(result.dy[-1], 0.0)

    def test_noncommuting_step_propagators_are_multiplied_in_time_order(self) -> None:
        # h=1.  B(0) and B(1) do not commute, so reversing their propagators
        # gives a visibly different answer.
        y0 = 0.4
        v0 = -0.8
        result = _solve(
            equation_id="propagator-order",
            b=lambda x: x,
            c=lambda x: 1.0 + x,
            g=lambda x: x,
            x0=0.0,
            x1=2.0,
            y0=y0,
            v0=v0,
            n_steps=2,
        )
        self.assertTrue(result.success, result.message)

        matrix_0 = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        matrix_1 = np.array(
            [[0.0, 1.0, 0.0], [-2.0, -1.0, 1.0], [0.0, 0.0, 0.0]]
        )
        initial = np.array([y0, v0, 1.0])
        expected = oracle_expm(matrix_1) @ oracle_expm(matrix_0) @ initial
        reversed_order = oracle_expm(matrix_0) @ oracle_expm(matrix_1) @ initial

        np.testing.assert_allclose(
            [result.y[-1], result.dy[-1]],
            expected[:2],
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        self.assertGreater(
            np.linalg.norm(expected[:2] - reversed_order[:2]),
            1.0e-3,
        )


class ProductIntegralDiagnosticsTests(unittest.TestCase):
    def test_success_diagnostics_and_augmented_invariant(self) -> None:
        n_steps = 23
        result = _solve(
            equation_id="diagnostics",
            b=lambda x: 0.2 * x,
            c=lambda x: 1.0 + 0.1 * x,
            g=lambda x: math.sin(x),
            y0=0.25,
            v0=-0.5,
            n_steps=n_steps,
        )

        self.assertTrue(result.success, result.message)
        self.assertEqual(result.equation_id, "diagnostics")
        self.assertEqual(result.method_id, METHOD_ID)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.failure_reason, "")
        self.assertEqual(result.steps_requested, n_steps)
        self.assertEqual(result.steps_completed, n_steps)
        self.assertEqual(result.coefficient_points_evaluated, n_steps)
        self.assertEqual(result.scalar_coefficient_calls, 4 * n_steps)
        self.assertEqual(result.expm_evals, n_steps)
        self.assertEqual(result.matvec_evals, n_steps)
        self.assertEqual(result.n_steps, n_steps)
        self.assertAlmostEqual(result.h, 1.0 / n_steps)
        self.assertEqual(len(result.x), n_steps + 1)
        self.assertEqual(len(result.y), n_steps + 1)
        self.assertEqual(len(result.dy), n_steps + 1)
        self.assertEqual(result.x[0], 0.0)
        self.assertEqual(result.x[-1], 1.0)
        self.assertEqual(result.y[0], 0.25)
        self.assertEqual(result.dy[0], -0.5)
        self.assertLessEqual(result.max_invariant_drift, INVARIANT_FAIL_TOL)

    def test_invariant_is_not_reset_between_steps(self) -> None:
        drift_per_step = 4.0e-9
        perturbed_identity = np.diag([1.0, 1.0, 1.0 + drift_per_step])

        with patch(
            "src.numerical_methods.product_integral_solver.scipy.linalg.expm",
            return_value=perturbed_identity,
        ):
            result = _solve(equation_id="invariant-accumulation", n_steps=3)

        self.assertFalse(result.success)
        self.assertEqual(result.status, "invariant_violation")
        self.assertEqual(result.failure_reason, "invariant_violation")
        self.assertEqual(result.steps_completed, 2)
        self.assertEqual(result.expm_evals, 3)
        self.assertEqual(result.matvec_evals, 3)
        self.assertGreater(result.max_invariant_drift, INVARIANT_FAIL_TOL)
        self.assertIn("step 2", result.message)


class ProductIntegralFailureTests(unittest.TestCase):
    def assertFailureStatus(self, result, status: str) -> None:  # noqa: N802
        self.assertFalse(result.success)
        self.assertEqual(result.status, status)
        self.assertEqual(result.failure_reason, status)
        self.assertTrue(result.message)

    def test_invalid_n_steps_is_reported_as_invalid_input(self) -> None:
        for n_steps in (0, -1, 1.5, True):
            with self.subTest(n_steps=n_steps):
                result = _solve(n_steps=n_steps)
                self.assertFailureStatus(result, "invalid_input")
                self.assertEqual(result.steps_completed, 0)
                self.assertEqual(result.expm_evals, 0)
                self.assertEqual(result.matvec_evals, 0)
                self.assertIn("n_steps", result.message)

    def test_other_invalid_inputs_are_reported(self) -> None:
        cases = (
            {"x0": 1.0, "x1": 1.0},
            {"x0": 1.0, "x1": 0.0},
            {"a_min": 0.0},
            {"a": 1.0},
            {"x0": 1.0 + 1.0j},
            {"x0": math.nan},
            {"x1": math.inf},
            {"a_min": math.nan},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = _solve(**arguments)
                self.assertFailureStatus(result, "invalid_input")
                self.assertEqual(result.steps_completed, 0)
                self.assertEqual(result.expm_evals, 0)
                self.assertEqual(result.matvec_evals, 0)

    def test_a_too_small_at_first_and_later_points(self) -> None:
        first = _solve(
            equation_id="small-a-first",
            a=lambda _x: 5.0e-4,
            n_steps=4,
        )
        self.assertFailureStatus(first, "a_too_small")
        self.assertEqual(first.steps_completed, 0)
        self.assertEqual(first.scalar_coefficient_calls, 4)
        self.assertEqual(first.coefficient_points_evaluated, 1)
        self.assertEqual(first.expm_evals, 0)
        self.assertIn("step 0", first.message)
        self.assertIn("x=0", first.message)

        later = _solve(
            equation_id="small-a-later",
            a=lambda x: 5.0e-4 if x >= 0.5 else 1.0,
            n_steps=4,
        )
        self.assertFailureStatus(later, "a_too_small")
        self.assertEqual(later.steps_completed, 2)
        self.assertEqual(len(later.x), 3)
        self.assertEqual(later.scalar_coefficient_calls, 12)
        self.assertEqual(later.coefficient_points_evaluated, 3)
        self.assertEqual(later.expm_evals, 2)
        self.assertEqual(later.matvec_evals, 2)
        self.assertIn("step 2", later.message)
        self.assertIn("x=0.5", later.message)

    def test_nan_and_inf_coefficients_are_reported(self) -> None:
        for name, arguments in (
            ("nan-b", {"b": lambda _x: math.nan}),
            ("inf-g", {"g": lambda _x: math.inf}),
        ):
            with self.subTest(name=name):
                result = _solve(equation_id=name, **arguments)
                self.assertFailureStatus(result, "nonfinite_coefficient")
                self.assertEqual(result.steps_completed, 0)
                self.assertEqual(result.coefficient_points_evaluated, 0)
                self.assertEqual(result.expm_evals, 0)
                self.assertEqual(result.matvec_evals, 0)
                self.assertIn("step 0", result.message)
                self.assertIn("x=0", result.message)

    def test_callable_exception_is_reported(self) -> None:
        def broken_coefficient(_x):
            raise ArithmeticError("deliberate test exception")

        result = _solve(c=broken_coefficient)
        self.assertFailureStatus(result, "coefficient_evaluation_failed")
        self.assertEqual(result.scalar_coefficient_calls, 3)
        self.assertEqual(result.coefficient_points_evaluated, 0)
        self.assertEqual(result.expm_evals, 0)
        self.assertEqual(result.matvec_evals, 0)
        self.assertIn("coefficient c", result.message)
        self.assertIn("ArithmeticError", result.message)

    def test_nonreal_coefficient_is_not_silently_cast(self) -> None:
        result = _solve(g=lambda _x: 1.0 + 1.0j)
        self.assertFailureStatus(result, "coefficient_evaluation_failed")
        self.assertEqual(result.scalar_coefficient_calls, 4)
        self.assertEqual(result.expm_evals, 0)
        self.assertEqual(result.matvec_evals, 0)
        self.assertIn("non-real", result.message)

    def test_nonfinite_augmented_matrix_is_reported(self) -> None:
        result = _solve(
            a=lambda _x: 1.0e-3,
            c=lambda _x: sys.float_info.max,
        )
        self.assertFailureStatus(result, "nonfinite_matrix")
        self.assertEqual(result.coefficient_points_evaluated, 1)
        self.assertEqual(result.expm_evals, 0)
        self.assertEqual(result.matvec_evals, 0)
        self.assertIn("step 0", result.message)

    def test_nonfinite_initial_state_is_reported(self) -> None:
        for name, arguments in (
            ("nan-y", {"y0": math.nan}),
            ("inf-v", {"v0": math.inf}),
            ("complex-y", {"y0": 1.0 + 1.0j}),
        ):
            with self.subTest(name=name):
                result = _solve(equation_id=name, **arguments)
                self.assertFailureStatus(result, "nonfinite_state")
                self.assertEqual(result.x, [])
                self.assertEqual(result.expm_evals, 0)
                self.assertEqual(result.matvec_evals, 0)

    def test_matrix_exponential_exception_is_reported_and_counted(self) -> None:
        with patch(
            "src.numerical_methods.product_integral_solver.scipy.linalg.expm",
            side_effect=RuntimeError("deliberate expm failure"),
        ):
            result = _solve(equation_id="expm-exception")

        self.assertFailureStatus(result, "matrix_exponential_failed")
        self.assertEqual(result.coefficient_points_evaluated, 1)
        self.assertEqual(result.expm_evals, 1)
        self.assertEqual(result.matvec_evals, 0)
        self.assertIn("RuntimeError", result.message)

    def test_nonfinite_matrix_exponential_result_is_reported(self) -> None:
        bad_propagator = np.eye(3)
        bad_propagator[0, 0] = math.inf
        with patch(
            "src.numerical_methods.product_integral_solver.scipy.linalg.expm",
            return_value=bad_propagator,
        ):
            result = _solve(equation_id="nonfinite-expm")

        self.assertFailureStatus(result, "nonfinite_matrix")
        self.assertEqual(result.expm_evals, 1)
        self.assertEqual(result.matvec_evals, 0)

    def test_nonreal_matrix_exponential_result_is_not_silently_cast(self) -> None:
        complex_propagator = np.eye(3, dtype=complex)
        complex_propagator[0, 0] += 1.0j
        with patch(
            "src.numerical_methods.product_integral_solver.scipy.linalg.expm",
            return_value=complex_propagator,
        ):
            result = _solve(equation_id="complex-expm")

        self.assertFailureStatus(result, "nonfinite_matrix")
        self.assertEqual(result.expm_evals, 1)
        self.assertEqual(result.matvec_evals, 0)

    def test_nonfinite_propagated_state_is_reported_and_matvec_is_counted(self) -> None:
        # The genuine finite matrix exponential for v'=v contains e.  Acting
        # on this finite initial state then overflows, without mocking solver
        # internals.
        result = _solve(
            equation_id="nonfinite-state",
            b=lambda _x: -1.0,
            y0=0.5 * sys.float_info.max,
            v0=0.5 * sys.float_info.max,
            n_steps=1,
        )

        self.assertFailureStatus(result, "nonfinite_state")
        self.assertEqual(result.expm_evals, 1)
        self.assertEqual(result.matvec_evals, 1)
        self.assertEqual(result.steps_completed, 0)
        self.assertIn("step 0", result.message)


if __name__ == "__main__":
    unittest.main()
