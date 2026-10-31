import math
import unittest

from rk4_solver import solve_rk4


class RK4SolverTests(unittest.TestCase):
    def test_harmonic_oscillator_has_fourth_order_convergence(self) -> None:
        errors = []
        for n_steps in (10, 20, 40, 80):
            result = solve_rk4(
                equation_id="harmonic",
                a=lambda _x: 1.0,
                b=lambda _x: 0.0,
                c=lambda _x: 1.0,
                g=lambda _x: 0.0,
                x0=0.0,
                x1=1.0,
                y0=0.0,
                v0=1.0,
                n_steps=n_steps,
            )
            self.assertTrue(result.success)
            self.assertEqual(result.nfev, 4 * n_steps)
            self.assertEqual(len(result.x), n_steps + 1)
            error = max(
                abs(result.y[-1] - math.sin(1.0)),
                abs(result.dy[-1] - math.cos(1.0)),
            )
            errors.append(error)
        observed_orders = [
            math.log(left / right, 2.0) for left, right in zip(errors, errors[1:])
        ]
        self.assertTrue(all(order > 3.8 for order in observed_orders), observed_orders)

    def test_variable_coefficients_with_known_exponential_solution(self) -> None:
        # (1+x^2)y'' + (1+x)y' + (2-x)y = (4+x^2)e^x, y=e^x.
        result = solve_rk4(
            equation_id="variable_known_solution",
            a=lambda x: 1.0 + x * x,
            b=lambda x: 1.0 + x,
            c=lambda x: 2.0 - x,
            g=lambda x: (4.0 + x * x) * math.exp(x),
            x0=0.0,
            x1=1.0,
            y0=1.0,
            v0=1.0,
            n_steps=200,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.nfev, 800)
        max_error = max(
            max(abs(y - math.exp(x)) for x, y in zip(result.x, result.y)),
            max(abs(v - math.exp(x)) for x, v in zip(result.x, result.dy)),
        )
        self.assertLess(max_error, 1.0e-8)

    def test_a_too_small_is_reported(self) -> None:
        result = solve_rk4(
            equation_id="small_a",
            a=lambda _x: 5.0e-4,
            b=lambda _x: 0.0,
            c=lambda _x: 0.0,
            g=lambda _x: 1.0,
            x0=0.0,
            x1=1.0,
            y0=0.0,
            v0=0.0,
            n_steps=10,
            a_min=1.0e-3,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "a_too_small")
        self.assertEqual(result.nfev, 1)
        self.assertEqual(len(result.x), 1)

    def test_nan_and_inf_are_reported_as_nonfinite_state(self) -> None:
        for label, y0, forcing, expected_nfev in (
            ("nan_initial", math.nan, lambda _x: 0.0, 0),
            ("inf_forcing", 0.0, lambda _x: math.inf, 1),
        ):
            with self.subTest(label=label):
                result = solve_rk4(
                    equation_id=label,
                    a=lambda _x: 1.0,
                    b=lambda _x: 0.0,
                    c=lambda _x: 0.0,
                    g=forcing,
                    x0=0.0,
                    x1=1.0,
                    y0=y0,
                    v0=0.0,
                    n_steps=10,
                )
                self.assertFalse(result.success)
                self.assertEqual(result.failure_reason, "nonfinite_state")
                self.assertEqual(result.nfev, expected_nfev)


if __name__ == "__main__":
    unittest.main()
