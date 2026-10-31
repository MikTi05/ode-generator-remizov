import unittest

import numpy as np

from remizov_benchmark_loader import (
    load_remizov_benchmark,
    load_remizov_problem,
    make_verification_target_x,
)


class RemizovBenchmarkLoaderTests(unittest.TestCase):
    def test_loader_creates_only_global_remizov_problem_data(self) -> None:
        problems = load_remizov_benchmark()
        self.assertEqual([problem.problem_id for problem in problems], ["R1", "R2", "R3", "R4", "R5"])
        for problem in problems:
            self.assertFalse(hasattr(problem, "ivp_y0"))
            self.assertFalse(hasattr(problem, "ivp_v0"))
            self.assertEqual(problem.bounds_certification, "analytic")
            self.assertEqual(problem.bounds_domain, "all_real_x")
            points = np.array([-1.0, 0.0, 1.0])
            for function in (problem.a, problem.b, problem.c_remizov, problem.g_remizov):
                values = function(points)
                self.assertEqual(values.shape, points.shape)
                self.assertTrue(np.all(np.isfinite(values)))

    def test_constant_r1_coefficients_are_broadcast(self) -> None:
        problem = load_remizov_problem("R1")
        points = np.linspace(-2.0, 2.0, 9)
        np.testing.assert_array_equal(problem.a(points), np.ones_like(points))
        np.testing.assert_array_equal(problem.b(points), np.zeros_like(points))
        np.testing.assert_array_equal(problem.c_remizov(points), np.zeros_like(points))

    def test_verification_target_grid_is_fixed_257_points(self) -> None:
        problem = load_remizov_problem("R3")
        target = make_verification_target_x(problem)
        self.assertEqual(target.size, 257)
        self.assertEqual(target[0], -2.0)
        self.assertEqual(target[-1], 2.0)
        self.assertTrue(np.all(np.diff(target) > 0.0))

    def test_unknown_problem_id_raises(self) -> None:
        with self.assertRaises(KeyError):
            load_remizov_problem("missing")


if __name__ == "__main__":
    unittest.main()
