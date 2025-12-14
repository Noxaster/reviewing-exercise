import unittest
import numpy as np

from heat_solver import (
    SimulationConfig,
    HeatEquationSolver,
    ConfigurationError,
)


class TestHeatEquationSolver(unittest.TestCase):
    """Unit tests for the HeatEquationSolver."""

    def test_constant_solution_with_equal_boundaries(self) -> None:
        """If initial field and boundaries are constant and equal, u should not change.

        PDE: u_t = alpha * u_xx
        If u(x,0) = C and boundaries are C, then u(x,t) = C is an exact solution.
        Numerical scheme should preserve this constant field (up to roundoff).
        """
        config = SimulationConfig(
            length=1.0,
            total_time=0.1,
            alpha=0.01,
            nx=21,
            nt=50,
            initial_temperature=50.0,
            boundary_left=50.0,
            boundary_right=50.0,
        )
        solver = HeatEquationSolver(config)
        u_initial = solver.u.copy()
        u_final = solver.run()

        # Constant should be preserved
        self.assertTrue(np.allclose(u_initial, u_final, atol=1e-12))

    def test_stability_condition_enforced(self) -> None:
        """Check that the solver raises an error when stability is violated."""
        # Make dt large to violate r <= 0.5
        config = SimulationConfig(
            length=1.0,
            total_time=10.0,  # large time with few steps -> large dt
            alpha=1.0,
            nx=11,
            nt=1,  # dt = 10.0
            initial_temperature=0.0,
            boundary_left=0.0,
            boundary_right=0.0,
        )
        from heat_solver import ConfigurationError

        with self.assertRaises(ConfigurationError):
            HeatEquationSolver(config)

    def test_energy_decreases_with_zero_boundaries(self) -> None:
        """For zero Dirichlet boundaries, total energy should not increase.

        Here, 'energy' is approximated as sum(u) * dx.
        With boundaries fixed at 0 and diffusion, the total energy must be
        non-increasing.
        """
        config = SimulationConfig(
            length=1.0,
            total_time=0.1,
            alpha=0.01,
            nx=51,
            nt=100,
            initial_temperature=1.0,
            boundary_left=0.0,
            boundary_right=0.0,
        )
        solver = HeatEquationSolver(config)
        initial_energy = solver.u.sum() * solver.dx
        solver.run()
        final_energy = solver.u.sum() * solver.dx

        self.assertTrue(
            final_energy <= initial_energy or np.isclose(final_energy, initial_energy, atol=1e-12),
            f"Final energy ({final_energy}) should not exceed initial energy ({initial_energy}) except for roundoff."
        )


if __name__ == "__main__":
    unittest.main()