from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class SimulationConfig:
    """Configuration parameters for the 1D heat equation simulation.

    Attributes
    ----------
    length : float
        Length of the rod (domain) in spatial units.
    total_time : float
        Total simulation time.
    alpha : float
        Thermal diffusivity.
    nx : int
        Number of spatial grid points.
    nt : int
        Number of time steps.
    initial_temperature : float
        Initial temperature of the rod interior.
    boundary_left : float
        Fixed boundary temperature at the left end.
    boundary_right : float
        Fixed boundary temperature at the right end.
    """

    length: float
    total_time: float
    alpha: float
    nx: int
    nt: int
    initial_temperature: float
    boundary_left: float
    boundary_right: float


class ConfigurationError(Exception):
    """Custom exception for invalid configuration."""
    pass


class HeatEquationSolver:
    """Solver for the 1D heat equation using FTCS finite differences.

    The PDE is u_t = alpha * u_xx, solved on a 1D rod with Dirichlet
    boundary conditions using forward-time, central-space (FTCS).

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration parameters.

    Attributes
    ----------
    config : SimulationConfig
        Stored simulation configuration.
    dx : float
        Spatial grid spacing.
    dt : float
        Time step size.
    r : float
        Stability parameter alpha * dt / dx^2.
    x : np.ndarray
        Spatial grid points.
    u : np.ndarray
        Temperature field at current time step.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.dx, self.dt = self._compute_discretization()
        self.r = self.config.alpha * self.dt / (self.dx ** 2)
        self._check_stability()
        self.x, self.u = self._initialize_field()

    def _compute_discretization(self) -> Tuple[float, float]:
        """Compute spatial and temporal step sizes.

        Returns
        -------
        dx : float
            Spatial grid spacing.
        dt : float
            Time step size.

        Raises
        ------
        ConfigurationError
            If nx or nt are invalid or time/length are non-positive.
        """
        if self.config.nx < 3:
            raise ConfigurationError("nx must be at least 3 (2 boundaries + at least 1 interior point).")
        if self.config.nt < 1:
            raise ConfigurationError("nt must be at least 1.")
        if self.config.length <= 0:
            raise ConfigurationError("length must be positive.")
        if self.config.total_time <= 0:
            raise ConfigurationError("total_time must be positive.")

        dx = self.config.length / (self.config.nx - 1)
        dt = self.config.total_time / self.config.nt
        return dx, dt

    def _check_stability(self) -> None:
        """Check the CFL-like stability condition for FTCS.

        Notes
        -----
        For FTCS scheme for the heat equation, stability requires:

            r = alpha * dt / dx^2 <= 0.5

        Raises
        ------
        ConfigurationError
            If the stability condition is violated.
        """
        if self.r > 0.5:
            raise ConfigurationError(
                f"Stability condition violated: r = {self.r:.3f} > 0.5. "
                "Reduce dt or increase spatial resolution."
            )

    def _initialize_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize spatial grid and temperature field.

        Returns
        -------
        x : np.ndarray
            Spatial grid points (size nx).
        u : np.ndarray
            Initial temperature field (size nx).
        """
        x = np.linspace(0.0, self.config.length, self.config.nx)
        u = np.full_like(x, self.config.initial_temperature, dtype=float)

        # Apply Dirichlet boundary conditions
        u[0] = self.config.boundary_left
        u[-1] = self.config.boundary_right

        return x, u

    def step(self) -> None:
        """Advance the solution by one time step using FTCS scheme.

        Notes
        -----
        The discrete update is:

            u_i^{n+1} = u_i^n + r * (u_{i+1}^n - 2 u_i^n + u_{i-1}^n)

        for 1 <= i <= nx-2. Boundary values are kept fixed.
        """
        u_new = self.u.copy()
        # Vectorized interior update: indices 1..nx-2
        u_new[1:-1] = (
            self.u[1:-1]
            + self.r * (self.u[2:] - 2.0 * self.u[1:-1] + self.u[:-2])
        )

        # Enforce Dirichlet boundary conditions
        u_new[0] = self.config.boundary_left
        u_new[-1] = self.config.boundary_right

        self.u = u_new

    def run(self) -> np.ndarray:
        """Run the time integration for nt time steps.

        Returns
        -------
        np.ndarray
            Final temperature distribution (size nx).
        """
        for _ in range(self.config.nt):
            self.step()
        return self.u

    def save_to_csv(self, filepath: str | Path) -> None:
        """Save the final temperature distribution to a CSV file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Output file path for saving the temperature distribution.

        Notes
        -----
        The CSV file will have two columns: x, u.
        """
        filepath = Path(filepath)
        data = np.column_stack((self.x, self.u))
        np.savetxt(filepath, data, delimiter=",", header="x,u", comments="")


def load_config(config_path: str | Path) -> SimulationConfig:
    """Load simulation configuration from a JSON file.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the JSON configuration file.

    Returns
    -------
    SimulationConfig
        Parsed simulation configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ConfigurationError
        If required fields are missing or invalid.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON: {exc}") from exc

    required_fields = [
        "length", "total_time", "alpha", "nx", "nt",
        "initial_temperature", "boundary_left", "boundary_right",
    ]
    missing = [k for k in required_fields if k not in data]
    if missing:
        raise ConfigurationError(f"Missing required config fields: {missing}")

    return SimulationConfig(
        length=float(data["length"]),
        total_time=float(data["total_time"]),
        alpha=float(data["alpha"]),
        nx=int(data["nx"]),
        nt=int(data["nt"]),
        initial_temperature=float(data["initial_temperature"]),
        boundary_left=float(data["boundary_left"]),
        boundary_right=float(data["boundary_right"]),
    )


def main(config_path: str = "config.json", output_path: str = "output.csv") -> None:
    """Run the 1D heat equation solver from a configuration file.

    Parameters
    ----------
    config_path : str, optional
        Path to the JSON configuration file, by default "config.json".
    output_path : str, optional
        Path to the output CSV file, by default "output.csv".
    """
    try:
        config = load_config(config_path)
        solver = HeatEquationSolver(config)
        solver.run()
        solver.save_to_csv(output_path)
        print(f"Simulation completed. Results saved to {output_path}")
    except (FileNotFoundError, ConfigurationError, ValueError) as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    # Simple CLI-style usage: python heat_solver.py
    main()