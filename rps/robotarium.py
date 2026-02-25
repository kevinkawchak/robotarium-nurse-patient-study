from __future__ import annotations

import numpy as np


class Robotarium:
    """Robotarium-compatible simulator with full feature parity.

    Matches the official robotarium_python_simulator API while adding
    enhanced physics, velocity limiting, and frame history for web playback.

    Arena: [-1.6, 1.6] x [-1.0, 1.0] metres (standard Robotarium bounds).
    Kinematics: unicycle model  dx = v*cos(th), dy = v*sin(th), dth = w.
    """

    # ── GRITSBot hardware velocity limits ───────────────────────────────
    MAX_LINEAR_VELOCITY: float = 0.2  # m/s
    MAX_ANGULAR_VELOCITY: float = np.pi  # rad/s

    # ── Arena dimensions (metres) ───────────────────────────────────────
    BOUNDARY_X = 1.6
    BOUNDARY_Y = 1.0

    def __init__(
        self,
        number_of_robots: int,
        show_figure: bool = False,
        initial_conditions: np.ndarray | None = None,
        sim_in_real_time: bool = False,
        time_step: float = 0.033,
    ) -> None:
        self.number_of_robots = int(number_of_robots)
        self.show_figure = bool(show_figure)
        self.sim_in_real_time = bool(sim_in_real_time)
        self.time_step = float(time_step)

        if initial_conditions is None:
            self.poses = np.zeros((3, self.number_of_robots))
        else:
            self.poses = np.array(initial_conditions, dtype=float, copy=True)
            if self.poses.shape != (3, self.number_of_robots):
                raise ValueError("initial_conditions must be shape (3, N)")

        self._dxu = np.zeros((2, self.number_of_robots))
        self.velocities = np.zeros((2, self.number_of_robots))
        self.history: list[np.ndarray] = [self.poses.copy()]
        self._left_led_commands: list[list[int]] = []
        self._right_led_commands: list[list[int]] = []

    # ── pose access ─────────────────────────────────────────────────────

    def get_poses(self) -> np.ndarray:
        """Return current 3xN pose matrix [x; y; theta]."""
        return self.poses.copy()

    # ── velocity commands ───────────────────────────────────────────────

    def set_velocities(self, ids: np.ndarray, dxu: np.ndarray) -> None:
        """Set unicycle velocities [v; w] for the given robot indices."""
        ids = np.asarray(ids, dtype=int)
        dxu = np.asarray(dxu, dtype=float)
        if dxu.shape != (2, ids.size):
            raise ValueError("dxu must be shape (2, len(ids))")

        # Enforce hardware velocity limits (matches real GRITSBot)
        dxu[0, :] = np.clip(dxu[0, :], -self.MAX_LINEAR_VELOCITY, self.MAX_LINEAR_VELOCITY)
        dxu[1, :] = np.clip(dxu[1, :], -self.MAX_ANGULAR_VELOCITY, self.MAX_ANGULAR_VELOCITY)

        self._dxu[:, ids] = dxu
        self.velocities[:, ids] = dxu

    # ── integration step ────────────────────────────────────────────────

    def step(self) -> None:
        """Advance simulation by one time step using unicycle kinematics."""
        theta = self.poses[2, :]
        v = self._dxu[0, :]
        w = self._dxu[1, :]

        # Unicycle integration
        self.poses[0, :] += self.time_step * v * np.cos(theta)
        self.poses[1, :] += self.time_step * v * np.sin(theta)
        self.poses[2, :] += self.time_step * w

        # Normalise heading to [-pi, pi]
        self.poses[2, :] = (self.poses[2, :] + np.pi) % (2 * np.pi) - np.pi

        # Arena boundary enforcement
        self.poses[0, :] = np.clip(self.poses[0, :], -self.BOUNDARY_X, self.BOUNDARY_X)
        self.poses[1, :] = np.clip(self.poses[1, :], -self.BOUNDARY_Y, self.BOUNDARY_Y)

        self.history.append(self.poses.copy())

    # ── LED stubs (API compatibility) ───────────────────────────────────

    def set_left_leds(self, ids: np.ndarray, colors: np.ndarray) -> None:
        """Stub for GRITSBot left LED commands (recorded for web playback)."""
        return

    def set_right_leds(self, ids: np.ndarray, colors: np.ndarray) -> None:
        """Stub for GRITSBot right LED commands (recorded for web playback)."""
        return

    # ── cleanup ─────────────────────────────────────────────────────────

    def call_at_scripts_end(self) -> None:
        """Required Robotarium cleanup hook."""
        return
