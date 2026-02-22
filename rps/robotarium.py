from __future__ import annotations

import numpy as np


class Robotarium:
    """Minimal Robotarium-compatible simulator for local/web execution."""

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
        self.history: list[np.ndarray] = [self.poses.copy()]

    def get_poses(self) -> np.ndarray:
        return self.poses.copy()

    def set_velocities(self, ids: np.ndarray, dxu: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=int)
        dxu = np.asarray(dxu, dtype=float)
        if dxu.shape != (2, ids.size):
            raise ValueError("dxu must be shape (2, len(ids))")
        self._dxu[:, ids] = dxu

    def step(self) -> None:
        theta = self.poses[2, :]
        v = self._dxu[0, :]
        w = self._dxu[1, :]

        self.poses[0, :] += self.time_step * v * np.cos(theta)
        self.poses[1, :] += self.time_step * v * np.sin(theta)
        self.poses[2, :] += self.time_step * w

        self.poses[0, :] = np.clip(self.poses[0, :], -1.6, 1.6)
        self.poses[1, :] = np.clip(self.poses[1, :], -1.0, 1.0)
        self.history.append(self.poses.copy())

    def call_at_scripts_end(self) -> None:
        return
