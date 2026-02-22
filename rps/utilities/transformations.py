from __future__ import annotations

import numpy as np


def create_si_to_uni_dynamics(
    linear_velocity_gain: float = 1.0,
    angular_velocity_limit: float = np.pi,
):
    def converter(dxi: np.ndarray, x: np.ndarray) -> np.ndarray:
        theta = x[2, :]
        desired_heading = np.arctan2(dxi[1, :], dxi[0, :])
        heading_error = desired_heading - theta
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        speed = np.linalg.norm(dxi, axis=0) * linear_velocity_gain
        omega = np.clip(2.0 * heading_error, -angular_velocity_limit, angular_velocity_limit)
        return np.vstack((speed, omega))

    return converter
