from __future__ import annotations

import numpy as np


def create_si_position_controller(
    x_velocity_gain: float = 1.0,
    y_velocity_gain: float = 1.0,
    velocity_magnitude_limit: float = 0.12,
):
    def controller(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = target - current
        dxi = np.vstack((x_velocity_gain * error[0, :], y_velocity_gain * error[1, :]))
        mag = np.linalg.norm(dxi, axis=0)
        over = mag > velocity_magnitude_limit
        if np.any(over):
            dxi[:, over] *= velocity_magnitude_limit / mag[over]
        return dxi

    return controller
