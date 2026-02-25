from __future__ import annotations

import numpy as np


def create_si_position_controller(
    x_velocity_gain: float = 1.0,
    y_velocity_gain: float = 1.0,
    velocity_magnitude_limit: float = 0.12,
):
    """Single-integrator position controller with velocity saturation.

    Returns a function controller(current_2xN, target_2xN) -> dxi_2xN.
    """

    def controller(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        error = target - current
        dxi = np.vstack((x_velocity_gain * error[0, :], y_velocity_gain * error[1, :]))
        mag = np.linalg.norm(dxi, axis=0)
        over = mag > velocity_magnitude_limit
        if np.any(over):
            dxi[:, over] *= velocity_magnitude_limit / mag[over]
        return dxi

    return controller


def create_clf_unicycle_position_controller(
    linear_velocity_gain: float = 0.8,
    angular_velocity_gain: float = 3.0,
    velocity_magnitude_limit: float = 0.15,
    angular_velocity_limit: float = np.pi,
):
    """CLF-based unicycle position controller (matches Robotarium API).

    Returns a function controller(states_3xN, targets_2xN) -> dxu_2xN.
    """

    def controller(states: np.ndarray, targets: np.ndarray) -> np.ndarray:
        n = states.shape[1]
        dxu = np.zeros((2, n))

        for i in range(n):
            dx = targets[0, i] - states[0, i]
            dy = targets[1, i] - states[1, i]
            dist = np.sqrt(dx**2 + dy**2)

            desired_heading = np.arctan2(dy, dx)
            heading_error = desired_heading - states[2, i]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

            dxu[0, i] = linear_velocity_gain * dist * np.cos(heading_error)
            dxu[1, i] = angular_velocity_gain * heading_error

        # Velocity limiting
        dxu[0, :] = np.clip(dxu[0, :], -velocity_magnitude_limit, velocity_magnitude_limit)
        dxu[1, :] = np.clip(dxu[1, :], -angular_velocity_limit, angular_velocity_limit)

        return dxu

    return controller


def create_clf_unicycle_pose_controller(
    linear_velocity_gain: float = 0.8,
    angular_velocity_gain: float = 3.0,
    rotation_error_gain: float = 0.4,
    velocity_magnitude_limit: float = 0.15,
    angular_velocity_limit: float = np.pi,
):
    """CLF-based unicycle pose controller (position + heading, matches Robotarium API).

    Returns a function controller(states_3xN, targets_3xN) -> dxu_2xN.
    """

    def controller(states: np.ndarray, targets: np.ndarray) -> np.ndarray:
        n = states.shape[1]
        dxu = np.zeros((2, n))

        for i in range(n):
            dx = targets[0, i] - states[0, i]
            dy = targets[1, i] - states[1, i]
            dist = np.sqrt(dx**2 + dy**2)

            desired_heading = np.arctan2(dy, dx)
            heading_error = desired_heading - states[2, i]
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

            if dist > 0.02:
                dxu[0, i] = linear_velocity_gain * dist * np.cos(heading_error)
                dxu[1, i] = angular_velocity_gain * heading_error
            else:
                target_heading_error = targets[2, i] - states[2, i]
                target_heading_error = (target_heading_error + np.pi) % (2 * np.pi) - np.pi
                dxu[0, i] = 0.0
                dxu[1, i] = rotation_error_gain * target_heading_error

        dxu[0, :] = np.clip(dxu[0, :], -velocity_magnitude_limit, velocity_magnitude_limit)
        dxu[1, :] = np.clip(dxu[1, :], -angular_velocity_limit, angular_velocity_limit)

        return dxu

    return controller
