from __future__ import annotations

import numpy as np


def create_si_to_uni_dynamics(
    linear_velocity_gain: float = 1.0,
    angular_velocity_limit: float = np.pi,
):
    """Convert single-integrator velocities to unicycle commands.

    Maps 2D velocity vectors [vx; vy] to unicycle commands [v; omega]
    using desired heading tracking.

    Returns a function converter(dxi_2xN, poses_3xN) -> dxu_2xN.
    """

    def converter(dxi: np.ndarray, x: np.ndarray) -> np.ndarray:
        theta = x[2, :]
        desired_heading = np.arctan2(dxi[1, :], dxi[0, :])
        heading_error = desired_heading - theta
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        speed = np.linalg.norm(dxi, axis=0) * linear_velocity_gain
        omega = np.clip(2.0 * heading_error, -angular_velocity_limit, angular_velocity_limit)
        return np.vstack((speed, omega))

    return converter


def create_si_to_uni_dynamics_with_obstacles(
    linear_velocity_gain: float = 1.0,
    angular_velocity_limit: float = np.pi,
    projection_distance: float = 0.05,
):
    """SI to unicycle conversion using near-identity diffeomorphism.

    More suitable when barrier certificates are active, as it accounts
    for the projection used by unicycle barriers.

    Returns a function converter(dxi_2xN, poses_3xN) -> dxu_2xN.
    """

    def converter(dxi: np.ndarray, x: np.ndarray) -> np.ndarray:
        n = x.shape[1]
        theta = x[2, :]
        dxu = np.zeros((2, n))

        for i in range(n):
            c = np.cos(theta[i])
            s = np.sin(theta[i])
            dxu[0, i] = c * dxi[0, i] + s * dxi[1, i]
            dxu[1, i] = (-s * dxi[0, i] + c * dxi[1, i]) / max(projection_distance, 1e-6)

        dxu[0, :] *= linear_velocity_gain
        dxu[1, :] = np.clip(dxu[1, :], -angular_velocity_limit, angular_velocity_limit)
        return dxu

    return converter


def create_uni_to_si_dynamics(
    projection_distance: float = 0.05,
):
    """Convert unicycle commands back to single-integrator velocities.

    Useful for applying SI barrier certificates to unicycle robots.

    Returns a function converter(dxu_2xN, poses_3xN) -> dxi_2xN.
    """

    def converter(dxu: np.ndarray, x: np.ndarray) -> np.ndarray:
        n = x.shape[1]
        theta = x[2, :]
        dxi = np.zeros((2, n))

        dxi[0, :] = dxu[0, :] * np.cos(theta) - projection_distance * dxu[1, :] * np.sin(theta)
        dxi[1, :] = dxu[0, :] * np.sin(theta) + projection_distance * dxu[1, :] * np.cos(theta)

        return dxi

    return converter
