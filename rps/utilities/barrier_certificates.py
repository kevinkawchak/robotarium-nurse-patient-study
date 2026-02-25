from __future__ import annotations

import numpy as np


def create_single_integrator_barrier_certificate(
    safety_radius: float = 0.17,
    barrier_gain: float = 100.0,
    magnitude_limit: float = 0.2,
):
    """Single-integrator barrier certificate (no boundary, matches Robotarium API).

    Applies pairwise repulsive corrections when robots are within safety_radius.
    """

    def barrier(dxi: np.ndarray, x: np.ndarray) -> np.ndarray:
        safe = np.array(dxi, dtype=float, copy=True)
        n = x.shape[1]

        for i in range(n):
            for j in range(i + 1, n):
                diff = x[:2, i] - x[:2, j]
                dist = np.linalg.norm(diff)
                if dist < safety_radius and dist > 1e-6:
                    push = (safety_radius - dist) * (diff / dist)
                    safe[:, i] += 0.3 * push
                    safe[:, j] -= 0.3 * push

        # Magnitude limiting
        mag = np.linalg.norm(safe, axis=0)
        over = mag > magnitude_limit
        if np.any(over):
            safe[:, over] *= magnitude_limit / mag[over]

        return safe

    return barrier


def create_single_integrator_barrier_certificate_with_boundary(
    safety_radius: float = 0.17,
    boundary_margin: float = 0.05,
    magnitude_limit: float = 0.2,
):
    """Single-integrator barrier certificate with boundary enforcement.

    Applies pairwise repulsive corrections and arena boundary reflection.
    Arena bounds: [-1.6, 1.6] x [-1.0, 1.0].
    """

    def barrier(dxi: np.ndarray, x: np.ndarray) -> np.ndarray:
        safe = np.array(dxi, dtype=float, copy=True)
        n = x.shape[1]

        for i in range(n):
            for j in range(i + 1, n):
                diff = x[:2, i] - x[:2, j]
                dist = np.linalg.norm(diff)
                if dist < safety_radius and dist > 1e-6:
                    push = (safety_radius - dist) * (diff / dist)
                    safe[:, i] += 0.3 * push
                    safe[:, j] -= 0.3 * push

            # Boundary enforcement
            if x[0, i] < -1.6 + boundary_margin:
                safe[0, i] = abs(safe[0, i])
            if x[0, i] > 1.6 - boundary_margin:
                safe[0, i] = -abs(safe[0, i])
            if x[1, i] < -1.0 + boundary_margin:
                safe[1, i] = abs(safe[1, i])
            if x[1, i] > 1.0 - boundary_margin:
                safe[1, i] = -abs(safe[1, i])

        return safe

    return barrier


def create_unicycle_barrier_certificate(
    safety_radius: float = 0.15,
    projection_distance: float = 0.05,
    barrier_gain: float = 100.0,
    magnitude_limit: float = 0.2,
):
    """Unicycle barrier certificate (no boundary, matches Robotarium API).

    Projects unicycle positions forward by projection_distance, applies
    SI barrier, then returns corrected unicycle velocities.
    """

    si_barrier = create_single_integrator_barrier_certificate(
        safety_radius=safety_radius,
        barrier_gain=barrier_gain,
        magnitude_limit=magnitude_limit,
    )

    def barrier(dxu: np.ndarray, x: np.ndarray) -> np.ndarray:
        n = x.shape[1]
        theta = x[2, :]

        # Project positions forward (account for heading)
        x_proj = np.zeros((2, n))
        x_proj[0, :] = x[0, :] + projection_distance * np.cos(theta)
        x_proj[1, :] = x[1, :] + projection_distance * np.sin(theta)

        # Convert unicycle velocities to SI velocities at projected point
        dxi = np.zeros((2, n))
        dxi[0, :] = dxu[0, :] * np.cos(theta) - projection_distance * dxu[1, :] * np.sin(theta)
        dxi[1, :] = dxu[0, :] * np.sin(theta) + projection_distance * dxu[1, :] * np.cos(theta)

        # Apply SI barrier
        dxi_safe = si_barrier(dxi, x_proj)

        # Convert back to unicycle
        out = np.zeros((2, n))
        for i in range(n):
            c = np.cos(theta[i])
            s = np.sin(theta[i])
            out[0, i] = c * dxi_safe[0, i] + s * dxi_safe[1, i]
            out[1, i] = (-s * dxi_safe[0, i] + c * dxi_safe[1, i]) / max(projection_distance, 1e-6)

        return out

    return barrier


def create_unicycle_barrier_certificate_with_boundary(
    safety_radius: float = 0.15,
    projection_distance: float = 0.05,
    boundary_margin: float = 0.05,
    magnitude_limit: float = 0.2,
):
    """Unicycle barrier certificate with boundary (matches Robotarium API).

    Projects unicycle positions forward, applies SI barrier with boundary
    enforcement, then returns corrected unicycle velocities.
    """

    si_barrier = create_single_integrator_barrier_certificate_with_boundary(
        safety_radius=safety_radius,
        boundary_margin=boundary_margin,
        magnitude_limit=magnitude_limit,
    )

    def barrier(dxu: np.ndarray, x: np.ndarray) -> np.ndarray:
        n = x.shape[1]
        theta = x[2, :]

        x_proj = np.zeros((2, n))
        x_proj[0, :] = x[0, :] + projection_distance * np.cos(theta)
        x_proj[1, :] = x[1, :] + projection_distance * np.sin(theta)

        dxi = np.zeros((2, n))
        dxi[0, :] = dxu[0, :] * np.cos(theta) - projection_distance * dxu[1, :] * np.sin(theta)
        dxi[1, :] = dxu[0, :] * np.sin(theta) + projection_distance * dxu[1, :] * np.cos(theta)

        dxi_safe = si_barrier(dxi, x_proj)

        out = np.zeros((2, n))
        for i in range(n):
            c = np.cos(theta[i])
            s = np.sin(theta[i])
            out[0, i] = c * dxi_safe[0, i] + s * dxi_safe[1, i]
            out[1, i] = (-s * dxi_safe[0, i] + c * dxi_safe[1, i]) / max(projection_distance, 1e-6)

        return out

    return barrier
