from __future__ import annotations

import numpy as np


def create_single_integrator_barrier_certificate_with_boundary(
    safety_radius: float = 0.17,
    boundary_margin: float = 0.05,
):
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
