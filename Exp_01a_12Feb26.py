"""
================================================================================
  ROBOTARIUM SWARM EXPERIMENT: Emergent Doctor-Patient Medical Response
  Georgia Tech Robotarium — https://www.robotarium.gatech.edu/experiment
================================================================================

  14 robots — 5 Doctors (indices 0-4) and 9 Patients (indices 5-13)
  demonstrate emergent swarm-based triage and medical response.

  BACKGROUND IMAGE:
    Upload "nvidia_gtc_background.png" as the arena background when submitting
    this experiment. The image is projected onto the arena surface.

  TIMELINE:
  ──────────────────────────────────────────────────────────────────────────
  [0–8s]   PHASE 1 — DISTRESS SIGNAL
     Patients scattered across the arena oscillate (medical distress).
     Doctors hold a tight formation on the left ("hospital base").

  [8–20s]  PHASE 2 — DISPATCH & TRIAGE SWEEP
     Doctors break formation and disperse using repulsion-based coverage.
     Each doctor locks onto the nearest unclaimed patient (greedy triage).
     Patient oscillations dampen as doctors approach.

  [20–38s] PHASE 3 — TREATMENT & STABILIZATION
     Assigned doctors orbit their patient at close range (active treatment).
     Unassigned patients flock toward the nearest doctor-patient cluster,
     forming self-organizing triage groups (emergent behavior).

  [38–50s] PHASE 4 — EVACUATION CONVOY
     All clusters migrate toward the arena center (the projected image).
     Doctors lead from the front; patients follow in convoy formation.

  [50–60s] PHASE 5 — RECOVERY FORMATION
     All 14 robots converge to a symmetric ring centered on the logo.
  ──────────────────────────────────────────────────────────────────────────

  EMERGENT BEHAVIORS:
    1. Self-organized spatial coverage (repulsion-based dispersion)
    2. Decentralized task allocation (nearest-neighbor greedy matching)
    3. Orbiting/shepherding (doctor-patient interaction)
    4. Flocking & attraction (unassigned patients cluster to groups)
    5. Convoy formation & collective migration
    6. Symmetric consensus formation (final ring)

================================================================================
"""

import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
N = 14                       # Total robots
NUM_DOCTORS = 5              # Robots 0..4
NUM_PATIENTS = 9             # Robots 5..13
DOCTOR_IDS = list(range(NUM_DOCTORS))
PATIENT_IDS = list(range(NUM_DOCTORS, N))

# Timing (~30 Hz → 0.033s per iteration)
TOTAL_ITERATIONS = 1800      # ~60 seconds
PHASE_1_END = 240            # 0-8s
PHASE_2_END = 600            # 8-20s
PHASE_3_END = 1140           # 20-38s
PHASE_4_END = 1500           # 38-50s
# Phase 5: 1500-1800         # 50-60s

# Arena bounds (Robotarium: [-1.6, 1.6] x [-1.0, 1.0])
ARENA_X = [-1.6, 1.6]
ARENA_Y = [-1.0, 1.0]

# Safety parameters (must exceed barrier certificate safety_radius=0.17)
ORBIT_RADIUS = 0.25
FLOCK_STOP_DIST = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# INITIAL CONDITIONS — 3×N [x; y; theta]
# Doctors: clustered on the left ("hospital base")
# Patients: scattered across the arena in a 3×3 grid pattern
# All pairwise distances >= 0.43 (verified safe; minimum required ~0.3)
# ─────────────────────────────────────────────────────────────────────────────
doctor_x = np.array([-1.30, -1.30, -0.95, -0.95, -0.95])
doctor_y = np.array([ 0.30, -0.30,  0.00,  0.55, -0.55])

patient_x = np.array([ 0.00,  0.50,  1.10,  0.00,  0.50,  1.10, -0.20,  0.50,  1.10])
patient_y = np.array([ 0.60,  0.60,  0.60,  0.00,  0.00,  0.00, -0.60, -0.60, -0.60])

initial_conditions = np.array([
    np.concatenate([doctor_x, patient_x]),
    np.concatenate([doctor_y, patient_y]),
    np.zeros(N)
])

# ─────────────────────────────────────────────────────────────────────────────
# ROBOTARIUM INIT
# ─────────────────────────────────────────────────────────────────────────────
r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True
)

# ─────────────────────────────────────────────────────────────────────────────
# CONTROLLERS & SAFETY (using verified Robotarium API function names)
# ─────────────────────────────────────────────────────────────────────────────
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics(
    linear_velocity_gain=1, angular_velocity_limit=np.pi
)
si_position_controller = create_si_position_controller(
    x_velocity_gain=1, y_velocity_gain=1, velocity_magnitude_limit=0.12
)

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND IMAGE NOTE
# ─────────────────────────────────────────────────────────────────────────────
# Upload "nvidia_gtc_background.png" via the Robotarium experiment page.
# It will be projected onto the physical arena beneath the robots.
# The final ring formation (Phase 5) is centered over this projected image.

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (numpy-only — no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def clamp_to_arena(pos, margin=0.15):
    """Clamp a 2D position to stay within the arena boundaries."""
    out = np.copy(pos)
    out[0] = np.clip(out[0], ARENA_X[0] + margin, ARENA_X[1] - margin)
    out[1] = np.clip(out[1], ARENA_Y[0] + margin, ARENA_Y[1] - margin)
    return out


def distress_offsets(t, n):
    """Jittery oscillation offsets simulating patient distress."""
    amp = 0.015
    offsets = np.zeros((2, n))
    for i in range(n):
        phase = i * 2.0 * np.pi / n
        offsets[0, i] = amp * np.sin(0.05 * t * 2 * np.pi + phase)
        offsets[1, i] = amp * np.cos(0.05 * t * 2 * np.pi + phase * 1.3)
    return offsets


def greedy_assignment(doc_pos, pat_pos):
    """
    Greedy nearest-neighbor: each doctor claims closest unclaimed patient.
    Uses numpy only (no scipy).
    Returns: dict {doc_local_idx: pat_local_idx}, set of claimed patient indices
    """
    n_doc = doc_pos.shape[1]
    n_pat = pat_pos.shape[1]

    # Pairwise distance matrix via numpy broadcasting
    dx = doc_pos[0, :, None] - pat_pos[0, None, :]   # n_doc × n_pat
    dy = doc_pos[1, :, None] - pat_pos[1, None, :]
    dist = np.sqrt(dx**2 + dy**2)

    assignment = {}
    claimed = set()

    for _ in range(min(n_doc, n_pat)):
        best_val = np.inf
        best_d, best_p = -1, -1
        for d in range(n_doc):
            if d in assignment:
                continue
            for p in range(n_pat):
                if p in claimed:
                    continue
                if dist[d, p] < best_val:
                    best_val = dist[d, p]
                    best_d, best_p = d, p
        if best_d >= 0:
            assignment[best_d] = best_p
            claimed.add(best_p)

    return assignment, claimed


def ring_formation(n, center, radius):
    """Evenly-spaced positions around a ring."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.zeros((2, n))
    pos[0, :] = center[0] + radius * np.cos(angles)
    pos[1, :] = center[1] + radius * np.sin(angles)
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTED DATA
# ─────────────────────────────────────────────────────────────────────────────
patient_home = np.array([patient_x, patient_y])
doctor_home = np.array([doctor_x, doctor_y])
final_ring = ring_formation(N, center=np.array([0.0, 0.0]), radius=0.6)

# Persistent assignment (computed in Phase 2 onward)
assignment = {}
claimed = set()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
for t in range(TOTAL_ITERATIONS):

    # Current poses: 3×N [x; y; theta]
    x = r.get_poses()
    xi = x[:2, :]                       # 2D positions (2×N)
    dxi = np.zeros((2, N))              # SI velocity commands (2×N)

    # =================================================================
    # PHASE 1: DISTRESS SIGNAL (0–8 sec)
    # =================================================================
    if t < PHASE_1_END:

        # Doctors hold at hospital base
        dxi[:, :NUM_DOCTORS] = si_position_controller(
            xi[:, :NUM_DOCTORS], doctor_home
        )

        # Patients oscillate (distress)
        offsets = distress_offsets(t, NUM_PATIENTS)
        targets = patient_home + offsets
        for i in range(NUM_PATIENTS):
            targets[:, i] = clamp_to_arena(targets[:, i])
        dxi[:, NUM_DOCTORS:] = si_position_controller(
            xi[:, NUM_DOCTORS:], targets
        )

    # =================================================================
    # PHASE 2: DISPATCH & TRIAGE SWEEP (8–20 sec)
    # =================================================================
    elif t < PHASE_2_END:
        progress = (t - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)

        if progress < 0.4:
            # Doctors disperse via mutual repulsion
            doc_targets = np.copy(xi[:, :NUM_DOCTORS])
            for i in range(NUM_DOCTORS):
                repulsion = np.zeros(2)
                for j in range(NUM_DOCTORS):
                    if i == j:
                        continue
                    diff = xi[:, i] - xi[:, j]
                    d = max(np.linalg.norm(diff), 0.01)
                    repulsion += diff / (d ** 2)
                rn = np.linalg.norm(repulsion)
                if rn > 0:
                    repulsion = repulsion / rn * 0.3
                doc_targets[:, i] = clamp_to_arena(xi[:, i] + repulsion)

            dxi[:, :NUM_DOCTORS] = si_position_controller(
                xi[:, :NUM_DOCTORS], doc_targets
            )
        else:
            # Compute & cache assignment
            assignment, claimed = greedy_assignment(
                xi[:, :NUM_DOCTORS], xi[:, NUM_DOCTORS:]
            )

            # Doctors move toward assigned patient
            for d_idx, p_idx in assignment.items():
                g_pat = NUM_DOCTORS + p_idx
                target = xi[:, g_pat].reshape(2, 1)
                dxi[:, d_idx:d_idx+1] = si_position_controller(
                    xi[:, d_idx].reshape(2, 1), target
                )

        # Patients: dampened distress
        damping = max(0.0, 1.0 - 0.8 * progress)
        offsets = distress_offsets(t, NUM_PATIENTS) * damping
        targets = patient_home + offsets
        for i in range(NUM_PATIENTS):
            targets[:, i] = clamp_to_arena(targets[:, i])
        dxi[:, NUM_DOCTORS:] = si_position_controller(
            xi[:, NUM_DOCTORS:], targets
        )

    # =================================================================
    # PHASE 3: TREATMENT & STABILIZATION (20–38 sec)
    # =================================================================
    elif t < PHASE_3_END:

        # Refresh assignment
        assignment, claimed = greedy_assignment(
            xi[:, :NUM_DOCTORS], xi[:, NUM_DOCTORS:]
        )

        # --- Doctors: orbit assigned patient ---
        orbit_speed = 0.04
        for d_idx, p_idx in assignment.items():
            g_pat = NUM_DOCTORS + p_idx
            center = xi[:, g_pat]
            angle = orbit_speed * t + d_idx * (2 * np.pi / NUM_DOCTORS)
            orb = center + ORBIT_RADIUS * np.array([np.cos(angle), np.sin(angle)])
            orb = clamp_to_arena(orb)
            dxi[:, d_idx:d_idx+1] = si_position_controller(
                xi[:, d_idx].reshape(2, 1), orb.reshape(2, 1)
            )

        # --- Claimed patients: hold still ---
        for p_idx in claimed:
            g_pat = NUM_DOCTORS + p_idx
            dxi[:, g_pat] = np.zeros(2)

        # --- Unclaimed patients: flock toward nearest cluster ---
        unclaimed = set(range(NUM_PATIENTS)) - claimed
        if len(assignment) > 0:
            centers = []
            for d_idx, p_idx in assignment.items():
                g_pat = NUM_DOCTORS + p_idx
                c = 0.5 * (xi[:, d_idx] + xi[:, g_pat])
                centers.append(c)
            centers = np.array(centers).T  # 2 × num_clusters

            for p_local in unclaimed:
                g_pat = NUM_DOCTORS + p_local
                pos = xi[:, g_pat]
                dists = np.linalg.norm(centers - pos.reshape(2, 1), axis=0)
                nearest = np.argmin(dists)
                target = centers[:, nearest]

                if dists[nearest] > FLOCK_STOP_DIST:
                    vel = si_position_controller(
                        pos.reshape(2, 1), target.reshape(2, 1)
                    )
                    dxi[:, g_pat:g_pat+1] = vel * 0.5
                else:
                    dxi[:, g_pat] = np.zeros(2)

    # =================================================================
    # PHASE 4: EVACUATION CONVOY (38–50 sec)
    # =================================================================
    elif t < PHASE_4_END:
        progress = (t - PHASE_3_END) / (PHASE_4_END - PHASE_3_END)
        center = np.array([0.0, 0.0])

        # Refresh assignment
        assignment, claimed = greedy_assignment(
            xi[:, :NUM_DOCTORS], xi[:, NUM_DOCTORS:]
        )

        # Doctors: lead toward center
        for d_idx, p_idx in assignment.items():
            g_pat = NUM_DOCTORS + p_idx
            pat_pos = xi[:, g_pat]
            blend = 0.3 + 0.5 * progress
            doc_target = (1 - blend) * pat_pos + blend * center

            # Offset so doctor leads from the front
            direction = center - pat_pos
            dn = np.linalg.norm(direction)
            if dn > 0.01:
                doc_target = doc_target + 0.12 * (direction / dn)

            doc_target = clamp_to_arena(doc_target)
            dxi[:, d_idx:d_idx+1] = si_position_controller(
                xi[:, d_idx].reshape(2, 1), doc_target.reshape(2, 1)
            )

        # Unassigned doctors also move to center
        for d_idx in range(NUM_DOCTORS):
            if d_idx not in assignment:
                dxi[:, d_idx:d_idx+1] = si_position_controller(
                    xi[:, d_idx].reshape(2, 1), center.reshape(2, 1)
                )

        # All patients: convoy toward center
        for p_local in range(NUM_PATIENTS):
            g_pat = NUM_DOCTORS + p_local
            pos = xi[:, g_pat]
            blend = 0.2 + 0.6 * progress
            target = (1 - blend) * pos + blend * center
            target = clamp_to_arena(target)
            dxi[:, g_pat:g_pat+1] = si_position_controller(
                pos.reshape(2, 1), target.reshape(2, 1)
            )

    # =================================================================
    # PHASE 5: RECOVERY FORMATION (50–60 sec)
    # =================================================================
    else:
        for i in range(N):
            dxi[:, i:i+1] = si_position_controller(
                xi[:, i].reshape(2, 1), final_ring[:, i].reshape(2, 1)
            )

    # ─────────────────────────────────────────────────────────────────
    # SAFETY: barrier certificates + SI→unicycle conversion
    # ─────────────────────────────────────────────────────────────────
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(N), dxu)
    r.step()

# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED: Robotarium server cleanup
# ─────────────────────────────────────────────────────────────────────────────
r.call_at_scripts_end()
