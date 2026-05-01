"""
================================================================================
  ROBOTARIUM SWARM EXPERIMENT v2: Emergent Doctor-Patient Medical Response
  Georgia Tech Robotarium — https://www.robotarium.gatech.edu/experiment
================================================================================

  16 GRITSBots — 6 Doctors (indices 0-5) and 10 Patients (indices 6-15)
  demonstrate emergent swarm-based triage and medical response.

  ──────────────────────────────────────────────────────────────────────────
  WHAT'S NEW IN v2 (vs. Exp_01a_12Feb26.py)
  ──────────────────────────────────────────────────────────────────────────
   * Real-world speeds. The prior script capped the single-integrator velocity
     at 0.12 m/s, which made every phase finish ~3x slower than the AI-narrated
     timeline. v2 commands at the GRITSBot maximums published by GA Tech:
        linear  : 0.20 m/s  (20 cm/s)
        angular : π rad/s   (Robotarium hard clip; physical bots ~3.6 rad/s)
   * 16 robots (6 Doctors + 10 Patients) — was 14 (5 + 9).
   * Initial conditions re-laid out so all 16 pairwise distances are >= 0.35 m
     (the Robotarium minimum spacing for safe spawn).
   * Orbit / convoy / flock gains scaled to take advantage of the higher speed
     ceiling so each phase visibly reaches its stable configuration before the
     next phase begins.
  ──────────────────────────────────────────────────────────────────────────

  TIMELINE — 1800 iterations @ 30 Hz = 60 s wall-clock on Robotarium servers
  ──────────────────────────────────────────────────────────────────────────
  [0–8 s]   PHASE 1 — DISTRESS SIGNAL
     t=0.0s : sim spawns; 6 doctors hold a 2x3 grid at the "hospital base"
              on the left wall; 10 patients sit in a scattered field on the
              right (4-2-4 grid).
     t=0.5s : patients begin small-amplitude oscillations (medical distress).
     t=4.0s : distress oscillations are at full amplitude; doctors still
              holding formation.
     t=8.0s : end of phase — doctors are stationary, all patients oscillating.

  [8–20 s]  PHASE 2 — DISPATCH & TRIAGE SWEEP
     t=8.0s  : doctors break formation, mutual-repulsion dispersion across
               the arena (self-organized spatial coverage).
     t=12.5s : repulsion phase ends; greedy nearest-neighbor assignment runs.
               Each of the 6 doctors locks onto its closest unclaimed patient.
     t=12.5–20.0s : doctors translate at full speed (≤0.20 m/s) toward their
               assigned patient. Patient distress oscillation damps linearly
               from 1.0 → 0.2.
     t=20.0s : all 6 doctors are within ~0.30 m of their assigned patient.

  [20–38 s] PHASE 3 — TREATMENT & STABILIZATION
     t=20–30s : doctors orbit their assigned patient at radius 0.25 m
                (active treatment). Claimed patients hold position.
     t=20–30s : the 4 unclaimed patients flock toward the nearest
                doctor-patient cluster centroid (emergent triage groups).
     t=30–38s : groups stabilize; unclaimed patients stop at ≥0.30 m from
                the cluster (no overlap with the orbiting doctor).

  [38–50 s] PHASE 4 — EVACUATION CONVOY
     t=38–44s : every cluster begins a coordinated translation toward the
                arena origin (the projected logo). Doctors lead from the
                front by a +0.12 m offset toward the center.
     t=44–50s : convoy compresses as it nears the origin; assignments
                refresh each iteration so the closest doctor stays paired
                with its patient.

  [50–60 s] PHASE 5 — RECOVERY FORMATION
     t=50–58s : all 16 robots converge to evenly-spaced points on a
                ring of radius 0.65 m centered on the origin.
     t=58–60s : ring is fully formed; small final corrections only.
  ──────────────────────────────────────────────────────────────────────────

  EMERGENT BEHAVIORS DEMONSTRATED
    1. Self-organized spatial coverage (mutual repulsion dispersion)
    2. Decentralized task allocation (greedy nearest-neighbor matching)
    3. Orbiting / shepherding (doctor-patient interaction)
    4. Flocking & attraction (unassigned patients cluster to groups)
    5. Convoy formation & collective migration
    6. Symmetric consensus formation (final ring)

  ROBOTARIUM SUBMISSION NOTES (https://www.robotarium.gatech.edu/experiment)
    * Run-time: 1800 iterations ~= 60 s (well under the 600 s server cap).
    * Robot count: 16 (within the 20-robot fleet maximum).
    * All commanded velocities are saturated by the SI barrier certificate
      (magnitude_limit=0.2 m/s) before being passed to set_velocities,
      matching the platform's published hardware limits.
    * No third-party imports beyond numpy and rps.* — no scipy, no cvxopt
      calls in user code.
    * Only public rps API entry points are used: get_poses, set_velocities,
      step, call_at_scripts_end.
================================================================================
"""

import numpy as np

import rps.robotarium as robotarium
from rps.utilities import barrier_certificates as bc
from rps.utilities import controllers as ctl
from rps.utilities import transformations as tr

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
N = 16  # Total robots
NUM_DOCTORS = 6  # Robots 0..5
NUM_PATIENTS = 10  # Robots 6..15
DOCTOR_IDS = list(range(NUM_DOCTORS))
PATIENT_IDS = list(range(NUM_DOCTORS, N))

# Robotarium hardware caps (per https://www.robotarium.gatech.edu/experiment)
MAX_LINEAR_SPEED = 0.20  # m/s
MAX_ANGULAR_SPEED = np.pi  # rad/s — Robotarium clips here; physical bot ~3.6

# Timing — 30 Hz step (0.0333 s/iter)
TOTAL_ITERATIONS = 1800  # 60 s
PHASE_1_END = 240  # 0–8s
PHASE_2_END = 600  # 8–20s
PHASE_3_END = 1140  # 20–38s
PHASE_4_END = 1500  # 38–50s
# Phase 5 runs from 1500–1800 (50–60s)

# Arena bounds (Robotarium: [-1.6, 1.6] x [-1.0, 1.0])
ARENA_X = [-1.6, 1.6]
ARENA_Y = [-1.0, 1.0]

# Behaviour parameters
ORBIT_RADIUS = 0.25  # > barrier safety_radius (0.17)
FLOCK_STOP_DIST = 0.32

# ─────────────────────────────────────────────────────────────────────────────
# INITIAL CONDITIONS — 3xN [x; y; theta]
# Doctors: 2 columns x 3 rows on the left wall (hospital base)
# Patients: 4 + 2 + 4 grid on the right side of the arena
# Verified: minimum pairwise distance = 0.35 m  (Robotarium needs >= 0.30 m)
# ─────────────────────────────────────────────────────────────────────────────
doctor_x = np.array([-1.30, -1.30, -1.30, -0.95, -0.95, -0.95])
doctor_y = np.array([0.55, 0.00, -0.55, 0.55, 0.00, -0.55])

patient_x = np.array(
    [-0.20, 0.30, 0.80, 1.30, 0.30, 0.80, -0.20, 0.30, 0.80, 1.30]
)
patient_y = np.array(
    [0.65, 0.65, 0.65, 0.65, 0.00, 0.00, -0.65, -0.65, -0.65, -0.65]
)

initial_conditions = np.array(
    [
        np.concatenate([doctor_x, patient_x]),
        np.concatenate([doctor_y, patient_y]),
        np.zeros(N),
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# ROBOTARIUM INIT
# ─────────────────────────────────────────────────────────────────────────────
r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONTROLLERS & SAFETY
#   * SI position controller saturates at MAX_LINEAR_SPEED (0.20 m/s).
#   * SI -> unicycle transform caps angular velocity at the Robotarium clip.
#   * Barrier certificate enforces inter-robot collision avoidance and the
#     arena boundary, again with magnitude_limit=0.20 m/s.
# ─────────────────────────────────────────────────────────────────────────────
si_barrier_cert = bc.create_single_integrator_barrier_certificate_with_boundary(
    safety_radius=0.17, magnitude_limit=MAX_LINEAR_SPEED
)
si_to_uni_dyn = tr.create_si_to_uni_dynamics(
    linear_velocity_gain=1.0, angular_velocity_limit=MAX_ANGULAR_SPEED
)
si_position_controller = ctl.create_si_position_controller(
    x_velocity_gain=1.2,
    y_velocity_gain=1.2,
    velocity_magnitude_limit=MAX_LINEAR_SPEED,
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (numpy-only — no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────
def clamp_to_arena(pos, margin=0.15):
    out = np.copy(pos)
    out[0] = np.clip(out[0], ARENA_X[0] + margin, ARENA_X[1] - margin)
    out[1] = np.clip(out[1], ARENA_Y[0] + margin, ARENA_Y[1] - margin)
    return out


def distress_offsets(t, n):
    """Jittery oscillation offsets simulating patient distress."""
    amp = 0.018
    offsets = np.zeros((2, n))
    for i in range(n):
        phase = i * 2.0 * np.pi / n
        offsets[0, i] = amp * np.sin(0.05 * t * 2 * np.pi + phase)
        offsets[1, i] = amp * np.cos(0.05 * t * 2 * np.pi + phase * 1.3)
    return offsets


def greedy_assignment(doc_pos, pat_pos):
    """Each doctor claims its closest unclaimed patient (numpy-only)."""
    n_doc = doc_pos.shape[1]
    n_pat = pat_pos.shape[1]

    dx = doc_pos[0, :, None] - pat_pos[0, None, :]
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
final_ring = ring_formation(N, center=np.array([0.0, 0.0]), radius=0.65)

assignment = {}
claimed = set()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
for t in range(TOTAL_ITERATIONS):
    x = r.get_poses()
    xi = x[:2, :]
    dxi = np.zeros((2, N))

    # =================================================================
    # PHASE 1: DISTRESS SIGNAL  (0–8 s)
    # =================================================================
    if t < PHASE_1_END:
        dxi[:, :NUM_DOCTORS] = si_position_controller(
            xi[:, :NUM_DOCTORS], doctor_home
        )

        offsets = distress_offsets(t, NUM_PATIENTS)
        targets = patient_home + offsets
        for i in range(NUM_PATIENTS):
            targets[:, i] = clamp_to_arena(targets[:, i])
        dxi[:, NUM_DOCTORS:] = si_position_controller(xi[:, NUM_DOCTORS:], targets)

    # =================================================================
    # PHASE 2: DISPATCH & TRIAGE SWEEP  (8–20 s)
    # =================================================================
    elif t < PHASE_2_END:
        progress = (t - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)

        if progress < 0.4:
            # Mutual-repulsion dispersion across the arena
            doc_targets = np.copy(xi[:, :NUM_DOCTORS])
            for i in range(NUM_DOCTORS):
                repulsion = np.zeros(2)
                for j in range(NUM_DOCTORS):
                    if i == j:
                        continue
                    diff = xi[:, i] - xi[:, j]
                    d = max(np.linalg.norm(diff), 0.01)
                    repulsion += diff / (d**2)
                rn = np.linalg.norm(repulsion)
                if rn > 0:
                    repulsion = repulsion / rn * 0.40
                doc_targets[:, i] = clamp_to_arena(xi[:, i] + repulsion)
            dxi[:, :NUM_DOCTORS] = si_position_controller(
                xi[:, :NUM_DOCTORS], doc_targets
            )
        else:
            # Greedy nearest-neighbor assignment + intercept
            assignment, claimed = greedy_assignment(
                xi[:, :NUM_DOCTORS], xi[:, NUM_DOCTORS:]
            )
            for d_idx, p_idx in assignment.items():
                g_pat = NUM_DOCTORS + p_idx
                target = xi[:, g_pat].reshape(2, 1)
                dxi[:, d_idx : d_idx + 1] = si_position_controller(
                    xi[:, d_idx].reshape(2, 1), target
                )

        # Patient distress damps linearly 1.0 → 0.2
        damping = max(0.2, 1.0 - 0.8 * progress)
        offsets = distress_offsets(t, NUM_PATIENTS) * damping
        targets = patient_home + offsets
        for i in range(NUM_PATIENTS):
            targets[:, i] = clamp_to_arena(targets[:, i])
        dxi[:, NUM_DOCTORS:] = si_position_controller(xi[:, NUM_DOCTORS:], targets)

    # =================================================================
    # PHASE 3: TREATMENT & STABILIZATION  (20–38 s)
    # =================================================================
    elif t < PHASE_3_END:
        assignment, claimed = greedy_assignment(
            xi[:, :NUM_DOCTORS], xi[:, NUM_DOCTORS:]
        )

        # Doctors orbit their assigned patient
        orbit_speed = 0.06  # rad/iter — scaled up for 0.20 m/s top speed
        for d_idx, p_idx in assignment.items():
            g_pat = NUM_DOCTORS + p_idx
            center = xi[:, g_pat]
            angle = orbit_speed * t + d_idx * (2 * np.pi / NUM_DOCTORS)
            orb = center + ORBIT_RADIUS * np.array([np.cos(angle), np.sin(angle)])
            orb = clamp_to_arena(orb)
            dxi[:, d_idx : d_idx + 1] = si_position_controller(
                xi[:, d_idx].reshape(2, 1), orb.reshape(2, 1)
            )

        # Claimed patients hold position
        for p_idx in claimed:
            g_pat = NUM_DOCTORS + p_idx
            dxi[:, g_pat] = np.zeros(2)

        # Unclaimed patients flock to the nearest cluster
        unclaimed = set(range(NUM_PATIENTS)) - claimed
        if len(assignment) > 0:
            centers = []
            for d_idx, p_idx in assignment.items():
                g_pat = NUM_DOCTORS + p_idx
                centers.append(0.5 * (xi[:, d_idx] + xi[:, g_pat]))
            centers = np.array(centers).T  # 2 x num_clusters

            for p_local in unclaimed:
                g_pat = NUM_DOCTORS + p_local
                pos = xi[:, g_pat]
                dists = np.linalg.norm(centers - pos.reshape(2, 1), axis=0)
                nearest = int(np.argmin(dists))
                target = centers[:, nearest]

                if dists[nearest] > FLOCK_STOP_DIST:
                    vel = si_position_controller(
                        pos.reshape(2, 1), target.reshape(2, 1)
                    )
                    dxi[:, g_pat : g_pat + 1] = vel * 0.7
                else:
                    dxi[:, g_pat] = np.zeros(2)

    # =================================================================
    # PHASE 4: EVACUATION CONVOY  (38–50 s)
    # =================================================================
    elif t < PHASE_4_END:
        progress = (t - PHASE_3_END) / (PHASE_4_END - PHASE_3_END)
        center = np.array([0.0, 0.0])

        assignment, claimed = greedy_assignment(
            xi[:, :NUM_DOCTORS], xi[:, NUM_DOCTORS:]
        )

        # Doctors lead from the front
        for d_idx, p_idx in assignment.items():
            g_pat = NUM_DOCTORS + p_idx
            pat_pos = xi[:, g_pat]
            blend = 0.3 + 0.5 * progress
            doc_target = (1 - blend) * pat_pos + blend * center

            direction = center - pat_pos
            dn = np.linalg.norm(direction)
            if dn > 0.01:
                doc_target = doc_target + 0.12 * (direction / dn)

            doc_target = clamp_to_arena(doc_target)
            dxi[:, d_idx : d_idx + 1] = si_position_controller(
                xi[:, d_idx].reshape(2, 1), doc_target.reshape(2, 1)
            )

        # Any unassigned doctor (n_doc <= n_pat so unlikely) heads to center
        for d_idx in range(NUM_DOCTORS):
            if d_idx not in assignment:
                dxi[:, d_idx : d_idx + 1] = si_position_controller(
                    xi[:, d_idx].reshape(2, 1), center.reshape(2, 1)
                )

        # All patients converge on the origin (the projected logo)
        for p_local in range(NUM_PATIENTS):
            g_pat = NUM_DOCTORS + p_local
            pos = xi[:, g_pat]
            blend = 0.2 + 0.6 * progress
            target = (1 - blend) * pos + blend * center
            target = clamp_to_arena(target)
            dxi[:, g_pat : g_pat + 1] = si_position_controller(
                pos.reshape(2, 1), target.reshape(2, 1)
            )

    # =================================================================
    # PHASE 5: RECOVERY FORMATION  (50–60 s)
    # =================================================================
    else:
        for i in range(N):
            dxi[:, i : i + 1] = si_position_controller(
                xi[:, i].reshape(2, 1), final_ring[:, i].reshape(2, 1)
            )

    # ─────────────────────────────────────────────────────────────────
    # SAFETY: barrier certificates + SI -> unicycle conversion
    # ─────────────────────────────────────────────────────────────────
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)
    r.set_velocities(np.arange(N), dxu)
    r.step()

# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED: Robotarium server cleanup
# ─────────────────────────────────────────────────────────────────────────────
r.call_at_scripts_end()
