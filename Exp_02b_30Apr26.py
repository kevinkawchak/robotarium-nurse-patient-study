"""
================================================================================
  ROBOTARIUM SWARM EXPERIMENT v2b: Real-Time Doctor-Patient Medical Response
  Georgia Tech Robotarium - https://www.robotarium.gatech.edu/experiment
================================================================================

  16 GRITSBots - 6 Doctors (indices 0-5) and 10 Patients (indices 6-15)
  demonstrate emergent swarm-based triage and medical response.

  MOTIVATION
    Exp_01a_12Feb26.py capped the single-integrator velocity at 0.12 m/s,
    which made every phase finish ~3x slower than the AI-narrated timeline.
    v2b commands at the GRITSBot maximums published by GA Tech:

        linear  : 0.20 m/s  (20 cm/s)
        angular : pi rad/s  (Robotarium hard clip; physical bot ~3.6 rad/s)

    All controllers, barrier certificates and SI->unicycle transforms route
    through these caps so the run reaches its narrated wall-clock instead of
    being silently clipped to 60% speed.

  STEP-BY-STEP TIMELINE  (1800 iterations @ 30 Hz = 60 s wall-clock)
  ──────────────────────────────────────────────────────────────────────────
  PHASE 1 - DISTRESS SIGNAL  (0..8 s)
    * t = 0.0 s  - sim spawns. 6 doctors hold a 2x3 grid on the left wall
                   ("hospital base"). 10 patients sit in a 4-2-4 scatter on
                   the right side of the arena.
    * t = 0.5 s  - patients begin small-amplitude oscillations (1.8 cm,
                   0.05 Hz) simulating medical distress.
    * t = 4.0 s  - distress oscillation is at full amplitude. Doctors still
                   holding formation, motors idle.
    * t = 8.0 s  - phase ends. Doctors stationary; all patients oscillating.

  PHASE 2 - DISPATCH & TRIAGE SWEEP  (8..20 s)
    * t =  8.0 s - doctors break formation. Mutual-repulsion dispersion
                   spreads them across the arena (self-organized coverage).
    * t = 12.5 s - repulsion phase ends; greedy nearest-neighbor matcher
                   assigns each of the 6 doctors its closest unclaimed
                   patient (decentralized task allocation).
    * t = 12.5..20.0 s
                 - doctors translate at the 0.20 m/s cap toward the
                   assigned patient. Distress amplitude damps linearly
                   1.0 -> 0.2 as help arrives.
    * t = 20.0 s - every doctor is within ~0.30 m of its assigned patient.

  PHASE 3 - TREATMENT & STABILIZATION  (20..38 s)
    * t = 20..30 s - doctors orbit their assigned patient at radius 0.25 m
                     (active treatment). Claimed patients hold position.
    * t = 20..30 s - the 4 unclaimed patients flock toward the nearest
                     doctor-patient cluster centroid (emergent triage
                     groups self-organize without a central planner).
    * t = 30..38 s - clusters stabilize; unclaimed patients halt at
                     >= 0.32 m so they don't collide with the orbit.

  PHASE 4 - EVACUATION CONVOY  (38..50 s)
    * t = 38..44 s - every cluster begins a coordinated translation toward
                     the arena origin (the projected logo). Doctors lead
                     from the front by a +0.12 m offset along the travel
                     direction (convoy formation).
    * t = 44..50 s - convoy compresses as it nears the origin. Assignments
                     refresh each iteration so the closest doctor stays
                     paired with its patient.

  PHASE 5 - RECOVERY FORMATION  (50..60 s)
    * t = 50..58 s - all 16 robots converge to evenly-spaced points on a
                     ring of radius 0.65 m centered on the origin
                     (symmetric consensus).
    * t = 58..60 s - ring is fully formed; small final corrections only.
  ──────────────────────────────────────────────────────────────────────────

  EMERGENT BEHAVIOURS DEMONSTRATED
    1. Self-organized spatial coverage (mutual-repulsion dispersion)
    2. Decentralized task allocation (greedy nearest-neighbor matching)
    3. Orbiting / shepherding (doctor-patient interaction)
    4. Flocking & attraction (unassigned patients cluster to groups)
    5. Convoy formation & collective migration
    6. Symmetric consensus formation (final ring)

  ROBOTARIUM SUBMISSION COMPLIANCE
    * Robot count: 16 (within the 20-robot fleet maximum).
    * Run-time: 1800 iterations ~= 60 s (well under the 600 s server cap).
    * Min initial spacing: 0.35 m (Robotarium needs >= 0.30 m).
    * Linear cap: 0.20 m/s; angular cap: pi rad/s - matches platform spec.
    * Barrier certificate: safety_radius=0.17 with magnitude_limit=0.20.
    * Imports: only numpy and rps.* - no scipy, cvxopt, or external pkgs.
    * Public API only: get_poses, set_velocities, step, call_at_scripts_end.
================================================================================
"""

import numpy as np

import rps.robotarium as robotarium
from rps.utilities import barrier_certificates as bc
from rps.utilities import controllers as ctl
from rps.utilities import transformations as tr

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION - fleet, hardware caps, timing, behaviour gains
# ─────────────────────────────────────────────────────────────────────────────
N = 16  # Total robots
NUM_DOCTORS = 6  # Robots 0..5
NUM_PATIENTS = 10  # Robots 6..15
DOCTOR_IDS = list(range(NUM_DOCTORS))
PATIENT_IDS = list(range(NUM_DOCTORS, N))

# Robotarium hardware caps (per https://www.robotarium.gatech.edu/experiment)
MAX_LINEAR_SPEED = 0.20  # m/s
MAX_ANGULAR_SPEED = np.pi  # rad/s - Robotarium clip; physical bot ~3.6

# Timing - 30 Hz step (0.0333 s/iter)
STEP_HZ = 30.0
DT = 1.0 / STEP_HZ
TOTAL_ITERATIONS = 1800  # 60 s
PHASE_1_END = int(8 * STEP_HZ)  # 240 - 0..8s
PHASE_2_END = int(20 * STEP_HZ)  # 600 - 8..20s
PHASE_3_END = int(38 * STEP_HZ)  # 1140 - 20..38s
PHASE_4_END = int(50 * STEP_HZ)  # 1500 - 38..50s
# Phase 5 runs from PHASE_4_END to TOTAL_ITERATIONS (50..60s)

# Arena bounds (Robotarium: [-1.6, 1.6] x [-1.0, 1.0])
ARENA_X = (-1.6, 1.6)
ARENA_Y = (-1.0, 1.0)
ARENA_MARGIN = 0.15

# Behaviour parameters
SAFETY_RADIUS = 0.17  # barrier certificate inter-robot safety distance
ORBIT_RADIUS = 0.25  # > SAFETY_RADIUS so doctor stays clear of patient
ORBIT_RATE = 0.06  # rad / iteration -> ~1.8 rad/s, visible at 0.20 m/s
FLOCK_STOP_DIST = 0.32  # unclaimed patients halt outside this radius
DISPERSION_GAIN = 0.40  # mutual-repulsion target step size
DISPERSION_FRACTION = 0.4  # fraction of phase 2 spent dispersing
DISTRESS_AMP = 0.018  # m - distress oscillation amplitude
DISTRESS_FREQ = 0.05  # Hz
LEAD_OFFSET = 0.12  # m - doctor leads patient toward center in convoy
RING_RADIUS = 0.65  # final ring radius

# ─────────────────────────────────────────────────────────────────────────────
# INITIAL CONDITIONS - 3xN [x; y; theta]
# Doctors: 2 columns x 3 rows on the left wall ("hospital base")
# Patients: 4-2-4 grid scattered on the right side of the arena
# Verified: minimum pairwise distance = 0.35 m (Robotarium needs >= 0.30 m)
# ─────────────────────────────────────────────────────────────────────────────
doctor_x = np.array([-1.30, -1.30, -1.30, -0.95, -0.95, -0.95])
doctor_y = np.array([0.55, 0.00, -0.55, 0.55, 0.00, -0.55])

patient_x = np.array([-0.20, 0.30, 0.80, 1.30, 0.30, 0.80, -0.20, 0.30, 0.80, 1.30])
patient_y = np.array([0.65, 0.65, 0.65, 0.65, 0.00, 0.00, -0.65, -0.65, -0.65, -0.65])

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
#   * SI position controller saturates at MAX_LINEAR_SPEED.
#   * SI -> unicycle transform caps angular velocity at MAX_ANGULAR_SPEED.
#   * Barrier cert enforces inter-robot collision avoidance and arena bounds.
# ─────────────────────────────────────────────────────────────────────────────
si_barrier_cert = bc.create_single_integrator_barrier_certificate_with_boundary(
    safety_radius=SAFETY_RADIUS,
    magnitude_limit=MAX_LINEAR_SPEED,
)
si_to_uni_dyn = tr.create_si_to_uni_dynamics(
    linear_velocity_gain=1.0,
    angular_velocity_limit=MAX_ANGULAR_SPEED,
)
si_position_controller = ctl.create_si_position_controller(
    x_velocity_gain=1.2,
    y_velocity_gain=1.2,
    velocity_magnitude_limit=MAX_LINEAR_SPEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (numpy-only - no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────
def clamp_to_arena(pos, margin=ARENA_MARGIN):
    """Clamp a 2D position (or 2xN array) to stay within the arena bounds."""
    out = np.copy(pos)
    out[0] = np.clip(out[0], ARENA_X[0] + margin, ARENA_X[1] - margin)
    out[1] = np.clip(out[1], ARENA_Y[0] + margin, ARENA_Y[1] - margin)
    return out


def distress_offsets(t, n):
    """Vectorized jittery oscillation offsets simulating patient distress."""
    phases = np.arange(n) * (2.0 * np.pi / n)
    base = DISTRESS_FREQ * t * 2.0 * np.pi
    offsets = np.empty((2, n))
    offsets[0, :] = DISTRESS_AMP * np.sin(base + phases)
    offsets[1, :] = DISTRESS_AMP * np.cos(base + phases * 1.3)
    return offsets


def greedy_assignment(doc_pos, pat_pos):
    """Greedy nearest-neighbor matcher.

    Each iteration picks the globally smallest doctor->patient distance via
    a single np.argmin, then masks that row + column. No Python triple-loop.
    Returns ({}, set()) if either fleet is empty so callers can fall back.
    """
    n_doc = doc_pos.shape[1]
    n_pat = pat_pos.shape[1]
    if n_doc == 0 or n_pat == 0:
        return {}, set()

    dx = doc_pos[0, :, None] - pat_pos[0, None, :]
    dy = doc_pos[1, :, None] - pat_pos[1, None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    assignment = {}
    claimed = set()
    for _ in range(min(n_doc, n_pat)):
        flat = int(np.argmin(dist))
        d, p = divmod(flat, n_pat)
        if not np.isfinite(dist[d, p]):
            break
        assignment[d] = p
        claimed.add(p)
        dist[d, :] = np.inf
        dist[:, p] = np.inf
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
final_ring = ring_formation(N, center=np.array([0.0, 0.0]), radius=RING_RADIUS)

assignment = {}
claimed = set()
# True once phase 2's dispersion handed off to the intercept sub-phase. While
# this flag is set, phase 3 reuses the same matching every tick so a doctor
# already orbiting a patient never gets reshuffled to a different one mid-run.
assignment_locked = False


# ─────────────────────────────────────────────────────────────────────────────
# PER-PHASE CONTROLLERS
#   Each phase function takes (t, xi) and returns a 2xN dxi command.
#   Greedy `assignment` / `claimed` are updated as a side effect through the
#   module-level globals so phases can re-use the previous matching when it
#   is still valid.
# ─────────────────────────────────────────────────────────────────────────────
def phase1_distress(t, xi):
    """0..8s: doctors hold base, patients oscillate in distress."""
    dxi = np.zeros((2, N))
    dxi[:, :NUM_DOCTORS] = si_position_controller(xi[:, :NUM_DOCTORS], doctor_home)

    targets = clamp_to_arena(patient_home + distress_offsets(t, NUM_PATIENTS))
    dxi[:, NUM_DOCTORS:] = si_position_controller(xi[:, NUM_DOCTORS:], targets)
    return dxi


def _doctor_dispersion_targets(xi):
    """Mutual-repulsion target positions for the 6 doctors (vectorized).

    Pairwise diffs as a (2, n_doc, n_doc) tensor -> inverse-square
    contributions -> sum across the j axis -> unit-normalize -> scale.
    """
    doc_xi = xi[:, :NUM_DOCTORS]
    diffs = doc_xi[:, :, None] - doc_xi[:, None, :]  # (2, n, n)
    dists = np.linalg.norm(diffs, axis=0)  # (n, n)
    np.fill_diagonal(dists, np.inf)  # no self-repulsion
    dists = np.maximum(dists, 0.01)  # avoid divide-by-zero
    contrib = diffs / (dists * dists)[None, :, :]
    repulsion = contrib.sum(axis=2)  # (2, n)
    mag = np.linalg.norm(repulsion, axis=0)
    nonzero = mag > 0
    repulsion[:, nonzero] = repulsion[:, nonzero] / mag[nonzero] * DISPERSION_GAIN
    return clamp_to_arena(doc_xi + repulsion)


def phase2_dispatch(t, xi):
    """8..20s: doctors disperse then sprint to greedily-matched patients."""
    global assignment, claimed, assignment_locked
    dxi = np.zeros((2, N))
    progress = (t - PHASE_1_END) / (PHASE_2_END - PHASE_1_END)

    if progress < DISPERSION_FRACTION:
        doc_targets = _doctor_dispersion_targets(xi)
        dxi[:, :NUM_DOCTORS] = si_position_controller(xi[:, :NUM_DOCTORS], doc_targets)
        assignment_locked = False
    else:
        # Compute the matching once at the dispersion -> intercept handoff and
        # reuse it for the rest of phase 2 + all of phase 3 so a doctor
        # already steering toward patient P never gets bounced to patient Q.
        if not assignment_locked:
            assignment, claimed = greedy_assignment(
                xi[:, :NUM_DOCTORS], xi[:, NUM_DOCTORS:]
            )
            assignment_locked = True
        for d_idx, p_idx in assignment.items():
            g_pat = NUM_DOCTORS + p_idx
            dxi[:, d_idx : d_idx + 1] = si_position_controller(
                xi[:, d_idx].reshape(2, 1), xi[:, g_pat].reshape(2, 1)
            )

    # Distress amplitude damps linearly 1.0 -> 0.2 across the phase
    damping = max(0.2, 1.0 - 0.8 * progress)
    targets = clamp_to_arena(patient_home + distress_offsets(t, NUM_PATIENTS) * damping)
    dxi[:, NUM_DOCTORS:] = si_position_controller(xi[:, NUM_DOCTORS:], targets)
    return dxi


def phase3_treatment(t, xi):
    """20..38s: doctors orbit patients, unclaimed patients flock to clusters."""
    global assignment, claimed, assignment_locked
    dxi = np.zeros((2, N))
    # Reuse phase 2's matching unless we somehow arrived without one.
    if not assignment_locked or not assignment:
        assignment, claimed = greedy_assignment(
            xi[:, :NUM_DOCTORS], xi[:, NUM_DOCTORS:]
        )
        assignment_locked = True

    # Doctors orbit their assigned patient
    for d_idx, p_idx in assignment.items():
        g_pat = NUM_DOCTORS + p_idx
        center = xi[:, g_pat]
        angle = ORBIT_RATE * t + d_idx * (2 * np.pi / NUM_DOCTORS)
        orb = clamp_to_arena(center + ORBIT_RADIUS * np.array([np.cos(angle), np.sin(angle)]))
        dxi[:, d_idx : d_idx + 1] = si_position_controller(
            xi[:, d_idx].reshape(2, 1), orb.reshape(2, 1)
        )

    # Claimed patients hold position
    for p_idx in claimed:
        dxi[:, NUM_DOCTORS + p_idx] = np.zeros(2)

    # Unclaimed patients flock to the nearest cluster centroid
    unclaimed = set(range(NUM_PATIENTS)) - claimed
    if assignment and unclaimed:
        centers = np.array(
            [
                0.5 * (xi[:, d_idx] + xi[:, NUM_DOCTORS + p_idx])
                for d_idx, p_idx in assignment.items()
            ]
        ).T
        for p_local in unclaimed:
            g_pat = NUM_DOCTORS + p_local
            pos = xi[:, g_pat]
            dists = np.linalg.norm(centers - pos.reshape(2, 1), axis=0)
            nearest = int(np.argmin(dists))
            target = centers[:, nearest]
            if dists[nearest] > FLOCK_STOP_DIST:
                vel = si_position_controller(pos.reshape(2, 1), target.reshape(2, 1))
                dxi[:, g_pat : g_pat + 1] = vel * 0.7
            else:
                dxi[:, g_pat] = np.zeros(2)
    return dxi


def phase4_evacuation(t, xi):
    """38..50s: clusters convoy toward the origin, doctors lead from the front."""
    global assignment, claimed
    dxi = np.zeros((2, N))
    progress = (t - PHASE_3_END) / (PHASE_4_END - PHASE_3_END)
    origin = np.array([0.0, 0.0])

    assignment, claimed = greedy_assignment(xi[:, :NUM_DOCTORS], xi[:, NUM_DOCTORS:])

    # Doctors lead from the front of each cluster
    for d_idx, p_idx in assignment.items():
        g_pat = NUM_DOCTORS + p_idx
        pat_pos = xi[:, g_pat]
        blend = 0.3 + 0.5 * progress
        doc_target = (1 - blend) * pat_pos + blend * origin

        direction = origin - pat_pos
        dn = np.linalg.norm(direction)
        if dn > 0.01:
            doc_target = doc_target + LEAD_OFFSET * (direction / dn)
        doc_target = clamp_to_arena(doc_target)
        dxi[:, d_idx : d_idx + 1] = si_position_controller(
            xi[:, d_idx].reshape(2, 1), doc_target.reshape(2, 1)
        )

    # Any unassigned doctor (n_doc <= n_pat so unlikely) heads to origin
    for d_idx in range(NUM_DOCTORS):
        if d_idx not in assignment:
            dxi[:, d_idx : d_idx + 1] = si_position_controller(
                xi[:, d_idx].reshape(2, 1), origin.reshape(2, 1)
            )

    # All patients converge on the origin
    for p_local in range(NUM_PATIENTS):
        g_pat = NUM_DOCTORS + p_local
        pos = xi[:, g_pat]
        blend = 0.2 + 0.6 * progress
        target = clamp_to_arena((1 - blend) * pos + blend * origin)
        dxi[:, g_pat : g_pat + 1] = si_position_controller(
            pos.reshape(2, 1), target.reshape(2, 1)
        )
    return dxi


def phase5_recovery(t, xi):
    """50..60s: every robot snaps onto its slot in the recovery ring."""
    return si_position_controller(xi, final_ring)


def select_phase(t):
    if t < PHASE_1_END:
        return phase1_distress
    if t < PHASE_2_END:
        return phase2_dispatch
    if t < PHASE_3_END:
        return phase3_treatment
    if t < PHASE_4_END:
        return phase4_evacuation
    return phase5_recovery


# Below this magnitude (m/s) the SI command is treated as "stop" and the
# unicycle command is forced to zero. Without this guard, claimed patients
# in phase 3 (dxi == 0) would still receive a heading-error torque from
# si_to_uni_dyn (arctan2(0, 0) = 0 desired heading) and spin in place.
SI_STOP_THRESHOLD = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP - just a phase dispatch + the safety stack
# ─────────────────────────────────────────────────────────────────────────────
for t in range(TOTAL_ITERATIONS):
    x = r.get_poses()
    xi = x[:2, :]

    dxi = select_phase(t)(t, xi)

    # SAFETY: barrier certificates + SI -> unicycle conversion
    dxi = si_barrier_cert(dxi, x[:2, :])
    dxu = si_to_uni_dyn(dxi, x)

    # Zero-velocity guard: stop the wheels (v=0, omega=0) for any robot whose
    # SI command rounds to zero, so it doesn't pirouette toward heading 0.
    stopped = np.linalg.norm(dxi, axis=0) < SI_STOP_THRESHOLD
    if np.any(stopped):
        dxu[:, stopped] = 0.0

    r.set_velocities(np.arange(N), dxu)
    r.step()

# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED: Robotarium server cleanup
# ─────────────────────────────────────────────────────────────────────────────
r.call_at_scripts_end()
