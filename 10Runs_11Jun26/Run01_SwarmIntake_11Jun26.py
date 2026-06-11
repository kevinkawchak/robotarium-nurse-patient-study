"""
================================================================================
  RUN 01 / 10 - SWARM FLOCKING PATIENT INTAKE (Boids + Shepherding)
  Georgia Tech Robotarium - https://www.robotarium.gatech.edu/experiment
================================================================================

  CLINICAL TRIAL OBJECTIVE
    Phase-0 cohort intake: 8 ambulatory patient robots in the waiting area
    self-organize as a flock (boids rules). 4 nurse robots shepherd the
    flock across the ward to two intake bays, where 2 doctor robots screen
    and enroll the cohort. Demonstrates crowd-flow management for trial
    enrollment without any centralized path planner.

  FLEET (14 robots, 2-5 min run)
    Robots  0-1   : DOCTORS  (intake bays, left wall)        LED blue
    Robots  2-5   : NURSES   (shepherds)                     LED green
    Robots  6-13  : PATIENTS (flocking cohort)               LED amber/red

  ALGORITHM PATTERN: swarm intelligence (Reynolds boids: cohesion,
    separation, alignment) + shepherding pressure fields (nurses position
    behind the flock relative to the drive goal, like sheepdogs).

  REAL-ROBOT TIMING ASSUMPTIONS
    * 15 s standby head time before tasks begin (real fleet start delay).
    * Linear speed planned at 0.14 m/s (30% below the 0.20 m/s platform max).
    * Angular speed planned at 1.8 rad/s (50% below the 3.6 rad/s max).
    * Wheel-speed rescaling keeps every command inside actuator limits.

  STEP-BY-STEP TIMELINE (4545 iterations @ 0.033 s = 150 s wall-clock)
  ----------------------------------------------------------------------------
  STANDBY (0..15 s) - real-robot start delay buffer
    * t =  0 s  - all 14 robots hold their start poses, motors idle.
    * t =  0-10 s - LEDs breathe dim white at 0.5 Hz (system check pattern).
    * t = 10-15 s - LEDs ramp from white to role colors (role announce):
                    doctors to blue, nurses to green, patients to amber.
  PHASE 1 - FREE FLOCKING (15..45 s)
    * t = 15 s  - patients begin boids flocking in the right half: cohesion
                  toward neighbors within 0.65 m, separation inside 0.40 m,
                  alignment with neighbor headings, slow wander bias.
    * t = 22 s  - flock visibly coalesces (mean nearest-neighbor distance
                  contracts to ~0.23 m, riding the barrier floor); nurses
                  hold the staging column, doctors hold the intake bays.
    * t = 30 s  - flock circulates the right half as 1-2 coherent groups;
                  polarization metric printed (~0.2-0.4 while milling: the
                  wander bias deliberately keeps the flock turning).
    * t = 45 s  - phase ends with a cohesive, wandering patient flock.
  PHASE 2 - SHEPHERDING DRIVE (45..90 s)
    * t = 45 s  - nurses leave staging, arc around the flock (~0.45 m behind
                  it relative to the bays), split 2-up / 2-down.
    * t = 55 s  - nurse pressure (repulsion within 0.55 m) starts the flock
                  drifting left at the derated 0.14 m/s budget.
    * t = 70 s  - flock crosses mid-arena (x ~ 0.0) and begins splitting
                  into an upper and a lower stream (emergent lane split).
    * t = 90 s  - both streams are inside the left half, near the bays.
  PHASE 3 - BAY DOCKING (90..120 s)
    * t = 90 s  - each patient is greedily assigned an arc slot around its
                  bay doctor (radius 0.42 m); doctors pulse bright blue.
    * t = 100 s - patients file onto their arc slots; LEDs turn green within
                  0.50 m of the bay center (enrolled).
    * t = 120 s - both bays filled (expected 4 patients per bay).
  PHASE 4 - COHORT HOLD + ALL-CLEAR (120..150 s)
    * t = 120 s - nurses take perimeter guard posts beside the bays.
    * t = 130 s - LED all-clear wave sweeps across the fleet (green pulse
                  traveling by robot index, 0.5 Hz).
    * t = 150 s - run ends; debug/cleanup hooks called.
  ----------------------------------------------------------------------------

  EMERGENT BEHAVIORS DEMONSTRATED (printed as metrics during the run)
    1. Flock self-assembly: nearest-neighbor distance contracts without any
       assigned formation (boids only).
    2. Polarization surge: the order parameter sits at ~0.2-0.4 while
       the flock mills freely, then jumps above 0.9 the moment shepherd
       pressure gives the group a shared direction (printed at 90 s).
    3. Shepherding flow: flock moves AWAY from nurses it never communicates
       with - pure local repulsion produces goal-directed transport.
    4. Lane split: a single flock divides into two bay streams purely from
       the geometry of nurse pressure (no patient is told its bay).
    5. Arc queueing: greedy slot capture forms orderly intake arcs.

  LED CONFIGURATION (updates every iteration)
    DOCTORS : solid blue (0,90,255); 1 Hz bright-blue pulse while docking.
    NURSES  : solid green (0,200,80); magenta (200,0,255) while actively
              herding (any patient within 0.50 m).
    PATIENTS: amber (255,150,0) while flocking; 2 Hz red blink if isolated
              (> 0.80 m from every other patient); green (90,255,90) once
              enrolled at a bay.

  ROBOTARIUM SUBMISSION COMPLIANCE
    * 14 robots (max 20); 150 s run (max 600 s); min start spacing 0.36 m.
    * get_poses() exactly once per step(); ends with the platform hook.
    * Single barrier certificate stack keeps inter-robot distance >= 0.20 m
      and robots inside [-1.6, 1.6] x [-1.0, 1.0].
    * Imports: numpy + rps only. Arena/figure background forced white.
    * RNPS_FAST_SIM=1 / RNPS_MAX_ITERS are local pre-flight knobs only; the
      Robotarium server never sets them and runs the full real-time script.
================================================================================
"""

import os
import random
import sys
from pathlib import Path

import numpy as np

try:  # installed rps (Robotarium server or simulator checkout)
    import rps.robotarium as robotarium
    from rps.utilities import barrier_certificates as bc
    from rps.utilities import controllers as ctl
    from rps.utilities import transformations as tr
except ImportError:  # running from inside 10Runs_11Jun26/: use the repo-root rps
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import rps.robotarium as robotarium
    from rps.utilities import barrier_certificates as bc
    from rps.utilities import controllers as ctl
    from rps.utilities import transformations as tr

# ----------------------------------------------------------------------------
# rps API RESOLVER - the production server / GTERNAL fork export short-form
# names while the repo-local stub exports long-form names. Probe both.
# ----------------------------------------------------------------------------


def resolve_rps(module, *candidate_names):
    for name in candidate_names:
        fn = getattr(module, name, None)
        if fn is not None:
            return fn
    raise RuntimeError(f"None of {candidate_names!r} found in {module.__name__}.")


# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
RUN_SEED = 1101
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)

NUM_DOCTORS = 2
NUM_NURSES = 4
NUM_PATIENTS = 8
N = NUM_DOCTORS + NUM_NURSES + NUM_PATIENTS  # 14
DOC = np.arange(0, NUM_DOCTORS)
NUR = np.arange(NUM_DOCTORS, NUM_DOCTORS + NUM_NURSES)
PAT = np.arange(NUM_DOCTORS + NUM_NURSES, N)

# Hardware caps and derated planning speeds (real robots run ~30% / ~50% slow)
HW_MAX_LINEAR = 0.20  # m/s, GRITSBot platform maximum
ROBOT_BASE_LENGTH = 0.11  # m, wheelbase used for wheel-speed budgeting
PLAN_LINEAR = 0.14  # m/s  = 70% of platform max
PLAN_ANGULAR = 1.8  # rad/s = 50% of the 3.6 rad/s platform max

DT = 0.033  # s per iteration (Robotarium time step)
STANDBY_SECONDS = 15.0  # real-fleet start delay buffer
TOTAL_SECONDS = 150.0


def sec(seconds):
    """Convert wall-clock seconds to an iteration index."""
    return int(round(seconds / DT))


TOTAL_ITERATIONS = sec(TOTAL_SECONDS)
T_STANDBY = sec(STANDBY_SECONDS)
T_FLOCK = sec(45.0)
T_SHEPHERD = sec(90.0)
T_DOCK = sec(120.0)

# Behaviour gains
FLOCK_RADIUS = 0.65  # m, boids neighborhood
SEP_RADIUS = 0.40  # m, boids separation zone
COHESION_GAIN = 0.55
SEPARATION_GAIN = 0.012
ALIGNMENT_GAIN = 0.35
BIAS_GAIN = 0.30
FLOCK_SPEED = 0.12  # m/s, patient cruise budget while flocking
NURSE_PUSH_RADIUS = 0.55  # m, shepherd pressure zone
NURSE_PUSH_GAIN = 0.035
HERD_STANDOFF = 0.45  # m, nurse position behind the flock
BAY_RADIUS = 0.42  # m, intake arc radius around each doctor
ENROLL_RADIUS = 0.50  # m, LED turns green inside this bay distance
SAFETY_RADIUS = 0.20  # m, barrier certificate inter-robot distance

ARENA = np.array([-1.6, 1.6, -1.0, 1.0])
MARGIN = 0.15

# ----------------------------------------------------------------------------
# INITIAL CONDITIONS (3xN [x; y; theta]) - min pairwise spacing 0.36 m
# Doctors on the left wall bays, nurses in a staging column, patients in a
# 4x2 grid on the right half. Headings face the first travel direction.
# ----------------------------------------------------------------------------
doctor_xy = np.array([[-1.30, -1.30], [0.45, -0.45]])
nurse_xy = np.array([[-0.85, -0.85, -0.85, -0.85], [0.66, 0.22, -0.22, -0.66]])
patient_xy = np.array(
    [
        [0.27, 0.63, 0.99, 1.35, 0.27, 0.63, 0.99, 1.35],
        [0.40, 0.40, 0.40, 0.40, -0.40, -0.40, -0.40, -0.40],
    ]
)
initial_conditions = np.vstack(
    [
        np.hstack([doctor_xy[0], nurse_xy[0], patient_xy[0]]),
        np.hstack([doctor_xy[1], nurse_xy[1], patient_xy[1]]),
        np.hstack([np.zeros(NUM_DOCTORS), np.zeros(NUM_NURSES), np.full(NUM_PATIENTS, np.pi)]),
    ]
)

_d = np.linalg.norm(
    initial_conditions[:2, :, None] - initial_conditions[:2, None, :], axis=0
) + np.eye(N)
assert _d.min() >= 0.35, f"initial spacing {_d.min():.3f} m violates the 0.35 m rule"

BAY_CENTERS = doctor_xy.copy()  # intake bays = doctor posts

# ----------------------------------------------------------------------------
# ROBOTARIUM INIT (RNPS_FAST_SIM=1 -> headless fast pre-flight; unset on server)
# ----------------------------------------------------------------------------
_FAST = os.environ.get("RNPS_FAST_SIM", "0") == "1"
_ITER_CAP = int(os.environ.get("RNPS_MAX_ITERS", str(TOTAL_ITERATIONS)))

r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=not _FAST,
    initial_conditions=initial_conditions,
    sim_in_real_time=not _FAST,
)

# Arena rendering: force a white arena/figure background where a figure exists.
_fig = getattr(r, "_fig", None) or getattr(r, "figure", None)
_axes = getattr(r, "_axes_handle", None) or getattr(r, "axes", None)
if _fig is not None:
    _fig.patch.set_facecolor("white")
if _axes is not None:
    _axes.set_facecolor("white")

# ----------------------------------------------------------------------------
# LED SHIM - one writer for every rps flavor (3xN RGB matrix, 0-255).
# ----------------------------------------------------------------------------
_led_left = getattr(r, "set_left_leds", None)
_led_right = getattr(r, "set_right_leds", None)
_led_array = getattr(r, "_leds", None)
_ALL_IDS = np.arange(N)


def write_leds(colors):
    colors = np.clip(np.asarray(colors, dtype=float), 0.0, 255.0)
    if callable(_led_left):
        _led_left(_ALL_IDS, colors)
    if callable(_led_right):
        _led_right(_ALL_IDS, colors)
    if _led_array is not None and getattr(_led_array, "shape", None) == colors.shape:
        _led_array[:, :] = colors


LED_DOCTOR = np.array([0.0, 90.0, 255.0])
LED_DOCTOR_PULSE = np.array([140.0, 200.0, 255.0])
LED_NURSE = np.array([0.0, 200.0, 80.0])
LED_NURSE_HERD = np.array([200.0, 0.0, 255.0])
LED_PATIENT = np.array([255.0, 150.0, 0.0])
LED_PATIENT_LOST = np.array([255.0, 30.0, 0.0])
LED_PATIENT_OK = np.array([90.0, 255.0, 90.0])
ROLE_COLORS = np.zeros((3, N))
ROLE_COLORS[:, DOC] = LED_DOCTOR[:, None]
ROLE_COLORS[:, NUR] = LED_NURSE[:, None]
ROLE_COLORS[:, PAT] = LED_PATIENT[:, None]

# ----------------------------------------------------------------------------
# CONTROLLERS & SAFETY STACK
# ----------------------------------------------------------------------------
make_si_barrier = resolve_rps(
    bc,
    "create_si_barrier_certificate_with_boundary",
    "create_single_integrator_barrier_certificate_with_boundary",
)
si_barrier = make_si_barrier(safety_radius=SAFETY_RADIUS, magnitude_limit=PLAN_LINEAR)
si_to_uni = tr.create_si_to_uni_dynamics(
    linear_velocity_gain=1.0, angular_velocity_limit=PLAN_ANGULAR
)
si_position = ctl.create_si_position_controller(
    x_velocity_gain=1.2, y_velocity_gain=1.2, velocity_magnitude_limit=PLAN_LINEAR
)


def wheel_safe(dxu):
    """Clamp [v; w] to derated caps, then rescale pairs to honor wheel limits.

    GRITSBot wheel speeds are (2v +/- L*w) / (2r); keeping |2v| + L|w| inside
    2 * HW_MAX_LINEAR guarantees both wheels stay legal for any v/w mix.
    """
    out = np.copy(dxu)
    out[0, :] = np.clip(out[0, :], -PLAN_LINEAR, PLAN_LINEAR)
    out[1, :] = np.clip(out[1, :], -PLAN_ANGULAR, PLAN_ANGULAR)
    demand = 2.0 * np.abs(out[0, :]) + ROBOT_BASE_LENGTH * np.abs(out[1, :])
    # 0.1% margin so the rescaled command never rounds onto the exact limit.
    budget = 2.0 * HW_MAX_LINEAR * 0.999
    over = demand > budget
    if np.any(over):
        out[:, over] *= budget / demand[over]
    return out


def clamp_arena(p, margin=MARGIN):
    out = np.copy(p)
    out[0] = np.clip(out[0], ARENA[0] + margin, ARENA[1] - margin)
    out[1] = np.clip(out[1], ARENA[2] + margin, ARENA[3] - margin)
    return out


def cap_speed(v, limit):
    mag = np.linalg.norm(v, axis=0)
    over = mag > limit
    if np.any(over):
        v = np.copy(v)
        v[:, over] *= limit / mag[over]
    return v


# ----------------------------------------------------------------------------
# BEHAVIOURS
# ----------------------------------------------------------------------------
patient_vel = np.zeros((2, NUM_PATIENTS))  # last commanded SI velocity (alignment memory)
bay_assignment = None  # patient -> (bay index, slot position), fixed at docking start


def boids_velocity(p, t_sec):
    """Reynolds boids for the patient flock, with a slow wander bias."""
    diffs = p[:, :, None] - p[:, None, :]  # (2, K, K) i minus j
    dists = np.linalg.norm(diffs, axis=0) + np.eye(NUM_PATIENTS) * 1e9
    neigh = dists < FLOCK_RADIUS

    dxi = np.zeros((2, NUM_PATIENTS))
    # Wander bias: anchor orbiting the right-half center keeps the flock inside.
    anchor = np.array([0.75 + 0.25 * np.sin(0.10 * t_sec), 0.30 * np.sin(0.07 * t_sec)])
    for i in range(NUM_PATIENTS):
        nb = neigh[i]
        if np.any(nb):
            centroid = p[:, nb].mean(axis=1)
            dxi[:, i] += COHESION_GAIN * (centroid - p[:, i])
            dxi[:, i] += ALIGNMENT_GAIN * (patient_vel[:, nb].mean(axis=1) - patient_vel[:, i])
        close = dists[i] < SEP_RADIUS
        if np.any(close):
            rel = p[:, i : i + 1] - p[:, close]
            d2 = np.maximum(np.linalg.norm(rel, axis=0) ** 2, 1e-4)
            dxi[:, i] += SEPARATION_GAIN * (rel / d2).sum(axis=1)
        dxi[:, i] += BIAS_GAIN * (anchor - p[:, i])
    return cap_speed(dxi, FLOCK_SPEED)


def nurse_pressure(p_pat, p_nur):
    """Shepherd push: patients flee nurses inside NURSE_PUSH_RADIUS."""
    push = np.zeros((2, NUM_PATIENTS))
    for i in range(NUM_PATIENTS):
        rel = p_pat[:, i : i + 1] - p_nur
        d = np.linalg.norm(rel, axis=0)
        act = d < NURSE_PUSH_RADIUS
        if np.any(act):
            d2 = np.maximum(d[act] ** 2, 1e-4)
            push[:, i] = NURSE_PUSH_GAIN * (rel[:, act] / d2).sum(axis=1)
    return push


def herd_targets(p_pat, t_sec):
    """Nurses stand behind their half-flock relative to its bay (sheepdog posts)."""
    targets = np.zeros((2, NUM_NURSES))
    for half, bay in enumerate(BAY_CENTERS.T):  # half 0 = upper bay, 1 = lower bay
        mask = p_pat[1] >= 0 if half == 0 else p_pat[1] < 0
        group = p_pat[:, mask] if np.any(mask) else p_pat
        centroid = group.mean(axis=1)
        away = centroid - bay
        away = away / max(np.linalg.norm(away), 1e-6)
        perp = np.array([-away[1], away[0]])
        base = centroid + HERD_STANDOFF * away
        wig = 0.06 * np.sin(1.2 * t_sec)  # slow weave keeps pressure moving
        targets[:, 2 * half] = base + perp * (0.25 + wig)
        targets[:, 2 * half + 1] = base - perp * (0.25 - wig)
    return clamp_arena(targets)


def greedy_slots(p_pat):
    """One-shot greedy patient -> bay arc slot assignment at docking start."""
    slots = []
    for bay in BAY_CENTERS.T:
        angles = np.linspace(-np.pi / 3.0, np.pi / 3.0, 4)  # arcs open toward +x
        for a in angles:
            slots.append(bay + BAY_RADIUS * np.array([np.cos(a), np.sin(a)]))
    slots = np.array(slots).T  # (2, 8)
    cost = np.linalg.norm(p_pat[:, :, None] - slots[:, None, :], axis=0)
    assign = np.full(NUM_PATIENTS, -1)
    for _ in range(NUM_PATIENTS):
        flat = int(np.argmin(cost))
        i, s = divmod(flat, NUM_PATIENTS)
        assign[i] = s
        cost[i, :] = np.inf
        cost[:, s] = np.inf
    return slots, assign


GUARD_POSTS = np.array([[-0.70, -0.70, -0.70, -0.70], [0.80, 0.30, -0.30, -0.80]])


def flock_metrics(p, v):
    """Polarization order parameter and mean nearest-neighbor distance."""
    mags = np.linalg.norm(v, axis=0)
    moving = mags > 1e-6
    pol = 0.0
    if np.any(moving):
        pol = float(np.linalg.norm((v[:, moving] / mags[moving]).mean(axis=1)))
    d = np.linalg.norm(p[:, :, None] - p[:, None, :], axis=0) + np.eye(NUM_PATIENTS) * 1e9
    return pol, float(d.min(axis=1).mean())


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
print(f"Run01 SwarmIntake: {N} robots, {TOTAL_SECONDS:.0f} s, seed {RUN_SEED}")
metric_marks = {sec(s) for s in (30.0, 45.0, 60.0, 75.0, 90.0)}

for t in range(min(TOTAL_ITERATIONS, _ITER_CAP)):
    x = r.get_poses()
    pos = x[:2, :]
    t_sec = t * DT
    dxi = np.zeros((2, N))
    leds = ROLE_COLORS.copy()

    if t < T_STANDBY:
        # STANDBY: hold poses; LED system-check breathing, then role announce.
        breathe = 0.25 + 0.55 * 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.5 * t_sec))
        white = np.full((3, N), 255.0) * breathe
        if t_sec < 10.0:
            leds = white
        else:
            blend = (t_sec - 10.0) / 5.0
            leds = (1.0 - blend) * white + blend * ROLE_COLORS
    else:
        p_pat = pos[:, PAT]
        p_nur = pos[:, NUR]

        if t < T_FLOCK:
            # PHASE 1: free boids flocking; staff hold posts.
            dxi[:, PAT] = boids_velocity(p_pat, t_sec)
            dxi[:, NUR] = si_position(p_nur, nurse_xy)
            dxi[:, DOC] = si_position(pos[:, DOC], doctor_xy)
        elif t < T_SHEPHERD:
            # PHASE 2: nurses shepherd; patients = boids + nurse pressure.
            dxi[:, PAT] = cap_speed(
                boids_velocity(p_pat, t_sec) + nurse_pressure(p_pat, p_nur), FLOCK_SPEED
            )
            dxi[:, NUR] = si_position(p_nur, herd_targets(p_pat, t_sec))
            dxi[:, DOC] = si_position(pos[:, DOC], doctor_xy)
            leds[:, NUR] = LED_NURSE[:, None]
            for k, n_idx in enumerate(NUR):
                if np.min(np.linalg.norm(p_pat - pos[:, n_idx : n_idx + 1], axis=0)) < 0.50:
                    leds[:, n_idx] = LED_NURSE_HERD
        else:
            # PHASES 3-4: dock onto bay arcs, then hold with nurse guard posts.
            if bay_assignment is None:
                bay_assignment = greedy_slots(p_pat)
                counts = [int((bay_assignment[1] < 4).sum()), int((bay_assignment[1] >= 4).sum())]
                print(f"[t={t_sec:5.1f}s] docking starts: bay split upper/lower = {counts}")
            slots, assign = bay_assignment
            dxi[:, PAT] = si_position(p_pat, slots[:, assign])
            nurse_goal = GUARD_POSTS if t >= T_DOCK else herd_targets(p_pat, t_sec)
            dxi[:, NUR] = si_position(p_nur, nurse_goal)
            dxi[:, DOC] = si_position(pos[:, DOC], doctor_xy)
            pulse = 0.5 * (1.0 + np.sin(2.0 * np.pi * 1.0 * t_sec))
            leds[:, DOC] = (LED_DOCTOR + pulse * (LED_DOCTOR_PULSE - LED_DOCTOR))[:, None]
            if t >= T_DOCK:
                wave = 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.5 * t_sec - np.arange(N) * 0.6))
                leds = leds * (0.55 + 0.45 * wave)

        # Patient LED status (flock health / enrollment), all task phases.
        d_pp = np.linalg.norm(p_pat[:, :, None] - p_pat[:, None, :], axis=0)
        d_pp += np.eye(NUM_PATIENTS) * 1e9
        d_bay = np.linalg.norm(p_pat[:, :, None] - BAY_CENTERS[:, None, :], axis=0).min(axis=1)
        blink = 0.5 * (1.0 + np.sin(2.0 * np.pi * 2.0 * t_sec))
        for k, p_idx in enumerate(PAT):
            if t >= T_SHEPHERD and d_bay[k] < ENROLL_RADIUS:
                leds[:, p_idx] = LED_PATIENT_OK
            elif d_pp[k].min() > 0.80:
                leds[:, p_idx] = LED_PATIENT_LOST * (0.35 + 0.65 * blink)

        patient_vel = dxi[:, PAT].copy()
        if t in metric_marks:
            pol, nn = flock_metrics(pos[:, PAT], patient_vel)
            print(f"[t={t_sec:5.1f}s] flock polarization={pol:.2f} mean-NN={nn:.2f} m")

    # SAFETY STACK: barrier -> unicycle map -> stop guard -> wheel budget.
    dxi = si_barrier(dxi, x)
    dxu = si_to_uni(dxi, x)
    stopped = np.linalg.norm(dxi, axis=0) < 1e-4
    if np.any(stopped):
        dxu[:, stopped] = 0.0
    dxu = wheel_safe(dxu)

    write_leds(leds)
    r.set_velocities(_ALL_IDS, dxu)
    r.step()

# Final bay tally for the experiment log.
final_assign = bay_assignment[1] if bay_assignment is not None else None
if final_assign is not None:
    print(
        f"Run01 complete: bay occupancy upper={int((final_assign < 4).sum())}, "
        f"lower={int((final_assign >= 4).sum())} of {NUM_PATIENTS} patients"
    )

# Platform cleanup hooks (server expects call_at_scripts_end; fork exposes debug).
_end_hook = getattr(r, "call_at_scripts_end", None)
if callable(_end_hook):
    _end_hook()
_debug_hook = getattr(r, "debug", None)
if callable(_debug_hook):
    _debug_hook()
