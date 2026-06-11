"""
================================================================================
  RUN 06 / 10 - GRAPH CONSENSUS VITALS SYNCHRONIZATION
  Georgia Tech Robotarium - https://www.robotarium.gatech.edu/experiment
================================================================================

  CLINICAL TRIAL OBJECTIVE
    Distributed vitals reconciliation: every robot carries a scalar
    'vital estimate' (e.g., a cohort biomarker reading). Robots only
    exchange values with neighbors inside a 0.55 m communication disk,
    so 3 isolated triage rings (one doctor + 3 patients each) can only
    reach ring-local agreement. 4 nurse robots physically ferry between
    rings, carrying their value with them - the ONLY channel through
    which global consensus can emerge.

  FLEET (16 robots, 2.75 min run)
    Robots  0-2   : DOCTORS  (triage ring hubs)              LED vital hue
    Robots  3-6   : NURSES   (information ferries)           LED vital hue, blinking
    Robots  7-15  : PATIENTS (3 per ring)                    LED vital hue

  ALGORITHM PATTERN: delta-disk graph consensus, v_i += eps * sum_j
    (v_j - v_i) over neighbors within 0.55 m (eps = 0.03 per 0.033 s
    step), gated ON only after the rings are spatially separated, plus
    formation control (triage rings) and periodic ferry routes (A<->B,
    B<->C, C<->A, and one roving nurse A->B->C). Ring centers A(-1.05,
    0.45), B(1.05, 0.45), C(0.00, -0.55) sit > 0.55 m apart at closest
    slot approach, so rings are true communication islands.

  REAL-ROBOT TIMING ASSUMPTIONS
    * 15 s standby head time before tasks begin (real fleet start delay).
    * Linear speed planned at 0.14 m/s (30% below the 0.20 m/s platform max).
    * Angular speed planned at 1.8 rad/s (50% below the 3.6 rad/s max).
    * Wheel-speed rescaling keeps every command inside actuator limits.

  STEP-BY-STEP TIMELINE (5000 iterations @ 0.033 s = 165 s wall-clock)
  ----------------------------------------------------------------------------
  STANDBY (0..15 s) - real-robot start delay buffer
    * t =  0-10 s - all 16 robots hold; LEDs breathe dim white at 0.5 Hz.
    * t = 10-15 s - LEDs ramp white -> vital-value colors (each robot
                    reveals its own initial reading: red=low, green=high).
  PHASE 1 - TRIAGE RING FORMATION (15..40 s)
    * t = 15 s  - patients leave the central waiting block for their ring
                  slots, TRIAGED BY SEVERITY: lowest 3 readings to ring A,
                  middle 3 to ring B, highest 3 to ring C (radius 0.36 m
                  around the doctors); nurses hold the arena-corner relay
                  posts. Vitals exchange is OFF during formation, so every
                  robot still shows its own initial hue while traveling.
    * t = 32 s  - rings essentially formed, three mixed-hue clusters.
    * t = 40 s  - vitals exchange goes live (printed ring means show the
                  pre-mix state).
  PHASE 2 - NURSE FERRY CONSENSUS (40..130 s)
    * t = 40-45 s - intra-ring mixing: each ring's LEDs blend to a shared
                  hue within seconds; three DIFFERENT hues persist
                  (consensus islands).
    * t = 40 s  - nurses simultaneously begin ferry routes between ring
                  contact points (0.48 m from ring centers), dwelling 4 s
                  per contact.
    * t ~ 50-60 s - first ferry contacts: ring means start dragging
                  toward each other in steps that coincide with nurse
                  arrivals (printed per contact).
    * t ~ 60-115 s - repeated ferry cycles; inter-ring spread decays
                  stepwise while intra-ring disagreement stays near zero.
    * t = 130 s - inter-ring spread expected < 0.10 (printed every 15 s).
  PHASE 3 - WARD MERGE (130..150 s)
    * t = 130 s - all 16 robots greedily claim slots on one 0.78 m ring
                  around the origin; chain connectivity completes the
                  final convergence.
    * t = 150 s - merged ring formed; global disagreement < 0.02.
  PHASE 4 - SYNC COMPLETE (150..165 s)
    * t = 150 s - LEDs pulse brightness in unison at 0.5 Hz, all robots
                  showing the SAME converged hue (vitals reconciled).
    * t = 165 s - run ends; debug/cleanup hooks called.
  ----------------------------------------------------------------------------

  EMERGENT BEHAVIORS DEMONSTRATED (printed as metrics during the run)
    1. Consensus islands: with only local communication, agreement forms
       per-ring but NOT globally - a red-ish critical ring, an amber ring,
       and a green ring, each internally uniform.
    2. Information ferrying: global convergence emerges from nurse
       mobility alone; disagreement decays in steps aligned with ferry
       contacts, not smoothly.
    3. Mixing-rate jump: the physical merge (denser graph) visibly
       accelerates convergence in the final phase.
    4. The converged value is an emergent weighted average no robot was
       given - it depends on ferry schedules and contact durations.

  LED CONFIGURATION (updates every iteration)
    ALL     : hue encodes the robot's CURRENT vital value - red (255,30,0)
              at 0.0 through amber (255,170,0) at 0.5 to green (90,255,90)
              at 1.0. Watching the hues equalize IS watching consensus.
    NURSES  : same hue but blinking at 1.5 Hz while traveling between
              rings (ferry in motion); solid while dwelling at a contact.
    DOCTORS : same hue with a white flash while a nurse is in contact
              range (acknowledging the exchange).
    ALL     : synchronized 0.5 Hz brightness pulse in the final phase.

  ROBOTARIUM SUBMISSION COMPLIANCE
    * 16 robots (max 20); 165 s run (max 600 s); min start spacing 0.36 m.
    * get_poses() exactly once per step(); ends with the platform hook.
    * Barrier certificates keep inter-robot distance >= 0.20 m and robots
      inside [-1.6, 1.6] x [-1.0, 1.0]. Arena/figure forced white.
    * Imports: numpy + rps only.
    * RNPS_FAST_SIM=1 / RNPS_MAX_ITERS are local pre-flight knobs only.
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


def resolve_rps(module, *candidate_names):
    for name in candidate_names:
        fn = getattr(module, name, None)
        if fn is not None:
            return fn
    raise RuntimeError(f"None of {candidate_names!r} found in {module.__name__}.")


# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
RUN_SEED = 1106
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)

NUM_DOCTORS = 3
NUM_NURSES = 4
NUM_PATIENTS = 9
N = NUM_DOCTORS + NUM_NURSES + NUM_PATIENTS  # 16
DOC = np.arange(0, NUM_DOCTORS)
NUR = np.arange(NUM_DOCTORS, NUM_DOCTORS + NUM_NURSES)
PAT = np.arange(NUM_DOCTORS + NUM_NURSES, N)

HW_MAX_LINEAR = 0.20
ROBOT_BASE_LENGTH = 0.11
PLAN_LINEAR = 0.14
PLAN_ANGULAR = 1.8

DT = 0.033
TOTAL_SECONDS = 165.0


def sec(seconds):
    return int(round(seconds / DT))


TOTAL_ITERATIONS = sec(TOTAL_SECONDS)
T_STANDBY = sec(15.0)
T_FERRY = sec(40.0)
T_MERGE = sec(130.0)
T_SYNC = sec(150.0)

# Consensus parameters
COMM_RADIUS = 0.55  # m, delta-disk communication range
CONSENSUS_EPS = 0.03  # per-step mixing gain (eps * max_degree < 1)
FERRY_DWELL = 4.0  # s at each ring contact

SAFETY_RADIUS = 0.20
ARENA = np.array([-1.6, 1.6, -1.0, 1.0])
MARGIN = 0.15

# ----------------------------------------------------------------------------
# WARD GEOMETRY (min pairwise start spacing 0.36 m)
# ----------------------------------------------------------------------------
RING_CENTERS = np.array([[-1.05, 1.05, 0.00], [0.45, 0.45, -0.55]])
RING_RADIUS = 0.36
SLOT_ANGLES = np.deg2rad([90.0, 210.0, 330.0])

# Patient initial vitals, drawn first so triage can sort on them: the 3 lowest
# readings go to ring A, middle 3 to ring B, highest 3 to ring C. This maximizes
# inter-ring contrast, so the consensus-island phase is unmistakable on LEDs.
PATIENT_VITALS_0 = np.random.uniform(0.10, 0.90, NUM_PATIENTS)
_severity_order = np.argsort(PATIENT_VITALS_0)
RING_OF_PATIENT = np.empty(NUM_PATIENTS, dtype=int)
RING_OF_PATIENT[_severity_order] = np.repeat(np.arange(3), 3)

ring_slots = np.zeros((2, NUM_PATIENTS))
for ring in range(3):
    members = _severity_order[3 * ring : 3 * ring + 3]
    for j, p in enumerate(members):
        c = RING_CENTERS[:, ring]
        a = SLOT_ANGLES[j]
        ring_slots[:, p] = c + RING_RADIUS * np.array([np.cos(a), np.sin(a)])

waiting_block = np.array(
    [
        [-0.45, 0.00, 0.45, -0.45, 0.00, 0.45, -0.45, 0.00, 0.45],
        [0.56, 0.56, 0.56, 0.20, 0.20, 0.20, -0.16, -0.16, -0.16],
    ]
)
nurse_posts = np.array([[-1.40, 1.40, -1.40, 1.40], [0.85, 0.85, -0.85, -0.85]])

initial_conditions = np.vstack(
    [
        np.hstack([RING_CENTERS[0], nurse_posts[0], waiting_block[0]]),
        np.hstack([RING_CENTERS[1], nurse_posts[1], waiting_block[1]]),
        np.zeros(N),
    ]
)
_d = np.linalg.norm(
    initial_conditions[:2, :, None] - initial_conditions[:2, None, :], axis=0
) + np.eye(N)
assert _d.min() >= 0.35, f"initial spacing {_d.min():.3f} m violates the 0.35 m rule"

# Ferry routes: nurse k cycles through these ring indices.
FERRY_ROUTES = [[0, 1], [1, 2], [2, 0], [0, 1, 2]]


def contact_point(ring_idx, toward_ring):
    """Nurse dwell point: 0.55 m from the ring center toward the next ring."""
    c = RING_CENTERS[:, ring_idx]
    d = RING_CENTERS[:, toward_ring] - c
    d = d / max(np.linalg.norm(d), 1e-6)
    return c + 0.48 * d


# ----------------------------------------------------------------------------
# VITAL STATES (the consensus variables)
# ----------------------------------------------------------------------------
vitals = np.zeros(N)
vitals[DOC] = 0.80  # staff calibrated readings
vitals[NUR] = 0.80
vitals[PAT] = PATIENT_VITALS_0


def consensus_step(pos):
    """One delta-disk consensus update on the vitals vector."""
    global vitals
    diff = pos[:, :, None] - pos[:, None, :]
    adj = (np.linalg.norm(diff, axis=0) < COMM_RADIUS) & ~np.eye(N, dtype=bool)
    lap_update = adj @ vitals - adj.sum(axis=1) * vitals
    vitals = vitals + CONSENSUS_EPS * lap_update


def ring_means():
    out = []
    for ring in range(3):
        members = np.concatenate([[DOC[ring]], PAT[RING_OF_PATIENT == ring]])
        out.append(float(vitals[members].mean()))
    return out


# ----------------------------------------------------------------------------
# ROBOTARIUM INIT + SHIMS (white arena, LED writer, safety stack)
# ----------------------------------------------------------------------------
_FAST = os.environ.get("RNPS_FAST_SIM", "0") == "1"
_ITER_CAP = int(os.environ.get("RNPS_MAX_ITERS", str(TOTAL_ITERATIONS)))

r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=not _FAST,
    initial_conditions=initial_conditions,
    sim_in_real_time=not _FAST,
)
_fig = getattr(r, "_fig", None) or getattr(r, "figure", None)
_axes = getattr(r, "_axes_handle", None) or getattr(r, "axes", None)
if _fig is not None:
    _fig.patch.set_facecolor("white")
if _axes is not None:
    _axes.set_facecolor("white")

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


LED_LOW = np.array([255.0, 30.0, 0.0])
LED_MID = np.array([255.0, 170.0, 0.0])
LED_HIGH = np.array([90.0, 255.0, 90.0])
LED_FLASH = np.array([255.0, 255.0, 255.0])


def vital_color(values):
    v = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    out = np.zeros((3, v.size))
    lo = v < 0.5
    bl = v[lo] / 0.5
    out[:, lo] = LED_LOW[:, None] * (1.0 - bl) + LED_MID[:, None] * bl
    hi = ~lo
    bh = (v[hi] - 0.5) / 0.5
    out[:, hi] = LED_MID[:, None] * (1.0 - bh) + LED_HIGH[:, None] * bh
    return out


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
    """Clamp [v; w] to derated caps, then rescale pairs to honor wheel limits."""
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


def greedy_slot_match(points, slots):
    """Greedy nearest matching of K points onto K slots (returns slot order)."""
    k = points.shape[1]
    cost = np.linalg.norm(points[:, :, None] - slots[:, None, :], axis=0)
    assign = np.full(k, -1)
    for _ in range(k):
        flat = int(np.argmin(cost))
        i, s = divmod(flat, k)
        assign[i] = s
        cost[i, :] = np.inf
        cost[:, s] = np.inf
    return assign


# ----------------------------------------------------------------------------
# FERRY FSM (per nurse): TRAVEL -> DWELL -> next leg
# ----------------------------------------------------------------------------
ferry_leg = [0] * NUM_NURSES  # index into the route
ferry_state = ["TRAVEL"] * NUM_NURSES
ferry_timer = [0.0] * NUM_NURSES
ferry_contacts = 0

merge_assign = None
merge_slots = None

# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
print(f"Run06 ConsensusVitals: {N} robots, {TOTAL_SECONDS:.0f} s, seed {RUN_SEED}")
print(f"  initial vitals: patients {np.round(vitals[PAT], 2).tolist()}, staff 0.8")
status_marks = {sec(s) for s in (40.0, 55.0, 70.0, 85.0, 100.0, 115.0, 130.0, 150.0, 163.0)}

for t in range(min(TOTAL_ITERATIONS, _ITER_CAP)):
    x = r.get_poses()
    pos = x[:2, :]
    t_sec = t * DT
    dxi = np.zeros((2, N))

    if t >= T_FERRY:
        # Vitals exchange goes live only once the rings are separated; mixing
        # during formation (everyone crossing the center) would erase the
        # ring-local consensus phase the experiment is built to show.
        consensus_step(pos)

    base_colors = vital_color(vitals)
    leds = base_colors.copy()

    if t < T_STANDBY:
        breathe = 0.25 + 0.55 * 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.5 * t_sec))
        white = np.full((3, N), 255.0) * breathe
        leds = (
            white
            if t_sec < 10.0
            else (1.0 - (t_sec - 10.0) / 5.0) * white + ((t_sec - 10.0) / 5.0) * base_colors
        )
    else:
        if t < T_MERGE:
            # Doctors hold ring centers; patients hold/seek ring slots.
            dxi[:, DOC] = si_position(pos[:, DOC], RING_CENTERS)
            dxi[:, PAT] = si_position(pos[:, PAT], ring_slots)
            if t < T_FERRY:
                # PHASE 1: nurses hold relay posts while rings form.
                dxi[:, NUR] = si_position(pos[:, NUR], nurse_posts)
            else:
                # PHASE 2: ferry routes between ring contact points.
                for k in range(NUM_NURSES):
                    route = FERRY_ROUTES[k]
                    here = route[ferry_leg[k] % len(route)]
                    nxt = route[(ferry_leg[k] + 1) % len(route)]
                    target = contact_point(here, nxt)
                    n_pos = pos[:, NUR[k]]
                    if ferry_state[k] == "TRAVEL":
                        if np.linalg.norm(n_pos - target) < 0.12:
                            ferry_state[k] = "DWELL"
                            ferry_timer[k] = t_sec
                            ferry_contacts += 1
                            means = " ".join(f"{m:.2f}" for m in ring_means())
                            print(
                                f"[t={t_sec:5.1f}s] ferry contact #{ferry_contacts} "
                                f"(nurse {k} at ring {here}); ring means [{means}]"
                            )
                    else:  # DWELL
                        if t_sec - ferry_timer[k] >= FERRY_DWELL:
                            ferry_state[k] = "TRAVEL"
                            ferry_leg[k] += 1
                    dxi[:, NUR[k] : NUR[k] + 1] = si_position(
                        n_pos.reshape(2, 1), target.reshape(2, 1)
                    )
                    if ferry_state[k] == "TRAVEL" and int(t_sec * 3) % 2 == 0:
                        leds[:, NUR[k]] *= 0.25  # ferry-in-motion blink
                # Doctor white flash while a nurse is in contact range.
                for ring in range(3):
                    c = RING_CENTERS[:, ring]
                    if any(np.linalg.norm(pos[:, NUR[k]] - c) < 0.60 for k in range(NUM_NURSES)):
                        leds[:, DOC[ring]] = LED_FLASH
        else:
            # PHASES 3-4: merge everyone onto one 16-slot ring.
            if merge_assign is None:
                ang = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
                merge_slots = np.vstack([0.78 * np.cos(ang), 0.78 * np.sin(ang)])
                merge_assign = greedy_slot_match(pos, merge_slots)
                print(f"[t={t_sec:5.1f}s] ward merge begins (16-slot ring)")
            dxi[:, :] = si_position(pos, merge_slots[:, merge_assign])
            if t >= T_SYNC:
                pulse = 0.55 + 0.45 * np.sin(2.0 * np.pi * 0.5 * t_sec)
                leds = leds * pulse

        if t in status_marks:
            means = " ".join(f"{m:.2f}" for m in ring_means())
            spread = max(ring_means()) - min(ring_means())
            disagreement = float(np.abs(vitals - vitals.mean()).max())
            print(
                f"[t={t_sec:5.1f}s] ring means [{means}] spread={spread:.3f} "
                f"global disagreement={disagreement:.3f}"
            )

    dxi = si_barrier(dxi, x)
    dxu = si_to_uni(dxi, x)
    stopped = np.linalg.norm(dxi, axis=0) < 1e-4
    if np.any(stopped):
        dxu[:, stopped] = 0.0
    dxu = wheel_safe(dxu)

    write_leds(leds)
    r.set_velocities(_ALL_IDS, dxu)
    r.step()

final_disagreement = float(np.abs(vitals - vitals.mean()).max())
print(
    f"Run06 complete: {ferry_contacts} ferry contacts, converged vital "
    f"{vitals.mean():.3f}, final disagreement {final_disagreement:.4f}"
)

_end_hook = getattr(r, "call_at_scripts_end", None)
if callable(_end_hook):
    _end_hook()
_debug_hook = getattr(r, "debug", None)
if callable(_debug_hook):
    _debug_hook()
