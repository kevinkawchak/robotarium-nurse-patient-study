"""
================================================================================
  RUN 08 / 10 - ARTIFICIAL POTENTIAL FIELD ISOLATION WARD
  Georgia Tech Robotarium - https://www.robotarium.gatech.edu/experiment
================================================================================

  CLINICAL TRIAL OBJECTIVE
    Infection-control study: 6 contagious patient robots must reach an
    isolation cell block while NEVER crowding each other - they obey a
    social-distancing repulsion field at all times. No patient is ever
    assigned a cell: cell attraction + mutual repulsion sort them into
    6 distinct cells on their own. 3 nurse robots sweep a 3-lane
    disinfection corridor that the patients must cross, and 3 doctor
    robots run a staggered inspection circuit once the ward settles.

  FLEET (12 robots, 3 min run)
    Robots  0-2   : DOCTORS  (inspection circuit)            LED blue/cyan
    Robots  3-5   : NURSES   (patrol laps)                   LED green/white
    Robots  6-11  : PATIENTS (distancing + cell capture)     LED red/amber/green

  ALGORITHM PATTERN: artificial potential fields. Patients feel
    (a) inverse-square repulsion from every other patient inside 0.55 m
    (distancing), (b) attraction to the NEAREST UNCROWDED cell (a cell
    is crowded if another patient is closer to it), and (c) wall
    repulsion. Nurses sweep vertical corridor lanes (x = -0.10, 0.12,
    0.34) via waypoint toggling; doctors run a cell-to-cell inspection
    tour with 5.5 s dwells, approaching from the inter-row midline.

  REAL-ROBOT TIMING ASSUMPTIONS
    * 15 s standby head time before tasks begin (real fleet start delay).
    * Linear speed planned at 0.14 m/s (30% below the 0.20 m/s platform max).
    * Angular speed planned at 1.8 rad/s (50% below the 3.6 rad/s max).
    * Wheel-speed rescaling keeps every command inside actuator limits.

  STEP-BY-STEP TIMELINE (5455 iterations @ 0.033 s = 180 s wall-clock)
  ----------------------------------------------------------------------------
  STANDBY (0..15 s) - real-robot start delay buffer
    * t =  0-10 s - all 12 robots hold; LEDs breathe dim white at 0.5 Hz.
    * t = 10-15 s - LEDs ramp white -> role colors (role announce).
  PHASE 1 - DISTANCING DISPERSAL (15..40 s)
    * t = 15 s  - patients (clustered 2x3 in the left waiting area) turn
                  on mutual repulsion and spread out; LEDs run red while
                  any neighbor is inside 0.45 m, amber once spaced.
    * t = 28 s  - patient lattice forms (min pairwise distance printed);
                  nurses sweep the corridor lanes continuously.
    * t = 40 s  - distancing satisfied (expected min distance > 0.45 m).
  PHASE 2 - EMERGENT CELL CAPTURE (40..100 s)
    * t = 40 s  - cell attraction switches on toward the 3x2 isolation
                  block (right half). Patients funnel right while the
                  repulsion field keeps them spaced - expect single-file
                  streams and yield maneuvers around the patrol lane.
    * t ~ 50-90 s - patients thread between the sweeping nurses (yield
                  maneuvers); cells fill one by one (each capture
                  printed); closer patients win contested cells, losers
                  re-target the next free cell (conflict resolution).
    * t = 100 s - all 6 patients in 6 distinct cells (occupancy printed);
                  cell LEDs green.
  PHASE 3 - DOCTOR INSPECTION CIRCUIT (100..165 s)
    * t = 100 s - doctor 0 departs for the top-left cell pair (cells 0,
                  1), approaching from the aisle above the block; 5.5 s
                  dwell at a 0.30 m standoff per cell (LED cyan).
    * t = 106 s - doctor 2 departs to sweep the entire bottom row right
                  to left (cells 5, 4, 3) from the aisle below.
    * t = 124 s - doctor 1 takes the high aisle (via 0.55, 0.88) to the
                  far top cell (cell 2) after doctor 0 has cleared the
                  lane. A 30 s per-stage timeout skips a blocked stage
                  rather than shoving through the barrier field.
    * t ~ 135-160 s - all 6 cells inspected; doctors return to posts;
                  nurses sweep the corridor throughout (half-lap counts
                  printed).
  PHASE 4 - WARD SEALED (165..180 s)
    * t = 165 s - doctors at posts, nurses park at their lane ends;
                  patients hold cells.
    * t = 170 s - LED seal sweep: synchronized green pulse front moves
                  across the ward (by x position).
    * t = 180 s - run ends; debug/cleanup hooks called.
  ----------------------------------------------------------------------------

  EMERGENT BEHAVIORS DEMONSTRATED (printed as metrics during the run)
    1. Self-organized allocation: 6 patients claim 6 distinct cells with
       no assignment, no auction, no communication - only fields.
    2. Conflict resolution: contested cells resolve by proximity; losing
       patients smoothly re-target (re-target events printed).
    3. Social distancing under transport: the repulsion field keeps
       patients predominantly outside the 0.45 m distancing radius while
       the cohort migrates (transient funneling dips are flagged by red
       LEDs; the barrier certificate enforces a hard 0.20 m floor).
    4. Corridor etiquette: patients crossing the 3-lane disinfection
       corridor yield to sweeping nurses (repulsion term), producing
       stop-and-go crossing waves.
    5. Lattice formation: phase 1 produces a near-uniform triangular
       lattice from pure repulsion in a bounded region.

  LED CONFIGURATION (updates every iteration)
    DOCTORS : blue (0,90,255) at posts/in transit; cyan (0,220,255)
              while dwelling at an inspected cell.
    NURSES  : green (0,200,80) / pale-green (220,255,220) alternation
              every completed corridor lap (lap parity visible).
    PATIENTS: red (255,30,0) whenever another patient is inside 0.45 m
              (distancing violation pressure); amber (255,150,0) spaced
              but uncelled; green (90,255,90) settled in a cell.

  ROBOTARIUM SUBMISSION COMPLIANCE
    * 12 robots (max 20); 180 s run (max 600 s); min start spacing 0.36 m.
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
RUN_SEED = 1108
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)

NUM_DOCTORS = 3
NUM_NURSES = 3
NUM_PATIENTS = 6
N = NUM_DOCTORS + NUM_NURSES + NUM_PATIENTS  # 12
DOC = np.arange(0, NUM_DOCTORS)
NUR = np.arange(NUM_DOCTORS, NUM_DOCTORS + NUM_NURSES)
PAT = np.arange(NUM_DOCTORS + NUM_NURSES, N)

HW_MAX_LINEAR = 0.20
ROBOT_BASE_LENGTH = 0.11
PLAN_LINEAR = 0.14
PLAN_ANGULAR = 1.8

DT = 0.033
TOTAL_SECONDS = 180.0


def sec(seconds):
    return int(round(seconds / DT))


TOTAL_ITERATIONS = sec(TOTAL_SECONDS)
T_STANDBY = sec(15.0)
T_CAPTURE = sec(40.0)
T_INSPECT = sec(100.0)
T_SEAL = sec(165.0)

# Potential-field parameters
DISTANCING_R = 0.55  # m, patient-patient repulsion range
DISTANCING_ALERT = 0.45  # m, LED red below this separation
REPULSE_GAIN = 0.020
NURSE_REPULSE_R = 0.45  # patients yield to the patrol lane
NURSE_REPULSE_GAIN = 0.012
ATTRACT_GAIN = 0.50
WALL_R = 0.25
WALL_GAIN = 0.008
CELL_CAPTURE_R = 0.16  # settled when inside this radius of a cell
INSPECT_DWELL = 5.5  # s per inspected cell

SAFETY_RADIUS = 0.20
ARENA = np.array([-1.6, 1.6, -1.0, 1.0])
MARGIN = 0.15

# ----------------------------------------------------------------------------
# WARD GEOMETRY (min pairwise start spacing 0.36 m)
# Isolation block: 3x2 cells in the right half; patrol lane is a rectangle
# around the block; patients start clustered in the left waiting area.
# ----------------------------------------------------------------------------
CELLS = np.array(
    [
        [0.55, 1.00, 1.45, 0.55, 1.00, 1.45],
        [0.38, 0.38, 0.38, -0.38, -0.38, -0.38],
    ]
)
# Disinfection corridor: 3 vertical patrol lanes between the waiting area and
# the cell block. Patients MUST cross the corridor to reach their cells.
LANE_X = np.array([-0.10, 0.12, 0.34])
LANE_TOP = 0.70
LANE_BOT = -0.70
DOCTOR_POSTS = np.array([[-1.40, -1.40, -1.40], [0.55, 0.00, -0.55]])
patient_start = np.array(
    [
        [-1.00, -0.62, -1.00, -0.62, -1.00, -0.62],
        [0.55, 0.55, 0.00, 0.00, -0.55, -0.55],
    ]
)
nurse_start = np.array([[LANE_X[0], LANE_X[1], LANE_X[2]], [LANE_TOP, LANE_BOT, 0.0]])

initial_conditions = np.vstack(
    [
        np.hstack([DOCTOR_POSTS[0], nurse_start[0], patient_start[0]]),
        np.hstack([DOCTOR_POSTS[1], nurse_start[1], patient_start[1]]),
        np.zeros(N),
    ]
)
_d = np.linalg.norm(
    initial_conditions[:2, :, None] - initial_conditions[:2, None, :], axis=0
) + np.eye(N)
assert _d.min() >= 0.35, f"initial spacing {_d.min():.3f} m violates the 0.35 m rule"

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


LED_DOCTOR = np.array([0.0, 90.0, 255.0])
LED_DOC_INSPECT = np.array([0.0, 220.0, 255.0])
LED_NURSE_A = np.array([0.0, 200.0, 80.0])
LED_NURSE_B = np.array([220.0, 255.0, 220.0])
LED_PAT_CLOSE = np.array([255.0, 30.0, 0.0])
LED_PAT_SPACED = np.array([255.0, 150.0, 0.0])
LED_PAT_CELL = np.array([90.0, 255.0, 90.0])
ROLE_COLORS = np.zeros((3, N))
ROLE_COLORS[:, DOC] = LED_DOCTOR[:, None]
ROLE_COLORS[:, NUR] = LED_NURSE_A[:, None]
ROLE_COLORS[:, PAT] = LED_PAT_SPACED[:, None]

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


def cap_speed(v, limit):
    mag = np.linalg.norm(v, axis=0)
    over = mag > limit
    if np.any(over):
        v = np.copy(v)
        v[:, over] *= limit / mag[over]
    return v


# ----------------------------------------------------------------------------
# POTENTIAL FIELDS
# ----------------------------------------------------------------------------
def patient_fields(p_pat, p_nur, attract_on, settled, cell_target):
    """Distancing repulsion + nurse-lane repulsion + wall + cell attraction."""
    dxi = np.zeros((2, NUM_PATIENTS))
    for i in range(NUM_PATIENTS):
        f = np.zeros(2)
        rel = p_pat[:, i : i + 1] - p_pat
        d = np.linalg.norm(rel, axis=0)
        mask = (d > 1e-6) & (d < DISTANCING_R)
        if np.any(mask):
            f += REPULSE_GAIN * (rel[:, mask] / (d[mask] ** 2)).sum(axis=1)
        if not settled[i]:  # settled patients are isolated; the lane can pass by
            rel_n = p_pat[:, i : i + 1] - p_nur
            d_n = np.linalg.norm(rel_n, axis=0)
            mask_n = d_n < NURSE_REPULSE_R
            if np.any(mask_n):
                f += NURSE_REPULSE_GAIN * (rel_n[:, mask_n] / (d_n[mask_n] ** 2)).sum(axis=1)
        # Wall repulsion (keeps the lattice off the boundary barrier).
        f[0] += WALL_GAIN / max(p_pat[0, i] - ARENA[0], WALL_R) ** 2
        f[0] -= WALL_GAIN / max(ARENA[1] - p_pat[0, i], WALL_R) ** 2
        f[1] += WALL_GAIN / max(p_pat[1, i] - ARENA[2], WALL_R) ** 2
        f[1] -= WALL_GAIN / max(ARENA[3] - p_pat[1, i], WALL_R) ** 2
        if attract_on and cell_target[i] >= 0:
            to_cell = CELLS[:, cell_target[i]] - p_pat[:, i]
            gain = ATTRACT_GAIN if not settled[i] else 1.0
            f += gain * to_cell
        dxi[:, i] = f
    return cap_speed(dxi, 0.12)


def retarget_cells(p_pat, settled, cell_target):
    """Each unsettled patient targets the nearest cell it is closest to.

    A settled patient sits on its cell and is therefore the closest robot to
    it, which removes that cell from contention automatically. If patient i
    is not the closest to ANY free cell (it lost every contest), it heads for
    the nearest free cell anyway; the winner settles first and i re-targets
    on a later call - proximity alone resolves all conflicts.
    """
    events = 0
    d = np.linalg.norm(p_pat[:, :, None] - CELLS[:, None, :], axis=0)  # (P, C)
    closest_patient = d.argmin(axis=0)  # per cell
    occupied = {cell_target[j] for j in range(NUM_PATIENTS) if settled[j]}
    free = [c for c in range(6) if c not in occupied]
    for i in range(NUM_PATIENTS):
        if settled[i] or not free:
            continue
        mine = [c for c in free if closest_patient[c] == i]
        pool = mine if mine else free
        pick = min(pool, key=lambda c: d[i, c])
        if pick != cell_target[i]:
            if cell_target[i] >= 0:
                events += 1
            cell_target[i] = pick
    return events


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
print(f"Run08 PotentialIsolation: {N} robots, {TOTAL_SECONDS:.0f} s, seed {RUN_SEED}")

nurse_heading_up = [False, True, True]  # initial lane directions (staggered)
nurse_laps = [0, 0, 0]
settled = [False] * NUM_PATIENTS
cell_target = [-1] * NUM_PATIENTS
retarget_events = 0
captures = 0
min_dist_log = []
# Inspection tours, partitioned so no doctor crosses another's aisle at
# derated speed: doctor 0 takes the top-left pair, doctor 2 sweeps the whole
# bottom row right-to-left, doctor 1 reaches the far top cell via a high
# aisle 'via' waypoint after doctor 0 has cleared the top lane.
doc_tour = [
    [("dwell", 0), ("dwell", 1)],
    [("via", (0.55, 0.88)), ("dwell", 2)],
    [("dwell", 5), ("dwell", 4), ("dwell", 3)],
]
DOC_START = [100.0, 124.0, 106.0]  # s, per-doctor circuit start times
doc_stage = [0] * NUM_DOCTORS  # index into its tour
doc_dwell_t = [-1.0] * NUM_DOCTORS
doc_stage_start = [-1.0] * NUM_DOCTORS  # approach timeout bookkeeping
APPROACH_TIMEOUT = 30.0  # s; covers worst-case derated travel before skipping
status_marks = {sec(s) for s in (28.0, 40.0, 70.0, 100.0, 130.0, 160.0)}

for t in range(min(TOTAL_ITERATIONS, _ITER_CAP)):
    x = r.get_poses()
    pos = x[:2, :]
    t_sec = t * DT
    dxi = np.zeros((2, N))
    leds = ROLE_COLORS.copy()

    if t < T_STANDBY:
        breathe = 0.25 + 0.55 * 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.5 * t_sec))
        white = np.full((3, N), 255.0) * breathe
        leds = (
            white
            if t_sec < 10.0
            else (1.0 - (t_sec - 10.0) / 5.0) * white + ((t_sec - 10.0) / 5.0) * ROLE_COLORS
        )
    else:
        p_pat = pos[:, PAT]
        p_nur = pos[:, NUR]
        attract_on = t >= T_CAPTURE

        # Cell retargeting (every 0.5 s) + capture detection.
        if attract_on and t % sec(0.5) == 0:
            retarget_events += retarget_cells(p_pat, settled, cell_target)
        for i in range(NUM_PATIENTS):
            if attract_on and not settled[i] and cell_target[i] >= 0:
                if np.linalg.norm(p_pat[:, i] - CELLS[:, cell_target[i]]) < CELL_CAPTURE_R:
                    settled[i] = True
                    captures += 1
                    print(
                        f"[t={t_sec:5.1f}s] patient {i} captured cell {cell_target[i]} "
                        f"({captures}/6)"
                    )

        dxi[:, PAT] = patient_fields(p_pat, p_nur, attract_on, settled, cell_target)

        # Distancing telemetry.
        dpp = np.linalg.norm(p_pat[:, :, None] - p_pat[:, None, :], axis=0)
        dpp += np.eye(NUM_PATIENTS) * 1e9
        min_pair = float(dpp.min())
        min_dist_log.append(min_pair)

        # ---------------- nurses: vertical disinfection-lane laps ----------
        for k in range(NUM_NURSES):
            n_idx = NUR[k]
            if t >= T_SEAL:
                target = np.array([LANE_X[k], [LANE_TOP, LANE_BOT, LANE_TOP][k]])
            else:
                end_y = LANE_TOP if nurse_heading_up[k] else LANE_BOT
                target = np.array([LANE_X[k], end_y])
                if abs(pos[1, n_idx] - end_y) < 0.08 and abs(pos[0, n_idx] - LANE_X[k]) < 0.12:
                    nurse_heading_up[k] = not nurse_heading_up[k]
                    nurse_laps[k] += 1  # half-lap completed
            dxi[:, n_idx : n_idx + 1] = si_position(
                pos[:, n_idx].reshape(2, 1), target.reshape(2, 1)
            )
            parity = (nurse_laps[k] // 2) % 2 == 0
            leds[:, n_idx] = LED_NURSE_A if parity else LED_NURSE_B

        # ---------------- doctors: posts, then inspection circuit ----------
        # Doctors approach top-row cells from above and bottom-row cells from
        # below: the outer aisles keep dwell points clear of settled patients,
        # the corridor lanes, and (with the start stagger) each other.
        for dd in range(NUM_DOCTORS):
            d_idx = DOC[dd]
            target = DOCTOR_POSTS[:, dd]
            circuit_open = t_sec >= DOC_START[dd] and t < T_SEAL
            if circuit_open and doc_stage[dd] < len(doc_tour[dd]):
                kind, ref = doc_tour[dd][doc_stage[dd]]
                if kind == "via":
                    approach = np.array(ref)
                else:
                    cell = ref
                    side = 1.0 if CELLS[1, cell] > 0 else -1.0
                    approach = CELLS[:, cell] + np.array([0.0, side * 0.30])
                target = approach
                if doc_stage_start[dd] < 0.0:
                    doc_stage_start[dd] = t_sec
                arrive_r = 0.15 if kind == "via" else 0.10
                if np.linalg.norm(pos[:, d_idx] - approach) < arrive_r:
                    if kind == "via":
                        doc_stage[dd] += 1
                        doc_stage_start[dd] = -1.0
                    elif doc_dwell_t[dd] < 0.0:
                        doc_dwell_t[dd] = t_sec
                    elif t_sec - doc_dwell_t[dd] >= INSPECT_DWELL:
                        print(f"[t={t_sec:5.1f}s] doctor {dd} inspected cell {cell}")
                        doc_stage[dd] += 1
                        doc_dwell_t[dd] = -1.0
                        doc_stage_start[dd] = -1.0
                    if kind == "dwell" and doc_dwell_t[dd] >= 0.0:
                        leds[:, d_idx] = LED_DOC_INSPECT
                elif doc_dwell_t[dd] < 0.0 and t_sec - doc_stage_start[dd] > APPROACH_TIMEOUT:
                    print(f"[t={t_sec:5.1f}s] doctor {dd} skipped blocked stage {doc_stage[dd]}")
                    doc_stage[dd] += 1
                    doc_stage_start[dd] = -1.0
            dxi[:, d_idx : d_idx + 1] = si_position(
                pos[:, d_idx].reshape(2, 1), clamp_arena(target.reshape(2, 1))
            )

        # ---------------- patient LEDs ----------------
        for i in range(NUM_PATIENTS):
            if settled[i]:
                leds[:, PAT[i]] = LED_PAT_CELL
            elif dpp[i].min() < DISTANCING_ALERT:
                leds[:, PAT[i]] = LED_PAT_CLOSE
            else:
                leds[:, PAT[i]] = LED_PAT_SPACED

        if t >= sec(170.0):
            front = np.sin(2.0 * np.pi * 0.4 * t_sec - pos[0, :] * 2.0)
            leds = leds * (0.55 + 0.45 * 0.5 * (1.0 + front))

        if t in status_marks:
            print(
                f"[t={t_sec:5.1f}s] min patient spacing={min_pair:.2f} m "
                f"captures={captures}/6 retargets={retarget_events} laps={nurse_laps}"
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

occupied = sorted(cell_target[i] for i in range(NUM_PATIENTS) if settled[i])
worst_dist = min(min_dist_log) if min_dist_log else float("nan")
print(
    f"Run08 complete: {captures}/6 cells captured {occupied}, "
    f"{retarget_events} retargets, min spacing ever={worst_dist:.2f} m, "
    f"patrol laps={nurse_laps}"
)

_end_hook = getattr(r, "call_at_scripts_end", None)
if callable(_end_hook):
    _end_hook()
_debug_hook = getattr(r, "debug", None)
if callable(_debug_hook):
    _debug_hook()
