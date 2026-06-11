"""
================================================================================
  RUN 10 / 10 - LEADER-FOLLOWER DISCHARGE CONVOYS
  Georgia Tech Robotarium - https://www.robotarium.gatech.edu/experiment
================================================================================

  CLINICAL TRIAL OBJECTIVE
    End-of-trial discharge logistics: 6 recovered patient robots must be
    moved from the ward bays (right wall) to the discharge bay (left
    wall) through a serpentine corridor with two switchbacks. Each of
    the 3 nurse robots leads a 2-patient chain (leader-follower
    control); convoys depart at 35 s staggers. 1 doctor robot acts as
    route marshal: it pre-runs the corridor ahead of the first convoy,
    then supervises the discharge gate and performs the final ward
    sweep. String stability of the chains is measured live.

  FLEET (10 robots, 3.5 min run)
    Robot   0     : DOCTOR   (route marshal / gate)          LED blue
    Robots  1-3   : NURSES   (convoy leaders A, B, C)        LED green, synced blink
    Robots  4-9   : PATIENTS (2 followers per convoy)        LED amber, synced blink

  ALGORITHM PATTERN: leader-follower chain control on a waypoint route.
    Leader: waypoint switching (advance within 0.15 m). Follower i
    holds a 0.32 m standoff from its predecessor (nurse <- P1 <- P2),
    which propagates - and visibly damps - spacing disturbances down
    the chain at every switchback (string stability). Final docking is
    sequenced self-seating (followers release one at a time).

  REAL-ROBOT TIMING ASSUMPTIONS
    * 15 s standby head time before tasks begin (real fleet start delay).
    * Linear speed planned at 0.14 m/s (30% below the 0.20 m/s platform max).
    * Angular speed planned at 1.8 rad/s (50% below the 3.6 rad/s max).
    * Wheel-speed rescaling keeps every command inside actuator limits.

  ROUTE (serpentine, ~4.5 m): bays -> entrance (0.75, 0.62) ->
    (0.10, 0.62) -> (0.10, 0.00) -> (0.70, 0.00) -> (0.70, -0.62) ->
    (-0.20, -0.62) -> row entry (-0.75, y_row) -> slots at x = -1.20.
    Convoys B and C first merge up the bay-front lane to the entrance.

  STEP-BY-STEP TIMELINE (6364 iterations @ 0.033 s = 210 s wall-clock)
  ----------------------------------------------------------------------------
  STANDBY (0..15 s) - real-robot start delay buffer
    * t =  0-10 s - all 10 robots hold; LEDs breathe dim white at 0.5 Hz.
    * t = 10-15 s - LEDs ramp white -> role colors (role announce).
  PHASE 1 - HOOKUP + MARSHAL SWEEP (15..40 s)
    * t = 15 s  - nurses advance to the hook points in front of their
                  bay rows; as each arrives its convoy's synchronized
                  blink signature starts (convoys A/B/C blink at phase
                  offsets 0 / 120 / 240 degrees).
    * t = 20 s  - the doctor departs to pre-run the full serpentine
                  (route marshal), reaching the gate post by ~60 s.
  PHASE 2 - STAGGERED CONVOY RUNS (40..165 s)
    * t = 40 s  - CONVOY A departs (top row): nurse leads, P1 trails at
                  0.32 m, P2 trails P1; watch the chain stretch on the
                  straights and compress at the two switchbacks (gap
                  telemetry printed every 15 s).
    * t = 75 s  - CONVOY B departs (middle row) while A is mid-route:
                  two chains share the corridor, staggered one lane apart.
    * t ~ 85-100 s - A reaches its row entry and docks: P1 self-seats at
                  the inner slot, then P2 (sequenced release); nurse
                  parks at the row rest post; doctor flashes white per
                  seated patient.
    * t = 110 s - CONVOY C departs (bottom row).
    * t ~ 120-135 s - B docks; t ~ 150-175 s - C docks (its final
                  seating may overlap the start of the ward sweep).
                  Arrival prints include per-chain gap statistics.
  PHASE 3 - FINAL WARD SWEEP (165..195 s)
    * t = 165 s - doctor sweeps the discharge column (x = -0.90) from
                  top to bottom, white flash at each row (final check),
                  then returns to the gate post.
  PHASE 4 - DISCHARGE COMPLETE (195..210 s)
    * t = 195 s - LED ceremony: green wave sweeps the fleet by index;
                  all 6 patients hold discharge slots (solid green).
    * t = 210 s - run ends; debug/cleanup hooks called.
  ----------------------------------------------------------------------------

  EMERGENT BEHAVIORS DEMONSTRATED (printed as metrics during the run)
    1. String stability: spacing disturbances injected by the two
       switchbacks propagate down each chain and damp out - per-chain
       gap mean/max printed at arrival.
    2. Platoon interleaving: up to 2 convoys share the serpentine,
       separated purely by the departure stagger (no inter-convoy
       communication).
    3. Accordion effect: chains compress at corners and re-stretch on
       straights - visible in the periodic gap telemetry.
    4. Role choreography from local rules: hookup, transit, sequenced
       docking, and rest emerge from per-robot standoff rules plus a
       single waypoint list.

  LED CONFIGURATION (updates every iteration)
    DOCTOR  : blue (0,90,255); white flash for each patient seated at
              the gate; cyan during the final ward sweep.
    NURSES  : green (0,200,80) modulated by their convoy's synchronized
              blink signature while the convoy is hooked/in transit;
              solid green at the rest post after docking.
    PATIENTS: amber (255,150,0) with the same convoy-synchronized blink
              while hooked/in transit (chains read as one organism);
              solid green (90,255,90) once seated in a discharge slot.

  ROBOTARIUM SUBMISSION COMPLIANCE
    * 10 robots (max 20); 210 s run (max 600 s); min start spacing 0.36 m.
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
RUN_SEED = 1110
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)

NUM_DOCTORS = 1
NUM_NURSES = 3
NUM_PATIENTS = 6
N = NUM_DOCTORS + NUM_NURSES + NUM_PATIENTS  # 10
DOC = np.arange(0, NUM_DOCTORS)
NUR = np.arange(NUM_DOCTORS, NUM_DOCTORS + NUM_NURSES)
PAT = np.arange(NUM_DOCTORS + NUM_NURSES, N)

HW_MAX_LINEAR = 0.20
ROBOT_BASE_LENGTH = 0.11
PLAN_LINEAR = 0.14
PLAN_ANGULAR = 1.8

DT = 0.033
TOTAL_SECONDS = 210.0


def sec(seconds):
    return int(round(seconds / DT))


TOTAL_ITERATIONS = sec(TOTAL_SECONDS)
T_STANDBY = sec(15.0)
T_SWEEP = sec(165.0)
T_CEREMONY = sec(195.0)
DEPART_TIMES = [40.0, 75.0, 110.0]  # s, convoy A/B/C departures
MARSHAL_DEPART = 20.0  # s

CHAIN_GAP = 0.32  # m, target predecessor standoff
WPT_RADIUS = 0.15  # m, leader waypoint switch radius
SEAT_RADIUS = 0.15  # m, discharge slot capture

SAFETY_RADIUS = 0.20
ARENA = np.array([-1.6, 1.6, -1.0, 1.0])
MARGIN = 0.15

# ----------------------------------------------------------------------------
# WARD GEOMETRY (min pairwise start spacing 0.36 m)
# ----------------------------------------------------------------------------
ROW_Y = [0.62, 0.00, -0.62]  # convoy rows A, B, C
BAY_COLS = [1.02, 1.40]
HOOK_X = 0.78
STAGE_X = 0.60
# Corridor entrance sits on the hook line (x ~ 0.78) so convoy A's leader
# never has to double back toward its own followers' bays; convoys B and C
# climb/merge to the entrance along the bay-front lane first.
ROUTE = [
    np.array([0.75, 0.62]),
    np.array([0.10, 0.62]),
    np.array([0.10, 0.00]),
    np.array([0.70, 0.00]),
    np.array([0.70, -0.62]),
    np.array([-0.20, -0.62]),
]
ENTRY_X = -0.75
SLOT_X = -1.20
REST_X = -0.62
GATE_POST = np.array([-0.85, -0.31])
DOCTOR_START = np.array([0.10, 0.62])  # already on the top lane

patient_start = np.zeros((2, NUM_PATIENTS))
for k in range(3):
    patient_start[:, 2 * k] = [BAY_COLS[0], ROW_Y[k]]
    patient_start[:, 2 * k + 1] = [BAY_COLS[1], ROW_Y[k]]
nurse_start = np.array([[STAGE_X] * 3, ROW_Y])

initial_conditions = np.vstack(
    [
        np.hstack([[DOCTOR_START[0]], nurse_start[0], patient_start[0]]),
        np.hstack([[DOCTOR_START[1]], nurse_start[1], patient_start[1]]),
        np.zeros(N),
    ]
)
_d = np.linalg.norm(
    initial_conditions[:2, :, None] - initial_conditions[:2, None, :], axis=0
) + np.eye(N)
assert _d.min() >= 0.35, f"initial spacing {_d.min():.3f} m violates the 0.35 m rule"


def slots_for(k):
    """Discharge slot pair for convoy k (inner = P1, outer = P2)."""
    return (
        np.array([SLOT_X, ROW_Y[k] + 0.15]),
        np.array([SLOT_X, ROW_Y[k] - 0.15]),
    )


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
LED_DOC_SWEEP = np.array([0.0, 220.0, 255.0])
LED_NURSE = np.array([0.0, 200.0, 80.0])
LED_FLASH = np.array([255.0, 255.0, 255.0])
LED_PAT = np.array([255.0, 150.0, 0.0])
LED_PAT_DONE = np.array([90.0, 255.0, 90.0])
ROLE_COLORS = np.zeros((3, N))
ROLE_COLORS[:, DOC] = LED_DOCTOR[:, None]
ROLE_COLORS[:, NUR] = LED_NURSE[:, None]
ROLE_COLORS[:, PAT] = LED_PAT[:, None]

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


def standoff(target, source, dist):
    away = source - target
    n = np.linalg.norm(away)
    away = away / n if n > 1e-6 else np.array([1.0, 0.0])
    return target + dist * away


# ----------------------------------------------------------------------------
# CONVOY STATE
# ----------------------------------------------------------------------------
# Per-convoy: WAIT -> TRANSIT -> ENTRY -> SEAT1 -> SEAT2 -> DONE
convoy_state = ["WAIT"] * 3
convoy_wpt = [0] * 3
hooked = [False] * 3
convoy_gaps = [([], []) for _ in range(3)]  # per-convoy (leader-P1, P1-P2) gap samples
seated_count = 0
doc_flash_until = -1.0
# Doctor marshal: PRE_RUN -> GATE -> SWEEP -> GATE
doc_state = "HOLD"
doc_wpt = 0
sweep_row = 0

# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
print(f"Run10 ConvoyDischarge: {N} robots, {TOTAL_SECONDS:.0f} s, seed {RUN_SEED}")
telemetry_marks = {sec(s) for s in (55.0, 70.0, 85.0, 100.0, 115.0, 130.0, 145.0, 160.0)}

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
        # ---------------- convoys ----------------
        for k in range(3):
            n_idx = NUR[k]
            p1_idx, p2_idx = PAT[2 * k], PAT[2 * k + 1]
            n_pos = pos[:, n_idx]
            p1 = pos[:, p1_idx]
            p2 = pos[:, p2_idx]
            slot1, slot2 = slots_for(k)
            nurse_goal = np.array([HOOK_X, ROW_Y[k]])
            p1_goal = patient_start[:, 2 * k]
            p2_goal = patient_start[:, 2 * k + 1]

            if convoy_state[k] == "WAIT":
                if not hooked[k] and np.linalg.norm(n_pos - nurse_goal) < 0.12:
                    hooked[k] = True
                    print(f"[t={t_sec:5.1f}s] convoy {'ABC'[k]} hooked (blink sync on)")
                if t_sec >= DEPART_TIMES[k]:
                    convoy_state[k] = "TRANSIT"
                    convoy_wpt[k] = 0
                    print(f"[t={t_sec:5.1f}s] CONVOY {'ABC'[k]} departs")
            elif convoy_state[k] == "TRANSIT":
                wpt = ROUTE[convoy_wpt[k]]
                nurse_goal = wpt
                if np.linalg.norm(n_pos - wpt) < WPT_RADIUS:
                    convoy_wpt[k] += 1
                    if convoy_wpt[k] >= len(ROUTE):
                        convoy_state[k] = "ENTRY"
                p1_goal = standoff(n_pos, p1, CHAIN_GAP)
                p2_goal = standoff(p1, p2, CHAIN_GAP)
                convoy_gaps[k][0].append(float(np.linalg.norm(n_pos - p1)))
                convoy_gaps[k][1].append(float(np.linalg.norm(p1 - p2)))
            elif convoy_state[k] == "ENTRY":
                entry = np.array([ENTRY_X, ROW_Y[k]])
                nurse_goal = entry
                p1_goal = standoff(n_pos, p1, CHAIN_GAP)
                p2_goal = standoff(p1, p2, CHAIN_GAP)
                if np.linalg.norm(n_pos - entry) < WPT_RADIUS:
                    convoy_state[k] = "SEAT1"
                    g1, g2 = convoy_gaps[k]
                    print(
                        f"[t={t_sec:5.1f}s] convoy {'ABC'[k]} at discharge row; "
                        f"gap1 mean {np.mean(g1):.2f} max {np.max(g1):.2f} m, "
                        f"gap2 mean {np.mean(g2):.2f} max {np.max(g2):.2f} m"
                    )
            elif convoy_state[k] == "SEAT1":
                # Sequenced docking (one mover at a time, Run09 lesson).
                nurse_goal = np.array([REST_X, ROW_Y[k]])
                p1_goal = slot1
                p2_goal = standoff(n_pos, p2, CHAIN_GAP)
                if np.linalg.norm(p1 - slot1) < SEAT_RADIUS:
                    convoy_state[k] = "SEAT2"
                    seated_count += 1
                    doc_flash_until = t_sec + 1.0
            elif convoy_state[k] == "SEAT2":
                nurse_goal = np.array([REST_X, ROW_Y[k]])
                p1_goal = slot1
                p2_goal = slot2
                if np.linalg.norm(p2 - slot2) < SEAT_RADIUS:
                    convoy_state[k] = "DONE"
                    seated_count += 1
                    doc_flash_until = t_sec + 1.0
                    print(
                        f"[t={t_sec:5.1f}s] convoy {'ABC'[k]} discharged "
                        f"({seated_count}/{NUM_PATIENTS} patients seated)"
                    )
            else:  # DONE
                nurse_goal = np.array([REST_X, ROW_Y[k]])
                p1_goal = slot1
                p2_goal = slot2

            dxi[:, n_idx : n_idx + 1] = si_position(
                n_pos.reshape(2, 1), clamp_arena(nurse_goal.reshape(2, 1))
            )
            dxi[:, p1_idx : p1_idx + 1] = si_position(
                p1.reshape(2, 1), clamp_arena(np.asarray(p1_goal).reshape(2, 1))
            )
            dxi[:, p2_idx : p2_idx + 1] = si_position(
                p2.reshape(2, 1), clamp_arena(np.asarray(p2_goal).reshape(2, 1))
            )

            # Convoy LED signature: synchronized blink at phase offset k*120deg.
            in_chain = hooked[k] and convoy_state[k] in ("WAIT", "TRANSIT", "ENTRY", "SEAT1")
            if in_chain:
                blink = 0.55 + 0.45 * np.sin(2.0 * np.pi * 0.8 * t_sec + k * 2.0 * np.pi / 3.0)
                leds[:, n_idx] = LED_NURSE * blink
                leds[:, p1_idx] = LED_PAT * blink
                leds[:, p2_idx] = LED_PAT * blink
            if convoy_state[k] in ("SEAT2", "DONE"):
                leds[:, p1_idx] = LED_PAT_DONE
            if convoy_state[k] == "DONE":
                leds[:, p2_idx] = LED_PAT_DONE

        # ---------------- doctor marshal ----------------
        d_pos = pos[:, DOC[0]]
        doc_goal = DOCTOR_START
        if doc_state == "HOLD" and t_sec >= MARSHAL_DEPART:
            doc_state = "PRE_RUN"
            doc_wpt = 2  # starts at ROUTE[1] (top-lane west), so next is index 2
        if doc_state == "PRE_RUN":
            doc_goal = ROUTE[doc_wpt]
            if np.linalg.norm(d_pos - doc_goal) < WPT_RADIUS:
                doc_wpt += 1
                if doc_wpt >= len(ROUTE):
                    doc_state = "GATE"
                    print(f"[t={t_sec:5.1f}s] marshal sweep complete; doctor at gate")
        elif doc_state == "GATE":
            doc_goal = GATE_POST
            if t >= T_SWEEP:
                doc_state = "SWEEP"
                sweep_row = 0
        elif doc_state == "SWEEP":
            doc_goal = np.array([-0.90, [0.77, 0.0, -0.77][min(sweep_row, 2)]])
            leds[:, DOC[0]] = LED_DOC_SWEEP
            if np.linalg.norm(d_pos - doc_goal) < 0.12:
                sweep_row += 1
                doc_flash_until = t_sec + 0.8
                if sweep_row >= 3:
                    doc_state = "GATE2"
        else:
            doc_goal = GATE_POST
        dxi[:, DOC[0] : DOC[0] + 1] = si_position(
            d_pos.reshape(2, 1), clamp_arena(np.asarray(doc_goal).reshape(2, 1))
        )
        if t_sec < doc_flash_until and int(t_sec * 6) % 2 == 0:
            leds[:, DOC[0]] = LED_FLASH

        # ---------------- telemetry + ceremony ----------------
        if t in telemetry_marks:
            lines = []
            for k in range(3):
                if convoy_state[k] in ("TRANSIT", "ENTRY"):
                    g1 = np.linalg.norm(pos[:, NUR[k]] - pos[:, PAT[2 * k]])
                    g2 = np.linalg.norm(pos[:, PAT[2 * k]] - pos[:, PAT[2 * k + 1]])
                    lines.append(f"{'ABC'[k]}: gaps {g1:.2f}/{g2:.2f} m wpt {convoy_wpt[k]}")
            if lines:
                print(f"[t={t_sec:5.1f}s] " + "; ".join(lines))
        if t >= T_CEREMONY:
            wave = 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.5 * t_sec - np.arange(N) * 0.7))
            leds = leds * (0.5 + 0.5 * wave)

    dxi = si_barrier(dxi, x)
    dxu = si_to_uni(dxi, x)
    stopped = np.linalg.norm(dxi, axis=0) < 1e-4
    if np.any(stopped):
        dxu[:, stopped] = 0.0
    dxu = wheel_safe(dxu)

    write_leds(leds)
    r.set_velocities(_ALL_IDS, dxu)
    r.step()

print(
    f"Run10 complete: {seated_count}/{NUM_PATIENTS} patients discharged, "
    f"convoy states {convoy_state}"
)

_end_hook = getattr(r, "call_at_scripts_end", None)
if callable(_end_hook):
    _end_hook()
_debug_hook = getattr(r, "debug", None)
if callable(_debug_hook):
    _debug_hook()
