"""
================================================================================
  RUN 09 / 10 - SIMULATED ANNEALING BED REASSIGNMENT
  Georgia Tech Robotarium - https://www.robotarium.gatech.edu/experiment
================================================================================

  CLINICAL TRIAL OBJECTIVE
    Overnight ward rebalancing: 7 patient robots occupy 7 beds on a ring,
    but admissions scrambled everyone 3 beds away from their clinically
    ideal bed. Simulated annealing proposes pairwise bed swaps; every
    ACCEPTED swap is physically executed by a nurse escort team while
    the rest of the ward holds position. Doctors run the control desk,
    their LEDs displaying the cooling temperature. The ward literally
    anneals: chaotic early swaps (some making things worse) freeze into
    a near-optimal layout as the temperature drops.

  FLEET (13 robots, 2.75 min run)
    Robots  0-1   : DOCTORS  (control desk / temperature gauge)  LED red->blue
    Robots  2-5   : NURSES   (two alternating swap-escort teams)  LED green/violet
    Robots  6-12  : PATIENTS (bed ring, scrambled by +3)          LED fit color

  ALGORITHM PATTERN: simulated annealing over bed permutations.
    Energy E = sum_p w_p * ||bed(assign(p)) - bed(ideal(p))||, weights
    w_p ~ U(0.6, 1.4) seeded. Proposal: swap two patients' beds. Accept
    if dE < 0 or rand < exp(-dE / T); T starts at 1.6 and cools by
    x0.60 each epoch. One physical swap executes at a time; new
    proposals wait until the movers settle.

  REAL-ROBOT TIMING ASSUMPTIONS
    * 15 s standby head time before tasks begin (real fleet start delay).
    * Linear speed planned at 0.14 m/s (30% below the 0.20 m/s platform max).
    * Angular speed planned at 1.8 rad/s (50% below the 3.6 rad/s max).
    * Wheel-speed rescaling keeps every command inside actuator limits.

  STEP-BY-STEP TIMELINE (5000 iterations @ 0.033 s = 165 s wall-clock)
  ----------------------------------------------------------------------------
  STANDBY (0..15 s) - real-robot start delay buffer
    * t =  0-10 s - all 13 robots hold; LEDs breathe dim white at 0.5 Hz.
    * t = 10-15 s - LEDs ramp white -> role colors (role announce).
  PHASE 1 - BED AUDIT (15..30 s)
    * t = 15 s  - roll-call: each patient LED flashes white in sequence
                  (0.8 s per bed, sweeping the ring); doctors' LEDs go
                  full red (T = T0); initial energy printed.
    * t = 30 s  - audit complete; annealer armed.
  PHASE 2 - ANNEALING EPOCHS (30..130 s proposals; swaps drain to ~150 s)
    * t = 30 s  - epoch 1 (T=1.60): first swap proposed; accepted swaps
                  dispatch a 2-nurse escort team from the inner stations:
                  each nurse collects its patient, leads it along an
                  offset lane (movers pass on opposite sides), overshoots
                  the destination bed so the trailing patient lands on
                  it, and releases (~18-22 s per accepted swap; rejected
                  proposals just print). Expect 5-7 executed swaps.
    * t ~ 30-60 s - HOT PHASE: most proposals accepted, including
                  energy-INCREASING swaps (printed with dE > 0) - the
                  ward visibly churns.
    * t ~ 60-100 s - WARM PHASE: T has cooled ~10x; only near-neutral or
                  improving swaps pass; energy trends down.
    * t ~ 100-130 s - COLD PHASE: T < 0.1, greedy descent only; the
                  layout freezes near the optimum; proposals stop at
                  130 s and any in-flight swap finishes by ~150 s.
    * (nurse teams alternate between swaps: team A = nurses 0+1,
       team B = nurses 2+3 - workload sharing.)
  PHASE 3 - FROZEN LAYOUT (150..165 s)
    * t = 150 s - annealer locks; final energy + improvement printed.
    * t = 155 s - LED freeze sweep: patient fit colors pulse in unison;
                  doctor gauges fully blue (T ~= 0).
    * t = 165 s - run ends; debug/cleanup hooks called.
  ----------------------------------------------------------------------------

  EMERGENT BEHAVIORS DEMONSTRATED (printed as metrics during the run)
    1. Annealing schedule made physical: the accept/reject mix shifts
       from chaotic (uphill moves executed by real robots) to frozen -
       watch the ward churn, then settle.
    2. Escape from local structure: the +3 rotation is a deep
       permutation scramble; early uphill swaps are what allow the ring
       to unwind it (pure greedy from this start typically stalls
       higher - the energy trace shows the non-monotone path down).
    3. Two-lane passing: swap movers heading in opposite directions
       resolve head-on conflicts via offset escort lanes + barriers.
    4. Team rhythm: alternating escort teams produce a relay cadence
       no scheduler explicitly encodes.

  LED CONFIGURATION (updates every iteration)
    DOCTORS : continuous temperature gauge - red (255,60,0) at T0 fading
              to blue (0,80,255) as T cools; 3 white flashes when a
              proposal is ACCEPTED.
    NURSES  : green (0,200,80) at the inner-ring stations; violet
              (180,0,255) while collecting/escorting; white flash on
              seating a patient.
    PATIENTS: bed-fit color - green (90,255,90) when in their ideal bed,
              through amber to red (255,30,0) by weighted misplacement;
              swap movers blink white at 3 Hz while in motion.

  ROBOTARIUM SUBMISSION COMPLIANCE
    * 13 robots (max 20); 165 s run (max 600 s); min start spacing 0.36 m.
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
RUN_SEED = 1126
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)

NUM_DOCTORS = 2
NUM_NURSES = 4
NUM_PATIENTS = 7
N = NUM_DOCTORS + NUM_NURSES + NUM_PATIENTS  # 13
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
T_ANNEAL = sec(30.0)
T_FREEZE = sec(150.0)

# Simulated annealing parameters
SA_T0 = 1.60
SA_COOL = 0.60  # temperature multiplier per epoch
EPOCH_SECONDS = 8.0
SCRAMBLE_SHIFT = 3  # initial assignment: patient p starts in bed (p+3) % 7

SAFETY_RADIUS = 0.20
ARENA = np.array([-1.6, 1.6, -1.0, 1.0])
MARGIN = 0.15

# ----------------------------------------------------------------------------
# WARD GEOMETRY (min pairwise start spacing 0.36 m)
# ----------------------------------------------------------------------------
_bed_angles = np.linspace(0.0, 2.0 * np.pi, NUM_PATIENTS, endpoint=False) + np.pi / 2.0
BED_RING_C = np.array([0.10, 0.0])
BED_RING_R = 0.75
BEDS = BED_RING_C[:, None] + BED_RING_R * np.vstack([np.cos(_bed_angles), np.sin(_bed_angles)])
NURSE_STATIONS = np.array([[-0.18, 0.38, -0.18, 0.38], [0.28, 0.28, -0.28, -0.28]])
DESK = np.array([[1.35, 1.35], [0.25, -0.25]])

PREF_WEIGHT = np.random.uniform(0.6, 1.4, NUM_PATIENTS)
assign = [(p + SCRAMBLE_SHIFT) % NUM_PATIENTS for p in range(NUM_PATIENTS)]  # patient -> bed

initial_conditions = np.vstack(
    [
        np.hstack([DESK[0], NURSE_STATIONS[0], [BEDS[0, assign[p]] for p in range(NUM_PATIENTS)]]),
        np.hstack([DESK[1], NURSE_STATIONS[1], [BEDS[1, assign[p]] for p in range(NUM_PATIENTS)]]),
        np.zeros(N),
    ]
)
_d = np.linalg.norm(
    initial_conditions[:2, :, None] - initial_conditions[:2, None, :], axis=0
) + np.eye(N)
assert _d.min() >= 0.35, f"initial spacing {_d.min():.3f} m violates the 0.35 m rule"


def energy(a):
    """Weighted total misplacement of the current bed assignment."""
    return float(
        sum(
            PREF_WEIGHT[p] * np.linalg.norm(BEDS[:, a[p]] - BEDS[:, p]) for p in range(NUM_PATIENTS)
        )
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


LED_T_HOT = np.array([255.0, 60.0, 0.0])
LED_T_COLD = np.array([0.0, 80.0, 255.0])
LED_NURSE = np.array([0.0, 200.0, 80.0])
LED_NURSE_ESCORT = np.array([180.0, 0.0, 255.0])
LED_FLASH = np.array([255.0, 255.0, 255.0])
LED_FIT_BAD = np.array([255.0, 30.0, 0.0])
LED_FIT_MID = np.array([255.0, 170.0, 0.0])
LED_FIT_OK = np.array([90.0, 255.0, 90.0])
ROLE_COLORS = np.zeros((3, N))
ROLE_COLORS[:, DOC] = LED_T_HOT[:, None]
ROLE_COLORS[:, NUR] = LED_NURSE[:, None]
ROLE_COLORS[:, PAT] = LED_FIT_MID[:, None]


def fit_color(p):
    """Bed-fit color for patient p under the current assignment."""
    worst = PREF_WEIGHT[p] * 2.0 * BED_RING_R
    mis = PREF_WEIGHT[p] * float(np.linalg.norm(BEDS[:, assign[p]] - BEDS[:, p]))
    v = 1.0 - min(mis / worst, 1.0)
    if v < 0.5:
        return LED_FIT_BAD * (1.0 - v / 0.5) + LED_FIT_MID * (v / 0.5)
    return LED_FIT_MID * (1.0 - (v - 0.5) / 0.5) + LED_FIT_OK * ((v - 0.5) / 0.5)


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
# SA + SWAP-EXECUTION STATE
# ----------------------------------------------------------------------------
sa_temperature = SA_T0
epoch_idx = 0
e_now = energy(assign)
E0 = e_now
accepts = 0
rejects = 0
uphill_accepts = 0
swap = None  # dict describing the in-flight physical swap, or None
team_toggle = 0  # alternates nurse teams (0+1, then 2+3)
doc_flash_until = -1.0
nur_flash_until = np.full(NUM_NURSES, -1.0)


def propose_swap(t_sec):
    """One SA proposal; returns an executable swap dict if accepted."""
    global sa_temperature, epoch_idx, e_now, accepts, rejects, uphill_accepts
    global doc_flash_until, team_toggle
    i, j = random.sample(range(NUM_PATIENTS), 2)
    trial = list(assign)
    trial[i], trial[j] = trial[j], trial[i]
    d_e = energy(trial) - e_now
    accept = d_e < 0.0 or random.random() < np.exp(-d_e / max(sa_temperature, 1e-9))
    verdict = "ACCEPT" if accept else "reject"
    print(
        f"[t={t_sec:5.1f}s] epoch {epoch_idx:02d} T={sa_temperature:.3f}: "
        f"swap P{i}<->P{j} dE={d_e:+.3f} -> {verdict} (E={e_now:.3f})"
    )
    epoch_idx += 1
    sa_temperature *= SA_COOL
    if not accept:
        rejects += 1
        return None
    accepts += 1
    if d_e > 0:
        uphill_accepts += 1
    doc_flash_until = t_sec + 1.0
    team = (0, 1) if team_toggle == 0 else (2, 3)
    team_toggle = 1 - team_toggle
    e_now += d_e
    return {
        "movers": (i, j),
        "dest": (assign[j], assign[i]),  # destination beds (pre-commit values)
        "team": team,
        "stage": "COLLECT",
        "t0": t_sec,
    }


def lane_offset(a, b, sign):
    """Perpendicular offset point at the midpoint of segment a->b."""
    mid = 0.5 * (a + b)
    d = b - a
    n = np.linalg.norm(d)
    perp = np.array([-d[1], d[0]]) / n if n > 1e-6 else np.array([0.0, 1.0])
    return clamp_arena(mid + sign * 0.28 * perp)


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
print(
    f"Run09 AnnealingBeds: {N} robots, {TOTAL_SECONDS:.0f} s, seed {RUN_SEED}, "
    f"initial energy {E0:.3f}"
)

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
        # Temperature gauge + bed-fit base colors every task iteration.
        gauge = np.clip(sa_temperature / SA_T0, 0.0, 1.0)
        leds[:, DOC] = (LED_T_HOT * gauge + LED_T_COLD * (1.0 - gauge))[:, None]
        if t_sec < doc_flash_until and int(t_sec * 6) % 2 == 0:
            leds[:, DOC] = LED_FLASH[:, None]
        for p in range(NUM_PATIENTS):
            leds[:, PAT[p]] = fit_color(p)

        # Default motion: patients hold their assigned beds, nurses hold
        # stations, doctors hold the desk.
        bed_targets = np.column_stack([BEDS[:, assign[p]] for p in range(NUM_PATIENTS)])
        dxi[:, PAT] = si_position(pos[:, PAT], bed_targets)
        dxi[:, NUR] = si_position(pos[:, NUR], NURSE_STATIONS)
        dxi[:, DOC] = si_position(pos[:, DOC], DESK)

        if t < T_ANNEAL:
            # PHASE 1: bed audit roll-call (LED sweep around the ring).
            slot = int((t_sec - 15.0) / 0.8) % NUM_PATIENTS
            if int(t_sec * 4) % 2 == 0:
                leds[:, PAT[slot]] = LED_FLASH
        else:
            # PHASE 2: anneal while free; epoch boundary every 10 s.
            # Proposals stop at 130 s so the last accepted swap can finish
            # executing before the 150 s freeze.
            anneal_open = t < sec(130.0) and epoch_idx < 12
            if anneal_open and swap is None and (t - T_ANNEAL) % sec(EPOCH_SECONDS) == 0:
                swap = propose_swap(t_sec)

            if swap is not None:
                i, j = swap["movers"]
                bed_i, bed_j = swap["dest"]
                n1, n2 = swap["team"]
                p1, p2 = pos[:, PAT[i]], pos[:, PAT[j]]
                if swap["stage"] == "COLLECT":
                    t1 = standoff(p1, pos[:, NUR[n1]], 0.28)
                    t2 = standoff(p2, pos[:, NUR[n2]], 0.28)
                    near1 = np.linalg.norm(pos[:, NUR[n1]] - p1) < 0.36
                    near2 = np.linalg.norm(pos[:, NUR[n2]] - p2) < 0.36
                    if near1 and near2:
                        swap["stage"] = "VIA"
                        # Same sign for both: the movers' chords run in opposite
                        # directions, so equal signs put the two via points on
                        # OPPOSITE sides of the shared segment (passing lanes).
                        swap["via1"] = lane_offset(p1, BEDS[:, bed_i], +1.0)
                        swap["via2"] = lane_offset(p2, BEDS[:, bed_j], +1.0)
                elif swap["stage"] == "VIA":
                    t1, t2 = swap["via1"], swap["via2"]
                    if (
                        np.linalg.norm(pos[:, NUR[n1]] - swap["via1"]) < 0.22
                        and np.linalg.norm(pos[:, NUR[n2]] - swap["via2"]) < 0.22
                    ):
                        swap["stage"] = "SEAT1"
                else:  # SEAT1 / SEAT2 - movers dock ONE AT A TIME. Seating both
                    # simultaneously sends them head-on down the same chord and
                    # the cooperative barrier QP turns that into a slow mutual
                    # orbit; sequencing removes the opposing traffic entirely.
                    dir1 = BEDS[:, bed_i] - swap["via1"]
                    dir1 = dir1 / max(np.linalg.norm(dir1), 1e-6)
                    dir2 = BEDS[:, bed_j] - swap["via2"]
                    dir2 = dir2 / max(np.linalg.norm(dir2), 1e-6)
                    t1 = BEDS[:, bed_i] + 0.34 * np.array([-dir1[1], dir1[0]])
                    t2 = swap["via2"]  # second escort waits on its lane
                    if swap["stage"] == "SEAT2":
                        t2 = BEDS[:, bed_j] + 0.34 * np.array([-dir2[1], dir2[0]])
                    done1 = np.linalg.norm(p1 - BEDS[:, bed_i]) < 0.15
                    done2 = np.linalg.norm(p2 - BEDS[:, bed_j]) < 0.15
                    if swap["stage"] == "SEAT1" and done1:
                        swap["stage"] = "SEAT2"
                    elif swap["stage"] == "SEAT2" and done1 and done2:
                        assign[i], assign[j] = bed_i, bed_j
                        nur_flash_until[n1] = t_sec + 1.0
                        nur_flash_until[n2] = t_sec + 1.0
                        print(
                            f"[t={t_sec:5.1f}s] swap seated: P{i}->bed{bed_i}, "
                            f"P{j}->bed{bed_j} (E={e_now:.3f})"
                        )
                        swap = None
                if swap is not None and t_sec - swap["t0"] > 35.0:
                    # Safety valve: commit the exchange and let the movers
                    # self-seat via the default hold-your-bed controller.
                    assign[i], assign[j] = bed_i, bed_j
                    print(f"[t={t_sec:5.1f}s] swap timed out -> committed; movers self-seat")
                    swap = None
                if swap is not None:
                    # Escort kinematics: nurses lead, movers trail at 0.30 m.
                    dxi[:, NUR[n1] : NUR[n1] + 1] = si_position(
                        pos[:, NUR[n1]].reshape(2, 1), clamp_arena(t1.reshape(2, 1))
                    )
                    dxi[:, NUR[n2] : NUR[n2] + 1] = si_position(
                        pos[:, NUR[n2]].reshape(2, 1), clamp_arena(t2.reshape(2, 1))
                    )
                    if swap["stage"] == "VIA":
                        trail1 = standoff(pos[:, NUR[n1]], p1, 0.30)
                        trail2 = standoff(pos[:, NUR[n2]], p2, 0.30)
                        dxi[:, PAT[i] : PAT[i] + 1] = si_position(
                            p1.reshape(2, 1), clamp_arena(trail1.reshape(2, 1))
                        )
                        dxi[:, PAT[j] : PAT[j] + 1] = si_position(
                            p2.reshape(2, 1), clamp_arena(trail2.reshape(2, 1))
                        )
                    elif swap["stage"] in ("SEAT1", "SEAT2"):
                        dxi[:, PAT[i] : PAT[i] + 1] = si_position(
                            p1.reshape(2, 1), BEDS[:, bed_i].reshape(2, 1)
                        )
                        # The second mover holds beside its via lane until the
                        # first has docked, then takes its own clear run.
                        p2_goal = swap["via2"] if swap["stage"] == "SEAT1" else BEDS[:, bed_j]
                        p2_hold = standoff(pos[:, NUR[n2]], p2, 0.30)
                        p2_tgt = p2_hold if swap["stage"] == "SEAT1" else p2_goal
                        dxi[:, PAT[j] : PAT[j] + 1] = si_position(
                            p2.reshape(2, 1), clamp_arena(np.asarray(p2_tgt).reshape(2, 1))
                        )
                    leds[:, NUR[[n1, n2]]] = LED_NURSE_ESCORT[:, None]
                    if int(t_sec * 6) % 2 == 0:
                        leds[:, PAT[[i, j]]] = LED_FLASH[:, None]

        # Nurse seat-completion flashes.
        for k in range(NUM_NURSES):
            if t_sec < nur_flash_until[k] and int(t_sec * 6) % 2 == 0:
                leds[:, NUR[k]] = LED_FLASH

        # PHASE 3: frozen-layout pulse.
        if t >= sec(155.0):
            pulse = 0.55 + 0.45 * np.sin(2.0 * np.pi * 0.5 * t_sec)
            leds = leds * pulse

    dxi = si_barrier(dxi, x)
    dxu = si_to_uni(dxi, x)
    stopped = np.linalg.norm(dxi, axis=0) < 1e-4
    if np.any(stopped):
        dxu[:, stopped] = 0.0
    dxu = wheel_safe(dxu)

    write_leds(leds)
    r.set_velocities(_ALL_IDS, dxu)
    r.step()

e_final = energy(assign)  # energy of the assignment the ward physically holds
improvement = 100.0 * (E0 - e_final) / max(E0, 1e-9)
print(
    f"Run09 complete: E {E0:.3f} -> {e_final:.3f} ({improvement:.0f}% better), "
    f"{accepts} accepted ({uphill_accepts} uphill) / {rejects} rejected, "
    f"final assignment {assign}"
)

_end_hook = getattr(r, "call_at_scripts_end", None)
if callable(_end_hook):
    _end_hook()
_debug_hook = getattr(r, "debug", None)
if callable(_debug_hook):
    _debug_hook()
