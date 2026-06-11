"""
================================================================================
  RUN 05 / 10 - ANT COLONY OPTIMIZATION MEDICATION ROUNDS
  Georgia Tech Robotarium - https://www.robotarium.gatech.edu/experiment
================================================================================

  CLINICAL TRIAL OBJECTIVE
    Continuous medication delivery: 6 patient robots occupy a hexagonal
    bed ring; each bed's medication level decays over time. 4 nurse
    robots run ant-colony-optimized delivery rounds from a pharmacy
    dock, laying virtual pheromone on completed legs so efficient
    circuits emerge. 1 doctor robot supervises and physically escalates
    to any bed whose medication runs critically low.

  FLEET (11 robots, 3.25 min run)
    Robot   0     : DOCTOR   (supervisor / escalation)       LED blue
    Robots  1-4   : NURSES   (ACO delivery ants)             LED green/violet
    Robots  5-10  : PATIENTS (hexagon bed ring)              LED level color

  ALGORITHM PATTERN: ant colony optimization on the 6-bed tour graph.
    Edge choice probability ~ tau^alpha * (1/d)^beta * urgency^gamma with
    alpha=1.0, beta=2.2, gamma=1.5; pheromone deposit Q/d on completed
    legs; continuous evaporation rho=0.02/s; per-bed reservations stop
    two ants converging on one bed.

  REAL-ROBOT TIMING ASSUMPTIONS
    * 15 s standby head time before tasks begin (real fleet start delay).
    * Linear speed planned at 0.14 m/s (30% below the 0.20 m/s platform max).
    * Angular speed planned at 1.8 rad/s (50% below the 3.6 rad/s max).
    * Wheel-speed rescaling keeps every command inside actuator limits.

  STEP-BY-STEP TIMELINE (5909 iterations @ 0.033 s = 195 s wall-clock)
  ----------------------------------------------------------------------------
  STANDBY (0..15 s) - real-robot start delay buffer
    * t =  0-10 s - all 11 robots hold; LEDs breathe dim white at 0.5 Hz.
    * t = 10-15 s - LEDs ramp white -> role colors (role announce).
  PHASE 1 - SHIFT START (15..30 s)
    * t = 15 s  - nurses form up on the pharmacy dock (left wall);
                  doctor begins a slow supervision orbit at 0.45 m around
                  the ward center; patients hold beds, med levels decaying
                  (LEDs drifting green -> amber -> red).
    * t = 30 s  - pharmacy stocked; ACO round-routing goes live.
  PHASE 2 - ACO MEDICATION ROUNDS (30..180 s, continuous process)
    * t = 30 s  - all 6 beds are below the 0.75 restock threshold, so all
                  4 nurses immediately draw routes (urgency-weighted).
    * t ~ 33-40 s - bed 3 starts lowest (0.28); expect it served first or,
                  if the route draw misses it, the doctor escalates when
                  it falls below 0.20 (LED red pulse, level restored to
                  0.60) - emergent supervisor intervention.
    * t ~ 30-90 s - exploration: nurses try varied legs; deliveries flash
                  white; pheromone accumulates on short, frequently-needed
                  legs. Ward status printed every 30 s.
    * t ~ 90-150 s - exploitation: dominant pheromone edges form stable
                  delivery circuits (top edge share printed); average bed
                  wait time drops.
    * t = 150-180 s - steady-state rounds; escalations become rare since
                  the learned circuits keep every level above critical.
  PHASE 3 - END OF SHIFT (180..195 s)
    * t = 180 s - nurses finish their current leg and return to the
                  pharmacy dock; doctor returns to ward center.
    * t = 190 s - LED handoff wave (green pulse sweeping robot indices).
    * t = 195 s - run ends; debug/cleanup hooks called.
  ----------------------------------------------------------------------------

  EMERGENT BEHAVIORS DEMONSTRATED (printed as metrics during the run)
    1. Stigmergic route learning: pheromone concentration (top-edge share)
       grows while average medication wait time falls - no nurse ever
       plans a tour.
    2. Load balancing: per-nurse delivery counts stay within ~2 of each
       other purely through reservations + urgency weighting.
    3. Supervisor escalation: the doctor's red-pulse interventions emerge
       only when the colony's routing falls behind demand.
    4. Shift rhythm: delivery intervals self-organize into a steady beat
       matched to the 0.012/s decay rate.

  LED CONFIGURATION (updates every iteration)
    DOCTOR  : blue (0,90,255) orbiting; 2 Hz red pulse (255,40,0) during
              an escalation visit; blue at ward center end-of-shift.
    NURSES  : violet (180,0,255) while carrying a dose (en route/serving);
              1 s white flash on delivery; green (0,200,80) docked/idle.
    PATIENTS: continuous medication-level colormap - green (90,255,90) at
              1.0 down through amber (255,170,0) to red (255,30,0) near
              0.0; 1.5 s white flash when a dose is delivered.

  ROBOTARIUM SUBMISSION COMPLIANCE
    * 11 robots (max 20); 195 s run (max 600 s); min start spacing 0.36 m.
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
RUN_SEED = 1105
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)

NUM_DOCTORS = 1
NUM_NURSES = 4
NUM_PATIENTS = 6
N = NUM_DOCTORS + NUM_NURSES + NUM_PATIENTS  # 11
DOC = np.arange(0, NUM_DOCTORS)
NUR = np.arange(NUM_DOCTORS, NUM_DOCTORS + NUM_NURSES)
PAT = np.arange(NUM_DOCTORS + NUM_NURSES, N)

HW_MAX_LINEAR = 0.20
ROBOT_BASE_LENGTH = 0.11
PLAN_LINEAR = 0.14
PLAN_ANGULAR = 1.8

DT = 0.033
TOTAL_SECONDS = 195.0


def sec(seconds):
    return int(round(seconds / DT))


TOTAL_ITERATIONS = sec(TOTAL_SECONDS)
T_STANDBY = sec(15.0)
T_ROUNDS = sec(30.0)
T_SHIFT_END = sec(180.0)

# ACO parameters
ACO_ALPHA = 1.0
ACO_BETA = 2.2
ACO_GAMMA = 1.5  # urgency exponent
ACO_RHO = 0.02  # pheromone evaporation per second
ACO_Q = 0.6  # deposit scale (tau += Q / leg_length)
TAU_FLOOR = 0.05
RESTOCK_BELOW = 0.75  # beds below this level request a dose
MED_DECAY = 0.012  # level units per second
DELIVERY_HOLD = 3.0  # s spent serving at a bed
DOSE_LEVEL = 1.0
ESCALATE_BELOW = 0.20
ESCALATE_RESTORE = 0.60
ESCALATE_HOLD = 2.5  # s
NURSE_TIMEOUT = 25.0  # s before an unreachable claim is released

SAFETY_RADIUS = 0.20
ARENA = np.array([-1.6, 1.6, -1.0, 1.0])
MARGIN = 0.15

# ----------------------------------------------------------------------------
# WARD GEOMETRY (min pairwise start spacing 0.36 m)
# ----------------------------------------------------------------------------
_bed_angles = np.linspace(0.0, 2.0 * np.pi, NUM_PATIENTS, endpoint=False)
BEDS = np.vstack([1.05 * np.cos(_bed_angles), 0.62 * np.sin(_bed_angles)])
PHARMACY_DOCKS = np.array([[-1.40, -1.40, -1.40, -1.40], [0.54, 0.18, -0.18, -0.54]])
WARD_CENTER = np.array([0.0, 0.0])
DOCTOR_ORBIT_R = 0.45

initial_conditions = np.vstack(
    [
        np.hstack([[0.0], PHARMACY_DOCKS[0], BEDS[0]]),
        np.hstack([[0.0], PHARMACY_DOCKS[1], BEDS[1]]),
        np.zeros(N),
    ]
)
_d = np.linalg.norm(
    initial_conditions[:2, :, None] - initial_conditions[:2, None, :], axis=0
) + np.eye(N)
assert _d.min() >= 0.35, f"initial spacing {_d.min():.3f} m violates the 0.35 m rule"


def bed_standoff(bed_idx):
    """Nurse/doctor service point: 0.26 m inward of the bed (clear of patient)."""
    inward = WARD_CENTER - BEDS[:, bed_idx]
    inward = inward / max(np.linalg.norm(inward), 1e-6)
    return BEDS[:, bed_idx] + 0.26 * inward


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
LED_DOC_ESC = np.array([255.0, 40.0, 0.0])
LED_NURSE_IDLE = np.array([0.0, 200.0, 80.0])
LED_NURSE_DOSE = np.array([180.0, 0.0, 255.0])
LED_FLASH = np.array([255.0, 255.0, 255.0])
LED_LVL_LOW = np.array([255.0, 30.0, 0.0])
LED_LVL_MID = np.array([255.0, 170.0, 0.0])
LED_LVL_OK = np.array([90.0, 255.0, 90.0])
ROLE_COLORS = np.zeros((3, N))
ROLE_COLORS[:, DOC] = LED_DOCTOR[:, None]
ROLE_COLORS[:, NUR] = LED_NURSE_IDLE[:, None]
ROLE_COLORS[:, PAT] = LED_LVL_OK[:, None]


def level_color(levels):
    v = np.clip(np.asarray(levels, dtype=float), 0.0, 1.0)
    out = np.zeros((3, v.size))
    lo = v < 0.5
    bl = v[lo] / 0.5
    out[:, lo] = LED_LVL_LOW[:, None] * (1.0 - bl) + LED_LVL_MID[:, None] * bl
    hi = ~lo
    bh = (v[hi] - 0.5) / 0.5
    out[:, hi] = LED_LVL_MID[:, None] * (1.0 - bh) + LED_LVL_OK[:, None] * bh
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


# ----------------------------------------------------------------------------
# ACO STATE
# ----------------------------------------------------------------------------
med_level = np.array([0.55, 0.50, 0.45, 0.28, 0.50, 0.45])
tau = np.ones((NUM_PATIENTS, NUM_PATIENTS))
np.fill_diagonal(tau, 0.0)
bed_dist = np.linalg.norm(BEDS[:, :, None] - BEDS[:, None, :], axis=0) + np.eye(NUM_PATIENTS)

# Per-nurse FSM: state in {IDLE, TO_BED, DELIVER}; node = last bed (-1 = dock).
nurse_state = ["IDLE"] * NUM_NURSES
nurse_node = [-1] * NUM_NURSES
nurse_target_bed = [-1] * NUM_NURSES
nurse_timer = [0.0] * NUM_NURSES
nurse_deliveries = [0] * NUM_NURSES
claimed = set()
deliveries_total = 0
wait_accum = 0.0  # integral of (#beds below threshold) dt -> mean wait proxy
flash_pat_until = np.full(NUM_PATIENTS, -1.0)
flash_nur_until = np.full(NUM_NURSES, -1.0)

# Doctor FSM
doc_state = "ORBIT"
doc_bed = -1
doc_timer = 0.0
doc_cooldown = 0.0
escalations = 0


def choose_next_bed(k, t_sec):
    """ACO edge draw for nurse k from its current node (urgency-weighted)."""
    needy = [
        b
        for b in range(NUM_PATIENTS)
        if med_level[b] < RESTOCK_BELOW and b not in claimed and b != nurse_node[k]
    ]
    if not needy:
        return -1
    node = nurse_node[k]
    weights = []
    for b in needy:
        d = (
            bed_dist[node, b]
            if node >= 0
            else float(np.linalg.norm(PHARMACY_DOCKS[:, k] - BEDS[:, b]))
        )
        ph = tau[node, b] if node >= 0 else 1.0
        urgency = (1.0 - med_level[b]) ** ACO_GAMMA
        weights.append((ph**ACO_ALPHA) * ((1.0 / max(d, 0.05)) ** ACO_BETA) * urgency)
    weights = np.array(weights)
    pick = int(np.random.choice(len(needy), p=weights / weights.sum()))
    return needy[pick]


def top_edge_share():
    """Share of total pheromone held by the strongest edge (i<j)."""
    upper = tau[np.triu_indices(NUM_PATIENTS, k=1)]
    return float(upper.max() / max(upper.sum(), 1e-9))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
print(f"Run05 AntColonyMeds: {N} robots, {TOTAL_SECONDS:.0f} s, seed {RUN_SEED}")
status_marks = {sec(s) for s in (60.0, 90.0, 120.0, 150.0, 180.0)}

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
        # Ward physics: decay + evaporation run during all task phases.
        med_level = np.clip(med_level - MED_DECAY * DT, 0.0, 1.0)
        tau = np.maximum(tau * (1.0 - ACO_RHO * DT), TAU_FLOOR)
        np.fill_diagonal(tau, 0.0)
        wait_accum += float((med_level < RESTOCK_BELOW).sum()) * DT

        # Patients hold their beds (gentle 1 cm sway).
        sway = 0.01 * np.vstack(
            [
                np.sin(2.0 * np.pi * 0.1 * t_sec + _bed_angles),
                np.cos(2.0 * np.pi * 0.12 * t_sec + _bed_angles),
            ]
        )
        dxi[:, PAT] = si_position(pos[:, PAT], BEDS + sway)

        rounds_active = T_ROUNDS <= t < T_SHIFT_END

        # ---------------- nurse ACO finite-state machines -----------------
        for k in range(NUM_NURSES):
            n_idx = NUR[k]
            n_pos = pos[:, n_idx]
            if nurse_state[k] == "IDLE":
                target = PHARMACY_DOCKS[:, k]
                if rounds_active and t % sec(1.0) == 0:
                    nxt = choose_next_bed(k, t_sec)
                    if nxt >= 0:
                        nurse_target_bed[k] = nxt
                        claimed.add(nxt)
                        nurse_state[k] = "TO_BED"
                        nurse_timer[k] = t_sec
            elif nurse_state[k] == "TO_BED":
                b = nurse_target_bed[k]
                target = bed_standoff(b)
                if np.linalg.norm(n_pos - target) < 0.12:
                    nurse_state[k] = "DELIVER"
                    nurse_timer[k] = t_sec
                elif t_sec - nurse_timer[k] > NURSE_TIMEOUT:
                    claimed.discard(b)  # unreachable: release and rechoose
                    nurse_target_bed[k] = -1
                    nurse_state[k] = "IDLE"
            else:  # DELIVER
                b = nurse_target_bed[k]
                target = bed_standoff(b)
                if t_sec - nurse_timer[k] >= DELIVERY_HOLD:
                    med_level[b] = DOSE_LEVEL
                    deliveries_total += 1
                    nurse_deliveries[k] += 1
                    flash_pat_until[b] = t_sec + 1.5
                    flash_nur_until[k] = t_sec + 1.0
                    if nurse_node[k] >= 0:
                        d = bed_dist[nurse_node[k], b]
                        tau[nurse_node[k], b] += ACO_Q / max(d, 0.05)
                        tau[b, nurse_node[k]] = tau[nurse_node[k], b]
                    nurse_node[k] = b
                    claimed.discard(b)
                    nurse_target_bed[k] = -1
                    nurse_state[k] = "IDLE"
            if not rounds_active and nurse_state[k] == "IDLE":
                target = PHARMACY_DOCKS[:, k]
                nurse_node[k] = -1
            dxi[:, n_idx : n_idx + 1] = si_position(
                n_pos.reshape(2, 1), clamp_arena(target.reshape(2, 1))
            )
            carrying = nurse_state[k] != "IDLE"
            if t_sec < flash_nur_until[k] and int(t_sec * 6) % 2 == 0:
                leds[:, n_idx] = LED_FLASH
            elif carrying:
                leds[:, n_idx] = LED_NURSE_DOSE

        # ---------------- doctor supervision / escalation ------------------
        d_pos = pos[:, DOC[0]]
        if doc_state == "ORBIT":
            ang = 0.35 * t_sec
            target = WARD_CENTER + DOCTOR_ORBIT_R * np.array([np.cos(ang), np.sin(ang)])
            worst = int(np.argmin(med_level))
            nurse_near = any(
                np.linalg.norm(pos[:, NUR[j]] - BEDS[:, worst]) < 0.50 for j in range(NUM_NURSES)
            )
            if (
                rounds_active
                and med_level[worst] < ESCALATE_BELOW
                and not nurse_near
                and t_sec > doc_cooldown
            ):
                doc_state = "ESCALATE"
                doc_bed = worst
                doc_timer = -1.0
                escalations += 1
                print(
                    f"[t={t_sec:5.1f}s] DOCTOR ESCALATION -> bed {worst} "
                    f"(level {med_level[worst]:.2f})"
                )
        if doc_state == "ESCALATE":
            target = bed_standoff(doc_bed)
            if np.linalg.norm(d_pos - target) < 0.12:
                if doc_timer < 0.0:
                    doc_timer = t_sec
                elif t_sec - doc_timer >= ESCALATE_HOLD:
                    med_level[doc_bed] = max(med_level[doc_bed], ESCALATE_RESTORE)
                    flash_pat_until[doc_bed] = t_sec + 1.5
                    doc_state = "ORBIT"
                    doc_cooldown = t_sec + 5.0
            pulse = int(t_sec * 4) % 2 == 0
            leds[:, DOC[0]] = LED_DOC_ESC * (1.0 if pulse else 0.35)
        if not rounds_active and doc_state == "ORBIT" and t >= T_SHIFT_END:
            target = WARD_CENTER
        dxi[:, DOC[0] : DOC[0] + 1] = si_position(
            d_pos.reshape(2, 1), clamp_arena(target.reshape(2, 1))
        )

        # ---------------- patient LEDs + shift-end wave --------------------
        leds[:, PAT] = level_color(med_level)
        for b in range(NUM_PATIENTS):
            if t_sec < flash_pat_until[b] and int(t_sec * 6) % 2 == 0:
                leds[:, PAT[b]] = LED_FLASH
        if t_sec >= 190.0:
            wave = 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.8 * t_sec - np.arange(N) * 0.7))
            leds = leds * (0.5 + 0.5 * wave)

        if t in status_marks:
            lv = " ".join(f"{v:.2f}" for v in med_level)
            print(
                f"[t={t_sec:5.1f}s] levels [{lv}] deliveries={deliveries_total} "
                f"per-nurse={nurse_deliveries} top-edge={top_edge_share():.2f} "
                f"escalations={escalations}"
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

mean_wait = wait_accum / max(TOTAL_SECONDS - 15.0, 1.0)
print(
    f"Run05 complete: {deliveries_total} deliveries {nurse_deliveries}, "
    f"{escalations} escalations, mean beds-awaiting={mean_wait:.2f}, "
    f"top pheromone edge share={top_edge_share():.2f}"
)

_end_hook = getattr(r, "call_at_scripts_end", None)
if callable(_end_hook):
    _end_hook()
_debug_hook = getattr(r, "debug", None)
if callable(_debug_hook):
    _debug_hook()
