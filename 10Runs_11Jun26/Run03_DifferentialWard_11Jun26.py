"""
================================================================================
  RUN 03 / 10 - DIFFERENTIAL EVOLUTION WARD LAYOUT OPTIMIZATION
  Georgia Tech Robotarium - https://www.robotarium.gatech.edu/experiment
================================================================================

  CLINICAL TRIAL OBJECTIVE
    Ward design study: 4 doctor robots embody candidate treatment-station
    locations while differential evolution (DE) minimizes the total
    distance from 8 patient robots to their nearest station (a facility-
    location problem). Once the layout converges, 4 nurse robots escort
    each patient pair to its station and the ward goes operational.

  FLEET (16 robots, 3.5 min run)
    Robots  0-3   : DOCTORS  (mobile treatment stations)     LED blue
    Robots  4-7   : NURSES   (escort teams)                  LED green
    Robots  8-15  : PATIENTS (4 home clusters of 2)          LED red/amber

  ALGORITHM PATTERN: differential evolution rand/1/bin over 8-dimensional
    layout vectors (4 stations x [x, y]); population 24, F=0.6, CR=0.8,
    one generation every 1.5 s. Doctors continuously drive toward the
    current best layout, physically tracing the optimizer's search.

  REAL-ROBOT TIMING ASSUMPTIONS
    * 15 s standby head time before tasks begin (real fleet start delay).
    * Linear speed planned at 0.14 m/s (30% below the 0.20 m/s platform max).
    * Angular speed planned at 1.8 rad/s (50% below the 3.6 rad/s max).
    * Wheel-speed rescaling keeps every command inside actuator limits.

  STEP-BY-STEP TIMELINE (6364 iterations @ 0.033 s = 210 s wall-clock)
  ----------------------------------------------------------------------------
  STANDBY (0..15 s) - real-robot start delay buffer
    * t =  0-10 s - all 16 robots hold; LEDs breathe dim white at 0.5 Hz.
    * t = 10-15 s - LEDs ramp white -> role colors (role announce).
  PHASE 1 - DEPLOY (15..30 s)
    * t = 15 s  - doctors leave the wall toward the initial best layout
                  (random seeded population); nurses move to the corner
                  staging posts; patients hold home clusters with gentle
                  distress oscillation.
    * t = 30 s  - doctors at their first candidate stations (LED red-blue
                  blend: red = layout still moving a lot).
  PHASE 2 - LIVE OPTIMIZATION (30..120 s; 60 DE generations @ 1.5 s)
    * t = 30 s  - DE iterates rand/1/bin; every generation the incumbent
                  best layout updates the doctors' position targets.
    * t ~ 30-55 s - stations jump visibly between candidate sites
                  (exploration); fitness printed every 10 generations.
    * t ~ 55-90 s - jumps shrink; one station migrates toward each of the
                  4 patient clusters (emergent k-median solution).
    * t ~ 90-120 s - sub-centimeter refinements only; doctor LEDs turn
                  fully blue (layout frozen-in); fitness plateau printed.
  PHASE 3 - ESCORTED ADMISSION (120..180 s)
    * t = 120 s - final layout locked. Each nurse drives to its patient
                  pair's midpoint (LED magenta), collects the pair, and
                  leads it to the pair's nearest station (LED chain:
                  nurse magenta, patients amber while in transit).
    * t = 150 s - first pairs arrive; patients dock at station-side bed
                  slots 0.32 m from the doctor and turn green.
    * t = 180 s - all 8 patients docked at their stations.
  PHASE 4 - WARD OPERATIONAL (180..210 s)
    * t = 180 s - nurses retreat to perimeter watch posts; doctors hold
                  stations; patients hold bed slots.
    * t = 195 s - LED ward heartbeat: synchronized slow green pulse across
                  all 16 robots (system nominal).
    * t = 210 s - run ends; debug/cleanup hooks called.
  ----------------------------------------------------------------------------

  EMERGENT BEHAVIORS DEMONSTRATED (printed as metrics during the run)
    1. Facility location without clustering code: DE alone discovers a
       1-station-per-cluster layout (k-median structure emerges from the
       distance objective).
    2. Exploration -> exploitation: doctor travel distance per generation
       (station displacement) decays toward zero - printed every 10
       generations as 'station drift'.
    3. Anytime embodiment: doctors chase the incumbent best, so the fleet
       always shows the optimizer's current belief, including occasional
       visible 'mind changes' when a new basin wins.
    4. Convoy admission: simple follow-the-nurse rules produce orderly
       2-patient trains without trajectory planning.

  LED CONFIGURATION (updates every iteration)
    DOCTORS : red(255,60,0) <-> blue(0,90,255) lerp by recent station
              drift (red = searching, blue = settled); blue at lock.
    NURSES  : green (0,200,80) staged; magenta (200,0,255) while escorting;
              green again at watch posts.
    PATIENTS: amber (255,150,0) at home; red pulse (1 Hz) during distress
              oscillation peaks; white flash while joining an escort
              train; green (90,255,90) once docked at a station bed.

  ROBOTARIUM SUBMISSION COMPLIANCE
    * 16 robots (max 20); 210 s run (max 600 s); min start spacing 0.36 m.
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
RUN_SEED = 1103
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)

NUM_DOCTORS = 4
NUM_NURSES = 4
NUM_PATIENTS = 8
N = NUM_DOCTORS + NUM_NURSES + NUM_PATIENTS  # 16
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
T_DEPLOYED = sec(30.0)
T_LOCK = sec(120.0)
T_WARD = sec(180.0)
GEN_ITERS = sec(1.5)  # one DE generation every 1.5 s
NUM_GENERATIONS = 60

# DE parameters (rand/1/bin)
POP_SIZE = 24
DE_F = 0.6
DE_CR = 0.8
SEARCH_X = (-1.10, 1.10)  # station search box keeps stations off the walls
SEARCH_Y = (-0.70, 0.70)

SAFETY_RADIUS = 0.20
ARENA = np.array([-1.6, 1.6, -1.0, 1.0])
MARGIN = 0.15

# ----------------------------------------------------------------------------
# INITIAL CONDITIONS (min pairwise spacing 0.36 m)
# Patients sit in 4 home clusters of 2 (one per arena quadrant). Doctors
# start on the left wall, nurses on the bottom/top staging corners.
# ----------------------------------------------------------------------------
doctor_xy = np.array([[-1.40, -1.40, -1.40, -1.40], [0.66, 0.22, -0.22, -0.66]])
nurse_xy = np.array([[-1.00, -1.00, -1.00, -1.00], [0.85, 0.35, -0.35, -0.85]])
CLUSTER_CENTERS = np.array([[-0.45, 0.95, -0.45, 0.95], [0.55, 0.55, -0.55, -0.55]])
patient_xy = np.zeros((2, NUM_PATIENTS))
for c in range(4):
    patient_xy[:, 2 * c] = CLUSTER_CENTERS[:, c] + np.array([-0.19, 0.0])
    patient_xy[:, 2 * c + 1] = CLUSTER_CENTERS[:, c] + np.array([0.19, 0.0])
initial_conditions = np.vstack(
    [
        np.hstack([doctor_xy[0], nurse_xy[0], patient_xy[0]]),
        np.hstack([doctor_xy[1], nurse_xy[1], patient_xy[1]]),
        np.zeros(N),
    ]
)
_d = np.linalg.norm(
    initial_conditions[:2, :, None] - initial_conditions[:2, None, :], axis=0
) + np.eye(N)
assert _d.min() >= 0.35, f"initial spacing {_d.min():.3f} m violates the 0.35 m rule"

PATIENT_HOME = patient_xy.copy()
NURSE_WATCH = np.array([[-1.35, -1.35, 1.35, 1.35], [0.80, -0.80, 0.80, -0.80]])
DRIFT_PHASE = np.random.uniform(0.0, 2.0 * np.pi, NUM_PATIENTS)


# ----------------------------------------------------------------------------
# DIFFERENTIAL EVOLUTION (genome = [x0, y0, x1, y1, x2, y2, x3, y3])
# ----------------------------------------------------------------------------
def random_genome():
    g = np.zeros(2 * NUM_DOCTORS)
    g[0::2] = np.random.uniform(SEARCH_X[0], SEARCH_X[1], NUM_DOCTORS)
    g[1::2] = np.random.uniform(SEARCH_Y[0], SEARCH_Y[1], NUM_DOCTORS)
    return g


def genome_stations(g):
    return g.reshape(NUM_DOCTORS, 2).T  # (2, 4)


def layout_cost(g, pat_pos):
    """Sum over patients of distance to the nearest station + spacing penalty."""
    st = genome_stations(g)
    d = np.linalg.norm(pat_pos[:, :, None] - st[:, None, :], axis=0)  # (P, 4)
    cost = float(d.min(axis=1).sum())
    # Keep stations mutually workable (>= 0.55 m apart) so escorts never jam.
    sd = np.linalg.norm(st[:, :, None] - st[:, None, :], axis=0) + np.eye(NUM_DOCTORS) * 1e9
    cost += float(np.maximum(0.55 - sd.min(axis=1), 0.0).sum()) * 4.0
    return cost


def clip_genome(g):
    out = np.copy(g)
    out[0::2] = np.clip(out[0::2], SEARCH_X[0], SEARCH_X[1])
    out[1::2] = np.clip(out[1::2], SEARCH_Y[0], SEARCH_Y[1])
    return out


de_pop = [random_genome() for _ in range(POP_SIZE)]
de_cost = None  # lazily evaluated against live patient positions
best_genome = None
prev_best_stations = None
station_drift = 1.0  # EMA of per-generation station displacement (m)
generation_idx = 0
pair_station = None  # cluster pair -> station index, fixed at lock time


def de_generation(pat_pos):
    """One rand/1/bin DE generation against live patient positions."""
    global de_pop, de_cost, best_genome, prev_best_stations, station_drift
    if de_cost is None:
        de_cost = np.array([layout_cost(g, pat_pos) for g in de_pop])
    for i in range(POP_SIZE):
        a, b, c = np.random.choice([k for k in range(POP_SIZE) if k != i], 3, replace=False)
        mutant = clip_genome(de_pop[a] + DE_F * (de_pop[b] - de_pop[c]))
        cross = np.random.rand(2 * NUM_DOCTORS) < DE_CR
        cross[np.random.randint(2 * NUM_DOCTORS)] = True
        trial = np.where(cross, mutant, de_pop[i])
        trial_cost = layout_cost(trial, pat_pos)
        if trial_cost <= de_cost[i]:
            de_pop[i] = trial
            de_cost[i] = trial_cost
    best_genome = de_pop[int(np.argmin(de_cost))].copy()
    stations = genome_stations(best_genome)
    if prev_best_stations is not None:
        step = float(np.linalg.norm(stations - prev_best_stations, axis=0).mean())
        station_drift = 0.7 * station_drift + 0.3 * step
    prev_best_stations = stations.copy()
    return float(de_cost.min())


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


LED_DOC_SEARCH = np.array([255.0, 60.0, 0.0])
LED_DOC_SET = np.array([0.0, 90.0, 255.0])
LED_NURSE = np.array([0.0, 200.0, 80.0])
LED_NURSE_ESCORT = np.array([200.0, 0.0, 255.0])
LED_PAT_HOME = np.array([255.0, 150.0, 0.0])
LED_PAT_RED = np.array([255.0, 30.0, 0.0])
LED_PAT_TRAIN = np.array([255.0, 255.0, 255.0])
LED_PAT_DOCKED = np.array([90.0, 255.0, 90.0])
ROLE_COLORS = np.zeros((3, N))
ROLE_COLORS[:, DOC] = LED_DOC_SET[:, None]
ROLE_COLORS[:, NUR] = LED_NURSE[:, None]
ROLE_COLORS[:, PAT] = LED_PAT_HOME[:, None]

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


def patient_drift(t_sec):
    dx = 0.05 * np.sin(2.0 * np.pi * 0.05 * t_sec + DRIFT_PHASE)
    dy = 0.05 * np.cos(2.0 * np.pi * 0.04 * t_sec + DRIFT_PHASE)
    return clamp_arena(PATIENT_HOME + np.vstack([dx, dy]))


def bed_slots(stations):
    """Two bed positions flanking each station (left/right of the doctor)."""
    slots = np.zeros((2, NUM_PATIENTS))
    for s in range(NUM_DOCTORS):
        slots[:, 2 * s] = stations[:, s] + np.array([-0.32, 0.0])
        slots[:, 2 * s + 1] = stations[:, s] + np.array([0.32, 0.0])
    return clamp_arena(slots)


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
print(f"Run03 DifferentialWard: {N} robots, {TOTAL_SECONDS:.0f} s, seed {RUN_SEED}")
locked_stations = None
locked_slots = None

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
            else ((1.0 - (t_sec - 10.0) / 5.0) * white + ((t_sec - 10.0) / 5.0) * ROLE_COLORS)
        )
    else:
        pat_pos = pos[:, PAT]
        nur_pos = pos[:, NUR]
        doc_pos = pos[:, DOC]

        if best_genome is None:
            de_generation(pat_pos)  # generation 0 seeds the first deployment

        if t < T_LOCK:
            # PHASES 1-2: patients drift at home; doctors chase incumbent best.
            dxi[:, PAT] = si_position(pat_pos, patient_drift(t_sec))
            dxi[:, NUR] = si_position(nur_pos, nurse_xy)
            if t >= T_DEPLOYED and (t - T_DEPLOYED) % GEN_ITERS == 0:
                cost = de_generation(pat_pos)
                generation_idx += 1
                if generation_idx % 10 == 0 or generation_idx == 1:
                    print(
                        f"[t={t_sec:5.1f}s] DE gen {generation_idx:02d}: "
                        f"cost={cost:.3f} station drift={station_drift:.3f} m"
                    )
            stations = genome_stations(best_genome)
            dxi[:, DOC] = si_position(doc_pos, clamp_arena(stations))
            # Doctor LED: red while the layout is moving, blue as it settles.
            settle = np.clip(1.0 - station_drift / 0.25, 0.0, 1.0)
            leds[:, DOC] = (LED_DOC_SEARCH * (1.0 - settle) + LED_DOC_SET * settle)[:, None]
            # Patient LED: amber with a 1 Hz red pulse riding the drift.
            pulse = 0.5 * (1.0 + np.sin(2.0 * np.pi * 1.0 * t_sec))
            leds[:, PAT] = (LED_PAT_HOME * (1.0 - 0.45 * pulse) + LED_PAT_RED * 0.45 * pulse)[
                :, None
            ]
        else:
            # PHASE 3-4: lock layout, escort pairs to stations, hold the ward.
            if locked_stations is None:
                locked_stations = clamp_arena(genome_stations(best_genome))
                # Pair c (patients 2c, 2c+1) -> nearest free station (greedy).
                cost = np.linalg.norm(
                    CLUSTER_CENTERS[:, :, None] - locked_stations[:, None, :], axis=0
                )
                pair_station = np.full(4, -1)
                for _ in range(4):
                    flat = int(np.argmin(cost))
                    c_idx, s_idx = divmod(flat, NUM_DOCTORS)
                    pair_station[c_idx] = s_idx
                    cost[c_idx, :] = np.inf
                    cost[:, s_idx] = np.inf
                locked_slots = bed_slots(locked_stations)
                print(
                    f"[t={t_sec:5.1f}s] layout locked; cluster->station map "
                    f"{[int(s) for s in pair_station]}"
                )
            dxi[:, DOC] = si_position(doc_pos, locked_stations)
            leds[:, DOC] = LED_DOC_SET[:, None]

            if t < T_WARD:
                # Nurse k escorts cluster k: collect at the pair midpoint, then
                # lead toward the station; patients follow the nurse.
                for k in range(4):
                    pair = [2 * k, 2 * k + 1]
                    s_idx = int(pair_station[k])
                    station = locked_stations[:, s_idx]
                    mid = pat_pos[:, pair].mean(axis=1)
                    nurse_p = nur_pos[:, k]
                    if np.linalg.norm(nurse_p - mid) > 0.30:
                        nurse_goal = mid  # go collect the pair
                        pat_goal = pat_pos[:, pair]  # pair waits
                    else:
                        to_st = station - nurse_p
                        d_st = np.linalg.norm(to_st)
                        step_v = to_st / d_st * min(d_st, 0.35) if d_st > 1e-6 else to_st
                        nurse_goal = nurse_p + step_v  # lead out front
                        trail = nurse_p - 0.30 * (to_st / max(d_st, 1e-6))
                        side = np.array([-(to_st[1]), to_st[0]]) / max(d_st, 1e-6)
                        pat_goal = np.column_stack([trail + 0.16 * side, trail - 0.16 * side])
                        leds[:, NUR[k]] = LED_NURSE_ESCORT
                        leds[:, PAT[pair]] = LED_PAT_TRAIN[:, None]
                    arrived = np.linalg.norm(nurse_p - station) < 0.45
                    if arrived:
                        nurse_goal = (
                            station
                            + (nurse_p - station)
                            / max(np.linalg.norm(nurse_p - station), 1e-6)
                            * 0.45
                        )
                        pat_goal = locked_slots[:, [2 * s_idx, 2 * s_idx + 1]]
                    dxi[:, NUR[k] : NUR[k] + 1] = si_position(
                        nurse_p.reshape(2, 1), clamp_arena(nurse_goal.reshape(2, 1))
                    )
                    dxi[:, PAT[pair]] = si_position(pat_pos[:, pair], clamp_arena(pat_goal))
            else:
                dxi[:, NUR] = si_position(nur_pos, NURSE_WATCH)
                slot_targets = locked_slots[
                    :, [2 * int(pair_station[k]) + i for k in range(4) for i in (0, 1)]
                ]
                dxi[:, PAT] = si_position(pat_pos, slot_targets)
                if t_sec >= 195.0:
                    beat = 0.6 + 0.4 * np.sin(2.0 * np.pi * 0.5 * t_sec)
                    leds[:, :] = (LED_PAT_DOCKED * beat)[:, None]

            # Docked patients show green regardless of sub-phase.
            if locked_slots is not None and t_sec < 195.0:
                for k in range(4):
                    s_idx = int(pair_station[k])
                    for j, p_local in enumerate([2 * k, 2 * k + 1]):
                        slot = locked_slots[:, 2 * s_idx + j]
                        if np.linalg.norm(pat_pos[:, p_local] - slot) < 0.18:
                            leds[:, PAT[p_local]] = LED_PAT_DOCKED

    dxi = si_barrier(dxi, x)
    dxu = si_to_uni(dxi, x)
    stopped = np.linalg.norm(dxi, axis=0) < 1e-4
    if np.any(stopped):
        dxu[:, stopped] = 0.0
    dxu = wheel_safe(dxu)

    write_leds(leds)
    r.set_velocities(_ALL_IDS, dxu)
    r.step()

if locked_stations is not None:
    final_d = np.linalg.norm(PATIENT_HOME[:, :, None] - locked_stations[:, None, :], axis=0).min(
        axis=1
    )
    print(
        f"Run03 complete: {generation_idx} DE generations, locked layout serves "
        f"patients at mean home distance {final_d.mean():.3f} m"
    )

_end_hook = getattr(r, "call_at_scripts_end", None)
if callable(_end_hook):
    _end_hook()
_debug_hook = getattr(r, "debug", None)
if callable(_debug_hook):
    _debug_hook()
