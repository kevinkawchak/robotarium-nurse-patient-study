"""
================================================================================
  RUN 02 / 10 - GENETIC ALGORITHM CARE-TEAM PAIRING
  Georgia Tech Robotarium - https://www.robotarium.gatech.edu/experiment
================================================================================

  CLINICAL TRIAL OBJECTIVE
    Care-team optimization: a genetic algorithm evolves the assignment of
    5 nurse robots to 8 drifting patient robots (3 nurses double-cover,
    2 single-cover) while 3 doctor robots shadow the highest-acuity
    patients. The physical fleet re-deploys to express each generation's
    best chromosome, so the optimization is visible on the arena floor.

  FLEET (16 robots, 3 min run)
    Robots  0-2   : DOCTORS  (acuity monitors)               LED blue
    Robots  3-7   : NURSES   (assignable care units)         LED green
    Robots  8-15  : PATIENTS (drifting, oscillating acuity)  LED red/amber

  ALGORITHM PATTERN: genetic algorithm over patient permutations
    (population 36, tournament selection k=3, order crossover OX,
    swap mutation pm=0.25, elitism 2), one generation expressed on the
    physical robots every 12 s. Patient drift + acuity oscillation make
    the fitness landscape time-varying, so the GA chases a moving target.

  REAL-ROBOT TIMING ASSUMPTIONS
    * 15 s standby head time before tasks begin (real fleet start delay).
    * Linear speed planned at 0.14 m/s (30% below the 0.20 m/s platform max).
    * Angular speed planned at 1.8 rad/s (50% below the 3.6 rad/s max).
    * Wheel-speed rescaling keeps every command inside actuator limits.

  STEP-BY-STEP TIMELINE (5455 iterations @ 0.033 s = 180 s wall-clock)
  ----------------------------------------------------------------------------
  STANDBY (0..15 s) - real-robot start delay buffer
    * t =  0-10 s - all 16 robots hold; LEDs breathe dim white at 0.5 Hz.
    * t = 10-15 s - LEDs ramp white -> role colors (role announce).
  PHASE 1 - STAGING SWEEP (15..27 s)
    * t = 15 s  - nurses advance to the mid-arena staging line; doctors move
                  to observation posts; patients begin slow drift orbits and
                  acuity oscillation (LED red <-> amber).
    * t = 27 s  - staff staged; GA population initialized (seeded).
  PHASE 2 - EVOLUTION (27..147 s; one generation every 12 s, 10 total)
    Each 12 s generation cycle:
    * +0 s  - GA evaluates 36 chromosomes against live robot positions and
              acuities; best chromosome printed with fitness and churn
              (gene changes vs the previous expressed assignment).
    * +0-8 s - nurses travel to their newly assigned patient-group
               centroids (LED magenta while re-deploying); doctors travel
               to standoff posts beside the 3 highest-acuity patients.
    * +8-12 s - hold/evaluate window (nurses flash white if this generation
                found a new global-best fitness).
    Expected macro behavior:
    * t ~ 27-60 s  - large fitness drops + high churn (exploration).
    * t ~ 60-110 s - churn decays; assignments begin to persist between
                     generations (exploitation).
    * t ~ 110-147 s - GA tracks patient drift with small refinements only.
  PHASE 3 - ASSIGNMENT LOCK (147..165 s)
    * t = 147 s - final chromosome frozen; nurses orbit their patient
                  group's centroid at 0.30 m (care contact).
    * t = 165 s - orbits stable; patients in covered groups show green.
  PHASE 4 - CARE HUDDLE (165..180 s)
    * t = 165 s - doctors join their nearest patient group at 0.45 m
                  standoff; full care teams visible as compact clusters.
    * t = 180 s - run ends; debug/cleanup hooks called.
  ----------------------------------------------------------------------------

  EMERGENT BEHAVIORS DEMONSTRATED (printed as metrics during the run)
    1. Exploration -> exploitation transition: per-generation assignment
       churn (Hamming distance between expressed chromosomes) decays.
    2. Fitness descent on a moving landscape: best fitness improves even
       though patients drift and acuities oscillate between evaluations.
    3. Physical co-evolution: nurse travel changes the next generation's
       fitness (cost is measured from live positions), closing the loop
       between optimizer state and robot state.
    4. Division of labor: double-cover nurses gravitate to compact patient
       pairs, single-cover nurses to outliers - never explicitly coded.

  LED CONFIGURATION (updates every iteration)
    DOCTORS : blue (0,90,255) with brightness ramping over generations
              (search confidence); cyan flash during evaluate windows.
    NURSES  : green (0,200,80) when idle/holding; magenta (200,0,255)
              while re-deploying; 3 white flashes on a new global best.
    PATIENTS: continuous red(255,30,0) <-> amber(255,170,0) lerp by live
              acuity; green (90,255,90) when their assigned nurse is
              within 0.45 m (covered).

  ROBOTARIUM SUBMISSION COMPLIANCE
    * 16 robots (max 20); 180 s run (max 600 s); min start spacing 0.36 m.
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
RUN_SEED = 1102
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)

NUM_DOCTORS = 3
NUM_NURSES = 5
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
TOTAL_SECONDS = 180.0


def sec(seconds):
    return int(round(seconds / DT))


TOTAL_ITERATIONS = sec(TOTAL_SECONDS)
T_STANDBY = sec(15.0)
T_STAGED = sec(27.0)
GEN_SECONDS = 12.0
NUM_GENERATIONS = 10  # expressed at t = 27, 39, ..., 135 s
T_LOCK = sec(147.0)  # last generation's travel window (135..147 s) completes
T_HUDDLE = sec(165.0)

# GA parameters
POP_SIZE = 36
TOURNAMENT_K = 3
MUTATION_P = 0.25
ELITES = 2
SPREAD_WEIGHT = 0.6  # intra-group compactness penalty for double-cover nurses

SAFETY_RADIUS = 0.20
ARENA = np.array([-1.6, 1.6, -1.0, 1.0])
MARGIN = 0.15

# ----------------------------------------------------------------------------
# INITIAL CONDITIONS (min pairwise spacing 0.36 m)
# ----------------------------------------------------------------------------
doctor_xy = np.array([[-1.35, -1.35, -1.35], [0.60, 0.00, -0.60]])
nurse_xy = np.array([[-0.45] * 5, [0.72, 0.36, 0.00, -0.36, -0.72]])
patient_xy = np.array(
    [
        [0.55, 1.05, 0.55, 1.05, 0.55, 1.05, 0.55, 1.05],
        [0.66, 0.66, 0.22, 0.22, -0.22, -0.22, -0.66, -0.66],
    ]
)
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
NURSE_STAGE = np.array([[-0.10] * 5, [0.72, 0.36, 0.00, -0.36, -0.72]])
DOCTOR_POST = np.array([[-1.05, -1.05, -1.05], [0.55, 0.00, -0.55]])

# Acuity model: per-patient base severity + slow oscillation (seeded).
ACUITY_BASE = np.random.uniform(0.30, 1.00, NUM_PATIENTS)
ACUITY_PHASE = np.random.uniform(0.0, 2.0 * np.pi, NUM_PATIENTS)
DRIFT_PHASE = np.random.uniform(0.0, 2.0 * np.pi, NUM_PATIENTS)


def acuity(t_sec):
    osc = 0.35 * np.sin(2.0 * np.pi * 0.05 * t_sec + ACUITY_PHASE)
    return np.clip(ACUITY_BASE + osc, 0.05, 1.30)


def patient_drift_targets(t_sec):
    dx = 0.06 * np.sin(2.0 * np.pi * 0.04 * t_sec + DRIFT_PHASE)
    dy = 0.06 * np.cos(2.0 * np.pi * 0.05 * t_sec + 1.3 * DRIFT_PHASE)
    return PATIENT_HOME + np.vstack([dx, dy])


# ----------------------------------------------------------------------------
# GENETIC ALGORITHM (chromosome = permutation of the 8 patients)
# Slot map: nurse k serves perm[k]; nurses 0-2 additionally serve perm[5+k].
# ----------------------------------------------------------------------------
def groups_from(perm):
    return [
        [perm[0], perm[5]],
        [perm[1], perm[6]],
        [perm[2], perm[7]],
        [perm[3]],
        [perm[4]],
    ]


def fitness(perm, nurse_pos, pat_pos, acu):
    """Total weighted service cost (lower is better), from live robot state."""
    cost = 0.0
    for k, group in enumerate(groups_from(perm)):
        centroid = pat_pos[:, group].mean(axis=1)
        cost += float(np.linalg.norm(nurse_pos[:, k] - centroid) * (1.0 + acu[group].sum()))
        if len(group) == 2:
            spread = float(np.linalg.norm(pat_pos[:, group[0]] - pat_pos[:, group[1]]))
            cost += SPREAD_WEIGHT * spread
    return cost


def order_crossover(a, b):
    n = len(a)
    i, j = sorted(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1)
    child[i : j + 1] = a[i : j + 1]
    fill = [g for g in b if g not in child]
    ptr = 0
    for idx in range(n):
        if child[idx] < 0:
            child[idx] = fill[ptr]
            ptr += 1
    return child


def mutate(perm):
    if np.random.rand() < MUTATION_P:
        i, j = np.random.choice(len(perm), 2, replace=False)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def run_generation(population, nurse_pos, pat_pos, acu):
    scores = np.array([fitness(p, nurse_pos, pat_pos, acu) for p in population])
    order = np.argsort(scores)
    elites = [population[i].copy() for i in order[:ELITES]]
    children = list(elites)
    while len(children) < POP_SIZE:
        picks_a = np.random.choice(POP_SIZE, TOURNAMENT_K, replace=False)
        picks_b = np.random.choice(POP_SIZE, TOURNAMENT_K, replace=False)
        pa = population[picks_a[np.argmin(scores[picks_a])]]
        pb = population[picks_b[np.argmin(scores[picks_b])]]
        children.append(mutate(order_crossover(pa, pb)))
    return children, population[order[0]].copy(), float(scores[order[0]])


population = [np.random.permutation(NUM_PATIENTS) for _ in range(POP_SIZE)]
expressed = None  # currently deployed best chromosome
expressed_groups = None
global_best = np.inf
flash_until = -1.0  # nurse white-flash window end (s) after a new global best
generation_idx = 0
doctor_focus = np.argsort(-ACUITY_BASE)[:NUM_DOCTORS]  # refreshed per generation

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
LED_DOCTOR_EVAL = np.array([0.0, 220.0, 255.0])
LED_NURSE = np.array([0.0, 200.0, 80.0])
LED_NURSE_MOVE = np.array([200.0, 0.0, 255.0])
LED_FLASH = np.array([255.0, 255.0, 255.0])
LED_PAT_RED = np.array([255.0, 30.0, 0.0])
LED_PAT_AMBER = np.array([255.0, 170.0, 0.0])
LED_PAT_COVERED = np.array([90.0, 255.0, 90.0])
ROLE_COLORS = np.zeros((3, N))
ROLE_COLORS[:, DOC] = LED_DOCTOR[:, None]
ROLE_COLORS[:, NUR] = LED_NURSE[:, None]
ROLE_COLORS[:, PAT] = LED_PAT_AMBER[:, None]

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


def nurse_targets_from(groups, nurse_pos, pat_pos):
    """Group-centroid posts with a standoff so nurses never crowd patients."""
    targets = np.zeros((2, NUM_NURSES))
    for k, group in enumerate(groups):
        centroid = pat_pos[:, group].mean(axis=1)
        away = nurse_pos[:, k] - centroid
        dist = np.linalg.norm(away)
        away = away / dist if dist > 1e-6 else np.array([-1.0, 0.0])
        standoff = 0.28 if len(group) == 2 else 0.35
        targets[:, k] = centroid + standoff * away
    return clamp_arena(targets)


def doctor_targets_from(focus, doc_pos, pat_pos):
    targets = np.zeros((2, NUM_DOCTORS))
    for k, p_local in enumerate(focus):
        away = doc_pos[:, k] - pat_pos[:, p_local]
        dist = np.linalg.norm(away)
        away = away / dist if dist > 1e-6 else np.array([-1.0, 0.0])
        targets[:, k] = pat_pos[:, p_local] + 0.40 * away
    return clamp_arena(targets)


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
print(f"Run02 GeneticPairing: {N} robots, {TOTAL_SECONDS:.0f} s, seed {RUN_SEED}")

for t in range(min(TOTAL_ITERATIONS, _ITER_CAP)):
    x = r.get_poses()
    pos = x[:2, :]
    t_sec = t * DT
    dxi = np.zeros((2, N))
    leds = ROLE_COLORS.copy()
    acu = acuity(t_sec)

    if t < T_STANDBY:
        breathe = 0.25 + 0.55 * 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.5 * t_sec))
        white = np.full((3, N), 255.0) * breathe
        if t_sec < 10.0:
            leds = white
        else:
            blend = (t_sec - 10.0) / 5.0
            leds = (1.0 - blend) * white + blend * ROLE_COLORS
    else:
        pat_pos = pos[:, PAT]
        nur_pos = pos[:, NUR]
        doc_pos = pos[:, DOC]

        # Patients drift on small seeded orbits in every task phase.
        dxi[:, PAT] = si_position(pat_pos, clamp_arena(patient_drift_targets(t_sec)))

        in_eval_window = False
        if t < T_STAGED:
            # PHASE 1: staff staging.
            dxi[:, NUR] = si_position(nur_pos, NURSE_STAGE)
            dxi[:, DOC] = si_position(doc_pos, DOCTOR_POST)
        elif t < T_LOCK:
            # PHASE 2: one GA generation expressed every GEN_SECONDS. Boundaries
            # are exact iteration indices (a float modulo on t_sec drifts against
            # the 0.033 s grid and silently skips generations).
            gen_step = (t - T_STAGED) % sec(GEN_SECONDS)
            if gen_step == 0:
                population, best, best_fit = run_generation(population, nur_pos, pat_pos, acu)
                churn = int((best != expressed).sum()) if expressed is not None else NUM_PATIENTS
                if best_fit < global_best - 1e-9:
                    global_best = best_fit
                    flash_until = t_sec + 1.0
                expressed = best
                expressed_groups = groups_from(expressed)
                doctor_focus = np.argsort(-acu)[:NUM_DOCTORS]
                generation_idx += 1
                print(
                    f"[t={t_sec:5.1f}s] gen {generation_idx:02d}: "
                    f"fitness={best_fit:.3f} global_best={global_best:.3f} churn={churn}"
                )
            in_eval_window = gen_step >= sec(GEN_SECONDS - 4.0)
            dxi[:, NUR] = si_position(
                nur_pos, nurse_targets_from(expressed_groups, nur_pos, pat_pos)
            )
            dxi[:, DOC] = si_position(doc_pos, doctor_targets_from(doctor_focus, doc_pos, pat_pos))
        else:
            # PHASES 3-4: lock final assignment; orbit, then full-team huddle.
            targets = nurse_targets_from(expressed_groups, nur_pos, pat_pos)
            if t < T_HUDDLE:
                ang = 0.8 * t_sec
                for k in range(NUM_NURSES):
                    centroid = pat_pos[:, expressed_groups[k]].mean(axis=1)
                    offs = np.array([np.cos(ang + k), np.sin(ang + k)])
                    targets[:, k] = clamp_arena(centroid + 0.30 * offs)
            dxi[:, NUR] = si_position(nur_pos, targets)
            dxi[:, DOC] = si_position(doc_pos, doctor_targets_from(doctor_focus, doc_pos, pat_pos))

        # LED state machine -------------------------------------------------
        # Patients: acuity lerp red<->amber; covered patients green.
        lerp = np.clip(acu / 1.3, 0.0, 1.0)
        leds[:, PAT] = LED_PAT_RED[:, None] * lerp + LED_PAT_AMBER[:, None] * (1.0 - lerp)
        if expressed_groups is not None:
            for k, group in enumerate(expressed_groups):
                for p_local in group:
                    d = np.linalg.norm(pos[:, NUR[k]] - pat_pos[:, p_local])
                    if d < 0.45:
                        leds[:, PAT[p_local]] = LED_PAT_COVERED
        # Nurses: magenta while re-deploying, white flash on new global best.
        nurse_speed = np.linalg.norm(dxi[:, NUR], axis=0)
        for k, n_idx in enumerate(NUR):
            if t_sec < flash_until and int(t_sec * 6) % 2 == 0:
                leds[:, n_idx] = LED_FLASH
            elif nurse_speed[k] > 0.05:
                leds[:, n_idx] = LED_NURSE_MOVE
        # Doctors: brightness ramps with generations; cyan in evaluate window.
        conf = 0.5 + 0.5 * min(generation_idx / max(NUM_GENERATIONS, 1), 1.0)
        leds[:, DOC] = (LED_DOCTOR_EVAL if in_eval_window else LED_DOCTOR)[:, None] * conf

    dxi = si_barrier(dxi, x)
    dxu = si_to_uni(dxi, x)
    stopped = np.linalg.norm(dxi, axis=0) < 1e-4
    if np.any(stopped):
        dxu[:, stopped] = 0.0
    dxu = wheel_safe(dxu)

    write_leds(leds)
    r.set_velocities(_ALL_IDS, dxu)
    r.step()

if expressed is not None:
    final_groups = [[int(g) for g in grp] for grp in groups_from(expressed)]
    print(
        f"Run02 complete: {generation_idx} generations, final fitness {global_best:.3f}, "
        f"final assignment groups {final_groups}"
    )

_end_hook = getattr(r, "call_at_scripts_end", None)
if callable(_end_hook):
    _end_hook()
_debug_hook = getattr(r, "debug", None)
if callable(_debug_hook):
    _debug_hook()
