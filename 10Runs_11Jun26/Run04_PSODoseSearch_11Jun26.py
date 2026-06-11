"""
================================================================================
  RUN 04 / 10 - PARTICLE SWARM OPTIMIZATION DOSE-RESPONSE SEARCH
  Georgia Tech Robotarium - https://www.robotarium.gatech.edu/experiment
================================================================================

  CLINICAL TRIAL OBJECTIVE
    Dose-response mapping: the arena floor encodes a hidden treatment
    efficacy field (a strong Gaussian optimum plus a weaker decoy peak).
    5 patient robots and 3 nurse robots act as embodied PSO particles
    that can only sample the field at their own position; 2 doctor
    robots monitor the swarm's global-best estimate. The swarm must
    find the true optimum and avoid locking onto the decoy.

  FLEET (10 robots, 2.5 min run)
    Robots  0-1   : DOCTORS  (global-best monitors)          LED blue
    Robots  2-4   : NURSES   (coordinator particles)         LED green
    Robots  5-9   : PATIENTS (trial-cohort particles)        LED red/amber

  ALGORITHM PATTERN: embodied particle swarm optimization. The robot
    positions ARE the particle states: v <- w*v + c1*r1*(pbest - x)
    + c2*r2*(gbest - x), inertia w annealed 0.9 -> 0.4, c1 = 1.6,
    c2 = 1.4 (patients) / c2 = 2.0 (nurses follow the social term more).
    Velocity updates every 2 s; samples taken from live robot positions.

  HIDDEN FIELD (unknown to the swarm)
    efficacy(p) = 1.00 * exp(-||p - (0.80, -0.25)||^2 / 0.20)   true peak
                + 0.68 * exp(-||p - (-0.85, 0.60)||^2 / 0.20)   decoy peak
    The broad decoy sits beside two lattice posts, so the swarm's FIRST
    gbest lands in the decoy basin and must be escaped.

  REAL-ROBOT TIMING ASSUMPTIONS
    * 15 s standby head time before tasks begin (real fleet start delay).
    * Linear speed planned at 0.14 m/s (30% below the 0.20 m/s platform max).
    * Angular speed planned at 1.8 rad/s (50% below the 3.6 rad/s max).
    * Wheel-speed rescaling keeps every command inside actuator limits.

  STEP-BY-STEP TIMELINE (4545 iterations @ 0.033 s = 150 s wall-clock)
  ----------------------------------------------------------------------------
  STANDBY (0..15 s) - real-robot start delay buffer
    * t =  0-10 s - all 10 robots hold; LEDs breathe dim white at 0.5 Hz.
    * t = 10-15 s - LEDs ramp white -> role colors (role announce).
  PHASE 1 - SEARCH LATTICE (15..25 s)
    * t = 15 s  - the 8 particles fan out to a seeded lattice covering the
                  arena; doctors take posts near the center line.
    * t = 25 s  - lattice reached; first field samples taken. The broad
                  decoy beside the two western posts wins the opening
                  samples: the initial gbest lands in the DECOY basin
                  (basin print).
  PHASE 2 - PSO SEARCH (25..105 s; velocity update every 2 s, 40 steps)
    * t ~ 25-30 s - DECOY ESCAPE: a particle whose personal best anchors
                  it near the true basin samples above the decoy ceiling
                  and the gbest jumps basins (second basin print).
    * t ~ 30-60 s - high-inertia exploration: western particles peel off
                  the decoy as the social term drags them across the
                  arena; gbest climbs 0.86 -> 1.00.
    * t = 60-105 s - inertia anneals; swarm contracts around the true
                  optimum; gbest efficacy printed every 10 s.
  PHASE 3 - SITE VERIFICATION (105..130 s)
    * t = 105 s - doctors travel to flank the gbest site at +/-0.35 m
                  (LED cyan while verifying).
    * t = 120 s - doctors hold the site; particles form a 0.45 m survey
                  ring around gbest.
    * t = 130 s - verification complete (doctor LED solid green).
  PHASE 4 - COHORT REPORT (130..150 s)
    * t = 130 s - ring holds; LED report: particles glow by their final
                  personal-best efficacy (dim red = poor, green = at
                  optimum), broadcasting the dose-response map.
    * t = 150 s - run ends; debug/cleanup hooks called.
  ----------------------------------------------------------------------------

  EMERGENT BEHAVIORS DEMONSTRATED (printed as metrics during the run)
    1. Stigmergy-free collective search: only 2 scalars (pbest, gbest)
       coordinate 8 robots into a converging search party.
    2. Decoy escape: particles initially trapped on the weaker peak are
       extracted by the social term once gbest crosses basins.
    3. Exploration -> exploitation: swarm dispersion (mean distance to
       swarm centroid) decays with annealed inertia - printed every 10 s.
    4. Role asymmetry: higher-social nurses converge first and 'anchor'
       the find; patients sweep wider before joining (visible spread).

  LED CONFIGURATION (updates every iteration)
    DOCTORS : blue (0,90,255) monitoring; cyan (0,220,255) verifying the
              site; solid green (60,255,120) once verified.
    NURSES  : green (0,200,80); white flash for 1 s when the swarm gbest
              improves; brightness rises with own pbest efficacy.
    PATIENTS: continuous red(255,30,0) -> amber(255,170,0) -> green
              (90,255,90) colormap of their CURRENT sampled efficacy;
              the gbest-holding particle blinks white at 2 Hz.

  ROBOTARIUM SUBMISSION COMPLIANCE
    * 10 robots (max 20); 150 s run (max 600 s); min start spacing 0.36 m.
    * get_poses() exactly once per step(); ends with the platform hook.
    * Barrier certificates keep inter-robot distance >= 0.23 m and robots
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
RUN_SEED = 1104
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)

NUM_DOCTORS = 2
NUM_NURSES = 3
NUM_PATIENTS = 5
N = NUM_DOCTORS + NUM_NURSES + NUM_PATIENTS  # 10
DOC = np.arange(0, NUM_DOCTORS)
NUR = np.arange(NUM_DOCTORS, NUM_DOCTORS + NUM_NURSES)
PAT = np.arange(NUM_DOCTORS + NUM_NURSES, N)
PARTICLES = np.concatenate([NUR, PAT])  # 8 embodied PSO particles
P = len(PARTICLES)

HW_MAX_LINEAR = 0.20
ROBOT_BASE_LENGTH = 0.11
PLAN_LINEAR = 0.14
PLAN_ANGULAR = 1.8

DT = 0.033
TOTAL_SECONDS = 150.0


def sec(seconds):
    return int(round(seconds / DT))


TOTAL_ITERATIONS = sec(TOTAL_SECONDS)
T_STANDBY = sec(15.0)
T_LATTICE = sec(25.0)
T_VERIFY = sec(105.0)
T_REPORT = sec(130.0)
PSO_ITERS = sec(2.0)  # one velocity update every 2 s

# PSO parameters
PSO_C1 = 1.6
PSO_C2_PATIENT = 1.4
PSO_C2_NURSE = 2.0
INERTIA_HI = 0.9
INERTIA_LO = 0.4
PSO_VMAX = 0.55  # m per update step (pre-scaling), caps particle jumps

# Hidden efficacy field: a narrow true peak set BETWEEN lattice posts and a
# broad decoy peak right next to two posts, so the swarm's first samples favor
# the decoy and the true optimum must be discovered by escaping it.
TRUE_PEAK = np.array([0.80, -0.25])
DECOY_PEAK = np.array([-0.85, 0.60])
TRUE_W = 0.20
DECOY_W = 0.20
DECOY_H = 0.68

SAFETY_RADIUS = 0.23  # extra margin: PSO jumps produce close crossings
ARENA = np.array([-1.6, 1.6, -1.0, 1.0])
MARGIN = 0.15

# ----------------------------------------------------------------------------
# INITIAL CONDITIONS (min pairwise spacing 0.36 m)
# ----------------------------------------------------------------------------
doctor_xy = np.array([[-0.20, 0.20], [0.90, 0.90]])
nurse_xy = np.array([[-1.20, -1.20, -1.20], [0.45, 0.00, -0.45]])
patient_xy = np.array([[-0.60, -0.20, 0.20, 0.60, 1.00], [-0.85, -0.85, -0.85, -0.85, -0.85]])
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

# Search lattice: 8 seeded posts covering the arena (one per particle).
LATTICE = np.array(
    [
        [-1.25, -0.45, 0.45, 1.25, -1.25, -0.45, 0.45, 1.25],
        [0.60, 0.60, 0.60, 0.60, -0.60, -0.60, -0.60, -0.60],
    ]
)
DOCTOR_POST = np.array([[-0.25, 0.25], [0.05, 0.05]])


def efficacy(points):
    """Hidden dose-response field sampled at 2xK positions (values 0..~1)."""
    pts = points.reshape(2, -1)
    d_true = ((pts - TRUE_PEAK[:, None]) ** 2).sum(axis=0)
    d_decoy = ((pts - DECOY_PEAK[:, None]) ** 2).sum(axis=0)
    return 1.00 * np.exp(-d_true / TRUE_W) + DECOY_H * np.exp(-d_decoy / DECOY_W)


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
LED_DOC_VERIFY = np.array([0.0, 220.0, 255.0])
LED_DOC_DONE = np.array([60.0, 255.0, 120.0])
LED_NURSE = np.array([0.0, 200.0, 80.0])
LED_FLASH = np.array([255.0, 255.0, 255.0])
LED_LOW = np.array([255.0, 30.0, 0.0])
LED_MID = np.array([255.0, 170.0, 0.0])
LED_HIGH = np.array([90.0, 255.0, 90.0])
ROLE_COLORS = np.zeros((3, N))
ROLE_COLORS[:, DOC] = LED_DOCTOR[:, None]
ROLE_COLORS[:, NUR] = LED_NURSE[:, None]
ROLE_COLORS[:, PAT] = LED_MID[:, None]


def efficacy_color(values):
    """Map efficacy 0..1 to red -> amber -> green (3xK)."""
    v = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    out = np.zeros((3, v.size))
    lo = v < 0.5
    blend_lo = v[lo] / 0.5
    out[:, lo] = LED_LOW[:, None] * (1.0 - blend_lo) + LED_MID[:, None] * blend_lo
    hi = ~lo
    blend_hi = (v[hi] - 0.5) / 0.5
    out[:, hi] = LED_MID[:, None] * (1.0 - blend_hi) + LED_HIGH[:, None] * blend_hi
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
# PSO STATE
# ----------------------------------------------------------------------------
pso_vel = np.zeros((2, P))
pbest_pos = LATTICE.copy()
pbest_val = np.full(P, -np.inf)
gbest_pos = LATTICE[:, 0].copy()
gbest_val = -np.inf
gbest_holder = -1  # particle index currently holding gbest
gbest_basin = None  # 'true' or 'decoy' (for basin-jump logging)
flash_until = -1.0
particle_targets = LATTICE.copy()
pso_step_count = 0
C2 = np.array([PSO_C2_NURSE] * NUM_NURSES + [PSO_C2_PATIENT] * NUM_PATIENTS)


def pso_update(particle_pos, t_sec):
    """One PSO velocity/target update from live sampled efficacies."""
    global pso_vel, pbest_pos, pbest_val, gbest_pos, gbest_val
    global gbest_holder, gbest_basin, flash_until, pso_step_count
    vals = efficacy(particle_pos)
    improved = vals > pbest_val
    pbest_val[improved] = vals[improved]
    pbest_pos[:, improved] = particle_pos[:, improved]
    k = int(np.argmax(pbest_val))
    if pbest_val[k] > gbest_val + 1e-12:
        gbest_val = float(pbest_val[k])
        gbest_pos = pbest_pos[:, k].copy()
        gbest_holder = k
        flash_until = t_sec + 1.0
        basin = (
            "true"
            if np.linalg.norm(gbest_pos - TRUE_PEAK) < np.linalg.norm(gbest_pos - DECOY_PEAK)
            else "decoy"
        )
        if basin != gbest_basin:
            print(f"[t={t_sec:5.1f}s] gbest basin -> {basin} peak (efficacy {gbest_val:.3f})")
            gbest_basin = basin
    progress = min(max(t_sec - 25.0, 0.0) / 80.0, 1.0)
    inertia = INERTIA_HI + (INERTIA_LO - INERTIA_HI) * progress
    r1 = np.random.rand(2, P)
    r2 = np.random.rand(2, P)
    pso_vel = (
        inertia * pso_vel
        + PSO_C1 * r1 * (pbest_pos - particle_pos)
        + C2[None, :] * r2 * (gbest_pos[:, None] - particle_pos)
    )
    speed = np.linalg.norm(pso_vel, axis=0)
    fast = speed > PSO_VMAX
    if np.any(fast):
        pso_vel[:, fast] *= PSO_VMAX / speed[fast]
    pso_step_count += 1
    return clamp_arena(particle_pos + pso_vel)


def swarm_dispersion(particle_pos):
    centroid = particle_pos.mean(axis=1, keepdims=True)
    return float(np.linalg.norm(particle_pos - centroid, axis=0).mean())


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
print(f"Run04 PSODoseSearch: {N} robots, {TOTAL_SECONDS:.0f} s, seed {RUN_SEED}")
report_marks = {sec(s) for s in (35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 105.0)}

for t in range(min(TOTAL_ITERATIONS, _ITER_CAP)):
    x = r.get_poses()
    pos = x[:2, :]
    t_sec = t * DT
    dxi = np.zeros((2, N))
    leds = ROLE_COLORS.copy()
    particle_pos = pos[:, PARTICLES]

    if t < T_STANDBY:
        breathe = 0.25 + 0.55 * 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.5 * t_sec))
        white = np.full((3, N), 255.0) * breathe
        leds = (
            white
            if t_sec < 10.0
            else (1.0 - (t_sec - 10.0) / 5.0) * white + ((t_sec - 10.0) / 5.0) * ROLE_COLORS
        )
    else:
        if t < T_LATTICE:
            # PHASE 1: fan out to the search lattice.
            particle_targets = LATTICE
            dxi[:, PARTICLES] = si_position(particle_pos, particle_targets)
            dxi[:, DOC] = si_position(pos[:, DOC], DOCTOR_POST)
        elif t < T_VERIFY:
            # PHASE 2: embodied PSO, velocity update every 2 s.
            if (t - T_LATTICE) % PSO_ITERS == 0:
                particle_targets = pso_update(particle_pos, t_sec)
                if pso_step_count == 1:
                    print(
                        f"[t={t_sec:5.1f}s] first samples: gbest efficacy {gbest_val:.3f} "
                        f"held by particle {gbest_holder}"
                    )
            dxi[:, PARTICLES] = si_position(particle_pos, particle_targets)
            dxi[:, DOC] = si_position(pos[:, DOC], DOCTOR_POST)
            if t in report_marks:
                print(
                    f"[t={t_sec:5.1f}s] gbest={gbest_val:.3f} "
                    f"dispersion={swarm_dispersion(particle_pos):.2f} m "
                    f"(step {pso_step_count})"
                )
        else:
            # PHASES 3-4: doctors verify the found site; particles ring it.
            flank = np.column_stack(
                [gbest_pos + np.array([-0.35, 0.0]), gbest_pos + np.array([0.35, 0.0])]
            )
            dxi[:, DOC] = si_position(pos[:, DOC], clamp_arena(flank))
            ring_angles = np.linspace(0.0, 2.0 * np.pi, P, endpoint=False) + np.pi / 8.0
            ring = gbest_pos[:, None] + 0.45 * np.vstack([np.cos(ring_angles), np.sin(ring_angles)])
            dxi[:, PARTICLES] = si_position(particle_pos, clamp_arena(ring))
            doc_close = np.linalg.norm(pos[:, DOC] - clamp_arena(flank), axis=0).max() < 0.10
            leds[:, DOC] = (LED_DOC_DONE if (t >= T_REPORT or doc_close) else LED_DOC_VERIFY)[
                :, None
            ]

        # LED state machine -------------------------------------------------
        live_vals = efficacy(particle_pos)
        if t >= T_REPORT:
            leds[:, PARTICLES] = efficacy_color(pbest_val)  # final report colors
        else:
            leds[:, PAT] = efficacy_color(live_vals[NUM_NURSES:])
            bright = 0.55 + 0.45 * np.clip(pbest_val[:NUM_NURSES], 0.0, 1.0)
            leds[:, NUR] = LED_NURSE[:, None] * bright[None, :]
        if t_sec < flash_until and int(t_sec * 6) % 2 == 0:
            leds[:, NUR] = LED_FLASH[:, None]
        if gbest_holder >= 0 and int(t_sec * 4) % 2 == 0 and t < T_REPORT:
            leds[:, PARTICLES[gbest_holder]] = LED_FLASH

    dxi = si_barrier(dxi, x)
    dxu = si_to_uni(dxi, x)
    stopped = np.linalg.norm(dxi, axis=0) < 1e-4
    if np.any(stopped):
        dxu[:, stopped] = 0.0
    dxu = wheel_safe(dxu)

    write_leds(leds)
    r.set_velocities(_ALL_IDS, dxu)
    r.step()

err = float(np.linalg.norm(gbest_pos - TRUE_PEAK))
print(
    f"Run04 complete: {pso_step_count} PSO steps, gbest efficacy {gbest_val:.3f}, "
    f"distance to true optimum {err:.3f} m ({'true' if err < 0.3 else 'decoy'} basin)"
)

_end_hook = getattr(r, "call_at_scripts_end", None)
if callable(_end_hook):
    _end_hook()
_debug_hook = getattr(r, "debug", None)
if callable(_debug_hook):
    _debug_hook()
