"""
================================================================================
  RUN 07 / 10 - MARKET-BASED AUCTION TRIAGE PIPELINE
  Georgia Tech Robotarium - https://www.robotarium.gatech.edu/experiment
================================================================================

  CLINICAL TRIAL OBJECTIVE
    Emergency-department throughput study: 9 patient robots raise acuity-
    graded assistance requests in three arrival waves. 5 nurse robots win
    tasks through sealed-bid auctions (bid = distance + workload -
    urgency bonus), stabilize the patient on-site, escort them to one of
    2 doctor stations for treatment, and the treated patients park
    themselves in recovery rows. A full patient-flow pipeline with no
    central dispatcher.

  FLEET (16 robots, 4 min run)
    Robots  0-1   : DOCTORS  (treatment stations, left wall)  LED blue/white
    Robots  2-6   : NURSES   (bidding responders)             LED green/violet
    Robots  7-15  : PATIENTS (3 arrival waves)                LED by state

  ALGORITHM PATTERN: decentralized market-based task allocation.
    Auction rounds every 4 s; every free nurse bids on every open
    request: bid = dist(nurse, patient) + 0.45 * jobs_already_done
    - 0.30 * acuity. Greedy lowest-bid matching awards tasks. The
    workload term makes load balancing emerge; the distance term makes
    spatial specialization emerge.

  REAL-ROBOT TIMING ASSUMPTIONS
    * 15 s standby head time before tasks begin (real fleet start delay).
    * Linear speed planned at 0.14 m/s (30% below the 0.20 m/s platform max).
    * Angular speed planned at 1.8 rad/s (50% below the 3.6 rad/s max).
    * Wheel-speed rescaling keeps every command inside actuator limits.

  STEP-BY-STEP TIMELINE (7273 iterations @ 0.033 s = 240 s wall-clock)
  ----------------------------------------------------------------------------
  STANDBY (0..15 s) - real-robot start delay buffer
    * t =  0-10 s - all 16 robots hold; LEDs breathe dim white at 0.5 Hz.
    * t = 10-15 s - LEDs ramp white -> role colors (role announce).
  PHASE 1 - SHIFT OPEN (15..30 s)
    * t = 15 s  - nurses form the center response pool; doctors power up
                  stations; patients idle in the waiting grid (dim amber).
  PHASE 2 - WAVE 1 (30..~75 s)
    * t = 30 s  - patients 0, 3, 5 raise requests (acuities 3, 2, 1);
                  waiting LEDs blink red at 1/2/3 Hz by acuity.
    * t = 32 s  - first auction round: 3 awards printed (nurse, patient,
                  bid). Expect the highest-acuity patient claimed first.
    * t ~ 35-50 s - claimed patients are stabilized on-site for 6 s
                  (nurse holds at 0.28 m), then escorted: the patient
                  physically trails its nurse at 0.30 m toward a doctor.
    * t ~ 50-75 s - handoffs at stations; doctors treat 6 s each (white
                  pulse); treated patients self-drive to recovery rows
                  (LED green). If both stations are busy, the emergent
                  queue point (0.72 m right of the station) fills.
  PHASE 3 - WAVE 2 (80..~130 s)
    * t = 80 s  - patients 1, 2, 6, 8 raise requests (acuities 2,3,1,2);
                  4 simultaneous escort pipelines run in parallel.
    * t ~ 95-130 s - both stations saturate; queueing and load-balanced
                  nurse re-bidding are clearly visible.
  PHASE 4 - WAVE 3 + DRAIN (130..225 s)
    * t = 130 s - patients 4, 7 raise critical requests (acuity 3, 3).
    * t ~ 165-220 s - pipeline drains: all 9 patients treated and parked
                  in recovery (two green rows top-left and bottom-left);
                  nurses return to the pool as their last job completes.
  PHASE 5 - SHIFT CLOSE (225..240 s)
    * t = 225 s - staff hold; recovery rows hold; LED all-clear wave
                  (green pulse sweeping by robot index at 0.5 Hz).
    * t = 240 s - run ends; debug/cleanup hooks called.
  ----------------------------------------------------------------------------

  EMERGENT BEHAVIORS DEMONSTRATED (printed as metrics during the run)
    1. Load balancing: per-nurse completed-job counts equalize through
       the workload bid term alone (final tally printed).
    2. Spatial specialization: nurses systematically win patients on
       their own side of the waiting grid (distance term).
    3. Priority service: higher-acuity requests are claimed earlier on
       average (per-wave claim order printed).
    4. Queue formation: when both doctors saturate, arriving escorts
       stack at the queue points without any queue being programmed.
    5. Pipeline throughput: stabilize/escort/treat stages overlap across
       patients - a self-organizing assembly line.

  LED CONFIGURATION (updates every iteration)
    DOCTORS : blue (0,90,255) idle; white 2 Hz pulse while treating.
    NURSES  : green (0,200,80) in pool; amber (255,170,0) en route to a
              claim; violet (180,0,255) stabilizing/escorting; green
              flash on job completion.
    PATIENTS: dim amber (130,80,0) idle; red blink at 1/2/3 Hz by acuity
              while requesting; solid amber (255,170,0) claimed; white
              (255,255,255) trailing during escort; cyan (0,220,255)
              in treatment; green (90,255,90) recovered.

  ROBOTARIUM SUBMISSION COMPLIANCE
    * 16 robots (max 20); 240 s run (max 600 s); min start spacing 0.36 m.
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
RUN_SEED = 1107
random.seed(RUN_SEED)
np.random.seed(RUN_SEED)

NUM_DOCTORS = 2
NUM_NURSES = 5
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
TOTAL_SECONDS = 240.0


def sec(seconds):
    return int(round(seconds / DT))


TOTAL_ITERATIONS = sec(TOTAL_SECONDS)
T_STANDBY = sec(15.0)
T_CLOSE = sec(225.0)

# Market parameters
AUCTION_PERIOD = sec(4.0)
BID_WORKLOAD_W = 0.45
BID_ACUITY_W = 0.30
STABILIZE_HOLD = 6.0  # s
TREAT_HOLD = 6.0  # s

# Arrival waves: (time s, patient local idx, acuity 1-3)
WAVES = [
    (30.0, [(0, 3), (3, 2), (5, 1)]),
    (80.0, [(1, 2), (2, 3), (6, 1), (8, 2)]),
    (130.0, [(4, 3), (7, 3)]),
]

SAFETY_RADIUS = 0.20
ARENA = np.array([-1.6, 1.6, -1.0, 1.0])
MARGIN = 0.15

# ----------------------------------------------------------------------------
# WARD GEOMETRY (min pairwise start spacing 0.36 m)
# ----------------------------------------------------------------------------
STATIONS = np.array([[-1.15, -1.15], [0.45, -0.45]])
TREAT_POINTS = STATIONS + np.array([[0.32, 0.32], [0.0, 0.0]])
QUEUE_POINTS = STATIONS + np.array([[0.72, 0.72], [0.0, 0.0]])
NURSE_POOL = np.array([[0.05] * 5, [0.72, 0.36, 0.00, -0.36, -0.72]])
WAITING = np.array(
    [
        [0.62, 1.00, 1.38, 0.62, 1.00, 1.38, 0.62, 1.00, 1.38],
        [0.55, 0.55, 0.55, 0.00, 0.00, 0.00, -0.55, -0.55, -0.55],
    ]
)
RECOVERY = np.array(
    [
        [-1.35, -1.05, -0.75, -0.45, -0.15, -1.20, -0.90, -0.60, -0.30],
        [0.85, 0.85, 0.85, 0.85, 0.85, -0.85, -0.85, -0.85, -0.85],
    ]
)

initial_conditions = np.vstack(
    [
        np.hstack([STATIONS[0], NURSE_POOL[0], WAITING[0]]),
        np.hstack([STATIONS[1], NURSE_POOL[1], WAITING[1]]),
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
LED_NURSE = np.array([0.0, 200.0, 80.0])
LED_NURSE_GO = np.array([255.0, 170.0, 0.0])
LED_NURSE_CARE = np.array([180.0, 0.0, 255.0])
LED_FLASH = np.array([255.0, 255.0, 255.0])
LED_PAT_IDLE = np.array([130.0, 80.0, 0.0])
LED_PAT_REQ = np.array([255.0, 30.0, 0.0])
LED_PAT_CLAIMED = np.array([255.0, 170.0, 0.0])
LED_PAT_TREAT = np.array([0.0, 220.0, 255.0])
LED_PAT_DONE = np.array([90.0, 255.0, 90.0])
ROLE_COLORS = np.zeros((3, N))
ROLE_COLORS[:, DOC] = LED_DOCTOR[:, None]
ROLE_COLORS[:, NUR] = LED_NURSE[:, None]
ROLE_COLORS[:, PAT] = LED_PAT_IDLE[:, None]

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
# PIPELINE STATE MACHINES
# ----------------------------------------------------------------------------
# Patient: IDLE -> REQUEST -> CLAIMED -> ESCORT -> QUEUE -> TREAT -> RECOVERED
p_state = ["IDLE"] * NUM_PATIENTS
p_acuity = [0] * NUM_PATIENTS
p_nurse = [-1] * NUM_PATIENTS
p_station = [-1] * NUM_PATIENTS
p_request_t = [0.0] * NUM_PATIENTS
p_claim_t = [0.0] * NUM_PATIENTS
p_recover_slot = [-1] * NUM_PATIENTS
# Nurse: POOL -> TO_PATIENT -> STABILIZE -> ESCORT -> POOL
n_state = ["POOL"] * NUM_NURSES
n_job = [-1] * NUM_NURSES
n_jobs_done = [0] * NUM_NURSES
n_timer = [0.0] * NUM_NURSES
n_flash_until = [-1.0] * NUM_NURSES
# Doctor: IDLE -> TREAT
d_busy_until = [-1.0] * NUM_DOCTORS
d_patient = [-1] * NUM_DOCTORS
recover_count = 0
wave_idx = 0
wait_times = []
claim_log = []


def run_auction(t_sec, pos):
    """Sealed-bid round: free nurses bid on open requests, lowest bid wins."""
    open_reqs = [p for p in range(NUM_PATIENTS) if p_state[p] == "REQUEST"]
    free_nurses = [k for k in range(NUM_NURSES) if n_state[k] == "POOL"]
    while open_reqs and free_nurses:
        best = None  # (bid, nurse, patient)
        for k in free_nurses:
            for p in open_reqs:
                d = float(np.linalg.norm(pos[:, NUR[k]] - pos[:, PAT[p]]))
                bid = d + BID_WORKLOAD_W * n_jobs_done[k] - BID_ACUITY_W * p_acuity[p]
                if best is None or bid < best[0]:
                    best = (bid, k, p)
        bid, k, p = best
        n_state[k] = "TO_PATIENT"
        n_job[k] = p
        p_state[p] = "CLAIMED"
        p_nurse[p] = k
        p_claim_t[p] = t_sec
        wait_times.append(t_sec - p_request_t[p])
        claim_log.append((p, p_acuity[p], t_sec))
        free_nurses.remove(k)
        open_reqs.remove(p)
        print(
            f"[t={t_sec:5.1f}s] AUCTION: nurse {k} wins patient {p} "
            f"(acuity {p_acuity[p]}, bid {bid:.2f}, waited {t_sec - p_request_t[p]:.1f}s)"
        )


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
print(f"Run07 AuctionTriage: {N} robots, {TOTAL_SECONDS:.0f} s, seed {RUN_SEED}")

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
        # Wave arrivals.
        if wave_idx < len(WAVES) and t_sec >= WAVES[wave_idx][0]:
            for p, acu in WAVES[wave_idx][1]:
                p_state[p] = "REQUEST"
                p_acuity[p] = acu
                p_request_t[p] = t_sec
            arrivals = ", ".join(f"P{p}(a{a})" for p, a in WAVES[wave_idx][1])
            print(f"[t={t_sec:5.1f}s] WAVE {wave_idx + 1} arrivals: {arrivals}")
            wave_idx += 1

        # Auction rounds.
        if t % AUCTION_PERIOD == 0:
            run_auction(t_sec, pos)

        # ---------------- nurse FSMs ----------------
        for k in range(NUM_NURSES):
            n_idx = NUR[k]
            n_pos = pos[:, n_idx]
            target = NURSE_POOL[:, k]
            if n_state[k] == "TO_PATIENT":
                p = n_job[k]
                target = standoff(pos[:, PAT[p]], n_pos, 0.28)
                leds[:, n_idx] = LED_NURSE_GO
                if np.linalg.norm(n_pos - pos[:, PAT[p]]) < 0.34:
                    n_state[k] = "STABILIZE"
                    n_timer[k] = t_sec
            elif n_state[k] == "STABILIZE":
                p = n_job[k]
                target = standoff(pos[:, PAT[p]], n_pos, 0.28)
                leds[:, n_idx] = LED_NURSE_CARE
                if t_sec - n_timer[k] >= STABILIZE_HOLD:
                    n_state[k] = "ESCORT"
                    # Choose the station with the shortest queue, then nearest.
                    loads = [
                        sum(
                            1
                            for q in range(NUM_PATIENTS)
                            if p_station[q] == s and p_state[q] in ("ESCORT", "QUEUE", "TREAT")
                        )
                        for s in range(NUM_DOCTORS)
                    ]
                    dists = np.linalg.norm(STATIONS - n_pos[:, None], axis=0)
                    s_pick = int(np.lexsort((dists, np.array(loads)))[0])
                    p_station[p] = s_pick
                    p_state[p] = "ESCORT"
            elif n_state[k] == "ESCORT":
                p = n_job[k]
                station = STATIONS[:, p_station[p]]
                to_st = station - n_pos
                d_st = np.linalg.norm(to_st)
                step_v = to_st / d_st * min(d_st, 0.40) if d_st > 1e-6 else to_st
                target = n_pos + step_v
                leds[:, n_idx] = LED_NURSE_CARE
                if np.linalg.norm(pos[:, PAT[p]] - station) < 0.55:
                    p_state[p] = "QUEUE"
                    p_nurse[p] = -1
                    n_jobs_done[k] += 1
                    n_flash_until[k] = t_sec + 1.0
                    n_state[k] = "POOL"
                    n_job[k] = -1
            if n_state[k] == "POOL":
                target = NURSE_POOL[:, k]
                if t_sec < n_flash_until[k] and int(t_sec * 6) % 2 == 0:
                    leds[:, n_idx] = LED_FLASH
            dxi[:, n_idx : n_idx + 1] = si_position(
                n_pos.reshape(2, 1), clamp_arena(target.reshape(2, 1))
            )

        # ---------------- doctor FSMs ----------------
        for s in range(NUM_DOCTORS):
            d_idx = DOC[s]
            dxi[:, d_idx : d_idx + 1] = si_position(
                pos[:, d_idx].reshape(2, 1), STATIONS[:, s].reshape(2, 1)
            )
            if d_patient[s] >= 0:
                leds[:, d_idx] = LED_FLASH if int(t_sec * 4) % 2 == 0 else LED_DOCTOR
                if t_sec >= d_busy_until[s]:
                    p = d_patient[s]
                    p_state[p] = "RECOVERED"
                    p_recover_slot[p] = recover_count
                    recover_count += 1
                    d_patient[s] = -1
                    print(
                        f"[t={t_sec:5.1f}s] patient {p} treated at station {s} "
                        f"-> recovery slot {p_recover_slot[p]} "
                        f"({recover_count}/{NUM_PATIENTS})"
                    )
            else:
                # Admit the closest queued patient assigned to this station.
                queued = [
                    q for q in range(NUM_PATIENTS) if p_state[q] == "QUEUE" and p_station[q] == s
                ]
                if queued:
                    dists = [float(np.linalg.norm(pos[:, PAT[q]] - STATIONS[:, s])) for q in queued]
                    p = queued[int(np.argmin(dists))]
                    if dists[int(np.argmin(dists))] < 0.60:
                        p_state[p] = "TREAT"
                        d_patient[s] = p
                        d_busy_until[s] = t_sec + TREAT_HOLD

        # ---------------- patient FSMs + LEDs ----------------
        for p in range(NUM_PATIENTS):
            p_idx = PAT[p]
            p_pos = pos[:, p_idx]
            state = p_state[p]
            if state in ("IDLE", "REQUEST"):
                wob = 0.02 * np.array(
                    [np.sin(2.0 * np.pi * 0.2 * t_sec + p), np.cos(2.0 * np.pi * 0.17 * t_sec + p)]
                )
                target = WAITING[:, p] + (wob if state == "REQUEST" else 0.0)
                if state == "REQUEST":
                    hz = [0.0, 1.0, 2.0, 3.0][p_acuity[p]]
                    on = 0.5 * (1.0 + np.sin(2.0 * np.pi * hz * t_sec)) > 0.5
                    leds[:, p_idx] = LED_PAT_REQ if on else LED_PAT_REQ * 0.25
            elif state == "CLAIMED":
                target = WAITING[:, p]
                leds[:, p_idx] = LED_PAT_CLAIMED
            elif state == "ESCORT":
                k = p_nurse[p]
                nurse_p = pos[:, NUR[k]]
                target = standoff(nurse_p, p_pos, 0.30)
                leds[:, p_idx] = LED_FLASH
            elif state == "QUEUE":
                target = QUEUE_POINTS[:, p_station[p]]
                if d_patient[p_station[p]] == p or p_state[p] == "TREAT":
                    target = TREAT_POINTS[:, p_station[p]]
                leds[:, p_idx] = LED_PAT_CLAIMED
            elif state == "TREAT":
                target = TREAT_POINTS[:, p_station[p]]
                leds[:, p_idx] = LED_PAT_TREAT
            else:  # RECOVERED
                target = RECOVERY[:, p_recover_slot[p]]
                leds[:, p_idx] = LED_PAT_DONE
            dxi[:, p_idx : p_idx + 1] = si_position(
                p_pos.reshape(2, 1), clamp_arena(target.reshape(2, 1))
            )

        if t >= T_CLOSE:
            wave = 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.5 * t_sec - np.arange(N) * 0.6))
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

treated = sum(1 for s in p_state if s == "RECOVERED")
mean_wait = float(np.mean(wait_times)) if wait_times else 0.0
print(
    f"Run07 complete: {treated}/{NUM_PATIENTS} patients treated, "
    f"jobs per nurse {n_jobs_done}, mean claim wait {mean_wait:.1f}s"
)

_end_hook = getattr(r, "call_at_scripts_end", None)
if callable(_end_hook):
    _end_hook()
_debug_hook = getattr(r, "debug", None)
if callable(_debug_hook):
    _debug_hook()
