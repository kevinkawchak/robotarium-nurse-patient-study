"""
Clinical Trial Nurse-Patient Robotarium Experiment
===================================================
Title: Trial_01_29Jan26
Duration: 60 seconds
Robots: 2 (Nurse and Patient)

This experiment simulates a clinical trial scenario where:
- Robot 0 (Nurse): Demonstrates careful navigation with obstacle avoidance,
  approaches the patient, pauses for interaction, then returns.
- Robot 1 (Patient): Exhibits dynamic behavior with varied movements that
  change based on environmental conditions.

Author: Generated for Robotarium Experiment Submission
Date: January 29, 2026
"""

import numpy as np

import rps.robotarium as robotarium
from rps.utilities import barrier_certificates as bc
from rps.utilities import controllers as ctl
from rps.utilities import transformations as tr

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

# Number of robots
N = 2

# Robot indices
NURSE = 0
PATIENT = 1

# Robotarium boundaries (approximate)
X_MIN, X_MAX = -0.6, 0.6
Y_MIN, Y_MAX = -0.35, 0.35

# Duration: 60 seconds at ~30 Hz = ~1800 iterations
EXPERIMENT_DURATION = 60  # seconds
ITERATION_RATE = 30  # Hz (approximate)
TOTAL_ITERATIONS = EXPERIMENT_DURATION * ITERATION_RATE

# Control parameters
VELOCITY_MAGNITUDE_LIMIT = 0.15  # Max robot speed (m/s)
CLOSE_ENOUGH = 0.05  # Distance threshold for waypoint arrival

# ============================================================================
# INITIALIZE ROBOTARIUM
# ============================================================================

# Initial positions: [x; y; theta] for each robot
# Nurse starts on the left, Patient starts on the right
initial_conditions = np.array(
    [
        [-0.4, 0.4],  # x positions
        [0.0, 0.0],  # y positions
        [0.0, np.pi],  # theta (orientations)
    ]
)

# Create Robotarium instance
r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    initial_conditions=initial_conditions,
    sim_in_real_time=True,
)

# ============================================================================
# SAFETY AND CONTROL UTILITIES
# ============================================================================

# Create barrier certificate for collision avoidance (robots and boundaries)
si_barrier_cert = bc.create_single_integrator_barrier_certificate_with_boundary()

# Create single-integrator position controller
si_position_controller = ctl.create_si_position_controller(
    x_velocity_gain=1.0, y_velocity_gain=1.0, velocity_magnitude_limit=VELOCITY_MAGNITUDE_LIMIT
)

# Create transformation from single-integrator to unicycle dynamics
si_to_uni_dyn = tr.create_si_to_uni_dynamics()

# ============================================================================
# WAYPOINT DEFINITIONS
# ============================================================================

# Define waypoints for the nurse's careful navigation path
# The nurse will navigate carefully, approach the patient, pause, then return
nurse_waypoints = np.array(
    [
        [-0.4, -0.2, 0.0, 0.2, 0.3, 0.2, 0.0, -0.2, -0.4],  # x coordinates
        [0.0, 0.1, 0.15, 0.1, 0.0, -0.1, -0.15, -0.1, 0.0],  # y coordinates
    ]
)

# Define waypoints for the patient's dynamic movement pattern
# The patient exhibits varied, environmentally-responsive movements
patient_waypoints = np.array(
    [
        [0.4, 0.35, 0.45, 0.3, 0.5, 0.35, 0.4],  # x coordinates
        [0.0, 0.15, -0.1, -0.2, 0.1, 0.2, 0.0],  # y coordinates
    ]
)

# ============================================================================
# STATE VARIABLES
# ============================================================================

# Current waypoint indices for each robot
nurse_waypoint_idx = 0
patient_waypoint_idx = 0

# Pause counters (for patient interaction simulation)
nurse_pause_counter = 0
NURSE_PAUSE_DURATION = 90  # iterations (~3 seconds pause at patient)

# Patient behavior state
patient_behavior_state = 0  # 0: normal, 1: responsive, 2: active

# Iteration counter
iteration = 0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return np.linalg.norm(pos1 - pos2)


def get_nurse_target(poses, waypoint_idx, pause_counter):
    """
    Get the current target for the nurse robot.
    Implements careful navigation with pausing behavior.
    """
    global nurse_pause_counter, nurse_waypoint_idx

    current_pos = poses[:2, NURSE]
    target = nurse_waypoints[:, waypoint_idx]

    # Check if reached current waypoint
    if get_distance(current_pos, target) < CLOSE_ENOUGH:
        # Special pause at waypoint 4 (near patient)
        if waypoint_idx == 4 and pause_counter < NURSE_PAUSE_DURATION:
            nurse_pause_counter += 1
            return target, waypoint_idx  # Stay at current position

        # Move to next waypoint
        if pause_counter >= NURSE_PAUSE_DURATION or waypoint_idx != 4:
            nurse_pause_counter = 0
            next_idx = (waypoint_idx + 1) % nurse_waypoints.shape[1]
            return nurse_waypoints[:, next_idx], next_idx

    return target, waypoint_idx


def get_patient_target(poses, waypoint_idx, nurse_pos, iteration):
    """
    Get the current target for the patient robot.
    Implements dynamic, environmentally-responsive behavior.
    """
    global patient_behavior_state

    current_pos = poses[:2, PATIENT]

    # Determine behavior state based on nurse proximity
    nurse_distance = get_distance(current_pos, nurse_pos)

    if nurse_distance < 0.25:
        # Nurse is close - patient becomes responsive (smaller movements)
        patient_behavior_state = 1
    elif nurse_distance < 0.4:
        # Nurse approaching - patient shows awareness
        patient_behavior_state = 2
    else:
        # Normal behavior
        patient_behavior_state = 0

    target = patient_waypoints[:, waypoint_idx]

    # Add dynamic variation based on behavior state
    variation = np.array([0.0, 0.0])
    if patient_behavior_state == 1:
        # Subtle movements when nurse is close
        variation = 0.02 * np.array([np.sin(iteration * 0.1), np.cos(iteration * 0.1)])
    elif patient_behavior_state == 2:
        # Medium activity when nurse approaching
        variation = 0.05 * np.array([np.sin(iteration * 0.15), np.cos(iteration * 0.2)])
    else:
        # Normal dynamic movement
        variation = 0.03 * np.array([np.sin(iteration * 0.08), np.cos(iteration * 0.12)])

    # Apply variation to target
    modified_target = target + variation

    # Clamp to boundaries with margin
    margin = 0.1
    modified_target[0] = np.clip(modified_target[0], X_MIN + margin, X_MAX - margin)
    modified_target[1] = np.clip(modified_target[1], Y_MIN + margin, Y_MAX - margin)

    # Check if reached current waypoint
    if get_distance(current_pos, target) < CLOSE_ENOUGH:
        # Move to next waypoint (with some randomness for dynamic behavior)
        if np.random.random() < 0.3:  # 30% chance to skip a waypoint
            next_idx = (waypoint_idx + 2) % patient_waypoints.shape[1]
        else:
            next_idx = (waypoint_idx + 1) % patient_waypoints.shape[1]
        return patient_waypoints[:, next_idx], next_idx

    return modified_target, waypoint_idx


# ============================================================================
# MAIN CONTROL LOOP
# ============================================================================

for iteration in range(TOTAL_ITERATIONS):
    # Get current robot poses [x; y; theta]
    poses = r.get_poses()

    # Single-integrator control states (x/y from unicycle states)
    x_si = poses[:2, :]

    # Initialize velocity array for single-integrator dynamics
    si_velocities = np.zeros((2, N))

    # ========================================================================
    # NURSE CONTROL (Robot 0): Careful Navigation & Patient Approach
    # ========================================================================

    nurse_target, nurse_waypoint_idx = get_nurse_target(
        poses, nurse_waypoint_idx, nurse_pause_counter
    )

    # Calculate velocity towards target
    nurse_velocity = si_position_controller(x_si[:, NURSE : NURSE + 1], nurse_target.reshape(2, 1))

    # Apply slower speed for careful navigation
    careful_speed_factor = 0.7
    if nurse_pause_counter > 0 and nurse_pause_counter < NURSE_PAUSE_DURATION:
        # Stationary during pause
        careful_speed_factor = 0.0

    si_velocities[:, NURSE : NURSE + 1] = nurse_velocity * careful_speed_factor

    # ========================================================================
    # PATIENT CONTROL (Robot 1): Dynamic Environment Behavior
    # ========================================================================

    nurse_pos = poses[:2, NURSE]
    patient_target, patient_waypoint_idx = get_patient_target(
        poses, patient_waypoint_idx, nurse_pos, iteration
    )

    # Calculate velocity towards target
    patient_velocity = si_position_controller(
        x_si[:, PATIENT : PATIENT + 1], patient_target.reshape(2, 1)
    )

    # Adjust speed based on behavior state
    if patient_behavior_state == 1:
        # Slower when nurse is very close
        speed_factor = 0.5
    elif patient_behavior_state == 2:
        # Moderate speed when nurse approaching
        speed_factor = 0.7
    else:
        # Normal dynamic speed
        speed_factor = 0.9

    si_velocities[:, PATIENT : PATIENT + 1] = patient_velocity * speed_factor

    # ========================================================================
    # APPLY SAFETY BARRIER CERTIFICATE
    # ========================================================================

    # Ensure collision avoidance between robots and with boundaries
    si_velocities = si_barrier_cert(si_velocities, x_si)

    # ========================================================================
    # TRANSFORM TO UNICYCLE DYNAMICS AND APPLY
    # ========================================================================

    # Convert single-integrator velocities to unicycle velocities
    dxu = si_to_uni_dyn(si_velocities, poses)

    # Set robot velocities
    r.set_velocities(np.arange(N), dxu)

    # Step the simulation forward
    r.step()

# ============================================================================
# CLEANUP
# ============================================================================

# Call at end of script for proper Robotarium server execution
r.call_at_scripts_end()
