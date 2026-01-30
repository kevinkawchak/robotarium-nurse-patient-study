# Clinical Trial RL Robotarium Experiment

## Experiment Overview

- **Title:** RL_ClinicalTrial_10Robots_600s
- **Duration:** 600 seconds (10 minutes)
- **Number of Robots:** 10 (4 Nurses + 6 Patients)
- **Platform:** Georgia Tech Robotarium (https://www.robotarium.gatech.edu)

## Description

This experiment implements a reinforcement learning-based clinical trial simulation where nurse robots learn optimal policies for delivering medicine to patient robots.

### Scenario

In a clinical trial setting:
- **4 Nurse Robots** must pick up medicine from a depot and deliver it to patients
- **6 Patient Robots** wait for medicine and exhibit dynamic behavior
- Nurses use **Q-learning** to optimize their task allocation strategy
- The experiment demonstrates both learning (training) and execution (exploitation) phases

## Robot Roles

### Nurse Robots (Robots 0-3)
- Navigate to medicine depot to collect medicine
- Travel to patients needing medicine
- Perform injection procedure (2-second pause)
- Return to depot for next medicine load
- Learn optimal patient assignment through Q-learning

### Patient Robots (Robots 4-9)
- Start in "needs medicine" state
- Exhibit slight movement (simulating restlessness)
- Remain still during injection
- Transition to "healthy" after receiving medicine
- Return to "needs medicine" after cooldown period (~10 seconds)

## LED Color Coding

| Color | Meaning |
|-------|---------|
| **BLUE** | Nurse: Idle/Searching |
| **GREEN** | Nurse: Has medicine, ready to inject |
| **CYAN-GREEN** | Nurse: Currently injecting |
| **YELLOW** | Nurse: Returning to depot |
| **RED** | Patient: Needs medicine |
| **CYAN** | Patient: Receiving injection |
| **WHITE** | Patient: Healthy (received medicine) |

## Experiment Phases

### Phase 1: Training (0-300 seconds)
- High exploration rate (ε-greedy with ε starting at 0.9)
- Nurses explore different patient assignments
- Q-table updates based on rewards
- Epsilon decays over time

### Phase 2: Exploitation (300-600 seconds)
- Low exploration rate (minimum ε = 0.1)
- Nurses primarily use learned optimal policy
- Demonstrates effectiveness of learned behavior

## Reinforcement Learning Details

### State Space
- Distance to nearest patient (discretized into 5 bins)
- Angle to nearest patient (discretized into 8 bins)
- Has medicine (binary)
- Number of patients needing medicine (0-6)

### Action Space
- Target patient 0-5 (go to specific patient)
- Go to depot (action 6)

### Reward Structure
- +100: Successful injection completion
- +50: Patient becomes healthy
- -0.1: Time penalty (encourages efficiency)
- -20: Collision penalty

### Hyperparameters
- Learning rate: 0.1
- Discount factor: 0.95
- Initial epsilon: 0.9
- Minimum epsilon: 0.1
- Epsilon decay: 0.9995

## Files

- `main.py` - Main experiment script (upload to Robotarium)

## Submission Instructions

1. Go to https://www.robotarium.gatech.edu/experiment
2. Fill in experiment details:
   - **Title:** RL_ClinicalTrial_10Robots_600s
   - **Estimated Duration:** 600 seconds
   - **Number of Robots:** 10
   - **Description:** RL-based clinical trial with nurses and patients
3. Upload `main.py`
4. Click "Submit Experiment"

## Technical Features

- **Barrier Certificates:** Collision avoidance between all robots and boundaries
- **Single-Integrator Dynamics:** Simplified control with unicycle transformation
- **Real-time Execution:** Synchronized with physical robot timing
- **Logging:** Progress reports every 30 seconds

## Expected Outcomes

- Nurses should learn to efficiently distribute themselves among patients
- Total injections should increase as learning progresses
- Exploitation phase should show more coordinated behavior
- LED patterns will visualize the clinical trial workflow

## Dependencies (Handled by Robotarium)

- NumPy
- Robotarium Python Simulator (rps)
- CVXOPT (for barrier certificates)

## Safety Features

- Barrier certificates prevent inter-robot collisions
- Boundary avoidance keeps robots in testbed
- Velocity limits ensure safe operation
- Graceful degradation if LED API unavailable

## Author

Generated for Robotarium Experiment Submission  
Date: January 29, 2026

## References

- Robotarium: https://www.robotarium.gatech.edu
- MARBLER (Multi-Agent RL Benchmark): https://github.com/GT-STAR-Lab/MARBLER
- Q-Learning: Watkins & Dayan (1992)
