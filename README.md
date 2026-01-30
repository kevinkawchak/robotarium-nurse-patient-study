# Robotarium Nurse Patient Experiment

Remote robotics experiments conducted on Georgia Tech's Robotarium platform using GRITSBots. (Nurse patient study shown below)

<picture>
  <img src="assets/robotarium-nurse-patient-study.jpg" alt="Main">
</picture>
Overview

This repository contains Python scripts and documentation for experiments run on the [Georgia Tech Robotarium](https://www.robotarium.gatech.edu/) - a remotely accessible, $2.5 million swarm robotics research platform funded by the National Science Foundation (NSF) and Office of Naval Research.

## Experiments

### Trial_01_29Jan26 - Clinical Trial Nurse-Patient Simulation
- **Duration:** 60 seconds
- **Robots:** 2 GRITSBot units
- **Description:** Original trial simulating clinical trial interactions between nurse and patient robots
- **Status:** ✅ Completed (January 30, 2026)

📹 **Video:** [Watch Experiment on Google Drive](https://drive.google.com/drive/folders/17Q9Va8vZhKIaC9hvBp60CX3MiqVKjEXY?usp=sharing)

Original Opus 4.5 Extended Prompt: “Create all the files necessary to run a 60 second GA Tech Robotarium physical robot experiment (https://www.robotarium.gatech.edu/experiment) using 2 robots with Python. One robot should be the clinical trial nurse, and the other robot should be the patient. “For instance, careful navigation may represent walking in a relatively straight line with avoidance of some obstacles; patient approach may be walking in a relatively straight line and then pausing; and dynamic environment might be walking with a larger array of movements that change based on conditions.“ Keep in mind that your Python based files must meet the Robotarium's requirements to run on their servers.”
 
Expected Script Behavior (Opus):
- Robot 0 (Nurse):
a) Navigates carefully through waypoints in a relatively straight path
b) Approaches the patient robot's area
c) Pauses for ~3 seconds to simulate patient interaction
d) Returns to the starting position
e) Operates at 70% speed for deliberate, careful movement
- Robot 1 (Patient): 
a) Exhibits dynamic, environmentally-responsive movements
b) Behavior changes based on nurse proximity:
c) Normal dynamic patterns when nurse is far
d) Moderate awareness activity when nurse approaches
e) Subtle, responsive movements when nurse is close
f) Includes sinusoidal variations for realistic human-like motion

Actual Robot Behavior (Similar to Expected):
- Robot 0 (Nurse, TOP): 
a) Demonstrates careful navigation, approaches the patient (0:50), pauses for interaction (1:06-1:09), then returns away from patient (1:15).
- Robot 1 (Patient, BOTTOM): 
a) Exhibits dynamic behavior with varied movements (0:28-1:03), (1:10-1:26) that become less pronounced when closer in proximity to nurse.

## Hardware Platform

The experiments utilize **GRITSBot** robots - custom miniature differential-drive robots developed by Georgia Tech's GRITS Lab.

### GRITSBot Specifications

| Component | Specification |
|-----------|---------------|
| **Main Processor** | ESP8266 (80/160 MHz, ~80kB DRAM, ~35kB IRAM) |
| **Motor Controller** | Atmega168/328 (8MHz, 16/32 KB flash, 2 KB RAM) |
| **WiFi** | IEEE 802.11 B/G/N (built into ESP8266) |
| **Motors** | Miniature stepper motors with high-resolution control |
| **Power** | LiPo battery with Qi wireless charging |
| **Sensors** | IR distance sensing, current/voltage monitoring |
| **Tracking** | Overhead camera system for global positioning |

### Main Board Components
- ESP8266 12-E with 4MB flash memory (OTA firmware updates)
- MCP73831 LiPo battery charging chip
- AP2112K-3.3V voltage regulator (600mA capacity)
- MCP1640 step-up converter for motors
- INA219 I2C current/voltage sensor
- ATECC108 encryption chip

### Motor Board Components
- Atmega168/328 microcontroller
- Two LB1836M motor drivers
- QRE1113 infrared line sensors
- STLM20 temperature sensors
- ZXCT1009 current sensors

For complete hardware documentation, see: [GRITSBot Hardware Design](https://github.com/robotarium/GRITSBot_hardware_design)

## Robotarium Platform

The Robotarium is a 725-square-foot facility housing nearly 100 rolling and flying swarm robots accessible to researchers worldwide.

### Key Features
- **Arena:** 12' x 14' white surface with integrated charging slots
- **Tracking:** Motion capture cameras for precise robot positioning
- **Charging:** 76 slots with 152 wireless charging cells
- **Safety:** Barrier certificates prevent robot collisions
- **Access:** Remote experiment submission via web interface

### Resources
- **Main Site:** https://www.robotarium.gatech.edu/
- **Demo Page:** https://www.robotarium.gatech.edu/demo
- **Python Simulator:** https://github.com/robotarium/robotarium_python_simulator
- **MATLAB Simulator:** https://github.com/robotarium/robotarium-matlab-simulator

## Running Experiments

### Prerequisites
- Python, NumPy, matplotlib, CVXOPT
- Robotarium Python Simulator (for local testing)

### Local Simulation
```bash
# Clone the Robotarium simulator
git clone https://github.com/robotarium/robotarium_python_simulator.git
cd robotarium_python_simulator
pip install .

# Run your experiment locally
python experiments/Trial_01_29Jan26/main.py
```

### Submitting to Robotarium
1. Go to https://www.robotarium.gatech.edu/experiment
2. Create account and log in
3. Fill in experiment parameters (title, duration, robot count)
4. Upload `main.py`
5. Submit and wait for execution
6. Receive video and log files via email/web interface

## References

1. Pickem, D., Lee, M., & Egerstedt, M. (2015). "The GRITSBot in its Natural Habitat – A Multi-Robot Testbed." IEEE ICRA 2015.

2. Wilson, S., et al. (2020). "The Robotarium: Globally impactful opportunities, challenges, and lessons learned in remote-access, distributed control of multirobot systems." IEEE Control Systems Magazine, 40(1):26–44.

3. Pickem, D., Glotfelter, P., Wang, L., et al. (2017). "The Robotarium: A remotely accessible swarm robotics research testbed." IEEE ICRA 2017.

## Acknowledgments

- Georgia Tech GRITS Lab
- National Science Foundation (Grants #1531195, #1544332)
- U.S. Office of Naval Research (Grant N00014-17-1-2323)

## License

This project is for educational and research purposes. See individual experiment directories for specific licensing.

## Contact

- **GitHub Repository:** [kevinkawchak](https://github.com/kevinkawchak)
- **Robotarium Support:** https://www.robotarium.gatech.edu/

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18435965-blue)](https://doi.org/10.5281/zenodo.18435965)

