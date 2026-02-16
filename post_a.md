## [LinkedIn 30Jan26 Post](https://www.linkedin.com/posts/kevin-kawchak-38b52a4a_the-georgia-tech-robotarium-allows-for-users-activity-7423102640668041216-6HnG)

The Georgia Tech Robotarium allows for users to complimentarily run real robots interactively and remotely by submitting Python or Matlab scripts; which is funded by NSF and ONR. (1) Once the script passes checks, the experiment is queued. After the experiment is complete: a video, data, and log files are available for download. Multiple robots can be utilized based on code specifications, as well as programmable LEDs to visualize algorithm state and diagnostics. Here, Opus 4.5 Extended was used to generate the required .py, which was then processed by Robotarium the next morning. (2)

Experiment Video, 1:26 (Nurse = Starts at Top of Screen, Patient = Starts at Bottom) https://lnkd.in/gFwVfJC3

Original Opus Prompt: “Create all the files necessary to run a 60 second GA Tech Robotarium physical robot experiment (https://lnkd.in/giDRaumj) using 2 robots with Python. One robot should be the clinical trial nurse, and the other robot should be the patient. “For instance, careful navigation may represent walking in a relatively straight line with avoidance of some obstacles; patient approach may be walking in a relatively straight line and then pausing; and dynamic environment might be walking with a larger array of movements that change based on conditions.“ Keep in mind that your Python based files must meet the Robotarium's requirements to run on their servers.”

Expected Script Behavior (Opus):
Robot 0 (Nurse, Starts at Top): 
a) Navigates carefully through waypoints in a relatively straight path 
b) Approaches the patient robot's area 
c) Pauses for ~3 seconds to simulate patient interaction 
d) Returns to the starting position 
e) Operates at 70% speed for deliberate, careful movement

Robot 1 (Patient, Starts at Bottom): 
a) Exhibits dynamic, environmentally-responsive movements 
b) Behavior changes based on nurse proximity 
c) Normal dynamic patterns when nurse is far 
d) Moderate awareness activity when nurse approaches 
e) Subtle, responsive movements when nurse is close 
f) Includes sinusoidal variations for realistic human-like motion

Actual Robot Behavior: 
Robot 0 (Nurse, Starts at Top), (Similar to Expected): 
a) Demonstrates careful navigation
b) Approaches the patient (0:50)
c) Pauses for interaction (1:06-1:09)
d) Returns away from patient (1:15)

Robot 1 (Patient, Starts at Bottom), (Similar to Expected): 
a) Exhibits dynamic behavior with varied movements (0:28-1:03), (1:10-1:26)
b) That become less pronounced when closer in proximity to nurse (1:03-1:10).

Future steps may include obtaining and running humanoids that others can access remotely for experimentation.

References:
1) “Robotarium.” Gatech.edu, 2025, www.robotarium.gatech.edu/.
2) kevinkawchak, GitHub, 2026, https://lnkd.in/gjmbHXch.

- Post by Kevin Kawchak
