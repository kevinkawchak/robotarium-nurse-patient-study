## [LinkedIn 13Feb26 Post](https://www.linkedin.com/posts/kevin-kawchak-38b52a4a_nvidiagtc-activity-7428177841550258176-90kd)

Thanks to the Georgia Tech Robotarium for access to the increased number of free robots and extended run times from prior attempts.

The 14 robot experiment shown in the video achieved some of the desired robot swarm behavior.
Opus Expectation:
5 Doctor Robots, 9 Patient Robots
* 0–8 sec Patients are scattered across the arena performing jittery oscillations. Doctors hold a tight cluster at the left "hospital base."
* 8–20 sec - Doctors disperse using Voronoi-like repulsion (swarm) for maximum spatial coverage, then each locks onto the nearest unattended patient. Patient oscillations dampen as doctors approach. 

Actual Video:
5 Doctor Robots, 9 Patient Robots
* 0–20 sec: 9 Patients on the right exhibit more jittery behavior than 5 Doctors on the left.
* 20–44 sec: Doctors disperse using Voronoi-like repulsion to lock onto unattended patients. Patient oscillations slightly dampen as doctors approach. Collision detection aborted experiment. https://lnkd.in/gw2C-My7

How to Scale Robotarium Experiments with Opus 4.6 Extended Prompts:
a) Ask Opus to meet the Robotarium's requirements to run the code on their servers. 
b) Have Opus run its generated script on the Robotarium GitHub simulator. Specify for only the Python or Matlab file be returned for download.
c) Be sure to ask the LLM to provide step by step bullet points regarding what the robots are anticipated to be doing at specific time points (then compare expected results to the actual video).
d) Submit to Robotarium, requesting 3x runtime that was provided as an estimate to Opus (the robots take time to reach desired positions). https://lnkd.in/gT_783SY

I am hoping to run larger robots and experiments with enhanced oncology trial context by #NVIDIAGTC, Carter Abdallah

- Post by Kevin Kawchak
