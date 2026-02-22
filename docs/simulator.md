# Simulator Design (v0.2.0)

The upgraded simulator is designed to mirror Robotarium experiment flow, while also being easy to run in GitHub Pages.

## Text Diagram

```text
User Experiment (.py)
      |
      v
+-------------------------------+
| Robotarium API compatibility  |
| - rps.robotarium.Robotarium   |
| - controllers                 |
| - barrier certificates        |
| - SI -> unicycle dynamics     |
+-------------------------------+
      |
      v
+-------------------------------+
| Simulation Core               |
| - Kinematics integration      |
| - Arena bounds enforcement    |
| - Inter-robot safety spacing  |
| - Frame history logging       |
+-------------------------------+
      |
      +-----------------------+
      |                       |
      v                       v
+-------------------+   +----------------------------+
| Local Python Run  |   | GitHub Pages Web Simulator |
| python script.py  |   | Upload .py + press Play    |
+-------------------+   +----------------------------+
```

## Requirements Parity Notes

- Safety certificates are included before velocity application.
- Differential-drive-compatible unicycle commands are generated from SI commands.
- Arena boundaries and inter-robot separation are enforced each step.
- Same experiment script structure (Robotarium API style) is preserved for pre-flight validation.
