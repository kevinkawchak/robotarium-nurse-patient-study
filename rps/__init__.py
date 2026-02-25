"""Lightweight Robotarium-compatible simulator package (v0.3.0).

Full API parity with robotarium_python_simulator including:
- Robotarium class (unicycle kinematics, velocity limits, frame history)
- SI and unicycle controllers (position, pose, CLF-based)
- Barrier certificates (SI/unicycle, with/without boundary)
- Dynamics transformations (SI <-> unicycle)
- Graph utilities (cycle, line, complete, random connected)
- Convergence helpers (at_pose, at_position)
"""

from .robotarium import Robotarium

__all__ = ["Robotarium"]
