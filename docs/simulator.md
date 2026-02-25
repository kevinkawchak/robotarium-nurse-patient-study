# Simulator Design (v0.3.0)

The v0.3.0 simulator provides full API parity with the official `robotarium_python_simulator` while running entirely in-browser via Pyodide or locally with only NumPy.

## Architecture Diagram

```text
+──────────────────────────────────────────────────────────────────────────+
|                     User Experiment Script (.py)                         |
|             (default: Exp_01a_12Feb26.py — 14 robot swarm)               |
+─────────────────────────────────┬────────────────────────────────────────+
                                  │
                                  v
+──────────────────────────────────────────────────────────────────────────+
|                  Robotarium API Compatibility Layer (rps/)                |
|                                                                          |
|  ┌─────────────────────┐  ┌───────────────────┐  ┌────────────────────┐  |
|  │  rps.robotarium     │  │  rps.utilities     │  │  rps.utilities     │  |
|  │  ─────────────────  │  │  .controllers      │  │  .barrier_certs    │  |
|  │  Robotarium class   │  │  ────────────────  │  │  ────────────────  │  |
|  │  - get_poses()      │  │  SI position ctrl  │  │  SI barrier (±bnd) │  |
|  │  - set_velocities() │  │  CLF uni position  │  │  Uni barrier(±bnd) │  |
|  │  - step()           │  │  CLF uni pose      │  │  Projected-point   │  |
|  │  - LED stubs        │  │                    │  │  diffeomorphism    │  |
|  └─────────────────────┘  └───────────────────┘  └────────────────────┘  |
|                                                                          |
|  ┌─────────────────────┐  ┌───────────────────────────────────────────┐  |
|  │  rps.utilities      │  │  rps.utilities.misc                       │  |
|  │  .transformations   │  │  ─────────────────────────────────────    │  |
|  │  ────────────────── │  │  Graph Laplacians (cycle, line, complete, │  |
|  │  SI → Uni dynamics  │  │    random connected)                      │  |
|  │  SI → Uni (w/ obs)  │  │  topological_neighbors                   │  |
|  │  Uni → SI dynamics  │  │  at_pose / at_position convergence       │  |
|  └─────────────────────┘  │  determine_marker_size                   │  |
|                            └───────────────────────────────────────────┘  |
+─────────────────────────────────┬────────────────────────────────────────+
                                  │
                                  v
+──────────────────────────────────────────────────────────────────────────+
|                        Physics + Safety Core                             |
|                                                                          |
|  - Unicycle integration: dx = v·cos(θ), dy = v·sin(θ), dθ = ω           |
|  - GRITSBot velocity limits: |v| ≤ 0.2 m/s, |ω| ≤ π rad/s             |
|  - Arena boundary clamp: [-1.6, 1.6] × [-1.0, 1.0] m                   |
|  - Heading normalisation: θ ∈ [-π, π]                                    |
|  - Pairwise collision avoidance (barrier certificates)                   |
|  - Frame history logging (every step → poses snapshot)                   |
+─────────────────────────────────┬────────────────────────────────────────+
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
                   v                             v
+────────────────────────────────+  +──────────────────────────────────────+
|   Local Python Execution       |  |   GitHub Pages Web Simulator v3      |
|   ────────────────────         |  |   ──────────────────────────────     |
|   python Exp_01a_12Feb26.py    |  |   Pyodide runtime (Python in WASM)  |
|   No GUI — headless CI/smoke   |  |   Canvas rendering with:            |
|   Validates safety + timing    |  |    - GRITSBot heading wedges        |
|                                |  |    - Motion trail history            |
+────────────────────────────────+  |    - Arena grid + labels             |
                                    |    - Phase timeline bar              |
                                    |    - Speed controls (0.5x–4x)       |
                                    |    - Pause / Resume / Restart        |
                                    |    - Drag-and-drop .py upload        |
                                    |    - Real-time info panel            |
                                    +──────────────────────────────────────+
```

## v0.2.0 → v0.3.0 Comparison

```text
┌────────────────────────────┬──────────────────────────┬─────────────────────────────────┐
│ Feature                    │ v0.2.0                   │ v0.3.0                          │
├────────────────────────────┼──────────────────────────┼─────────────────────────────────┤
│ Controllers                │ SI position only         │ + CLF unicycle position & pose  │
│ Barrier certificates       │ SI with boundary only    │ + SI plain, Uni plain, Uni+bnd  │
│ Transformations            │ SI → Uni only            │ + SI→Uni w/ obstacles, Uni→SI   │
│ Graph utilities            │ (none)                   │ cycle, line, complete, random    │
│ Convergence checkers       │ (none)                   │ at_pose, at_position             │
│ Velocity limits            │ (none)                   │ GRITSBot hardware limits        │
│ Heading normalisation      │ (none)                   │ [-π, π] after each step         │
│ LED API stubs              │ (none)                   │ set_left_leds, set_right_leds   │
│ Web robot display          │ Coloured circles         │ Heading wedges + numbered       │
│ Web trails                 │ (none)                   │ 80-frame fading trail history   │
│ Web arena                  │ Plain white              │ Grid + coordinate labels        │
│ Web phase display          │ (none)                   │ Colour-coded timeline bar       │
│ Web info panel             │ Frame count only         │ Robots, time, phase, speed      │
│ Web speed control          │ Fixed 1x                 │ 0.5x, 1x, 2x, 4x              │
│ Web playback controls      │ Play only                │ Play, Pause/Resume, Restart     │
│ Web file upload            │ Button only              │ Button + drag-and-drop overlay  │
│ Bootstrap API coverage     │ 3 functions              │ 16+ functions (full API)        │
└────────────────────────────┴──────────────────────────┴─────────────────────────────────┘
```

## Requirements Parity with Official Robotarium

The following real-Robotarium features are fully covered:

1. **Unicycle kinematics** — identical differential-drive model.
2. **Barrier certificates** — SI and unicycle variants, with/without boundary.
3. **Position & pose controllers** — SI proportional and CLF-based unicycle.
4. **Dynamics transformations** — bidirectional SI ↔ unicycle mappings.
5. **Arena boundaries** — hard clamp matching physical Robotarium.
6. **Velocity limiting** — GRITSBot hardware caps enforced in `set_velocities()`.
7. **Graph utilities** — Laplacian generators for consensus/formation algorithms.
8. **LED commands** — API stubs for code compatibility (no-op in simulation).
9. **Script structure** — identical `get_poses / set_velocities / step / call_at_scripts_end` loop.
