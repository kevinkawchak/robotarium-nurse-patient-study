# Simulator Design (v0.4.0)

The bundled simulator provides full API parity with the official `robotarium_python_simulator` while running entirely in-browser via Pyodide or locally with only NumPy. The v3.1 web interface adds an experiment selector for switching between the 2-robot nurse-patient and 14-robot swarm experiments.

## v0.4.0 - 10Runs_11Jun26 Experiment Suite

v0.4.0 adds ten standalone doctor/nurse/patient experiments (`10Runs_11Jun26/Run01..Run10`) that exercise the rps API across ten algorithm families. The scripts are dual-environment by construction:

- **API resolver:** barrier-certificate factories are looked up under both the production-server short-form name (`create_si_barrier_certificate_with_boundary`) and this repo's long-form stub name (`create_single_integrator_barrier_certificate_with_boundary`), the same pattern Exp_02c established.
- **LED shim:** per-role LED commands (0-255 RGB, 3xN) are routed to `set_left_leds`/`set_right_leds` where the API exposes them (production server, repo stub) and to the GTERNAL fork simulator's LED array otherwise, so role colors render in every environment.
- **End hooks:** both `call_at_scripts_end()` (classic API) and `debug()` (GTERNAL fork validator report) are called when present.
- **Safety stack:** SI barrier certificate with boundary -> SI-to-unicycle map -> zero-velocity spin guard -> wheel-speed budget rescaling (|2v| + L|w| <= 2 v_max with a 0.1% float margin), with planning caps derated to 0.14 m/s / 1.8 rad/s for real-fleet fidelity.
- **Pre-flight knobs:** `RNPS_FAST_SIM=1` (headless, unthrottled) and `RNPS_MAX_ITERS` (iteration cap) for local verification; unset on the Robotarium server.

```text
+------------------------------+--------------------------+------------------------------+
| Feature                      | v0.3.1                   | v0.4.0                       |
+------------------------------+--------------------------+------------------------------+
| Experiment scripts           | 2 (smoke-tested)         | + 10-run algorithm suite     |
| Algorithm families covered   | swarm phases             | 10 (GA, DE, PSO, ACO, ...)   |
| LED usage                    | stub API only            | per-role state machines      |
| Real-robot derating          | (none)                   | 0.14 m/s, 1.8 rad/s plans    |
| Fleet start delay handling   | (none)                   | 15 s standby in every run    |
| Wheel-limit guarantee        | clip in set_velocities   | pre-scaled wheel budget      |
| Verification flow            | CI smoke tests           | + twice-run GTERNAL fork     |
+------------------------------+--------------------------+------------------------------+
```

## Architecture Diagram

```text
+──────────────────────────────────────────────────────────────────────────+
|                     User Experiment Script (.py)                         |
|   Exp_01a_12Feb26.py (14-robot) / main.py (2-robot nurse)               |
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
|   Local Python Execution       |  |   GitHub Pages Web Simulator v3.1    |
|   ────────────────────         |  |   ──────────────────────────────     |
|   python Exp_01a_12Feb26.py    |  |   Pyodide runtime (Python in WASM)  |
|   python main.py               |  |   Canvas rendering with:            |
|   No GUI — headless CI/smoke   |  |    - Experiment selector dropdown   |
|   Validates safety + timing    |  |    - GRITSBot heading wedges        |
|                                |  |    - Motion trail history            |
+────────────────────────────────+  |    - Arena grid + labels             |
                                    |    - Dynamic phase timeline          |
                                    |    - Speed controls (0.5x–4x)       |
                                    |    - Pause / Resume / Restart        |
                                    |    - Drag-and-drop .py upload        |
                                    |    - Real-time info panel            |
                                    +──────────────────────────────────────+
```

## v0.3.0 → v0.3.1 Comparison

```text
┌──────────────────────────────┬───────────────────────────┬─────────────────────────────────┐
│ Feature                      │ v0.3.0                    │ v0.3.1                          │
├──────────────────────────────┼───────────────────────────┼─────────────────────────────────┤
│ Built-in experiments         │ 14-robot only (default)   │ 14-robot + 2-robot selectable   │
│ Experiment selector          │ (none)                    │ Dropdown in web UI              │
│ Phase timeline               │ Fixed 5-phase (swarm)     │ Dynamic per experiment          │
│ Legend labels                 │ Fixed Doctor/Patient      │ Adapts to experiment type       │
│ Robot rendering size         │ Fixed 10px                │ 14px for small experiments      │
│ FRAME_LOG reset              │ (none — reload required)  │ Automatic between runs          │
│ CI smoke tests               │ Exp_01a_12Feb26 only      │ + main.py (both experiments)    │
│ Dependency management        │ Inline pip install        │ requirements.txt                │
│ README diagrams              │ 1 (architecture)          │ 3 (architecture, arena, phases) │
│ README simulation link       │ (none)                    │ Top of page link                │
└──────────────────────────────┴───────────────────────────┴─────────────────────────────────┘
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
