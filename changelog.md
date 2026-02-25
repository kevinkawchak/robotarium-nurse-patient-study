# Changelog

## v0.3.0 - Upgraded Python Simulator v3 (Full API Parity + Visual Overhaul)

### Added

**Simulator backend (`rps/`)**
- `Robotarium` class: GRITSBot hardware velocity limits (0.2 m/s linear, pi rad/s angular), heading normalisation, velocity state tracking, LED stub API.
- `controllers.py`: CLF-based unicycle position controller (`create_clf_unicycle_position_controller`), CLF-based unicycle pose controller (`create_clf_unicycle_pose_controller`).
- `barrier_certificates.py`: plain SI barrier certificate (`create_single_integrator_barrier_certificate`), unicycle barrier certificate (`create_unicycle_barrier_certificate`), unicycle barrier certificate with boundary (`create_unicycle_barrier_certificate_with_boundary`). All use projected-point diffeomorphism.
- `transformations.py`: SI-to-unicycle with obstacles (`create_si_to_uni_dynamics_with_obstacles`), unicycle-to-SI reverse transform (`create_uni_to_si_dynamics`).
- `misc.py`: graph Laplacian generators (`cycle_GL`, `lineGL`, `completeGL`, `random_connectedGL`), `topological_neighbors`, convergence checkers (`at_pose`, `at_position`), `determine_marker_size`.

**GitHub Pages Web Simulator v3 (`docs/index.html`)**
- GRITSBot-shaped robots with heading wedge indicators and numbered labels.
- Robot trail visualisation (toggleable, 80-frame fading history).
- Arena grid overlay with coordinate axis labels (0.4m spacing).
- Phase timeline bar with colour-coded segments and active highlight.
- Real-time info bar: robot count, frame counter, elapsed time, current phase name, playback speed.
- Playback controls: Play, Pause/Resume, Restart buttons.
- Speed controls: 0.5x, 1x, 2x, 4x playback.
- Drag-and-drop `.py` file upload with overlay indicator.
- Doctor/Patient colour legend.
- Full rps API bootstrap (all new controllers, barriers, transforms, graph utils) embedded for browser execution.

**Documentation**
- Updated `docs/simulator.md` with v0.3.0 architecture diagram and old-vs-new comparison table.
- New `releases/v0.3.0.md` release notes.

### Changed
- `rps/robotarium.py`: now enforces hardware velocity limits in `set_velocities()`, normalises heading angle after integration, uses class constants for arena bounds.
- `rps/utilities/barrier_certificates.py`: added magnitude limiting parameter; restructured as foundation for unicycle barrier variants.
- `rps/__init__.py`, `rps/utilities/__init__.py`: updated docstrings listing all available submodules.
- `docs/index.html`: complete rewrite from basic circles to full v3 simulator interface.
- `README.md`: updated architecture diagram and quick-start section for v0.3.0.

### Compatibility
- `Exp_01a_12Feb26.py` runs unchanged against v0.3.0 (verified via CI smoke test).
- All new APIs match official `robotarium_python_simulator` function signatures.
- Passes ruff lint + format for Python 3.10, 3.11, 3.12.

---

## v0.2.0 - Improved Python Simulator (Web + Local)

### Added
- New in-repo `rps` Robotarium-compatible simulator package (`robotarium`, controllers, barrier certificates, transformations).
- New GitHub Pages simulator at `docs/index.html` with:
  - default loading of `Exp_01a_12Feb26.py`
  - `.py` upload/drag-and-play workflow
  - Pyodide runtime (no terminal required, iOS-friendly)
  - canvas visualization for doctors/patients trajectories.
- New release notes in `releases/v0.2.0.md`.
- New CI workflow and lint config to prevent lint/format failures across Python 3.10-3.12.

### Changed
- README updated with architecture and rollout details, including a text-based simulator diagram.

### Compatibility
- `Exp_01a_12Feb26.py` runs against the new simulator API as the default pre-flight test script.
