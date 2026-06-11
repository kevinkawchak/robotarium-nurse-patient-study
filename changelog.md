# Changelog

## v0.4.0 - 10Runs_11Jun26: Ten-Algorithm Doctor/Nurse/Patient Suite

### Added

**Experiment suite (`10Runs_11Jun26/`)**
- Ten new standalone Robotarium experiment scripts, each pairing a unique doctor/nurse/patient fleet mix (10-16 robots) with a different algorithm family and clinical-trial objective:
  - `Run01_SwarmIntake_11Jun26.py` - boids flocking + shepherding intake (2D/4N/8P, 150 s).
  - `Run02_GeneticPairing_11Jun26.py` - genetic algorithm care-team pairing (3D/5N/8P, 180 s).
  - `Run03_DifferentialWard_11Jun26.py` - differential evolution ward layout (4D/4N/8P, 210 s).
  - `Run04_PSODoseSearch_11Jun26.py` - embodied PSO dose-response search (2D/3N/5P, 150 s).
  - `Run05_AntColonyMeds_11Jun26.py` - ant colony medication rounds (1D/4N/6P, 195 s).
  - `Run06_ConsensusVitals_11Jun26.py` - ferry-driven vitals consensus (3D/4N/9P, 165 s).
  - `Run07_AuctionTriage_11Jun26.py` - market-based auction triage pipeline (2D/5N/9P, 240 s).
  - `Run08_PotentialIsolation_11Jun26.py` - potential-field isolation ward (3D/3N/6P, 180 s).
  - `Run09_AnnealingBeds_11Jun26.py` - simulated annealing bed reassignment (2D/4N/7P, 165 s).
  - `Run10_ConvoyDischarge_11Jun26.py` - leader-follower discharge convoys (1D/3N/6P, 210 s).
- Per-role LED state machines (doctors blue, nurses green/violet, patients red-amber-green) updating every iteration, with run-specific signals (temperature gauge, acuity blink rates, convoy blink signatures, consensus hue equalization) and a shared 15 s standby/role-announce sequence.
- Step-by-step expected-time bullet timelines, documented emergent behaviors, and live printed metrics in every script docstring.
- Shared compliance core: SI barrier certificates with boundary, single `get_poses()` per `step()`, >= 0.36 m start spacing asserts, derated planning speeds (0.14 m/s / 1.8 rad/s), wheel-speed budget rescaling with float-safe margin, zero-velocity spin guard, white arena background, and `call_at_scripts_end()`/`debug()` end hooks.
- Runtime rps API resolver (short-form server names + long-form repo-stub names), LED shim for `set_left_leds`/`set_right_leds`/fork LED array, and repo-root path bootstrap.
- Local pre-flight knobs `RNPS_FAST_SIM=1` and `RNPS_MAX_ITERS` (ignored on the Robotarium server).

**Documentation**
- README v0.4.0 section: run catalog table, suite architecture diagram, repository structure, and pre-flight instructions.
- `docs/simulator.md` v0.4.0 section and comparison table.
- New `releases/v0.4.0.md` release notes.

### Changed
- `changelog.md`: added v0.4.0 entry.
- `README.md`: restructured top section around the v0.4.0 suite; updated experiments and quick-start sections.
- Second-to-last commit of the release applies optimizations and bug fixes from the first full simulator pass: wheel-budget float margin on Run01/Run02, Run04 hidden-field reshaping (decoy escape demonstrated) and 0.23 m safety radius, Run05 pheromone gains + tour-concentration metric, Run01 flocking-gain revert with measured polarization narrative.

### Compatibility
- All ten scripts verified twice end-to-end in the GTERNAL fork of `robotarium_python_simulator` (initialization phase included): "No errors or warnings in your simulation! Your script will run on the Robotarium!" for every run.
- Scripts also import-resolve against the repo-local `rps` stub and the Robotarium production server API names unchanged.
- `main.py` and `Exp_01a_12Feb26.py` CI smoke tests unaffected; all files pass ruff lint + format for Python 3.10, 3.11, 3.12.

---

## v0.3.1 - Dual-Experiment Simulator + CI Hardening

### Added

**Web Simulator v3.1 (`docs/index.html`)**
- Experiment selector dropdown: choose between 14-robot swarm and 2-robot nurse-patient without uploading files.
- Dynamic legend and phase timeline that adapts to the loaded experiment (Doctor/Patient vs Nurse/Patient labels, different phase names and durations).
- FRAME_LOG reset between runs so the simulator can be replayed without page reload.
- Larger robot rendering for small experiments (2-4 robots).

**CI / Build (`ci.yml`, `requirements.txt`)**
- `requirements.txt` for explicit dependency management (`numpy>=1.24`).
- CI now installs dependencies from `requirements.txt` instead of inline pip install.
- Added second smoke test step: `python main.py` (2-robot experiment) runs alongside `Exp_01a_12Feb26.py`.

**Documentation**
- Web Simulator link at the top of `README.md` for one-click access.
- Three text diagrams in README: Simulator Architecture, Arena Top-Down View, Experiment Timeline Comparison.
- Explicit Exp_01a_12Feb26 experiment section in README with video link and details.
- Updated `docs/simulator.md` with v0.3.1 architecture and v0.3.0-to-v0.3.1 comparison table.
- New `releases/v0.3.1.md` release notes.
- Updated Quick Start instructions to reference both experiments and `requirements.txt`.

### Changed
- `docs/index.html`: title updated to v3.1, experiment selector added, legend and timeline made dynamic, init loads both built-in scripts.
- `.github/workflows/ci.yml`: dependency install split into pip upgrade + ruff + requirements.txt; two separate smoke test steps.
- `README.md`: restructured with simulation link at top, 3 diagrams, updated version references, added Exp_01a section, updated prerequisites and local simulation instructions.
- `changelog.md`: added v0.3.1 entry.
- @kevinkawchak added Exp_02_30Apr26.py using Claude Code on 2026-04-30.

### Compatibility
- Both `main.py` and `Exp_01a_12Feb26.py` run unchanged against v0.3.1.
- All files pass ruff lint + format for Python 3.10, 3.11, 3.12.

---

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
