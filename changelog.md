# Changelog

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
