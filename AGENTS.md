# Repository Guidelines

## Project Structure & Module Organization
- `tractseg/`: core Python package (models, data utilities, experiments, and API code).
- `bin/`: CLI entrypoints such as `TractSeg`, `Tracking`, `Tractometry`, and `ExpRunner`.
- `tests/`: unit tests (`test_functions.py`, `test_model_imports.py`).
- `scripts/` and `preprocessing/`: cluster job scripts and dataset preparation utilities.
- `tools/`: standalone, ad-hoc utility scripts not part of the installed CLI or cluster job scripts.
- `resources/` and `tractseg/resources/`: packaged templates/models used at runtime.

## Build, Test, and Development Commands
Use the `masam` conda environment for all Python invocations.
- Install editable package: `conda run -n masam python -m pip install -e .`
- Run a quick test file: `conda run -n masam python -m pytest -v tests/test_functions.py`

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and readable, explicit names.
- Keep code and comments in English.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_CASE` for constants.
- Prefer small, testable functions and add defensive checks around file I/O and array shape assumptions.
- There is no enforced formatter in CI; keep style consistent with surrounding files.

## Testing Guidelines
- Test framework: `pytest` (running existing `unittest.TestCase` suites).
- Add tests under `tests/` with filenames `test_*.py`.
- For numerical outputs, use tolerant comparisons (`np.allclose`) where stochastic steps are involved.

## Commit & Pull Request Guidelines
- History favors short, imperative commit subjects (e.g., `Fix CUDA device handling...`).
- Use focused commits: `<area>: <imperative summary>` (example: `tracking: fix seed handling in test mode`).
- Avoid vague subjects like `latest code`.
- PRs should include purpose/scope, exact validation commands run, linked issue(s), and sample output paths or screenshots for behavior/output changes.
