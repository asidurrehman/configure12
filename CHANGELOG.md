# Changelog

All notable changes to this project are documented here.

## [Unreleased]

Repository tidy-up. The optimisation logic and its results are unchanged.

### Added
- `config.yaml` for the run folder and main run settings, with a `--config` option to use another file.
- Docstrings for every function in `configure12_functions.py`.
- Automated tests (`tests/`) using a synthetic case and a stand-in for CityCAT, and a GitHub Actions workflow that runs them on Windows and Linux.
- `CITATION.cff`, `requirements.txt`, `pyproject.toml`, `.gitignore` and this changelog.

### Changed
- `configure12_functions.py` is imported from the script's own folder instead of `C:\z\configure12\codes`.
- `environment.yaml` replaced by a short, cross-platform conda environment.
- README rewritten with installation, input data layout, configuration, usage and output descriptions; fixed the article link.

## [1.2.0] - 2025

Version used for the article *Integrated evolutionary optimisation framework for explicit configuration of detention ponds in urban flood risk management* (International Journal of Disaster Risk Reduction, 131, 105901).
