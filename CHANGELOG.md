# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.2.0] - 2025-08-18
Minor release introducing canonical command names and removing legacy CLI subcommands.
Python API remains unchanged; only the CLI surface changed.

### Breaking
- Removed legacy subcommands:
  - dserve
  - infer
  - inferg
  - eval
  - download
- New canonical commands (use these going forward):
  - data-serve (replaces dserve)
  - data-download (replaces download)
  - forecast (replaces both infer and inferg; use -G/--gan and -s/--samples for ensembles)
  - backtest (replaces eval)

### Added
- Unified forecast entrypoint that supports:
  - Deterministic forecasting: -t YYYY-MM-DD
  - GAN ensemble forecasting: -G true -s N -t YYYY-MM-DDTHH:MM:SS
- Backtest command (operational day-by-day evaluation)
- Comprehensive documentation updates across README and docs/user/* and docs/technical/*.

### Changed
- All examples and guides now use: data-serve, data-download, forecast, backtest
- Time format explicitly documented:
  - forecast (deterministic): -t YYYY-MM-DD
  - forecast (GAN): -t YYYY-MM-DDTHH:MM:SS with -G true -s N
  - backtest: -t YYYY-MM-DD
- Version bumped to 0.2.0 in pyproject.toml

### Fixed
- Clarified dataset server binding: use --bind; --port is currently not used by implementation.

### Migration Guide
Replace old commands as follows:
- wxb dserve ...           → wxb data-serve ...
- wxb download ...         → wxb data-download ...
- wxb infer -t DATE ...    → wxb forecast -t DATE ...
- wxb inferg -t DATETIME ... -s N → wxb forecast -t DATETIME -G true -s N ...
- wxb eval -t DATE ...     → wxb backtest -t DATE ...

Time format reminders:
- DATE must be YYYY-MM-DD
- DATETIME must be YYYY-MM-DDTHH:MM:SS

## [0.1.21] - 2025-XX-XX
- Previous minor release (see repository history for details)
