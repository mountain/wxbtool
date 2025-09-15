# Project 003: Flexible Granularity and Data Path Support

Status: Implemented
Created: 2025-09-15
Implemented: 2025-09-15
Author: AI Assistant
Type: Architecture Decision Record & Technical Specification

1. Executive Summary

wxbtool previously assumed yearly-organized NetCDF files and hard-coded year-based path construction in WxDataset. This blocked usage of datasets organized by other temporal grains (monthly, daily, hourly, etc.).

This project introduces:
- Configurable data organization in Setting via:
  - granularity (yearly | quarterly | monthly | weekly | daily | hourly)
  - data_path_format (format string with placeholders)
- A stateless DataPathManager to generate candidate file paths from a pandas date_range.
- A refactor of WxDataset to use the manager and formats from Setting, removing hard-coded year-based paths while maintaining full backward compatibility.

Result: wxbtool now supports hour/day/week/month/quarter/year (and custom) dataset layouts via format strings with preserved defaults and test coverage.

2. Problem Statement (Recap)

- Hard-coded file naming and year-based iteration in wxbtool/data/dataset.py
- Missing configuration in Setting for temporal granularity and path naming
- Entanglement of dataset discovery and loading logic (violates SRP)

3. Design Goals

- Make dataset file layout configurable per Spec
- Support multiple temporal granularities without library code changes
- Preserve backward compatibility by default
- Keep WxDataset focused on loading/slicing; move discovery to a helper
- Provide a small, typed, and testable API

4. Final Design

4.1 Setting configuration (implemented)
- New properties (with defaults to preserve existing behaviour):
  - granularity: "yearly"
  - data_path_format: "{var}_{year}_{resolution}.nc"
- Users can override both in their Setting subclasses:
  - Monthly:
    granularity = "monthly"
    data_path_format = "{year}/{var}_{year}-{month:02d}_{resolution}.nc"
  - Daily:
    granularity = "daily"
    data_path_format = "{year}/{month:02d}/{var}_{year}-{month:02d}-{day:02d}_{resolution}.nc"
  - Hourly:
    granularity = "hourly"
    data_path_format = "{year}/{month:02d}/{day:02d}/{var}_{year}-{month:02d}-{day:02d}T{hour:02d}_{resolution}.nc"

4.2 DataPathManager (implemented)
- New file: wxbtool/data/path.py
- Input: root, var, resolution, data_path_format, date_range (pandas)
- Output: sorted unique candidate paths
- Supported placeholders: {var}, {resolution}, {year}, {month}, {day}, {hour}, {week}, {quarter}
- No existence check (delegated to dataset loader)

4.3 WxDataset refactor (implemented)
- Uses Setting.granularity to build a pandas date_range via a private freq map:
  yearly -> YS, quarterly -> QS, monthly -> MS, weekly -> W-MON, daily -> D, hourly -> H
- Uses DataPathManager to generate paths for each var and attempts to load any existing files
- load_2ddata/load_3ddata now accept absolute file paths (year parameter removed internally)
- Cache hashcode now includes granularity and data_path_format to avoid collisions

4.4 Placeholder validation
- Unknown placeholders raise a clear KeyError during path formatting in DataPathManager
- When unknown granularity is provided, WxDataset warns and defaults to daily frequency

5. Affected Areas (Implemented)

- wxbtool/nn/setting.py
  - Added granularity and data_path_format (defaults maintain existing behaviour)
- wxbtool/data/path.py
  - New helper with typed API for path generation
- wxbtool/data/dataset.py
  - Refactored load logic to use date_range + DataPathManager
  - Updated load_2ddata/load_3ddata signatures to accept file paths
  - Cache hash extended to include new config values
- pyproject.toml
  - Added pandas dependency (>=2.2.2)
- tests/test_path_manager.py
  - New unit tests for DataPathManager

6. Backward Compatibility

- Defaults reproduce the previous yearly layout:
  - data_path_format = "{var}_{year}_{resolution}.nc"
  - granularity = "yearly"
- No changes needed to existing Settings to keep current behaviour
- Cache directories now incorporate granularity and data_path_format; old caches remain unaffected

7. Usage Examples

7.1 Default yearly (no changes required)
- Directory:
  WXBHOME/{var}/{var}_{YYYY}_{resolution}.nc

7.2 Monthly
- Setting:
  class MySetting(Setting):
      def __init__(self):
          super().__init__()
          self.granularity = "monthly"
          self.data_path_format = "{year}/{var}_{year}-{month:02d}_{resolution}.nc"
- Directory example:
  WXBHOME/2m_temperature/1980/2m_temperature_1980-01_5.625deg.nc

7.3 Daily
- Setting:
  class MySetting(Setting):
      def __init__(self):
          super().__init__()
          self.granularity = "daily"
          self.data_path_format = "{year}/{month:02d}/{var}_{year}-{month:02d}-{day:02d}_{resolution}.nc"
- Directory example:
  WXBHOME/2m_temperature/1980/01/2m_temperature_1980-01-01_5.625deg.nc

8. Testing

- Unit tests:
  - tests/test_path_manager.py validates yearly, monthly, daily path generation and placeholder errors
- Integration tests:
  - Full test suite executed with flexible granularity code path integrated
- Command (local CI):
  - PYTHONPATH=. uv run pytest -q
- Result:
  - 37 passed, 61 warnings in 523.71s

9. Risks and Mitigations

- Inconsistent format vs actual files:
  - Early KeyError on unknown placeholders and debug logs for missing files
- Large date ranges generating many nonexistent paths:
  - Granularity controls frequency; documentation updated with best practices
- Cache collisions:
  - Mitigated by including granularity and data_path_format in hash

10. Engineering Protocol Compliance

- Clarity over Cleverness: simple, explicit configuration and helper
- Type Safety: typed APIs in DataPathManager
- Tests: unit + integration tests in suite
- Automation: Covered within existing pytest workflow
- Docs: README and user docs updated
- Atomic Changes: Confined to data loading and configuration modules

11. Implementation Status (Post-merge)

- Code:
  - Added wxbtool/data/path.py
  - Modified wxbtool/nn/setting.py
  - Modified wxbtool/data/dataset.py
  - Updated pyproject.toml (pandas dependency)
- Tests:
  - Added tests/test_path_manager.py
  - All tests passing
- Documentation:
  - README and docs/user/data_handling/overview.md updated

12. Migration Notes

- Existing users: no action required
- For non-yearly layouts, set granularity and data_path_format in your Setting
- Caches are now keyed by the new configuration; old caches remain intact

13. Appendix: Placeholder Reference

- var (str): variable directory/name
- resolution (str): e.g., "5.625deg"
- year (int), month (int), day (int), hour (int)
- week (int): ISO week number (1–53)
- quarter (int): 1–4
