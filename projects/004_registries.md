# Project 004: Official Registries for Variables and Normalizers

Status: Proposed
Created: 2025-09-16
Author: AI Assistant
Type: Architecture Decision Record & Technical Specification

1. Executive Summary

Custom datasets and research needs often require extending the set of supported variables and their normalization logic. Previously, these extensions were commonly achieved via monkeypatching (overwriting global dicts like `codes`, `code2var`, `normalizors`, `denormalizors` at import-time). This approach is brittle (import order dependent, non-idempotent, hard to audit), and it has surfaced as a “weird bug” when combined with Project 003’s flexible `granularity` and custom `data_path_format`.

This project introduces official, additive Registries with public APIs to register new variables and normalization functions safely and predictably. The Registries replace ad-hoc monkeypatches while remaining backward-compatible in the short term.

Outcomes:
- Stable registration APIs (no dict replacement).
- Idempotent behavior with explicit override semantics and logging.
- Early validation hooks to surface configuration errors (e.g., unknown variables) before data loading.
- Documentation and migration guide for users currently relying on monkeypatching.

2. Problem Statement

- Monkeypatching global dicts (`codes`, `code2var`, `normalizors`, `denormalizors`) is:
  - Fragile: order-of-import dependent; modules may hold stale references.
  - Opaque: overwrites are silent; debugging KeyError is difficult.
  - Non-idempotent: repeated evaluation can accidentally diverge state.
- With Project 003 (Flexible Granularity and Data Path), users more frequently customize variable sets and file layouts. When variable names don’t match built-in mappings, `WxDataset` eventually raises `KeyError` during data loading, often far from the root cause.
- Lack of a sanctioned extension point leads to duplicated patches, harder maintenance, and inconsistent behaviors across projects.

3. Design Goals

- Provide explicit, public registration APIs for:
  - Variable name ↔ code mapping (2D and 3D).
  - Normalization and denormalization functions.
- Ensure registrations are:
  - Additive (no wholesale dict replacements).
  - Idempotent, with explicit override control.
  - Logged with clear provenance for troubleshooting.
- Keep defaults backward-compatible; introduce deprecation path for monkeypatching.
- Enable early validation in `Setting` to fail-fast on unknown variables.

4. Final Design

4.1 Variable Registry

File: `wxbtool/data/variables.py` (extend existing module)

- New public APIs:
  - `register_var2d(name: str, code: str, *, override: bool = False) -> None`
  - `register_var3d(name: str, code: str, *, override: bool = False) -> None`
  - `register_alias(alias: str, target_name: str, *, override: bool = False) -> None`  [optional for future use]
  - `is_known_variable(name: str) -> bool`
  - `get_supported_variables() -> dict[str, list[str]]`
- Semantics:
  - Enforce consistency between `vars2d`/`vars3d` lists and `codes`/`code2var` dicts.
  - If a name already exists with a different dimension (2D vs 3D) or a code collision occurs, raise unless `override=True`; when overriding, log a warning with before/after values.
  - Duplicate/idempotent registrations are no-ops (logged at debug).
- Logging:
  - INFO on first-time registration.
  - WARNING on override.
  - DEBUG on idempotent no-op.

4.2 Normalizer Registry

File: `wxbtool/norms/meanstd.py` (extend existing module)

- New public APIs:
  - `register_normalizer(key: str, fn: Callable) -> None`
  - `register_denormalizer(key: str, fn: Callable) -> None`
  - `get_normalizer(key: str) -> Callable | None`
  - `get_denormalizer(key: str) -> Callable | None`
- Key can be a variable name or a code; internally normalize to a canonical key (e.g., prefer code when resolvable via `code2var`/`codes`).
- Idempotent behavior; override overwrites with WARNING.

4.3 Early Validation Hook (recommended usage)

- Settings may assert variable availability after construction:
  - Example:
    ```python
    from wxbtool.data.variables import is_known_variable, get_supported_variables
    missing = [var for var in self.vars if not is_known_variable(var)]
    if missing:
        raise KeyError(f"Unknown variable(s): {missing}. Known={get_supported_variables()}")
    ```

This surfaces configuration errors (such as misspelled or unregistered custom variables) before dataset loading.

4.4 Backward Compatibility

- Existing built-in mappings remain the same.
- Monkeypatching still “works” short-term but will be discouraged:
  - Phase A: Docs recommend using Registries.
  - Phase B: Emit `DeprecationWarning` when wholesale dict replacement is detected.
  - Phase C (major): Remove reliance on dict replacement behavior (TBD).

5. Affected Areas

- `wxbtool/data/variables.py`:
  - Add registration functions and logging.
  - No breaking changes to existing constants (`vars2d`, `vars3d`, `codes`, `code2var`).
- `wxbtool/norms/meanstd.py`:
  - Add registration/query functions; keep existing `normalizors`/`denormalizors` dicts.
- User Settings (custom modules):
  - Optional: add early validation to fail-fast.
- Documentation:
  - New project doc (this ADR).
  - Update user/technical docs to show how to extend via Registries.

6. Usage Examples

6.1 Register 2D variables (custom names)

```python
from wxbtool.data.variables import register_var2d

register_var2d("sea_surface_temperature", "sst")
register_var2d("surface_net_thermal_radiation", "str")
register_var2d("snow_depth", "sd")

# Layer-specific as 2D (data must actually expose such variables)
register_var2d("relative_humidity_700hPa", "r700")
register_var2d("u_component_of_wind_250hPa", "u250")
register_var2d("v_component_of_wind_250hPa", "v250")
register_var2d("vertical_velocity_500hPa", "w500")
register_var2d("vertical_velocity_700hPa", "w700")
```

6.2 Register 3D variables

```python
from wxbtool.data.variables import register_var3d

register_var3d("geopotential", "z")
register_var3d("temperature", "t")
register_var3d("relative_humidity", "r")
register_var3d("u_component_of_wind", "u")
register_var3d("v_component_of_wind", "v")
```

6.3 Register normalizers

```python
from wxbtool.norms.meanstd import register_normalizer, register_denormalizer

register_normalizer("snow_depth", my_sd_norm)  # by variable name
register_denormalizer("sd", my_sd_denorm)      # by code
```

6.4 Setting with monthly granularity and early validation

```python
from wxbtool.nn.setting import Setting
from wxbtool.data.variables import is_known_variable, get_supported_variables

class SettingMonthly(Setting):
    def __init__(self):
        super().__init__()
        self.granularity = "monthly"
        self.resolution = "1.40625deg"
        self.data_path_format = "{var}_{year}{month:02d}.nc"   # root/{var}/{var}_YYYYMM.nc

        self.vars = [
            "temperature_850hPa", "2m_temperature", "sea_surface_temperature",
            "geopotential_1000hPa", "geopotential_500hPa",
            "relative_humidity_850hPa", "relative_humidity_700hPa",
            "u_component_of_wind_850hPa", "u_component_of_wind_250hPa",
            "v_component_of_wind_850hPa", "v_component_of_wind_250hPa",
            "vertical_velocity_500hPa", "vertical_velocity_700hPa",
            "toa_incident_solar_radiation", "surface_net_thermal_radiation", "snow_depth",
        ]
        self.levels = []
        self.height = 0

        missing = [v for v in self.vars if not is_known_variable(v)]
        if missing:
            raise KeyError(f"Unknown variable(s): {missing}. Supported: {get_supported_variables()}")
```

7. Interaction with Project 003 (Flexible Granularity & Paths)

- Project 003 decouples “where to find files” from “what variables to load.” The Registries decouple “what variables exist” from “how they are declared internally.”
- Together, they provide robust support for:
  - Custom file layouts via `granularity` + `data_path_format`.
  - Custom variable naming and normalization via Registries.
- This combination mitigates common `KeyError` scenarios where a `Setting` lists variables that aren’t in built-in mappings.

8. Testing

- Unit tests:
  - variables registry: add/override/idempotence; cross-dimension conflicts; code collisions; logging assertions.
  - norms registry: add/override/query idempotence.
  - validation: ensure `Setting` early validation catches unknown variable names.
- Integration tests:
  - Minimal `Setting` with monthly granularity and a couple of registered variables; confirm `WxDataset` loads known files and skips missing gracefully.
- Local CI:
  - `PYTHONPATH=. uv run pytest -q`

9. Risks and Mitigations

- Risk: Silent divergence between variable name and NetCDF variable code.
  - Mitigation: registry enforces code mapping; optional sanity checks to verify dataset variable names against code.
- Risk: Users still rely on monkeypatching.
  - Mitigation: deprecation warnings and documentation; keep compatibility for one deprecation cycle.
- Risk: Overuse of override hides mistakes.
  - Mitigation: WARN on override with explicit before/after values; advise to avoid override in docs.

10. Engineering Protocol Compliance

- Clarity over Cleverness: explicit registration APIs with clear semantics.
- Type Safety: typed function signatures.
- Test Rigor: unit + integration tests.
- Focused Changes: confined to variables and norms modules; no behavioral change for default users.

11. Migration Plan

- Phase A (Current Release)
  - Introduce Registries and document usage.
  - Encourage Settings to adopt early validation.
  - Keep monkeypatches working, but update docs to prefer Registries.
- Phase B (Next Minor)
  - Emit `DeprecationWarning` when dicts are wholesale replaced.
- Phase C (Next Major)
  - Remove reliance on dict replacement; Registries are the only supported extension mechanism.

12. Appendix

- Canonical placeholders for `data_path_format` (from Project 003):
  - `{var}`, `{resolution}`, `{year}`, `{month}`, `{day}`, `{hour}`, `{week}`, `{quarter}`
- Pandas frequency mapping (from Project 003):
  - yearly → `YS`, quarterly → `QS`, monthly → `MS`, weekly → `W-MON`, daily → `D`, hourly → `H`
