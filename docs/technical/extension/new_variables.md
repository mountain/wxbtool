# Extending Variables via Official Registries

Status: Implemented (Project 004)
Updated: 2025-09-16

This page documents the official registries for:
- Variable name ↔ code mappings (2D/3D)
- Optional aliases for variable names
- Early validation patterns to fail-fast on configuration issues

These registries replace ad-hoc monkeypatching (e.g., overwriting global dicts at import-time) with explicit, idempotent, and logged APIs.

## Module

- wxbtool.data.variables

Key built-ins (unchanged and still supported):
- vars2d: list[str] — known 2D variable names
- vars3d: list[str] — known 3D variable names
- codes: dict[str, str] — forward mapping name -> code
- code2var: dict[str, str] — reverse mapping code -> name

New APIs:
- register_var2d(name: str, code: str, *, override: bool = False) -> None
- register_var3d(name: str, code: str, *, override: bool = False) -> None
- register_alias(alias: str, target_name: str, *, override: bool = False) -> None
- is_known_variable(name: str) -> bool
- get_supported_variables() -> dict[str, list[str]]

## Semantics

- Additive and idempotent:
  - Re-registering an existing mapping (same name and same code) is a no-op.
  - First-time registration logs at INFO; idempotent no-ops log at DEBUG.

- Conflict handling:
  - Dimensionality conflicts (2D vs 3D) require override=True and log a WARNING.
  - Code collisions (the same code mapped to another name) require override=True; this reassigns the code, removes the previous owner's forward mapping, and cleans up vars2d/vars3d lists appropriately with a WARNING.

- Aliases:
  - register_alias allows additional names for an existing target variable, without mutating code2var (kept one-to-one).
  - Alias redefinitions require override=True, logging a WARNING.

## Examples

Register 2D variables:
```python
from wxbtool.data.variables import register_var2d, is_known_variable, get_supported_variables

register_var2d("sea_surface_temperature", "sst")
assert is_known_variable("sea_surface_temperature")

# Idempotent re-registration
register_var2d("sea_surface_temperature", "sst")

# Conflict (code collision): requires override
register_var2d("sst_conflict", "sst", override=True)
```

Register 3D variables:
```python
from wxbtool.data.variables import register_var3d

register_var3d("geopotential", "z")
register_var3d("temperature", "t")
register_var3d("relative_humidity", "r")
```

Aliases:
```python
from wxbtool.data.variables import register_alias

register_alias("sst_alias", "sea_surface_temperature")
```

## Early Validation in Settings

For robust configurations, validate variable availability after constructing a Setting:

```python
from wxbtool.nn.setting import Setting
from wxbtool.data.variables import is_known_variable, get_supported_variables

class MySetting(Setting):
    def __init__(self):
        super().__init__()
        self.vars = [
            "temperature_850hPa",
            "2m_temperature",
            "sea_surface_temperature",
        ]
        missing = [v for v in self.vars if not is_known_variable(v)]
        if missing:
            raise KeyError(
                f"Unknown variable(s): {missing}. Supported: {get_supported_variables()}"
            )
```

## Interaction with Flexible Dataset Organization

This registry is independent from Project 003 (granularity and data_path_format). Use Project 003 to control where files are discovered; use this registry to define what variables exist and how they map to dataset codes.

## Migration Guidance

- Prefer using these registries over modifying global dicts directly.
- Keep override=True for rare, intentional remappings; avoid when possible.
- If you currently rely on monkeypatching, migrate to these APIs and add early validation to Settings to fail-fast.
