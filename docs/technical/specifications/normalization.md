# Normalization Systems and Registries

Status: Implemented (Project 004)
Updated: 2025-09-16

This page documents how normalization and denormalization are handled in wxbtool and how to extend them safely using the new registry APIs.

## Module

- wxbtool.norms.meanstd

Built-ins:
- A set of predefined normalization/denormalization functions for common variables (e.g., t2m, z500, u/v at various levels, etc.)
- Two mapping dicts for function dispatch:
  - normalizors: dict[str, Callable]
  - denormalizors: dict[str, Callable]

Note: Keys in these dicts are canonical “codes” (e.g., "t2m", "z500", "u250", "q925", etc.)

## New Registry APIs

Project 004 introduces explicit, idempotent, and override-aware registration APIs to extend normalization without monkeypatching:

- register_normalizer(key: str, fn: Callable, *, override: bool = False) -> None
- register_denormalizer(key: str, fn: Callable, *, override: bool = False) -> None
- get_normalizer(key: str) -> Optional[Callable]
- get_denormalizer(key: str) -> Optional[Callable]

Where key can be either:
- A variable name (e.g., "2m_temperature", "sea_surface_temperature"), or
- A canonical code (e.g., "t2m", "sst").

Internally, keys are normalized to a canonical “code” using the variable registry:
- If key is a known variable name in wxbtool.data.variables.codes, it’s mapped to its code.
- If key is already a known code, it’s used directly.
- Otherwise, the provided key is used as-is.

## Semantics

- Idempotency:
  - Registering the same function for the same canonical key is a no-op (logged at DEBUG).
- Override:
  - Replacing an existing function requires override=True and logs a WARNING.
- Queries:
  - get_normalizer/get_denormalizer accept either variable names or codes and resolve to the canonical code internally.

## Examples

Registering by variable name:
```python
from wxbtool.data.variables import register_var2d
from wxbtool.norms.meanstd import register_normalizer, register_denormalizer, get_normalizer

# Ensure variable-to-code mapping exists
register_var2d("sea_surface_temperature", "sst")

# Register normalization/denormalization
register_normalizer("sea_surface_temperature", lambda x: x)
register_denormalizer("sst", lambda x: x)

# Query by name or by code
assert get_normalizer("sea_surface_temperature") is not None
assert get_normalizer("sst") is not None
```

Overriding (requires explicit override=True):
```python
def my_norm(x): return x
def my_norm_v2(x): return x  # replacement

register_normalizer("sst", my_norm)
# This will raise unless override=True
register_normalizer("sst", my_norm_v2, override=True)
```

## Interaction with Variable Registry

These normalization registries complement the Variable Registry (see technical/extension/new_variables.md):
- Variable Registry defines what variables exist and how names map to codes.
- Normalization Registry defines how those variables are normalized/denormalized by code.

Using both together provides a safe, auditable, and predictable extension mechanism without import-order pitfalls.

## Migration Guidance

- Prefer registry APIs over direct dict modifications (monkeypatching).
- Use override=True sparingly and intentionally.
- When introducing a new variable, first register the variable mapping (name -> code), then register normalization functions if required.
