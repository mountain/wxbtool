# -*- coding: utf-8 -*-
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Optional alias mapping (kept separate to avoid mutating code2var one-to-one mapping)
_aliases: Dict[str, str] = {}


def resolve_name(name: str) -> str:
    """Resolve an alias to its canonical variable name if present."""
    return _aliases.get(name, name)


def is_var2d(name: str) -> bool:
    """Check if a variable (or its alias) is a known 2D variable."""
    if name in vars2d:
        return True
    canonical = resolve_name(name)
    return canonical in vars2d


def is_var3d(name: str) -> bool:
    """Check if a variable (or its alias) is a known 3D variable."""
    if name in vars3d:
        return True
    canonical = resolve_name(name)
    return canonical in vars3d


def get_code(name: str) -> str:
    """Get canonical code for a variable (or its alias)."""
    # Prefer direct mapping if present
    if name in codes:
        return codes[name]
    canonical = resolve_name(name)
    if canonical in codes:
        return codes[canonical]
    raise KeyError(f"No code mapping for variable '{name}' (resolved to '{canonical}')")


def get_var(code: str) -> str:
    """Get canonical variable name for a code."""
    if code in code2var:
        return code2var[code]
    raise KeyError(f"No variable mapping for code '{code}'")


vars2d = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_cloud_cover",
    "total_precipitation",
    "toa_incident_solar_radiation",
    "temperature_850hPa",
    "test_variable",
]

vars3d = [
    "geopotential",
    "temperature",
    "specific_humidity",
    "relative_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "vorticity",
    "potential_vorticity",
]

codes = {
    "geopotential": "z",
    "temperature": "t",
    "temperature_850hPa": "t",
    "specific_humidity": "q",
    "relative_humidity": "r",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vorticity": "vo",
    "potential_vorticity": "pv",
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_cloud_cover": "tcc",
    "total_precipitation": "tp",
    "toa_incident_solar_radiation": "tisr",
    "test_variable": "test",
}

code2var = {
    "z": "geopotential",
    "t": "temperature",
    "q": "specific_humidity",
    "r": "relative_humidity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "vo": "vorticity",
    "pv": "potential_vorticity",
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "tcc": "total_cloud_cover",
    "tp": "total_precipitation",
    "tisr": "toa_incident_solar_radiation",
    "test": "test_variable",
}


def split_name(composite):
    if (
        composite == "t2m"
        or composite == "u10"
        or composite == "v10"
        or composite == "tcc"
        or composite == "tp"
        or composite == "tisr"
        or composite == "test"
    ):
        return composite, ""
    else:
        if composite[:2] == "vo" or composite[:2] == "pv":
            return composite[:2], composite[2:]
        else:
            code, level = composite[:1], composite[1:]
            if code in code2var:
                return code, level
            else:
                return composite, ""


def is_known_variable(name: str) -> bool:
    """Return True if the variable name or alias is known to the registry."""
    if name in codes or name in _aliases:
        return True
    return False


def get_supported_variables() -> Dict[str, List[str]]:
    """Return a snapshot of supported variables grouped by dimensionality."""
    return {
        "vars2d": list(vars2d),
        "vars3d": list(vars3d),
    }


def _ensure_list_membership(name: str, lst: List[str]) -> None:
    if name not in lst:
        lst.append(name)


def _remove_from_list(name: str, lst: List[str]) -> None:
    if name in lst:
        lst.remove(name)


def register_var2d(name: str, code: str, *, override: bool = False) -> None:
    """
    Register (or update) a 2D variable mapping.

    - Ensures consistency between vars2d/vars3d and codes/code2var.
    - Idempotent if the exact same mapping already exists.
    - If conflicts occur, requires override=True and logs a WARNING.
    """
    if not isinstance(name, str) or not isinstance(code, str):
        raise TypeError("name and code must be strings")

    # If defined as 3D previously
    if name in vars3d:
        if not override:
            raise ValueError(
                f"Variable '{name}' already registered as 3D. Use override=True to change."
            )
        logger.warning("Overriding variable dimensionality: %s (3D -> 2D)", name)
        _remove_from_list(name, vars3d)

    existing_code = codes.get(name)
    if existing_code is not None:
        if existing_code == code:
            logger.debug("register_var2d: idempotent for %s=%s", name, code)
            _ensure_list_membership(name, vars2d)
            code2var.setdefault(code, name)
            return
        if not override:
            raise ValueError(
                f"Variable '{name}' already mapped to code '{existing_code}'. "
                f"Got '{code}'. Use override=True to change."
            )
        logger.warning(
            "Overriding variable code for %s: %s -> %s", name, existing_code, code
        )

    # Code collision: code already owned by another variable name
    owner = code2var.get(code)
    if owner is not None and owner != name:
        if not override:
            raise ValueError(
                f"Code '{code}' already mapped to variable '{owner}'. "
                f"Use override=True to reassign."
            )
        logger.warning("Reassigning code '%s' from '%s' to '%s'", code, owner, name)
        # Remove previous owner's forward mapping and from var lists
        if owner in codes and codes[owner] == code:
            del codes[owner]
        _remove_from_list(owner, vars2d)
        _remove_from_list(owner, vars3d)
        # Auto-alias old owner to the new canonical name so directory/name lookups still work
        _aliases[owner] = name
        logger.info(
            "Registered alias '%s' -> '%s' due to code reassignment", owner, name
        )

    # Apply mapping
    codes[name] = code
    code2var[code] = name
    _ensure_list_membership(name, vars2d)


def register_var3d(name: str, code: str, *, override: bool = False) -> None:
    """
    Register (or update) a 3D variable mapping.

    See register_var2d for semantics and conflict handling.
    """
    if not isinstance(name, str) or not isinstance(code, str):
        raise TypeError("name and code must be strings")

    # If defined as 2D previously
    if name in vars2d:
        if not override:
            raise ValueError(
                f"Variable '{name}' already registered as 2D. Use override=True to change."
            )
        logger.warning("Overriding variable dimensionality: %s (2D -> 3D)", name)
        _remove_from_list(name, vars2d)

    existing_code = codes.get(name)
    if existing_code is not None:
        if existing_code == code:
            logger.debug("register_var3d: idempotent for %s=%s", name, code)
            _ensure_list_membership(name, vars3d)
            code2var.setdefault(code, name)
            return
        if not override:
            raise ValueError(
                f"Variable '{name}' already mapped to code '{existing_code}'. "
                f"Got '{code}'. Use override=True to change."
            )
        logger.warning(
            "Overriding variable code for %s: %s -> %s", name, existing_code, code
        )

    # Code collision: code already owned by another variable name
    owner = code2var.get(code)
    if owner is not None and owner != name:
        if not override:
            raise ValueError(
                f"Code '{code}' already mapped to variable '{owner}'. "
                f"Use override=True to reassign."
            )
        logger.warning("Reassigning code '%s' from '%s' to '%s'", code, owner, name)
        # Remove previous owner's forward mapping and from var lists
        if owner in codes and codes[owner] == code:
            del codes[owner]
        _remove_from_list(owner, vars2d)
        _remove_from_list(owner, vars3d)
        # Auto-alias old owner to the new canonical name so directory/name lookups still work
        _aliases[owner] = name
        logger.info(
            "Registered alias '%s' -> '%s' due to code reassignment", owner, name
        )

    # Apply mapping
    codes[name] = code
    code2var[code] = name
    _ensure_list_membership(name, vars3d)


def register_alias(alias: str, target_name: str, *, override: bool = False) -> None:
    """
    Register an alias for a target variable name without changing code2var.

    Note:
    - Alias resolution is provided via _aliases; core lookups continue to use
      codes/code2var. This avoids breaking code that assumes a one-to-one reverse map.
    """
    if not isinstance(alias, str) or not isinstance(target_name, str):
        raise TypeError("alias and target_name must be strings")
    if not is_known_variable(target_name):
        raise KeyError(f"Target variable '{target_name}' is not known.")

    if alias in _aliases and _aliases[alias] == target_name:
        logger.debug("register_alias: idempotent for %s -> %s", alias, target_name)
        return

    if alias in _aliases and _aliases[alias] != target_name and not override:
        raise ValueError(
            f"Alias '{alias}' already mapped to '{_aliases[alias]}'. "
            "Use override=True to change."
        )

    if alias in _aliases and _aliases[alias] != target_name:
        logger.warning(
            "Overriding alias mapping: %s: %s -> %s",
            alias,
            _aliases[alias],
            target_name,
        )

    _aliases[alias] = target_name
