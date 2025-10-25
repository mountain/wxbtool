# -*- coding: utf-8 -*-
"""
Central utilities for managing global registries.

This module provides a clean_registries() function to reset registry state
across wxbtool's variable and normalization registries.

WARNING:
    This is a destructive operation intended for controlled scenarios such as
    tests or interactive sessions where you explicitly want to clear all
    registered variables and normalizers. Do NOT call this in normal training,
    inference, or production workflows.
"""
from __future__ import annotations

import logging

from wxbtool.data import variables
from wxbtool.norms import meanstd

logger = logging.getLogger(__name__)


def clean_registries() -> None:
    """
    Reset registry containers to empty lists/dicts across variables and norms.

    Clears (if present):
      - wxbtool.data.variables:
          * vars2d (list)
          * vars3d (list)
          * vars4d (list)           [optional; cleared only if defined]
          * codes (dict)
          * code2var (dict)
          * _aliases (dict; internal alias map)
      - wxbtool.norms.meanstd:
          * normalizors (dict)
          * denormalizors (dict)
    """
    # Variables module
    if hasattr(variables, "vars2d") and isinstance(variables.vars2d, list):
        variables.vars2d.clear()
        logger.debug("Cleared variables.vars2d")

    if hasattr(variables, "vars3d") and isinstance(variables.vars3d, list):
        variables.vars3d.clear()
        logger.debug("Cleared variables.vars3d")

    # Some codebases may have a vars4d; clear if present
    if hasattr(variables, "vars4d") and isinstance(getattr(variables, "vars4d"), list):
        variables.vars4d.clear()  # type: ignore[attr-defined]
        logger.debug("Cleared variables.vars4d")

    if hasattr(variables, "codes") and isinstance(variables.codes, dict):
        variables.codes.clear()
        logger.debug("Cleared variables.codes")

    if hasattr(variables, "code2var") and isinstance(variables.code2var, dict):
        variables.code2var.clear()
        logger.debug("Cleared variables.code2var")

    # Internal alias map
    if hasattr(variables, "_aliases") and isinstance(variables._aliases, dict):  # type: ignore[attr-defined]
        variables._aliases.clear()  # type: ignore[attr-defined]
        logger.debug("Cleared variables._aliases")

    # Normalization registries
    if hasattr(meanstd, "normalizors") and isinstance(meanstd.normalizors, dict):
        meanstd.normalizors.clear()
        logger.debug("Cleared meanstd.normalizors")

    if hasattr(meanstd, "denormalizors") and isinstance(meanstd.denormalizors, dict):
        meanstd.denormalizors.clear()
        logger.debug("Cleared meanstd.denormalizors")

    logger.info("All registries have been cleared.")
