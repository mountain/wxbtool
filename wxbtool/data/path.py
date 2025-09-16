# -*- coding: utf-8 -*-
"""
DataPathManager: generate dataset file paths from configurable format strings.

Supported placeholders in data_path_format:
- {var}: variable name (e.g., "2m_temperature")
- {resolution}: resolution string (e.g., "5.625deg")
- {year}: four-digit year (int)
- {month}: 1-12 (int), can be formatted as {month:02d}
- {day}: 1-31 (int), can be formatted as {day:02d}
- {hour}: 0-23 (int), can be formatted as {hour:02d}
- {week}: ISO week number 1-53 (int), can be formatted as {week:02d}
- {quarter}: 1-4 (int)

Example formats:
- Yearly (default): "{var}_{year}_{resolution}.nc"
- Monthly: "{year}/{var}_{year}-{month:02d}_{resolution}.nc"
- Daily: "{year}/{month:02d}/{var}_{year}-{month:02d}-{day:02d}_{resolution}.nc"
- Hourly: "{year}/{month:02d}/{day:02d}/{var}_{year}-{month:02d}-{day:02d}T{hour:02d}_{resolution}.nc"
"""
from __future__ import annotations

import os
from typing import Iterable, List

import pandas as pd


class DataPathManager:
    """Stateless helper that formats file paths based on a date range and format."""

    @staticmethod
    def _compute_fields(ts: pd.Timestamp, var: str, resolution: str) -> dict:
        # ISO week with Monday as the first day of the week
        week = int(ts.strftime("%V"))
        quarter = (ts.month - 1) // 3 + 1
        hour = int(getattr(ts, "hour", 0))
        return {
            "var": var,
            "resolution": resolution,
            "year": int(ts.year),
            "month": int(ts.month),
            "day": int(ts.day),
            "hour": hour,
            "week": week,
            "quarter": quarter,
        }

    @classmethod
    def get_file_paths(
        cls,
        root: str,
        var: str,
        resolution: str,
        data_path_format: str,
        date_range: Iterable[pd.Timestamp],
    ) -> List[str]:
        """Generate sorted unique file paths for a variable over a date range.

        Args:
            root: Dataset root directory
            var: Variable name
            resolution: Resolution string
            data_path_format: Python format string with placeholders
            date_range: Iterable of pandas Timestamps (e.g., pd.date_range(...))

        Returns:
            Sorted list of unique paths
        """
        paths = []
        for ts in date_range:
            fields = cls._compute_fields(ts, var, resolution)
            try:
                relative = data_path_format.format(**fields)
            except KeyError as e:
                raise KeyError(
                    f"Unknown placeholder {e} in data_path_format='{data_path_format}'. "
                    f"Supported keys: {sorted(fields.keys())}"
                )
            full_path = os.path.join(root, var, relative)
            paths.append(full_path)

        # Deduplicate while preserving order
        seen = set()
        unique_paths: List[str] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)

        return sorted(unique_paths)
