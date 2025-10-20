# -*- coding: utf-8 -*-

"""
Resolution configuration system for wxbtool

This module provides a centralized registry for different spatial resolutions,
eliminating hard-coded dimensions throughout the codebase and enabling
multi-resolution support.
"""

from typing import Any, Dict, Tuple


class ResolutionConfig:
    """Configuration registry for different spatial resolutions"""

    RESOLUTIONS: Dict[str, Dict[str, Any]] = {
        "5.625deg": {
            "lat_size": 32,
            "lon_size": 64,
            "lat_range": (87.1875, -87.1875),  # North to South
            "lon_range": (0, 354.375),  # West to East
            "lat_step": 5.625,
            "lon_step": 5.625,
            "description": "Standard 5.625 degree resolution (32x64 grid)",
        },
        "2.8125deg": {
            "lat_size": 64,
            "lon_size": 128,
            "lat_range": (87.1875, -87.1875),
            "lon_range": (0, 357.1875),
            "lat_step": 2.8125,
            "lon_step": 2.8125,
            "description": "High resolution 2.8125 degree (64x128 grid)",
        },
        "1.40625deg": {
            "lat_size": 128,
            "lon_size": 256,
            "lat_range": (87.1875, -87.1875),
            "lon_range": (0, 358.59375),
            "lat_step": 1.40625,
            "lon_step": 1.40625,
            "description": "Very high resolution 1.40625 degree (128x256 grid)",
        },
    }

    @classmethod
    def get_config(cls, resolution: str) -> Dict[str, Any]:
        """Get configuration for a specific resolution

        Args:
            resolution: Resolution string (e.g., "5.625deg")

        Returns:
            Dictionary containing resolution configuration

        Raises:
            ValueError: If resolution is not supported
        """
        if resolution not in cls.RESOLUTIONS:
            available = list(cls.RESOLUTIONS.keys())
            raise ValueError(
                f"Unsupported resolution: {resolution}. "
                f"Available resolutions: {available}"
            )
        return cls.RESOLUTIONS[resolution].copy()

    @classmethod
    def get_supported_resolutions(cls) -> list[str]:
        """Get list of supported resolution strings"""
        return list(cls.RESOLUTIONS.keys())

    @classmethod
    def get_spatial_shape(cls, resolution: str) -> Tuple[int, int]:
        """Get spatial shape (lat_size, lon_size) for a resolution

        Args:
            resolution: Resolution string

        Returns:
            Tuple of (lat_size, lon_size)
        """
        config = cls.get_config(resolution)
        return (config["lat_size"], config["lon_size"])

    @classmethod
    def get_lat_range(cls, resolution: str) -> Tuple[float, float]:
        """Get latitude range for a resolution

        Args:
            resolution: Resolution string

        Returns:
            Tuple of (lat_north, lat_south)
        """
        config = cls.get_config(resolution)
        return config["lat_range"]

    @classmethod
    def get_lon_range(cls, resolution: str) -> Tuple[float, float]:
        """Get longitude range for a resolution

        Args:
            resolution: Resolution string

        Returns:
            Tuple of (lon_west, lon_east)
        """
        config = cls.get_config(resolution)
        return config["lon_range"]


class ResolutionValidator:
    """Validation utilities for resolution configurations"""

    @staticmethod
    def validate_setting(setting) -> None:
        """Validate that a Setting object's spatial dimensions match its resolution

        Args:
            setting: Setting object to validate

        Raises:
            ValueError: If spatial dimensions don't match resolution configuration
        """
        if not hasattr(setting, "resolution"):
            raise ValueError("Setting object must have a 'resolution' attribute")

        try:
            expected = ResolutionConfig.get_config(setting.resolution)
        except ValueError as e:
            raise ValueError(f"Invalid resolution in setting: {e}")

        # Check spatial dimensions if they exist
        if hasattr(setting, "lat_size"):
            if setting.lat_size != expected["lat_size"]:
                raise ValueError(
                    f"lat_size mismatch: setting has {setting.lat_size}, "
                    f"expected {expected['lat_size']} for resolution {setting.resolution}"
                )

        if hasattr(setting, "lon_size"):
            if setting.lon_size != expected["lon_size"]:
                raise ValueError(
                    f"lon_size mismatch: setting has {setting.lon_size}, "
                    f"expected {expected['lon_size']} for resolution {setting.resolution}"
                )

    @staticmethod
    def validate_spatial_shape(shape: Tuple[int, int], resolution: str) -> None:
        """Validate that a spatial shape matches the expected resolution

        Args:
            shape: Tuple of (lat_size, lon_size)
            resolution: Resolution string

        Raises:
            ValueError: If shape doesn't match resolution
        """
        expected_shape = ResolutionConfig.get_spatial_shape(resolution)
        if shape != expected_shape:
            raise ValueError(
                f"Spatial shape {shape} doesn't match expected {expected_shape} "
                f"for resolution {resolution}"
            )
