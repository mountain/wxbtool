# -*- coding: utf-8 -*-

"""
Tests for the resolution configuration system
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

root = Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(root))
os.environ.setdefault("WXBHOME", str(root))

from wxbtool.core.resolution import ResolutionConfig  # noqa: E402
from wxbtool.core.resolution import ResolutionValidator  # noqa: E402
from wxbtool.core.setting import Setting  # noqa: E402


class TestResolutionConfig:
    """Test cases for ResolutionConfig class"""

    def test_get_config_valid_resolution(self):
        """Test getting configuration for valid resolution"""
        config = ResolutionConfig.get_config("5.625deg")

        assert config["lat_size"] == 32
        assert config["lon_size"] == 64
        assert config["lat_range"] == (87.1875, -87.1875)
        assert config["lon_range"] == (0, 354.375)
        assert config["lat_step"] == 5.625
        assert config["lon_step"] == 5.625
        assert "description" in config

    def test_get_config_invalid_resolution(self):
        """Test getting configuration for invalid resolution"""
        with pytest.raises(ValueError, match="Unsupported resolution"):
            ResolutionConfig.get_config("invalid_resolution")

    def test_get_supported_resolutions(self):
        """Test getting list of supported resolutions"""
        resolutions = ResolutionConfig.get_supported_resolutions()

        assert isinstance(resolutions, list)
        assert "5.625deg" in resolutions
        assert "2.8125deg" in resolutions
        assert len(resolutions) >= 2

    def test_get_spatial_shape(self):
        """Test getting spatial shape for resolutions"""
        shape_5625 = ResolutionConfig.get_spatial_shape("5.625deg")
        assert shape_5625 == (32, 64)

        shape_2812 = ResolutionConfig.get_spatial_shape("2.8125deg")
        assert shape_2812 == (64, 128)

    def test_get_lat_range(self):
        """Test getting latitude range for resolutions"""
        lat_range = ResolutionConfig.get_lat_range("5.625deg")
        assert lat_range == (87.1875, -87.1875)

    def test_get_lon_range(self):
        """Test getting longitude range for resolutions"""
        lon_range = ResolutionConfig.get_lon_range("5.625deg")
        assert lon_range == (0, 354.375)

    def test_config_immutability(self):
        """Test that config returns are copies (not references)"""
        config1 = ResolutionConfig.get_config("5.625deg")
        config2 = ResolutionConfig.get_config("5.625deg")

        # Modify one config
        config1["lat_size"] = 999

        # Original should be unchanged
        config2 = ResolutionConfig.get_config("5.625deg")
        assert config2["lat_size"] == 32


class TestResolutionValidator:
    """Test cases for ResolutionValidator class"""

    def test_validate_setting_valid(self):
        """Test validation of valid setting"""
        setting = Setting()
        setting.resolution = "5.625deg"
        setting.lat_size = 32
        setting.lon_size = 64

        # Should not raise any exception
        ResolutionValidator.validate_setting(setting)

    def test_validate_setting_missing_resolution(self):
        """Test validation of setting without resolution"""

        class MockSetting:
            pass

        setting = MockSetting()

        with pytest.raises(
            ValueError, match="Setting object must have a 'resolution' attribute"
        ):
            ResolutionValidator.validate_setting(setting)

    def test_validate_setting_invalid_resolution(self):
        """Test validation of setting with invalid resolution"""

        class MockSetting:
            def __init__(self):
                self.resolution = "invalid"

        setting = MockSetting()

        with pytest.raises(ValueError, match="Invalid resolution in setting"):
            ResolutionValidator.validate_setting(setting)

    def test_validate_setting_lat_size_mismatch(self):
        """Test validation of setting with wrong lat_size"""

        class MockSetting:
            def __init__(self):
                self.resolution = "5.625deg"
                self.lat_size = 16  # Wrong size

        setting = MockSetting()

        with pytest.raises(ValueError, match="lat_size mismatch"):
            ResolutionValidator.validate_setting(setting)

    def test_validate_setting_lon_size_mismatch(self):
        """Test validation of setting with wrong lon_size"""

        class MockSetting:
            def __init__(self):
                self.resolution = "5.625deg"
                self.lat_size = 32
                self.lon_size = 32  # Wrong size

        setting = MockSetting()

        with pytest.raises(ValueError, match="lon_size mismatch"):
            ResolutionValidator.validate_setting(setting)

    def test_validate_spatial_shape_valid(self):
        """Test validation of valid spatial shape"""
        # Should not raise exception
        ResolutionValidator.validate_spatial_shape((32, 64), "5.625deg")

    def test_validate_spatial_shape_invalid(self):
        """Test validation of invalid spatial shape"""
        with pytest.raises(ValueError, match="Spatial shape .* doesn't match expected"):
            ResolutionValidator.validate_spatial_shape((16, 32), "5.625deg")


class TestSetting:
    """Test cases for enhanced Setting class"""

    def test_spatial_config_loading(self):
        """Test that spatial configuration is loaded correctly"""
        setting = Setting()

        assert setting.resolution == "5.625deg"
        assert setting.lat_size == 32
        assert setting.lon_size == 64
        assert setting.spatial_shape == (32, 64)
        assert setting.total_spatial_size == 32 * 64
        assert setting.lat_range == (87.1875, -87.1875)
        assert setting.lon_range == (0, 354.375)

    def test_get_latitude_array(self):
        """Test latitude array generation"""
        setting = Setting()
        lat_array = setting.get_latitude_array()

        assert isinstance(lat_array, np.ndarray)
        assert len(lat_array) == 32
        assert lat_array[0] == 87.1875  # North
        assert lat_array[-1] == -87.1875  # South

    def test_get_longitude_array(self):
        """Test longitude array generation"""
        setting = Setting()
        lon_array = setting.get_longitude_array()

        assert isinstance(lon_array, np.ndarray)
        assert len(lon_array) == 64
        assert lon_array[0] == 0  # West
        assert lon_array[-1] == 354.375  # East

    def test_get_meshgrid_unnormalized(self):
        """Test meshgrid generation with actual coordinates"""
        setting = Setting()
        lon_grid, lat_grid = setting.get_meshgrid(normalized=False)

        assert lon_grid.shape == (32, 64)
        assert lat_grid.shape == (32, 64)
        assert lon_grid[0, 0] == 0  # West edge
        assert lat_grid[0, 0] == 87.1875  # North edge

    def test_get_meshgrid_normalized(self):
        """Test meshgrid generation with normalized coordinates"""
        setting = Setting()
        lon_grid, lat_grid = setting.get_meshgrid(normalized=True)

        assert lon_grid.shape == (32, 64)
        assert lat_grid.shape == (32, 64)
        assert lon_grid[0, 0] == 0.0
        assert lat_grid[0, 0] == 0.0
        assert lon_grid[0, -1] == 1.0
        assert lat_grid[-1, 0] == 1.0


class TestMultiResolutionSupport:
    """Test cases for multi-resolution functionality"""

    @pytest.mark.parametrize(
        "resolution,expected_shape",
        [
            ("5.625deg", (32, 64)),
            ("2.8125deg", (64, 128)),
            ("1.40625deg", (128, 256)),
        ],
    )
    def test_different_resolutions(self, resolution, expected_shape):
        """Test that different resolutions work correctly"""

        # Create a setting with custom resolution
        class CustomSetting(Setting):
            def __init__(self, resolution):
                self.root = ""
                self.resolution = resolution
                self._load_spatial_config()
                # Add other required attributes for completeness
                self.name = "test"
                self.step = 4
                self.input_span = 3
                self.pred_span = 1
                self.pred_shift = 72
                self.levels = ["500", "850"]
                self.height = 2

        setting = CustomSetting(resolution)

        assert setting.spatial_shape == expected_shape
        assert setting.lat_size == expected_shape[0]
        assert setting.lon_size == expected_shape[1]

        # Test coordinate arrays
        lat_array = setting.get_latitude_array()
        lon_array = setting.get_longitude_array()

        assert len(lat_array) == expected_shape[0]
        assert len(lon_array) == expected_shape[1]

        # Test meshgrid
        lon_grid, lat_grid = setting.get_meshgrid()
        assert lon_grid.shape == expected_shape
        assert lat_grid.shape == expected_shape
