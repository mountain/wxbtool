# -*- coding: utf-8 -*-

import importlib.util
import os
import pathlib
import sys
import unittest
import unittest.mock as mock
from unittest.mock import patch

import pytest

CONSTANTS_FILE = pathlib.Path(__file__).parent / "constants" / "constants_5.625deg.nc"
if importlib.util.find_spec("arghandler") is None or not CONSTANTS_FILE.exists():
    pytest.skip(
        "integration tests require arghandler and constants dataset",
        allow_module_level=True,
    )


class TestDataset(unittest.TestCase):
    """
    Optimized test cases for dataset functionality.

    These tests use smaller batch sizes, fewer workers, and consolidated test cases
    to reduce test execution time while still verifying functionality.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_dataset(self):
        """Test basic dataset functionality with minimal settings."""
        import wxbtool.wxb as wxb

        # Start dataset server
        testargs = [
            "wxb",
            "data-serve",
            "-m",
            "models.model",
            "-s",
            "Setting3d",
            "-t",
            "true",
            "-w",
            "2",  # Reduced workers
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()

        # Test with the dataset
        testargs = [
            "wxb",
            "test",
            "-m",
            "models.model",
            "-b",
            "5",  # Reduced batch size
            "-t",
            "true",
            "-c",
            "4",  # Reduced CPU threads
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_unix_socket(self):
        """Test Unix socket communication with minimal settings."""
        import wxbtool.wxb as wxb

        # Start dataset server with Unix socket
        testargs = [
            "wxb",
            "data-serve",
            "-m",
            "models.model",
            "-s",
            "Setting3d",
            "-t",
            "true",
            "-w",
            "2",  # Reduced workers
            "--bind",
            "unix:/tmp/test.sock",
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()

        # Test with the Unix socket
        testargs = [
            "wxb",
            "test",
            "-m",
            "models.model",
            "-b",
            "5",  # Reduced batch size
            "-t",
            "true",
            "-c",
            "4",  # Reduced CPU threads
            "--data",
            "unix:/tmp/test.sock",
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_climate_model(self):
        """Test climate model with minimal settings."""
        import wxbtool.wxb as wxb

        # Start dataset server for climate model
        testargs = [
            "wxb",
            "data-serve",
            "-m",
            "spec.spec",
            "-s",
            "Setting30d",
            "-t",
            "true",
            "-w",
            "2",  # Reduced workers
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()

        # Test climate model
        testargs = [
            "wxb",
            "test",
            "-m",
            "models.climate_30d",
            "-b",
            "10",  # Reduced batch size (from 30)
            "-t",
            "true",
            "-c",
            "4",  # Reduced CPU threads
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()
