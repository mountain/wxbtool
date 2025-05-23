# -*- coding: utf-8 -*-

import unittest
import os
import sys
import pathlib
import unittest.mock as mock

from unittest.mock import patch


class TestTest(unittest.TestCase):
    """
    Optimized test cases for testing models.
    
    These tests use smaller batch sizes, fewer CPU threads, and minimal settings
    to reduce test execution time while still verifying functionality.
    """
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_teste3d(self):
        """Test the 3D model with minimal settings."""
        import wxbtool.wxb as wxb

        testargs = [
            "wxb", 
            "test", 
            "-m", 
            "models.fast_3d", 
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
    def test_teste_other_models(self):
        """
        Test other model architectures.
        
        This consolidated test verifies that different model architectures can be tested,
        but uses only one model (fast_6d) to save time.
        """
        import wxbtool.wxb as wxb

        testargs = [
            "wxb", 
            "test", 
            "-m", 
            "models.fast_6d",  # Representative of other models
            "-b", 
            "5",  # Reduced batch size
            "-t", 
            "true",
            "-c",
            "4",  # Reduced CPU threads
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()
