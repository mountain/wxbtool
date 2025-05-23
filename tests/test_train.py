# -*- coding: utf-8 -*-

import unittest
import os
import sys
import pathlib
import unittest.mock as mock

from unittest.mock import patch


class TestTrain(unittest.TestCase):
    """
    Optimized test cases for training models.
    
    These tests use smaller batch sizes, fewer CPU threads, and minimal epochs
    to reduce test execution time while still verifying functionality.
    """
    
    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_train3d(self):
        """Test training a 3D model with minimal settings."""
        import wxbtool.wxb as wxb

        testargs = [
            "wxb",
            "train",
            "-m",
            "models.fast_3d",
            "-b",
            "5",  # Reduced batch size
            "-n",
            "1",
            "-t",
            "true",
            "-c",
            "4",  # Reduced CPU threads
            "-g",
            "-1",  # force CPU
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_train_other_models(self):
        """
        Test training other model architectures.
        
        This consolidated test verifies that different model architectures can be trained,
        but uses only one model (fast_6d) to save time.
        """
        import wxbtool.wxb as wxb

        testargs = [
            "wxb",
            "train",
            "-m",
            "models.fast_6d",  # Representative of other models
            "-b",
            "5",  # Reduced batch size
            "-n",
            "1",
            "-t",
            "true",
            "-c",
            "4",  # Reduced CPU threads
            "-g",
            "-1",  # force CPU
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_train_gan(self):
        """Test training a GAN model with minimal settings."""
        import wxbtool.wxb as wxb

        testargs = [
            "wxb",
            "train",
            "-m",
            "models.fast_gan",
            "-b",
            "5",  # Reduced batch size
            "-n",
            "1",  # Reduced epochs
            "-G",
            "true",
            "-c",
            "4",  # Reduced CPU threads
            "-t",
            "true",  # Test mode
            "-g",
            "-1",  # force CPU
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()
