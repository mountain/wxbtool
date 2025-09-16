# -*- coding: utf-8 -*-

import os
import pathlib
import unittest
import unittest.mock as mock

import numpy as np


class TestWindowArray(unittest.TestCase):
    """
    Test case for the WindowArray class.

    This test verifies the functionality of the WindowArray class with
    optimized test cases that cover the essential functionality.
    """

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_window_access(self):
        """Test window array access with different shift and step parameters."""
        from wxbtool.data.dataset import WindowArray

        # Create a smaller test array (4x2x3 instead of 6x2x3)
        # This is sufficient to test the functionality while reducing memory usage
        test_array = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)

        # Test with shift=0, step=2
        w = WindowArray(test_array, shift=0, step=2)
        self.assertEqual((2, 2, 3), w.shape)
        self.assertEqual((2, 3), w[0].shape)

        # Verify first window (index 0)
        np.testing.assert_array_equal(
            w[0], np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        )

        # Verify second window (index 1)
        np.testing.assert_array_equal(
            w[1], np.array([[12, 13, 14], [15, 16, 17]], dtype=np.float32)
        )

        # Test with shift=1, step=2
        w = WindowArray(test_array, shift=1, step=2)
        self.assertEqual((2, 2, 3), w.shape)

        # Verify first window (index 0)
        np.testing.assert_array_equal(
            w[0], np.array([[6, 7, 8], [9, 10, 11]], dtype=np.float32)
        )

        # Verify second window (index 1)
        np.testing.assert_array_equal(
            w[1], np.array([[18, 19, 20], [21, 22, 23]], dtype=np.float32)
        )
