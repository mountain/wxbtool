# -*- coding: utf-8 -*-

import unittest
import numpy as np


class TestWindowArray(unittest.TestCase):

    def test_window_access(self):
        from wxbtool.data.dataset import WindowArray

        # We have a 6x2x3 array:
        # i.e. 6 temporal samples, each with 2x3 pixels, the 6 samples are:
        # 0: [[0, 1, 2], [3, 4, 5]]
        # 1: [[6, 7, 8], [9, 10, 11]]
        # 2: [[12, 13, 14], [15, 16, 17]]
        # 3: [[18, 19, 20], [21, 22, 23]]
        # 4: [[24, 25, 26], [27, 28, 29]]
        # 5: [[30, 31, 32], [33, 34, 35]]

        # We want to create a window array with a step 2, each with 2x3 pixels
        # and a shift of 0 so the samples will be:
        # 0: [[0, 1, 2], [3, 4, 5]]
        # 1: [[12, 13, 14], [15, 16, 17]]
        # 2: [[24, 25, 26], [27, 28, 29]]

        w = WindowArray(
            np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3), shift=0, step=2
        )
        self.assertEqual((2, 2, 3), w.shape)
        self.assertEqual((2, 3), w[0].shape)
        self.assertEqual(0, np.array(w[0, 0, 0], dtype=np.float32))
        self.assertEqual(1, np.array(w[0, 0, 1], dtype=np.float32))
        self.assertEqual(2, np.array(w[0, 0, 2], dtype=np.float32))
        self.assertEqual(3, np.array(w[0, 1, 0], dtype=np.float32))
        self.assertEqual(4, np.array(w[0, 1, 1], dtype=np.float32))
        self.assertEqual(5, np.array(w[0, 1, 2], dtype=np.float32))

        w = WindowArray(
            np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3), shift=1, step=2
        )
        self.assertEqual((2, 2, 3), w.shape)
        self.assertEqual((2, 3), w[0].shape)
        self.assertEqual(6, np.array(w[0, 0, 0], dtype=np.float32))
        self.assertEqual(7, np.array(w[0, 0, 1], dtype=np.float32))
        self.assertEqual(8, np.array(w[0, 0, 2], dtype=np.float32))
        self.assertEqual(9, np.array(w[0, 1, 0], dtype=np.float32))
        self.assertEqual(10, np.array(w[0, 1, 1], dtype=np.float32))
        self.assertEqual(11, np.array(w[0, 1, 2], dtype=np.float32))

        w = WindowArray(
            np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3), shift=0, step=2
        )
        self.assertEqual((2, 2, 3), w.shape)
        self.assertEqual((2, 3), w[0].shape)
        self.assertEqual(12, np.array(w[1, 0, 0], dtype=np.float32))
        self.assertEqual(13, np.array(w[1, 0, 1], dtype=np.float32))
        self.assertEqual(14, np.array(w[1, 0, 2], dtype=np.float32))
        self.assertEqual(15, np.array(w[1, 1, 0], dtype=np.float32))
        self.assertEqual(16, np.array(w[1, 1, 1], dtype=np.float32))
        self.assertEqual(17, np.array(w[1, 1, 2], dtype=np.float32))

        w = WindowArray(
            np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3), shift=1, step=2
        )
        self.assertEqual((2, 2, 3), w.shape)
        self.assertEqual((2, 3), w[0].shape)
        self.assertEqual(18, np.array(w[1, 0, 0], dtype=np.float32))
        self.assertEqual(19, np.array(w[1, 0, 1], dtype=np.float32))
        self.assertEqual(20, np.array(w[1, 0, 2], dtype=np.float32))
        self.assertEqual(21, np.array(w[1, 1, 0], dtype=np.float32))
        self.assertEqual(22, np.array(w[1, 1, 1], dtype=np.float32))
        self.assertEqual(23, np.array(w[1, 1, 2], dtype=np.float32))
