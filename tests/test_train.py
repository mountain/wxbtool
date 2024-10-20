# -*- coding: utf-8 -*-

import unittest
import os
import sys
import pathlib
import unittest.mock as mock

from unittest.mock import patch


class TestTrain(unittest.TestCase):

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_train3d(self):
        import wxbtool.wxb as wxb

        testargs = ["wxb", "train", "-m", "models.fast_3d", "-b", "1", "-n", "1"]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_train6d(self):
        import wxbtool.wxb as wxb

        testargs = ["wxb", "train", "-m", "models.fast_6d", "-b", "1", "-n", "1"]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_train10d(self):
        import wxbtool.wxb as wxb

        testargs = ["wxb", "train", "-m", "models.fast_10d", "-b", "1", "-n", "1"]
        with patch.object(sys, "argv", testargs):
            wxb.main()
