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

        testargs = [
            "wxb",
            "train",
            "-m",
            "models.fast_3d",
            "-b",
            "10",
            "-n",
            "1",
            "-t",
            "true",
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_train6d(self):
        import wxbtool.wxb as wxb

        testargs = [
            "wxb",
            "train",
            "-m",
            "models.fast_6d",
            "-b",
            "10",
            "-n",
            "1",
            "-t",
            "true",
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_train10d(self):
        import wxbtool.wxb as wxb

        testargs = [
            "wxb",
            "train",
            "-m",
            "models.fast_10d",
            "-b",
            "10",
            "-n",
            "1",
            "-t",
            "true",
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_train10d_epoch2(self):
        import wxbtool.wxb as wxb

        testargs = ["wxb", "train", "-m", "models.fast_10d", "-b", "10", "-n", "2"]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_train10d_gan(self):
        import wxbtool.wxb as wxb

        testargs = ["wxb", "train", "-m", "models.fast_gan", "-b", "10", "-n", "2", "-G", "true"]
        with patch.object(sys, "argv", testargs):
            wxb.main()
