# -*- coding: utf-8 -*-

import unittest
import os
import sys
import pathlib
import unittest.mock as mock

from unittest.mock import patch


class TestTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_teste3d(self):
        import wxbtool.wxb as wxb

        testargs = ["wxb", "test", "-m", "models.fast_3d", "-b", "10", "-t", "true"]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_teste6d(self):
        import wxbtool.wxb as wxb

        testargs = ["wxb", "test", "-m", "models.fast_6d", "-b", "10", "-t", "true"]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_teste10d(self):
        import wxbtool.wxb as wxb

        testargs = ["wxb", "test", "-m", "models.fast_10d", "-b", "10", "-t", "true"]
        with patch.object(sys, "argv", testargs):
            wxb.main()
