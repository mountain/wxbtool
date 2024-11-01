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
    def test_dataset(self):
        import wxbtool.wxb as wxb

        testargs = [
            "wxb",
            "dserve",
            "-m",
            "models.model",
            "-s",
            "Setting3d",
            "-t",
            "true",
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()

        testargs = ["wxb", "test", "-m", "models.model", "-b", "10", "-t", "true"]
        with patch.object(sys, "argv", testargs):
            wxb.main()

    @mock.patch.dict(
        os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())}
    )
    def test_unix_socket(self):
        import wxbtool.wxb as wxb

        testargs = [
            "wxb",
            "dserve",
            "-m",
            "models.model",
            "-s",
            "Setting3d",
            "-t",
            "true",
            "--bind",
            "unix:/tmp/test.sock",
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()

        testargs = [
            "wxb",
            "test",
            "-m",
            "models.model",
            "-b",
            "10",
            "-t",
            "true",
            "--data",
            "unix:/tmp/test.sock",
        ]
        with patch.object(sys, "argv", testargs):
            wxb.main()
