# -*- coding: utf-8 -*-
import pandas as pd

from wxbtool.data.path import DataPathManager


def test_yearly_paths():
    root = "/data"
    var = "2m_temperature"
    resolution = "5.625deg"
    fmt = "{var}_{year}_{resolution}.nc"

    dr = pd.date_range("1980-01-01", "1982-12-31", freq="YS")
    paths = DataPathManager.get_file_paths(root, var, resolution, fmt, dr)

    assert paths == sorted(
        [
            f"{root}/{var}/2m_temperature_1980_{resolution}.nc",
            f"{root}/{var}/2m_temperature_1981_{resolution}.nc",
            f"{root}/{var}/2m_temperature_1982_{resolution}.nc",
        ]
    )


def test_monthly_paths():
    root = "/data"
    var = "geopotential"
    resolution = "5.625deg"
    fmt = "{year}/{var}_{year}-{month:02d}_{resolution}.nc"

    dr = pd.date_range("1980-01-01", "1980-03-31", freq="MS")
    paths = DataPathManager.get_file_paths(root, var, resolution, fmt, dr)

    assert paths == sorted(
        [
            f"{root}/{var}/1980/{var}_1980-01_{resolution}.nc",
            f"{root}/{var}/1980/{var}_1980-02_{resolution}.nc",
            f"{root}/{var}/1980/{var}_1980-03_{resolution}.nc",
        ]
    )


def test_daily_paths():
    root = "/data"
    var = "temperature"
    resolution = "5.625deg"
    fmt = "{year}/{month:02d}/{var}_{year}-{month:02d}-{day:02d}_{resolution}.nc"

    dr = pd.date_range("1981-12-30", "1982-01-02", freq="D")
    paths = DataPathManager.get_file_paths(root, var, resolution, fmt, dr)

    expected = [
        f"{root}/{var}/1981/12/{var}_1981-12-30_{resolution}.nc",
        f"{root}/{var}/1981/12/{var}_1981-12-31_{resolution}.nc",
        f"{root}/{var}/1982/01/{var}_1982-01-01_{resolution}.nc",
        f"{root}/{var}/1982/01/{var}_1982-01-02_{resolution}.nc",
    ]
    assert paths == sorted(expected)


def test_bad_placeholder_raises():
    root = "/data"
    var = "2m_temperature"
    resolution = "5.625deg"
    fmt = "{unknown_placeholder}.nc"
    dr = pd.date_range("1980-01-01", "1980-01-01", freq="D")

    try:
        DataPathManager.get_file_paths(root, var, resolution, fmt, dr)
        assert False, "Expected KeyError for unknown placeholder"
    except KeyError:
        assert True
