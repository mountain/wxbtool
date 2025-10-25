# -*- coding: utf-8 -*-


def norm_gpt(x):
    min, max = -10000, 500000
    return (x - min) / (max - min)


def norm_tmp(x):
    min, max = 173, 373
    return (x - min) / (max - min)


def norm_shm(x):
    x = x * (x > 0)
    min, max = 0, 0.1
    return (x - min) / (max - min)


def norm_rhm(x):
    x = x * (x > 0)
    min, max = 0.0, 200.0
    return (x - min) / (max - min)


def norm_u(x):
    min, max = -250.0, 250.0
    return (x - min) / (max - min)


def norm_v(x):
    min, max = -250.0, 250.0
    return (x - min) / (max - min)


def norm_tisr(tisr):
    return tisr / 5500000.0


def norm_tcc(tcc):
    return tcc


def denorm_gpt(x):
    return 510000 * x - 10000


def denorm_tmp(x):
    return 173 + 200.0 * x


def denorm_shm(x):
    return 0.1 * x


def denorm_rhm(x):
    return 200.0 * x


def denorm_u(x):
    return x * 500 - 250.0


def denorm_v(x):
    return x * 500 - 250.0


def denorm_tisr(tisr):
    return tisr * 5500000.0


def denorm_tcc(tcc):
    return tcc


def identical(x):
    return x


normalizors = {
    "geopotential": norm_gpt,
    "temperature": norm_tmp,
    "specific_humidity": norm_shm,
    "relative_humidity": norm_rhm,
    "u_component_of_wind": norm_u,
    "v_component_of_wind": norm_v,
    "toa_incident_solar_radiation": norm_tisr,
    "total_cloud_cover": norm_tcc,
    "2m_temperature": norm_tmp,
    "test_variable": identical,
    "test": identical,
    "data": identical,
}


denormalizors = {
    "geopotential": denorm_gpt,
    "temperature": denorm_tmp,
    "specific_humidity": denorm_shm,
    "relative_humidity": denorm_rhm,
    "u_component_of_wind": denorm_u,
    "v_component_of_wind": denorm_v,
    "toa_incident_solar_radiation": denorm_tisr,
    "total_cloud_cover": denorm_tcc,
    "2m_temperature": denorm_tmp,
    "test_variable": identical,
    "test": identical,
    "data": identical,
}

min_values = {
    "geopotential": -10000,
    "temperature": 173,
    "specific_humidity": 0,
    "relative_humidity": 0,
    "u_component_of_wind": -250,
    "v_component_of_wind": -250,
    "toa_incident_solar_radiation": 0,
    "total_cloud_cover": 0,
    "2m_temperature": 223,
    "test_variable": 0,
    "test": 0,
    "data": 0,
}

max_values = {
    "geopotential": 500000,
    "temperature": 373,
    "specific_humidity": 0.1,
    "relative_humidity": 200,
    "u_component_of_wind": 250,
    "v_component_of_wind": 250,
    "toa_incident_solar_radiation": 5500000,
    "total_cloud_cover": 1,
    "2m_temperature": 318,
    "test_variable": 1,
    "test": 1,
    "data": 1,
}
