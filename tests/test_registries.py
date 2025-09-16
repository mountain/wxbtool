import copy

import pytest

import wxbtool.data.variables as variables
import wxbtool.norms.meanstd as meanstd


@pytest.fixture(autouse=True)
def restore_registries():
    """
    Snapshot and restore global registries in variables and meanstd to avoid test leakage.
    """
    # Snapshot
    vars2d_orig = list(variables.vars2d)
    vars3d_orig = list(variables.vars3d)
    codes_orig = copy.deepcopy(variables.codes)
    code2var_orig = copy.deepcopy(variables.code2var)
    # meanstd registries
    normalizors_orig = copy.deepcopy(meanstd.normalizors)
    denormalizors_orig = copy.deepcopy(meanstd.denormalizors)

    yield

    # Restore
    variables.vars2d[:] = vars2d_orig
    variables.vars3d[:] = vars3d_orig
    variables.codes.clear()
    variables.codes.update(codes_orig)
    variables.code2var.clear()
    variables.code2var.update(code2var_orig)

    meanstd.normalizors.clear()
    meanstd.normalizors.update(normalizors_orig)
    meanstd.denormalizors.clear()
    meanstd.denormalizors.update(denormalizors_orig)


def test_is_known_variable_and_supported_snapshot():
    snap = variables.get_supported_variables()
    assert "vars2d" in snap and "vars3d" in snap
    # some known built-ins
    assert variables.is_known_variable("2m_temperature")
    assert variables.is_known_variable("geopotential")
    # unknown
    assert not variables.is_known_variable("totally_unknown_variable_foo")


def test_register_var2d_add_and_idempotent():
    name = "sea_surface_temperature"
    code = "sst"

    assert not variables.is_known_variable(name)
    variables.register_var2d(name, code)
    assert variables.is_known_variable(name)
    assert variables.codes[name] == code
    assert variables.code2var[code] == name
    assert name in variables.vars2d
    assert name not in variables.vars3d

    # idempotent
    variables.register_var2d(name, code)
    assert variables.codes[name] == code
    assert variables.code2var[code] == name
    assert name in variables.vars2d


def test_register_var3d_add_and_idempotent():
    name = "my3d_variable"
    code = "c3d"

    variables.register_var3d(name, code)
    assert variables.codes[name] == code
    assert variables.code2var[code] == name
    assert name in variables.vars3d
    assert name not in variables.vars2d

    # idempotent
    variables.register_var3d(name, code)
    assert variables.codes[name] == code
    assert variables.code2var[code] == name


def test_dimensionality_conflict_requires_override():
    name = "dim_conflict"
    variables.register_var2d(name, "c2d")
    with pytest.raises(ValueError):
        variables.register_var3d(name, "c3d", override=False)

    # with override switches dimensionality
    variables.register_var3d(name, "c3d", override=True)
    assert name in variables.vars3d and name not in variables.vars2d
    assert variables.codes[name] == "c3d"
    assert variables.code2var["c3d"] == name


def test_code_collision_requires_override_and_reassigns():
    # create an initial owner for a new code
    variables.register_var2d("owner_a", "cA")
    assert variables.code2var["cA"] == "owner_a"

    # attempt to take over same code without override
    with pytest.raises(ValueError):
        variables.register_var2d("owner_b", "cA", override=False)

    # now override and ensure reassignment
    variables.register_var2d("owner_b", "cA", override=True)
    assert variables.code2var["cA"] == "owner_b"
    assert variables.codes.get("owner_a") != "cA"
    # owner_a should be removed from var lists
    assert "owner_a" not in variables.vars2d
    assert "owner_a" not in variables.vars3d


def test_register_alias_and_is_known_variable():
    variables.register_var2d("sea_surface_temperature", "sst")
    # alias target must exist
    variables.register_alias("sst_alias", "sea_surface_temperature")
    # alias should be treated as known
    assert variables.is_known_variable("sst_alias")


def test_normalizer_registry_by_name_and_code():
    # Ensure a mapping exists so name resolves to code in normalizer registry
    variables.register_var2d("snow_depth", "sd")

    def norm_fn(x):
        return x

    def denorm_fn(x):
        return x

    # Register by variable name
    meanstd.register_normalizer("snow_depth", norm_fn)
    # Should be stored under code "sd"
    assert meanstd.get_normalizer("sd") is norm_fn
    assert (
        meanstd.get_normalizer("snow_depth") is norm_fn
    )  # resolves to code internally

    # Register denormalizer by code
    meanstd.register_denormalizer("sd", denorm_fn)
    assert meanstd.get_denormalizer("sd") is denorm_fn
    assert meanstd.get_denormalizer("snow_depth") is denorm_fn


def test_normalizer_override_semantics():
    variables.register_var2d("snow_depth", "sd")

    def fn1(x):
        return x

    def fn2(x):
        return x + 1 if hasattr(x, "__add__") else x

    meanstd.register_normalizer("snow_depth", fn1)
    with pytest.raises(ValueError):
        meanstd.register_normalizer("snow_depth", fn2, override=False)

    meanstd.register_normalizer("snow_depth", fn2, override=True)
    assert meanstd.get_normalizer("sd") is fn2

    meanstd.register_denormalizer("sd", fn1)
    with pytest.raises(ValueError):
        meanstd.register_denormalizer("sd", fn2, override=False)

    meanstd.register_denormalizer("sd", fn2, override=True)
    assert meanstd.get_denormalizer("sd") is fn2
