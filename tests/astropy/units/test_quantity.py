"""Unit test for quantity.py"""

import typing as ty

import astropy.units as u
import numpy as np
import numpy.testing as npt
import pydantic
import pytest

from scientific_pydantic.astropy.units import QuantityAdapter


class Unconstrained(pydantic.BaseModel):
    """Test model with an unconstrained quantity"""

    field: ty.Annotated[u.Quantity, QuantityAdapter()]


@pytest.mark.parametrize(
    ("adapter", "value", "truth"),
    [
        pytest.param(
            QuantityAdapter(), 5 << u.m, 5 << u.m, id="unconstrained-scalar-quantity"
        ),
        pytest.param(
            QuantityAdapter(), 5, u.Quantity(5), id="unconstrained-unitless-scalar"
        ),
        pytest.param(
            QuantityAdapter(), "1.234 um", 1.234 << u.um, id="unconstrained-str-scalar"
        ),
        pytest.param(
            QuantityAdapter(),
            [[[2.34]]] << u.s,
            [[[2.34]]] << u.s,
            id="unconstrained-vector-quantity",
        ),
        pytest.param(
            QuantityAdapter(),
            {"value": [1, 2]},
            u.Quantity([1, 2]),
            id="unconstrained-dict-unitless",
        ),
        pytest.param(
            QuantityAdapter(),
            {"value": [1, 2], "unit": "m"},
            [1, 2] << u.m,
            id="unconstrained-dict-m-str",
        ),
        pytest.param(
            QuantityAdapter(),
            {"value": [1, 2], "unit": u.m},
            [1, 2] << u.m,
            id="unconstrained-dict-m-unit",
        ),
        pytest.param(
            QuantityAdapter(scalar=True),
            1.234 << u.s,
            1.234 << u.s,
            id="scalar-True-quantity",
        ),
        pytest.param(
            QuantityAdapter(scalar=False),
            [1.234] << u.s,
            [1.234] << u.s,
            id="scalar-False-quantity",
        ),
        pytest.param(
            QuantityAdapter(u.m),
            [1, 2] << u.mm,
            [1, 2] << u.mm,
            id="equiv-m-quantity",
        ),
        pytest.param(
            QuantityAdapter(u.m, equivalencies=u.spectral()),
            1.234 << u.cm**-1,
            1.234 << u.cm**-1,
            id="equiv-m-quantity-spectral",
        ),
        pytest.param(
            QuantityAdapter(physical_type="mass"),
            [[2], [3], [4]] << u.kg,
            [[2], [3], [4]] << u.kg,
            id="physical-type-mass-quantity-kg",
        ),
        pytest.param(
            QuantityAdapter(ndim=2),
            [[2], [3], [4]] << u.kg,
            [[2], [3], [4]] << u.kg,
            id="ndim-2-quantity",
        ),
        # The finer points of shape validation are tested elsewhere
        pytest.param(
            QuantityAdapter(shape=(..., 2, 2)),
            [[[2, 3], [4, 5]]] << u.s,
            [[[2, 3], [4, 5]]] << u.s,
            id="shape-quantity-1x2x2",
        ),
        pytest.param(
            QuantityAdapter(u.m, gt=0, ge=1, le=2, lt=3),
            [1000, 1500, 2000] << u.mm,
            [1000, 1500, 2000] << u.mm,
            id="all-bounds-float-quantity",
        ),
        pytest.param(
            QuantityAdapter(u.m, gt=[[5], [6]]),
            [[5.1, 5.2], [6.1, 6.2]] << u.m,
            [[5.1, 5.2], [6.1, 6.2]] << u.m,
            id="gt-vector-no-unit-bounds",
        ),
        pytest.param(
            QuantityAdapter(clip=(0 << u.m, None)),
            [-1, 0, 2] << u.mm,
            [0, 0, 2] << u.mm,
            id="clip-(quantity,None)",
        ),
        pytest.param(
            QuantityAdapter(clip=(None, 1e-3 << u.m)),
            [-1, 0, 2] << u.mm,
            [-1, 0, 1] << u.mm,
            id="clip-(None, quantity)",
        ),
        pytest.param(
            QuantityAdapter(u.m, clip=(0, 1)),
            [-1, 0, 2] << u.m,
            [0, 0, 1] << u.m,
            id="clip-(float,float)",
        ),
        pytest.param(
            QuantityAdapter(clip=(0, 1) << u.m),
            [-1, 0, 2] << u.m,
            [0, 0, 1] << u.m,
            id="clip-Quantity",
        ),
    ],
)
def test_valid(adapter: QuantityAdapter, value: ty.Any, truth: u.Quantity) -> None:
    """Test for a valid quantity validation"""

    class Model(pydantic.BaseModel):
        field: ty.Annotated[u.Quantity, adapter]

    m = Model(field=value)
    npt.assert_allclose(m.field, truth)


@pytest.mark.parametrize(
    ("adapter", "value", "match"),
    [
        pytest.param(
            QuantityAdapter(), None, "invalid_quantity", id="unconstrained-None"
        ),
        pytest.param(
            QuantityAdapter(), {"foo": [1]}, "missing_value", id="dict-no-value"
        ),
        pytest.param(
            QuantityAdapter(), {"value": "hi"}, "invalid_quantity", id="dict-bad-value"
        ),
        pytest.param(
            QuantityAdapter(scalar=True),
            [1.234] << u.s,
            "scalar_error",
            id="scalar-True-quantity",
        ),
        pytest.param(
            QuantityAdapter(scalar=False),
            1.234 << u.s,
            "scalar_error",
            id="scalar-False-quantity",
        ),
        pytest.param(
            QuantityAdapter(u.m),
            [1, 2] << u.s,
            "astropy_unit_not_equivalent",
            id="equiv-m-quantity",
        ),
        pytest.param(
            QuantityAdapter(u.m, equivalencies=u.spectral()),
            1.234 << u.s,
            "astropy_unit_not_equivalent",
            id="equiv-m-quantity-spectral",
        ),
        pytest.param(
            QuantityAdapter(physical_type="mass"),
            [[2], [3], [4]] << u.m,
            "wrong_physical_type",
            id="physical-type-mass-quantity-m",
        ),
        pytest.param(
            QuantityAdapter(ndim=2),
            [[[2], [3], [4]]] << u.kg,
            "ndim_error",
            id="ndim-2-quantity-3d",
        ),
        *[
            pytest.param(
                QuantityAdapter(u.m, **{cmp: val}),  # type: ignore[bad-argument-type]
                [1000, 1500, 2000] << u.mm,
                "bounds_error",
                id=f"{cmp}-fail-{type(val).__name__}",
            )
            for cmp, val in (
                ("ge", 1.1),
                ("ge", 1.1 << u.m),
                ("gt", 1.0),
                ("gt", 1 << u.m),
                ("le", 1.9),
                ("le", 190 << u.cm),
                ("lt", 2),
                ("lt", 2e6 << u.um),
            )
        ],
        pytest.param(
            QuantityAdapter(u.m, gt=[[5], [6]]),
            [[5.1, 5.2], [5.9, 6.2]] << u.m,
            "bounds_error",
            id="gt-vector-no-unit-bounds",
        ),
        pytest.param(
            QuantityAdapter(u.m, gt=[[500], [600]] << u.cm),
            [[5.1, 5.2], [5.9, 6.2]] << u.m,
            "bounds_error",
            id="gt-vector-quantity-bounds",
        ),
        pytest.param(
            QuantityAdapter(clip=[500, 600] << u.cm),
            [[5.1, 5.2], [5.9, 6.2]] << u.s,
            "Can only apply 'clip' function to quantities with compatible dimensions",
            id="clip-mismatch-unit",
        ),
    ],
)
def test_invalid(adapter: QuantityAdapter, value: ty.Any, match: str) -> None:
    """Test an invalid quantity validation"""

    class Model(pydantic.BaseModel):
        field: ty.Annotated[u.Quantity, adapter]

    with pytest.raises(pydantic.ValidationError, match=match):
        Model(field=value)


@pytest.mark.parametrize(
    ("data", "mode", "truth"),
    [
        pytest.param(1.234 << u.m, "python", 1.234 << u.m, id="scalar-python"),
        pytest.param(
            [[[1, 2]]] << u.m, "python", [[[1, 2]]] << u.m, id="vector-python"
        ),
        pytest.param(
            1.234 << u.m, "json", {"value": 1.234, "unit": "m"}, id="scalar-json"
        ),
        pytest.param(
            [[[1, 2]]] << u.m,
            "json",
            {"value": [[[1, 2]]], "unit": "m"},
            id="vector-json",
        ),
    ],
)
def test_serialization(data: u.Quantity, mode: str, truth: ty.Any) -> None:
    """Test the serialization behavior"""
    m = Unconstrained(field=data)
    m_dict = m.model_dump(mode=mode)
    assert np.all(m_dict["field"] == truth)


@pytest.mark.parametrize(
    ("json_str", "truth"),
    [
        pytest.param("5", u.Quantity(5), id="5"),
        pytest.param('"5m"', 5 << u.m, id="5m"),
        pytest.param(
            '"1.234 ph/(m2 sr um)"',
            1.234 << (u.ph / (u.m**2 * u.sr * u.um)),
            id="1.234 ph/(m2 sr um)",
        ),
        pytest.param('{"value": 1.234, "unit": "s"}', 1.234 << u.s, id="1.234s-dict"),
        pytest.param(
            '{"value": [[[1.234, 2.345]]], "unit": "s"}',
            [[[1.234, 2.345]]] << u.s,
            id="vector-dict",
        ),
    ],
)
def test_json_parse(json_str: str, truth: u.Quantity) -> None:
    """Test JSON parsing"""
    m = Unconstrained.model_validate_json(f'{{"field":{json_str}}}')
    npt.assert_allclose(m.field, truth)


def test_json_schema() -> None:
    """Test the JSON schema"""
    # We haven't really tried to make this nice yet
    js = Unconstrained.model_json_schema()
    assert js["properties"]["field"] == {
        "title": "Field",
        "description": "An encoding of an astropy.units.Quantity",
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            {"scalar": True, "ndim": 1},
            "scalar=True and ndim=1 contradict",
            id="scalar-True-ndim-1",
        ),
        pytest.param(
            {"scalar": True, "shape": (1, 2)},
            r"scalar=True and shape=\(1, 2\) contradict",
            id="scalar-True-shape-(1,2)",
        ),
        pytest.param(
            {"scalar": False, "ndim": 0},
            r"scalar=False and ndim=0 contradict",
            id="scalar-False-ndim-0",
        ),
        pytest.param(
            {"scalar": False, "shape": ()},
            r"scalar=False and shape=\(\) contradict",
            id="scalar-False-shape-()",
        ),
        pytest.param(
            {"gt": 45},
            'If equivalent_unit is not defined, then "gt" must be a Quantity if given',
            id="no-equiv-unit-unitless-gt",
        ),
        pytest.param(
            {"clip": []},
            "clip must be a sequence of size 2, was 0",
            id="clip-empty",
        ),
        pytest.param(
            {"clip": [1, 2]},
            r'If equivalent_unit is not defined, then "clip\[0\]" must be a '
            "Quantity if given",
            id="no-equiv-unit-unitless-clip[0]",
        ),
        pytest.param(
            {"clip": [1, 2, 3] << u.m},
            "clip must be a sequence of size 2, was 3",
            id="clip-len-3",
        ),
    ],
)
def test_invalid_adapter_params(kwargs: dict[str, ty.Any], match: str) -> None:
    """Test invalid annotations"""
    with pytest.raises(pydantic.PydanticSchemaGenerationError, match=match):
        QuantityAdapter(**kwargs)


def test_serialize_as_unit() -> None:
    """Test serialize_as_type"""

    class Model(pydantic.BaseModel):
        field: ty.Annotated[u.Quantity, QuantityAdapter(serialize_as_unit=u.m)]

    m = Model(field=1 << u.mm)
    m_py = m.model_dump()
    assert m_py["field"].value == 1e-3
    assert m_py["field"].unit == u.m
    m_json = m.model_dump(mode="json")
    assert m_json["field"]["value"] == 1e-3
    assert m_json["field"]["unit"] == "m"
