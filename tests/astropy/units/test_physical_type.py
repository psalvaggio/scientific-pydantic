# tests/test_astropy_unit.py
"""Tests for the AstropyUnit pydantic adapter."""

import typing as ty

import astropy.units as u
import pydantic
import pytest

from scientific_pydantic.astropy.units import PhysicalTypeAdapter


class Model(pydantic.BaseModel):
    """A model for unit testing UnitAdapter"""

    t: ty.Annotated[u.PhysicalType, PhysicalTypeAdapter()]


@pytest.mark.parametrize(
    ("data", "truth"),
    [
        pytest.param("length", u.get_physical_type("length"), id="str-length"),
        pytest.param(
            u.get_physical_type("length"), u.get_physical_type("length"), id="pt-length"
        ),
        pytest.param("area", u.get_physical_type("area"), id="str-area"),
        pytest.param(u.m**2, u.get_physical_type("area"), id="unit-area"),
        pytest.param(5 << u.m**2, u.get_physical_type("area"), id="quantity-area"),
    ],
)
def test_valid(data: ty.Any, truth: u.PhysicalType) -> None:
    """Test positive validation cases"""
    m = Model(t=data)
    assert m.t == truth


@pytest.mark.parametrize(
    ("data", "match"),
    [
        pytest.param(
            "foo bar",
            'Could not parse "foo bar" as an astropy PhysicalType',
            id="bad-str",
        ),
        pytest.param(
            42,
            "Expected a string, astropy PhysicalType, or quantity-like object",
            id="bad-type",
        ),
    ],
)
def test_invalid(data: ty.Any, match: str) -> None:
    """Test positive validation cases"""
    with pytest.raises(pydantic.ValidationError, match=match):
        Model(t=data)


ROUNT_TRIP_TESTS = [
    pytest.param("length", id="str-length"),
    pytest.param(u.get_physical_type("length"), id="pt-length"),
    pytest.param(u.m**2, id="unit-area"),
    pytest.param(5 << u.m**2, id="quantity-area"),
]


@pytest.mark.parametrize("data", ROUNT_TRIP_TESTS)
def test_round_trip_python(data: ty.Any) -> None:
    """Test round-tripping through JSON"""
    original = Model(t=data)
    dump = original.model_dump()
    restored = Model.model_validate(dump)
    assert restored == original


@pytest.mark.parametrize("data", ROUNT_TRIP_TESTS)
def test_round_trip_json(data: ty.Any) -> None:
    """Test round-tripping through JSON"""
    original = Model(t=data)
    json_str = original.model_dump_json()
    restored = Model.model_validate_json(json_str)
    assert restored == original


def test_serialization() -> None:
    """Test serialization behavior"""
    m = Model(t=u.m / u.s)
    assert isinstance(m.model_dump()["t"], u.PhysicalType)
    assert isinstance(m.model_dump(mode="json")["t"], str)


def test_json_schema() -> None:
    """Test the JSON schema"""
    schema = Model.model_json_schema()
    assert schema["properties"]["t"] == {
        "title": "T",
        "type": "string",
        "description": "An astropy PhysicalType expressed as a string.",
        "examples": ["length", "area"],
    }


@pytest.mark.parametrize(
    "annotation",
    [int, str, u.Unit],
)
def test_with_other_bases(annotation: ty.Any) -> None:
    """Test that we can only use this with UnitBase"""
    with pytest.raises(pydantic.PydanticSchemaGenerationError):

        class A(pydantic.BaseModel):
            a: ty.Annotated[annotation, PhysicalTypeAdapter()]
