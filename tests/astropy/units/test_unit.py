# tests/test_astropy_unit.py
"""Tests for the AstropyUnit pydantic adapter."""

import typing as ty

import astropy.units as u
import pydantic
import pytest

from scientific_pydantic.astropy.units import UnitAdapter


class Model(pydantic.BaseModel):
    """A model for unit testing UnitAdapter"""

    any_unit: ty.Annotated[u.UnitBase, UnitAdapter()] | None = None
    velocity: ty.Annotated[u.UnitBase, UnitAdapter("m/s")] | None = None
    duration: ty.Annotated[u.UnitBase, UnitAdapter(u.s)] | None = None
    spectral: (
        ty.Annotated[u.UnitBase, UnitAdapter(u.m, equivalencies=u.spectral())] | None
    ) = None
    length: ty.Annotated[u.UnitBase, UnitAdapter(physical_type="length")] | None = None


@pytest.mark.parametrize(
    ("data", "truth"),
    [
        pytest.param(
            {"any_unit": u.m / u.s}, {"any_unit": u.m / u.s}, id="unconstrained-ident"
        ),
        pytest.param(
            {"any_unit": "m / s"}, {"any_unit": u.m / u.s}, id="unconstrained-str-m/s"
        ),
        pytest.param({"any_unit": "kg"}, {"any_unit": u.kg}, id="unconstrained-str-kg"),
        pytest.param(
            {"velocity": "m / s"}, {"velocity": u.m / u.s}, id="velocity-str-m/s"
        ),
        pytest.param({"duration": u.day}, {"duration": u.day}, id="duration-unit-day"),
        pytest.param({"duration": u.s}, {"duration": u.s}, id="duration-same-unit"),
        pytest.param({"spectral": u.nm}, {"spectral": u.nm}, id="spectral-nm"),
        pytest.param({"spectral": u.GHz}, {"spectral": u.GHz}, id="spectral-GHz"),
        pytest.param({"spectral": "cm-1"}, {"spectral": u.cm**-1}, id="spectral-cm-1"),
        pytest.param({"length": "km"}, {"length": u.km}, id="str-length-km"),
    ],
)
def test_valid(data: dict[str, ty.Any], truth: dict[str, u.UnitBase]) -> None:
    """Test positive validation cases"""
    m = Model(**data)
    for f in Model.model_fields:
        assert getattr(m, f) == truth.get(f)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        pytest.param(
            {"any_unit": "not_a_unit"}, "astropy_unit_parse_error", id="bad-str"
        ),
        pytest.param({"any_unit": 42}, "astropy_unit_type_error", id="bad-type"),
        pytest.param(
            {"velocity": u.kg}, "astropy_unit_not_equivalent", id="unit-not-equivalent"
        ),
        pytest.param(
            {"duration": "kg"}, "astropy_unit_not_equivalent", id="str-not-equivalent"
        ),
        pytest.param({"spectral": "s"}, "astropy_unit_not_equivalent", id="spectral-s"),
        pytest.param({"length": "s"}, "wrong_physical_type", id="length-s"),
    ],
)
def test_invalid(data: dict[str, ty.Any], match: str) -> None:
    """Test positive validation cases"""
    with pytest.raises(pydantic.ValidationError, match=match):
        Model(**data)


ROUNT_TRIP_TESTS = [
    pytest.param({"any_unit": u.km / u.s}, id="unconstrained-unit"),
    pytest.param({"duration": "day"}, id="equiv-unit"),
    pytest.param({"spectral": u.nm}, id="custom-equiv-unit"),
    pytest.param({"length": u.nm}, id="phys-type-unit"),
]


@pytest.mark.parametrize("data", ROUNT_TRIP_TESTS)
def test_round_trip_python(data: dict[str, ty.Any]) -> None:
    """Test round-tripping through JSON"""
    original = Model(**data)
    dump = original.model_dump()
    restored = Model.model_validate(dump)
    for f in Model.model_fields:
        assert getattr(restored, f) == getattr(original, f)


@pytest.mark.parametrize("data", ROUNT_TRIP_TESTS)
def test_round_trip_json(data: dict[str, ty.Any]) -> None:
    """Test round-tripping through JSON"""
    original = Model(**data)
    json_str = original.model_dump_json()
    restored = Model.model_validate_json(json_str)
    for f in Model.model_fields:
        assert getattr(restored, f) == getattr(original, f)


def test_serialization() -> None:
    """Test serialization behavior"""
    m = Model(any_unit=u.m / u.s)
    assert isinstance(m.model_dump()["any_unit"], u.UnitBase)
    assert isinstance(m.model_dump(mode="json")["any_unit"], str)


@pytest.mark.parametrize(
    ("adapter", "description", "examples"),
    [
        pytest.param(
            UnitAdapter(),
            "An astropy unit expressed as a string.",
            [
                "m / s",
                "km / h",
                "kg",
                "deg",
                "J / (kg K)",
            ],
            id="default",
        ),
        pytest.param(
            UnitAdapter(u.m),
            'An astropy unit expressed as a string. Must be equivalent to "m".',
            ["m"],
            id="equiv-meters",
        ),
        pytest.param(
            UnitAdapter(
                u.K,
                equivalencies=[
                    (u.K, u.C, lambda x: x - 273.15, lambda x: x + 273.15),
                    (
                        u.K,
                        u.F,
                        lambda x: (x - 273.15) * 9 / 5 + 32,
                        lambda x: (x - 32) * (5 / 9) + 273.15,
                    ),
                ],
                physical_type="temperature",
            ),
            (
                "An astropy unit expressed as a string. Must be equivalent "
                'to "K" (with custom equivalencies: K <-> C, K <-> F). Must be of '
                'type "temperature".'
            ),
            ["K"],
            id="equiv-temp",
        ),
    ],
)
def test_json_schema(
    adapter: UnitAdapter, description: str, examples: list[str]
) -> None:
    """Test the JSON schema"""

    class A(pydantic.BaseModel):
        a: ty.Annotated[u.UnitBase, adapter]

    schema = A.model_json_schema()
    props = schema["properties"]["a"]
    assert props["description"] == description
    assert props["examples"] == examples


@pytest.mark.parametrize(
    "annotation",
    [int, str, u.Unit],
)
def test_with_other_bases(annotation: ty.Any) -> None:
    """Test that we can only use this with UnitBase"""
    with pytest.raises(pydantic.PydanticSchemaGenerationError):

        class A(pydantic.BaseModel):
            a: ty.Annotated[annotation, UnitAdapter()]
