"""Unit test for range.py"""

import typing as ty

import pydantic
import pytest

from scientific_pydantic import RangeAdapter


class _Model(pydantic.BaseModel):
    """Test model using RangeAdapter."""

    r: RangeAdapter


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(range(5), range(5), id="range(5)"),
        pytest.param(":5", range(5), id=":5"),
        pytest.param(" : 5", range(5), id=" : 5"),
        pytest.param("1:5", range(1, 5), id="1:5"),
        pytest.param(" 1\t:\n5 : ", range(1, 5), id=" 1 : 5 : "),
        pytest.param("1:10:2", range(1, 10, 2), id="1:10:2"),
    ],
)
def test_range_validation(value: ty.Any, expected: range) -> None:
    """Valid inputs are converted to a range."""
    model = _Model(r=value)
    assert model.r == expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(123, id="int"),
        pytest.param("5", id="5"),
        pytest.param("random text", id="random text"),
        pytest.param("random:text:with colons", id="non-ints"),
        pytest.param("1:2:3:4", id="1:2:3:4"),
    ],
)
def test_range_validation_errors(value: ty.Any) -> None:
    """Invalid inputs raise ValidationError."""
    with pytest.raises(pydantic.ValidationError):
        _Model(r=value)


@pytest.mark.parametrize(
    ("value", "truth"),
    [
        pytest.param(range(5), ":5", id=":5"),
        pytest.param(range(1, 5), "1:5", id="1:5"),
        pytest.param(range(1, 4, 2), "1:4:2", id="1:4:2"),
    ],
)
def test_range_serialization(value: range, truth: str) -> None:
    """Range serializes to JSON-compatible mapping."""
    model = _Model(r=value)
    assert model.model_dump() == {"r": value}
    assert model.model_dump(mode="json") == {"r": truth}


def test_json_schema() -> None:
    """JSON schema is stable and well-defined."""
    schema = _Model.model_json_schema()
    r = schema["properties"]["r"]
    assert r["type"] == "string"
    assert "pattern" in r
