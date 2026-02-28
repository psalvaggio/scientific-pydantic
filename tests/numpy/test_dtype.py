"""Unit tests for numpy dtypes"""

import typing as ty

import numpy as np
import pydantic
import pytest

from scientific_pydantic.numpy import DTypeAdapter


class DefaultModel(pydantic.BaseModel):
    """Test model"""

    dtype: ty.Annotated[np.dtype, DTypeAdapter()]


def test_int32() -> None:
    """Test a scalar int32"""
    # This feels like a bug with the type checker, the schema for this is a
    # validator that accepts Any, so this kwarg shoud take Any.
    x = DefaultModel(dtype=">i4")  # type: ignore[bad-argument-type]
    assert x.dtype.byteorder == ">"
    assert x.dtype.itemsize == 4
    assert x.dtype.name == "int32"
    assert x.dtype.type is np.int32


@pytest.mark.parametrize(
    "value",
    [
        "int",
        "float32",
    ],
)
def test_roundtrip_json(value: ty.Any) -> None:
    """Test round-tripping through JSON"""
    x = DefaultModel(dtype=value)
    assert isinstance(x.dtype, np.dtype)
    x_json = x.model_dump_json()
    x2 = DefaultModel.model_validate_json(x_json)
    assert x2.dtype == x.dtype


def test_json_schema() -> None:
    """Test for the JSON schema"""
    js = DefaultModel.model_json_schema()
    assert js["properties"]["dtype"] == {
        "description": "NumPy dtype",
        "title": "Dtype",
        "type": "str",
    }


# Add more tests here
