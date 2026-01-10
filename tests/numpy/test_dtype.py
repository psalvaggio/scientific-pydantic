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
    x = DefaultModel(dtype=">i4")
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


# Add more tests here
