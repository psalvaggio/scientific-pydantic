"""Unit test for Numpy functionality"""

import typing as ty

import numpy as np
import pydantic
import pytest
from numpy.typing import ArrayLike

from scientific_pydantic.numpy import NDArrayAdapter


class DefaultModel(pydantic.BaseModel):
    """Model with an unconstrained ndarray"""

    arr: ty.Annotated[np.ndarray, NDArrayAdapter()]


@pytest.mark.parametrize(
    ("input_data", "dtype", "expected_dtype"),
    [
        pytest.param([[1, 2], [3, 4]], np.float64, np.float64, id="f64"),
        pytest.param([[1, 2], [3, 4]], float, np.float64, id="float"),
        pytest.param([[1, 2], [3, 4]], "float64", np.float64, id="f64-str"),
        pytest.param([[1.5, 2.5], [3.5, 4.5]], "int32", np.int32, id="i32-str"),
        pytest.param([1, 2, 3], "float32", np.float32, id="f32-str"),
        pytest.param(
            np.array([1, 2, 3], dtype=np.int64),
            "float64",
            np.float64,
            id="np-f64-str",
        ),
    ],
)
def test_dtype_conversion(
    input_data: ArrayLike,
    dtype: ty.Any,
    expected_dtype: np.dtype,
) -> None:
    """Test that dtype parameter correctly converts array types"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(dtype=dtype)]

    model = Model(arr=input_data)  # type: ignore[bad-argument-type]
    assert model.arr.dtype == expected_dtype


@pytest.mark.parametrize(
    ("input_data", "dtype"),
    [
        pytest.param([[1, 2], [3, 4]], "float64", id="f64"),
        pytest.param([1.0, 2.0, 3.0], "int32", id="i32"),
    ],
)
def test_dtype_none_preserves_type(input_data: ArrayLike, dtype: str) -> None:
    """Test that dtype=None preserves input type"""
    arr = np.array(input_data, dtype=dtype)
    model = DefaultModel(arr=arr)
    assert model.arr.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    ("input_data", "ndim", "should_pass"),
    [
        pytest.param([1, 2, 3], 1, True, id="1d-positive"),
        pytest.param([[1, 2], [3, 4]], 2, True, id="2d-positive"),
        pytest.param([[[1, 2]], [[3, 4]]], 3, True, id="3d-positive"),
        pytest.param(5, 0, True, id="scalar-positive"),  # scalar
        pytest.param([1, 2, 3], 2, False, id="1d-2d"),
        pytest.param([[1, 2], [3, 4]], 1, False, id="2d-1d"),
        pytest.param(5, 1, False, id="scalar-1d"),
    ],
)
def test_ndim_validation(
    *,
    input_data: ArrayLike,
    ndim: int,
    should_pass: bool,
) -> None:
    """Test that ndim parameter validates array dimensions"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(ndim=ndim)]

    if should_pass:
        x = Model(arr=input_data)  # type: ignore[bad-argument-type]
        assert x.arr.ndim == ndim
    else:
        with pytest.raises(pydantic.ValidationError) as exc_info:
            Model(arr=input_data)  # type: ignore[bad-argument-type]
        assert "dimension" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    ("input_data", "shape", "should_pass"),
    [
        pytest.param([[1, 2, 3], [4, 5, 6]], (2, 3), True, id="2x3"),
        pytest.param([[1, 2], [3, 4]], (2, 2), True, id="2x2"),
        pytest.param([[1, 2, 3]], (1, None), True, id="1xN"),
        pytest.param([[1, 2], [3, 4], [5, 6]], (None, 2), True, id="Nx2"),
        pytest.param([[1, 2, 3], [4, 5, 6]], (2, 4), False, id="2x3-2x4"),
        pytest.param([[1, 2], [3, 4]], (3, 2), False, id="2x2-3x2"),
    ],
)
def test_shape_exact(
    *,
    input_data: ArrayLike,
    shape: tuple[int | None, ...],
    should_pass: bool,
) -> None:
    """Test exact shape matching with int and None constraints"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(shape=shape)]

    if should_pass:
        x = Model(arr=input_data)  # type: ignore[bad-argument-type]
        for i, constraint in enumerate(shape):
            if constraint is not None:
                assert x.arr.shape[i] == constraint
    else:
        with pytest.raises(pydantic.ValidationError) as exc_info:
            Model(arr=input_data)  # type: ignore[bad-argument-type]
        assert "does not match spec" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    ("array_shape", "shape_constraint", "should_pass"),
    [
        pytest.param((5,), (range(3, 8),), True, id="range(3,8)"),
        pytest.param((5,), (range(3, 8, 2),), True, id="range(3,8,2)"),
        pytest.param((5,), (slice(3, 8),), True, id="slice(3,8)"),
        pytest.param((5,), (slice(3, 8, 2),), True, id="slice(3,8,2)"),
        pytest.param(
            (10, 5),
            (range(5, 15), slice(5, 6)),
            True,
            id="range(5,15)-slice(5,6)",
        ),
        pytest.param((2,), (range(3, 8),), False, id="2-range(3,8)"),
        pytest.param((15,), (range(5, 10),), False, id="15-range(5,10)"),
        pytest.param((15,), (range(5, 15),), False, id="15-range(5,15)"),
    ],
)
def test_shape_range(
    *,
    array_shape: tuple[int, ...],
    shape_constraint: tuple[range, ...],
    should_pass: bool,
) -> None:
    """Test shape validation with range constraints"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(shape=shape_constraint)]

    input_data = np.ones(array_shape)

    if should_pass:
        x = Model(arr=input_data)
        assert x.arr.shape == array_shape
    else:
        with pytest.raises(pydantic.ValidationError) as exc_info:
            Model(arr=input_data)
        assert "range" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    ("input_data", "constraint_name", "constraint_value", "should_pass"),
    [
        pytest.param([1, 2, 3], "ge", 0, True, id="[0,None)"),
        pytest.param([1, 2, 3], "ge", 1, True, id="[1,None)"),
        pytest.param([1, 2, 3], "ge", 2, False, id="[2,None)-fail"),
        pytest.param([1, 2, 3], "gt", 0, True, id="(0, None)"),
        pytest.param([1, 2, 3], "gt", 1, False, id="(1, None)-fail"),
        pytest.param([1, 2, 3], "le", 4, True, id="(None, 4]"),
        pytest.param([1, 2, 3], "le", 2, False, id="(None, 2]-fail"),
        pytest.param([1, 2, 3], "lt", 4, True, id="(None, 4)"),
        pytest.param([1, 2, 3], "lt", 3, False, id="(None, 3)-fail"),
    ],
)
def test_bounds_validation(
    *,
    input_data: ArrayLike,
    constraint_name: str,
    constraint_value: float,
    should_pass: bool,
) -> None:
    """Test that bounds constraints work correctly"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[
            np.ndarray, NDArrayAdapter(**{constraint_name: constraint_value})
        ]

    if should_pass:
        x = Model(arr=input_data)  # type: ignore[bad-argument-type]
        assert isinstance(x.arr, np.ndarray)
    else:
        with pytest.raises(pydantic.ValidationError) as exc_info:
            Model(arr=input_data)  # type: ignore[bad-argument-type]
        assert {
            "ge": " >= ",
            "gt": " > ",
            "lt": " < ",
            "le": " <= ",
        }[constraint_name] in str(exc_info.value).lower()


@pytest.mark.parametrize(
    ("input_data", "ge", "le", "should_pass"),
    [
        ([0.5, 0.7, 0.9], 0, 1, True),
        ([0, 0.5, 1], 0, 1, True),
        ([-0.1, 0.5, 0.9], 0, 1, False),
        ([0.5, 0.7, 1.1], 0, 1, False),
    ],
)
def test_combined_bounds(
    *,
    input_data: ArrayLike,
    ge: float,
    le: float,
    should_pass: bool,
) -> None:
    """Test combining multiple bounds constraints"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(ge=ge, le=le)]

    if should_pass:
        x = Model(arr=input_data)  # type: ignore[bad-argument-type]
        assert np.all(x.arr >= ge)
        assert np.all(x.arr <= le)
    else:
        with pytest.raises(pydantic.ValidationError):
            Model(arr=input_data)  # type: ignore[bad-argument-type]


@pytest.mark.parametrize(
    ("input_data", "clip_min", "clip_max", "expected"),
    [
        ([1, 2, 3, 4, 5], 2, 4, [2, 2, 3, 4, 4]),
        ([-5, 0, 5], -1, 1, [-1, 0, 1]),
        ([1, 2, 3], None, 2, [1, 2, 2]),
        ([1, 2, 3], 2, None, [2, 2, 3]),
        ([1, 2, 3], None, None, [1, 2, 3]),
    ],
)
def test_clipping(
    input_data: ArrayLike,
    clip_min: float | None,
    clip_max: float | None,
    expected: ArrayLike,
) -> None:
    """Test that clipping works correctly"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(clip=(clip_min, clip_max))]

    x = Model(arr=input_data)  # type: ignore[bad-argument-type]
    np.testing.assert_array_equal(x.arr, expected)


def test_dtype_and_shape() -> None:
    """Test combining dtype and shape constraints"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(dtype="float32", shape=(3, None))]

    x = Model(arr=[[1, 2], [3, 4], [5, 6]])  # type: ignore[bad-argument-type]
    assert x.arr.dtype == np.float32
    assert x.arr.shape[0] == 3


def test_bounds_and_clipping() -> None:
    """Test that validation happens before clipping"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(clip=(0, 10))]

    # Should clip values outside range
    x = Model(arr=[-5, 5, 15])  # type: ignore[bad-argument-type]
    np.testing.assert_array_equal(x.arr, [0, 5, 10])


def test_all_constraints() -> None:
    """Test all constraints together"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[
            np.ndarray,
            NDArrayAdapter(
                dtype="float64",
                ndim=2,
                shape=(3, None),
                clip=(-1, 1),
            ),
        ]

    x = Model(arr=[[0, 0.5], [0.7, -0.3], [-2, 2]])  # type: ignore[bad-argument-type]
    assert x.arr.dtype == np.float64
    assert x.arr.ndim == 2
    assert x.arr.shape[0] == 3
    assert np.all(x.arr >= -1)
    assert np.all(x.arr <= 1)


@pytest.mark.parametrize(
    "input_data",
    [
        [1, 2, 3],
        [[1, 2], [3, 4]],
        [[[1, 2]], [[3, 4]]],
    ],
)
def test_json_round_trip(input_data: ArrayLike) -> None:
    """Test that arrays can be serialized and deserialized"""
    model = DefaultModel(arr=input_data)  # type: ignore[bad-argument-type]
    json_str = model.model_dump_json()
    restored = DefaultModel.model_validate_json(json_str)

    np.testing.assert_array_equal(model.arr, restored.arr)


def test_json_with_dtype() -> None:
    """Test JSON serialization preserves dtype through validation"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(dtype="float64")]

    x = Model(arr=[1, 2, 3])  # type: ignore[bad-argument-type]
    json_str = x.model_dump_json()
    restored = Model.model_validate_json(json_str)

    assert restored.arr.dtype == np.float64


def test_json_schema_generation() -> None:
    """Test that JSON schema is generated correctly"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[
            np.ndarray, NDArrayAdapter(ndim=2, dtype="float64", ge=0, le=1)
        ]

    schema = Model.model_json_schema()

    assert "arr" in schema["properties"]
    assert schema["properties"]["arr"]["type"] == "array"
    assert "description" in schema["properties"]["arr"]


def test_empty_array() -> None:
    """Test handling of empty arrays"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(ndim=1)]

    x = Model(arr=[])  # type: ignore[bad-argument-type]
    assert x.arr.shape == (0,)


def test_scalar_array() -> None:
    """Test 0-dimensional (scalar) arrays"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(ndim=0)]

    x = Model(arr=5)  # type: ignore[bad-argument-type]
    assert x.arr.ndim == 0
    assert x.arr == 5


def test_invalid_dtype_conversion() -> None:
    """Test that invalid dtype conversions raise errors"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(dtype="M")]

    # Numpy will pretty much do any conversion, but strings to datetime's can
    # raise an error
    with pytest.raises(pydantic.ValidationError):
        Model(arr=["a", "b"])  # type: ignore[bad-argument-type]


def test_shape_constraint_dimension_mismatch() -> None:
    """Test error when shape constraint length doesn't match array dims"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(shape=(3, 3))]

    with pytest.raises(pydantic.ValidationError, match="does not match spec"):
        # 1D array with 2D shape constraint
        Model(arr=[1, 2, 3])  # type: ignore[bad-argument-type]


@pytest.mark.parametrize(
    "input_data",
    [
        pytest.param([1, [2, 3]], id="inhomogenous-input"),
    ],
)
def test_bad_input(input_data: ty.Any) -> None:
    """Test that non-convertible inputs raise appropriate errors"""
    with pytest.raises(pydantic.ValidationError):
        DefaultModel(arr=input_data)


def test_adaptor_invalid_ndim() -> None:
    """Test that negative ndim is rejected"""
    with pytest.raises(ValueError, match="ndim"):
        NDArrayAdapter(ndim=-1)


@pytest.mark.parametrize(
    ("shape", "input_shape", "should_pass"),
    [
        # --- Basic exact matching ---
        ([2, 3, 4], (2, 3, 4), True),
        ([2, 3, 4], (2, 3, 5), False),  # last dim wrong
        ([2, 3, 4], (2, 3), False),  # too few dims
        ([2, 3, 4], (2, 3, 4, 5), False),  # too many dims
        # --- None (any size, exactly one dim) ---
        ([None], (5,), True),
        ([None], (0,), True),  # zero-size dim
        ([None], (), False),  # must match exactly one
        ([None, None], (3, 7), True),
        ([2, None, 4], (2, 99, 4), True),
        ([2, None, 4], (2, 99, 5), False),  # last dim wrong
        # --- range ---
        ([range(3, 6)], (3,), True),
        ([range(3, 6)], (5,), True),
        ([range(3, 6)], (6,), False),  # range is exclusive
        ([range(3, 6)], (2,), False),
        ([range(1)], (0,), True),
        ([range(2, 10, 2)], (4,), True),  # step
        ([range(2, 10, 2)], (3,), False),  # step mismatch
        # --- slice ---
        ([slice(3, 6)], (3,), True),
        ([slice(3, 6)], (5,), True),
        ([slice(3, 6)], (6,), False),
        ([slice(None, 5)], (0,), True),  # unbounded below
        ([slice(None, 5)], (4,), True),
        ([slice(None, 5)], (5,), False),
        ([slice(3, None)], (3,), True),  # unbounded above
        ([slice(3, None)], (100,), True),
        ([slice(3, None)], (2,), False),
        ([slice(2, None, 2)], (4,), True),  # unbounded with step
        ([slice(2, None, 2)], (5,), False),
        # --- Single ellipsis ---
        ([...], (), True),  # matches 0 dims
        ([...], (3,), True),
        ([...], (3, 4, 5), True),
        ([..., 4], (4,), True),  # ellipsis matches 0
        ([..., 4], (3, 4), True),
        ([..., 4], (3, 5, 4), True),
        ([..., 4], (3, 5), False),
        ([2, ...], (2,), True),
        ([2, ...], (2, 3, 4), True),
        ([2, ...], (3,), False),
        ([2, ..., 4], (2, 4), True),  # ellipsis matches 0
        ([2, ..., 4], (2, 3, 4), True),
        ([2, ..., 4], (2, 3, 5, 4), True),
        ([2, ..., 4], (2, 3, 5), False),
        # --- Multiple ellipses ---
        ([..., ...], (), True),
        ([..., ...], (3, 4), True),
        ([..., 3, ...], (3,), True),  # both ellipses match 0
        ([..., 3, ...], (2, 3, 4), True),
        ([..., 3, ...], (2, 4), False),
        ([2, ..., 3, ..., 4], (2, 3, 4), True),  # both ellipses match 0
        ([2, ..., 3, ..., 4], (2, 5, 3, 6, 7, 4), True),
        ([2, ..., 3, ..., 4], (2, 4), False),  # missing the 3
        ([..., 2, ..., 2, ...], (2, 2), True),
        ([..., 2, ..., 2, ...], (2, 3, 2), True),
        ([..., 2, ..., 2, ...], (2,), False),  # only one 2
        # --- Empty shape spec ---
        ([], (), True),
        ([], (3,), False),
        # --- Zero-size dimensions ---
        ([0], (0,), True),
        ([0], (1,), False),
        ([..., 0], (3, 0), True),
        ([..., 0], (3, 1), False),
        # --- Mixed spec types ---
        ([2, range(3, 6), None, ..., 4], (2, 4, 99, 4), True),
        ([2, range(3, 6), None, ..., 4], (2, 4, 99, 1, 2, 4), True),
        ([2, range(3, 6), None, ..., 4], (2, 2, 99, 4), False),  # range fail
    ],
    ids=lambda x: (
        ("pass" if x else "fail")
        if isinstance(x, bool)
        else f"[{','.join(str(y) for y in x)}]"
    ),
)
def test_shape(*, shape: tuple, input_shape: tuple, should_pass: bool) -> None:
    """Test ellipsis shape specs"""

    class Model(pydantic.BaseModel):
        arr: ty.Annotated[np.ndarray, NDArrayAdapter(shape=shape)]

    data = np.zeros(input_shape)
    if should_pass:
        assert Model(arr=data).arr is data
    else:
        with pytest.raises(pydantic.ValidationError, match="does not match spec"):
            Model(arr=data)
