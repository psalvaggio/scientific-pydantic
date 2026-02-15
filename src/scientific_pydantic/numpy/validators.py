"""Validation logic"""

import typing as ty
from collections.abc import Sequence

import numpy as np
import pydantic
from numpy.typing import NDArray
from pydantic_core import PydanticCustomError

from ..range import RangeAdapter
from ..slice import IntSliceAdapter
from .dtype_adapter import DTypeAdapter


class DTypeValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for the array's data type"""

    dtype: ty.Annotated[np.dtype, DTypeAdapter()]

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply data type validation"""
        try:
            return arr.astype(self.dtype)
        except (ValueError, TypeError) as e:
            err = PydanticCustomError(
                "dtype_error",
                f"the array could not be converted to {self.dtype}",
            )
            raise err from e


class NDimValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for the number of dimensions"""

    ndim: int = pydantic.Field(ge=0)

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply ndim validation"""
        if arr.ndim != self.ndim:
            err = PydanticCustomError(
                "ndim_error",
                f"the array had {arr.ndim} dimension(s), expected {self.ndim}",
            )
            raise err
        return arr


class ShapeValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Data type for the shape of the array"""

    shape: Sequence[
        int
        | ty.Annotated[range, RangeAdapter()]
        | ty.Annotated[slice, IntSliceAdapter]
        | None
    ]

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply shape validation"""
        if len(self.shape) != arr.ndim:
            err = PydanticCustomError(
                "ndim_error",
                f"the array had {arr.ndim} dimension(s), expected {len(self.shape)}",
            )
            raise err

        for i, constraint in enumerate(self.shape):
            actual_size = arr.shape[i]

            if isinstance(constraint, int):  # Exact size required
                if actual_size != constraint:
                    err = PydanticCustomError(
                        "shape_error",
                        f"Dimension {i}: expected size {constraint}, got {actual_size}",
                    )
                    raise err
            elif isinstance(constraint, range):  # Size must be in range
                if actual_size not in constraint:
                    err = PydanticCustomError(
                        "shape_error",
                        f"Dimension {i}: size {actual_size} not in range {constraint}",
                    )
                    raise err
            elif isinstance(constraint, slice):  # Convert slice to range and check
                start = constraint.start if constraint.start is not None else 0
                stop = constraint.stop if constraint.stop is not None else float("inf")
                step = constraint.step if constraint.step is not None else 1

                # Check if actual_size would be in the range defined by slice
                if actual_size < start or actual_size >= stop:
                    err = PydanticCustomError(
                        "shape_error",
                        "Dimension {i}: size {actual_size} not in slice {constraint}",
                    )
                    raise err
                if step != 1 and (actual_size - start) % step != 0:
                    err = PydanticCustomError(
                        "shape_error",
                        f"Dimension {i}: size {actual_size} does not satisfy "
                        f"step {step} of slice {constraint}",
                    )
                    raise err

            # Otherwise it was None, so any size is fine

        return arr


class GtValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for a bounds check"""

    gt: float

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply gt validation"""
        if not np.all(arr > self.gt):
            err = PydanticCustomError(
                "bounds_error",
                f"Not all elements were greater than {self.gt}",
            )
            raise err
        return arr


class GeValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for a bounds check"""

    ge: float

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply ge validation"""
        if not np.all(arr >= self.ge):
            err = PydanticCustomError(
                "bounds_error",
                f"Not all elements were greater than or equal to {self.ge}",
            )
            raise err
        return arr


class LtValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for a bounds check"""

    lt: float

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply lt validation"""
        if not np.all(arr < self.lt):
            err = PydanticCustomError(
                "bounds_error",
                f"Not all elements were less than {self.lt}",
            )
            raise err
        return arr


class LeValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for a bounds check"""

    le: float

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply le validation"""
        if not np.all(arr <= self.le):
            err = PydanticCustomError(
                "bounds_error",
                f"Not all elements were less than or equal to {self.le}",
            )
            raise err
        return arr


class ClipValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for a clip check"""

    clip: tuple[float | None, float | None]

    def __call__(self, arr: NDArray) -> NDArray:
        """Apply clip validation"""
        if self.clip[0] is not None or self.clip[1] is not None:
            return np.clip(arr, self.clip[0], self.clip[1])
        return arr


class NDArrayValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for an ndarray"""

    dtype: DTypeValidator | None = None
    ndim: NDimValidator | None = None
    shape: ShapeValidator | None = None
    gt: GtValidator | None = None
    ge: GeValidator | None = None
    lt: LtValidator | None = None
    le: LeValidator | None = None
    clip: ClipValidator | None = None

    @classmethod
    def from_kwargs(cls, **kwargs) -> ty.Self:
        """Create from field/value pairs"""
        try:
            return cls(
                **{key: {key: val} for key, val in kwargs.items() if val is not None},
            )
        except pydantic.ValidationError as e:
            msg = "Invalid constraint value(s):\n" + (
                "\n".join(
                    f"{i + 1}. {err['loc'][0]} ({err['input']!r})- {err['msg']}"
                    for i, err in enumerate(e.errors())
                )
            )
            raise ValueError(msg) from None

    def __call__(self, arr: np.typing.ArrayLike) -> np.typing.NDArray:
        """Validate and optionally transform a numpy ndarray.

        Parameters
        ----------
        arr : ndarray
            The array to validate

        Returns
        -------
        ndarray
            The validated (and possibly transformed) array

        Raises
        ------
        ValueError
            If validation fails
        TypeError
            If dtype conversion fails
        """
        # Convert to ndarray if needed
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        for field in type(self).model_fields:
            val = getattr(self, field)
            if val is None:
                continue

            arr = val(arr)

        return arr
