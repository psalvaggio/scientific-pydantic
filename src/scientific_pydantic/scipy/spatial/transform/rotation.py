"""Pydantic adapter for scipy.spatial.transform.Rotation"""

from __future__ import annotations

import contextlib
import functools
import typing as ty
from collections.abc import Iterable, Mapping, Sequence

import pydantic
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, PydanticCustomError, core_schema

from scientific_pydantic.numpy import NDArrayAdapter
from scientific_pydantic.numpy.validators import validate_shape
from scientific_pydantic.version_check import version_ge

if ty.TYPE_CHECKING:
    import types

    from scipy.spatial.transform import Rotation

    from scientific_pydantic.ellipsis import EllipsisLiteral


class RotationAdapter:
    """Pydantic adapter for scipy.spatial.transform.Rotation.

    Serializes a Rotation as a quaternion (scalar-last, xyzw convention -
    the same convention scipy uses internally) and validates from:
    - A Rotation instance (passthrough)
    - A mapping with one of the following:
      - {
          "quat": array_like, shape (..., 4),
          "scalar_first": bool (default False),
        }
      - {
          "matrix": array_like, shape (..., 3, 3),
          "assume_valid": bool (default False, requires >= 1.17.0),
        }
      - {
          "rotvec": array_like, shape (..., 3),
          "degrees": bool (default False),
        }
      - {
          "mrp": array_like, shape (..., 3),
        }
      - {
          "euler": {
            "seq": str (see scipy docs),
            "angles": float | array_like, shape (..., [1 or 2 or 3]),
            "degrees": bool (default False),
          },
        }
      - {
          "davenport": {
            "axes": array_like, shape (3,) or (..., [1 or 2 or 3], 3),
            "order": "e" or "extrinsic" or "i" or "intrinsic"
            "angles": float | array_like, shape (..., [1 or 2 or 3]),
            "degrees": bool (default False),
          }
        }

    Usage
    -----
        import typing as ty
        from pydantic import BaseModel
        from scientific_pydantic.scipy.spatial.transform import RotationAdapter
        from scipy.spatial.transform import Rotation

        class Pose(BaseModel):
            rotation: ty.Annotated[Rotation, RotationAdapter()]

        pose = Pose(rotation={"quat": [0, 0, 0, 1]})
        pose.rotation          # scipy.spatial.transform.Rotation instance
        pose.model_dump()      # {"rotation": {"quat": [0.0, 0.0, 0.0, 1.0]}}
        pose.model_dump_json() # '{"rotation":{"quat":[0.0,0.0,0.0,1.0]}}'

    Parameters
    ----------
    single : bool | None
        If given as `True`, only single rotations will be accepted. Overrides
        `ndim` or `shape`.
    ndim : int | None
        If given, the dimensionaly of the rotations must be equal to the given
        value.
    shape : Sequence[int | range | slice | None] | None
        If given, provides a constraint on the shape of the given rotations.
        Overrides `ndim`.
    """

    def __init__(
        self,
        *,
        single: bool | None = None,
        ndim: int | None = None,
        shape: Sequence[EllipsisLiteral | int | range | slice | None] | None = None,
    ) -> None:

        self._shape_spec: (
            Sequence[types.EllipsisType | int | range | slice | None] | None
        ) = None
        if single:
            self._shape_spec = ()
        elif shape is not None:
            self._shape_spec = shape
        elif ndim is not None:
            self._shape_spec = (None,) * ndim

        if not _supports_shape() and (
            (shape is not None and len(shape) > 1)
            or (ndim is not None and ndim not in (0, 1))
        ):
            msg = "N-D shape constraints on Rotation require scipy >= 1.17.0"
            raise pydantic.PydanticSchemaGenerationError(msg)

    def __get_pydantic_core_schema__(
        self,
        source_type: ty.Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        """Get the pydantic schema for this type"""
        from scipy.spatial.transform import Rotation

        del handler

        if source_type is not Rotation:
            msg = (
                "RotationAdapter is only usable with "
                f"scipy.spatial.transform.Rotation, not {source_type}."
            )
            raise pydantic.PydanticSchemaGenerationError(msg)

        # Accept any Python object and run our validator.
        python_schema = core_schema.no_info_plain_validator_function(
            _validate_rotation,
        )
        if self._shape_spec is not None:
            spec = self._shape_spec

            def _val(x: Rotation) -> Rotation:
                shape = (
                    x.shape if _supports_shape() else (() if x.single else (len(x),))
                )
                if validate_shape(shape, spec):
                    return x

                err_t = "invalid_rotation_shape"
                msg = "Rotation object shape {shape} did not match spec {spec}"
                raise PydanticCustomError(err_t, msg, {"shape": shape, "spec": spec})

            python_schema = core_schema.chain_schema(
                [
                    python_schema,
                    core_schema.no_info_plain_validator_function(_val),
                ]
            )

        # When deserialising from JSON/dict Pydantic passes a Python object
        # after JSON parsing, so the same validator works for both paths.
        return core_schema.json_or_python_schema(
            json_schema=python_schema,
            python_schema=python_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _rotation_to_dict,
                when_used="json-unless-none",
                return_schema=core_schema.dict_schema(),
            ),
        )


def _rotation_to_dict(r: Rotation) -> dict:
    """Serialise to a quaternion dict (scalar-last, xyzw)."""
    return {"quat": r.as_quat().tolist()}


@functools.lru_cache
def _mapping_validator() -> pydantic.TypeAdapter:  # noqa: C901
    """Get the pydantic TypeAdapter"""
    import numpy as np  # noqa: TC002
    from scipy.spatial.transform import Rotation

    class Quat(pydantic.BaseModel, extra="forbid"):
        quat: ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., 4))]
        scalar_first: bool = False

        def __call__(self) -> Rotation:
            return Rotation.from_quat(self.quat, scalar_first=self.scalar_first)

    class Matrix(pydantic.BaseModel, extra="forbid"):
        matrix: ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., 3, 3))]
        assume_valid: bool = False

        def __call__(self) -> Rotation:
            return (
                Rotation.from_matrix(self.matrix, assume_valid=self.assume_valid)
                if _matrix_supports_assume_valid()
                else Rotation.from_matrix(self.matrix)
            )

        @pydantic.field_validator("assume_valid", mode="after")
        @classmethod
        def _validate_assume_valid(cls, val: bool) -> bool:  # noqa: FBT001
            """We can only pass assume_valid=True if scipy supports it"""
            if not _matrix_supports_assume_valid() and val:
                err_t = "not_supported"
                msg = "assume_valid=True is only supported in scipy >= 1.17.0"
                raise PydanticCustomError(err_t, msg)
            return val

    class Rotvec(pydantic.BaseModel, extra="forbid"):
        rotvec: ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., 3))]
        degrees: bool = False

        def __call__(self) -> Rotation:
            return Rotation.from_rotvec(self.rotvec, degrees=self.degrees)

    class Mrp(pydantic.BaseModel, extra="forbid"):
        mrp: ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., 3))]

        def __call__(self) -> Rotation:
            return Rotation.from_mrp(self.mrp)

    class Euler(pydantic.BaseModel, extra="forbid"):
        class Arg(pydantic.BaseModel, extra="forbid"):
            seq: str = pydantic.Field(pattern="^([xyz]{1,3}|[XYZ]{1,3})$")
            angles: (
                float
                | ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., slice(1, 4)))]
            )
            degrees: bool = False

        euler: Arg

        def __call__(self) -> Rotation:
            return Rotation.from_euler(
                self.euler.seq, self.euler.angles, degrees=self.euler.degrees
            )

    class Davenport(pydantic.BaseModel, extra="forbid"):
        class Arg(pydantic.BaseModel, extra="forbid"):
            axes: ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., slice(1, 4), 3))]
            order: ty.Literal["e", "extrinsic", "i", "intrinsic"]  # type: ignore[invalid-literal]
            angles: (
                float
                | ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., slice(1, 4)))]
            )
            degrees: bool = False

        davenport: Arg

        def __call__(self) -> Rotation:
            return Rotation.from_davenport(
                self.davenport.axes,
                self.davenport.order,
                self.davenport.angles,
                degrees=self.davenport.degrees,
            )

    def _disc(val: Mapping) -> str:
        if "quat" in val:
            return "from_quat"
        if "matrix" in val:
            return "from_matrix"
        if "rotvec" in val:
            return "from_rotvec"
        if "mrp" in val:
            return "from_mrp"
        if "euler" in val:
            return "from_euler"
        if "davenport" in val:
            return "from_davenport"

        err_t = "invalid_rotation_mapping"
        msg = (
            "no valid keys found in mapping, must have one of: "
            '"quat", "matrix", "rotvec", "mrp", "euler" or "davenport".'
        )
        raise PydanticCustomError(err_t, msg)

    return pydantic.TypeAdapter(
        ty.Annotated[
            ty.Annotated[Quat, pydantic.Tag("from_quat")]
            | ty.Annotated[Matrix, pydantic.Tag("from_matrix")]
            | ty.Annotated[Rotvec, pydantic.Tag("from_rotvec")]
            | ty.Annotated[Mrp, pydantic.Tag("from_mrp")]
            | ty.Annotated[Euler, pydantic.Tag("from_euler")]
            | ty.Annotated[Davenport, pydantic.Tag("from_davenport")],
            pydantic.Discriminator(_disc),
        ]
    )


@functools.lru_cache(maxsize=1)
def _ndarray_adaptor() -> pydantic.TypeAdapter:
    import numpy as np

    return pydantic.TypeAdapter(
        ty.Annotated[np.ndarray, NDArrayAdapter(dtype=np.float64, shape=(..., 4))]
    )


def _validate_rotation(value: ty.Any) -> Rotation:
    from scipy.spatial.transform import Rotation

    if isinstance(value, Rotation):
        return value

    if isinstance(value, Mapping):
        return _mapping_validator().validate_python(value)()

    if isinstance(value, Iterable):
        with contextlib.suppress(pydantic.ValidationError):
            return Rotation(_ndarray_adaptor().validate_python(value))

    err_t = "invalid_rotation_type"
    msg = (
        "Cannot convert {val_t} to scipy.spatial.transform.Rotation. "
        "Expected a Rotation, a quaternion array, or a mapping with one of: "
        '"quat", "matrix", "rotvec", "mrp", "euler" or "davenport".'
    )
    raise PydanticCustomError(err_t, msg, {"val_t": repr(type(value))})


@functools.cache
def _supports_shape() -> bool:
    import scipy

    return version_ge(scipy, (1, 17, 0))


def _matrix_supports_assume_valid() -> bool:
    return _supports_shape()  # also 1.17.0
