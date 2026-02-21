"""Pydantic adapter for scipy.spatial.transform.Rotation"""

from __future__ import annotations

import functools
import typing as ty
from collections.abc import Iterable, Mapping, Sequence

import pydantic
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, PydanticCustomError, core_schema

from scientific_pydantic.numpy import NDArrayAdapter

if ty.TYPE_CHECKING:
    from scipy.spatial.transform import Rotation


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
          "assume_valid": bool (default False),
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
        shape: Sequence[int | range | slice | None] | None = None,
    ) -> None:
        from scientific_pydantic.numpy.validators import NDArrayValidator

        kwargs = None
        if single:
            kwargs = {"ndim": 0}
        elif shape is not None:
            kwargs = {"shape": shape}
        elif ndim is not None:
            kwargs = {"ndim": ndim}
        self._np_validator = (
            NDArrayValidator.from_kwargs(**kwargs) if kwargs is not None else None
        )

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
        if self._np_validator is not None:
            python_schema = core_schema.chain_schema(
                [
                    python_schema,
                    core_schema.no_info_plain_validator_function(self._np_validator),
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
def _mapping_validator() -> pydantic.TypeAdapter:
    """Get the pydantic TypeAdapter"""
    if ty.TYPE_CHECKING:
        import numpy as np

    class Quat(pydantic.BaseModel, extra="forbid"):
        quat: ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., 4))]
        scalar_first: bool = False

    class Matrix(pydantic.BaseModel, extra="forbid"):
        matrix: ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., 3, 3))]
        assume_valid: bool = False

    class Rotvec(pydantic.BaseModel, extra="forbid"):
        rotvec: ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., 3))]
        degrees: bool = False

    class Mrp(pydantic.BaseModel, extra="forbid"):
        mrp: ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., 3))]

    class Euler(pydantic.BaseModel, extra="forbid"):
        class Arg(pydantic.BaseModel, extra="forbid"):
            seq: str = pydantic.Field(pattern="^[xyz]{1,3}|[XYZ]{1,3}$")
            angles: (
                float
                | ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., slice(1, 4)))]
            )
            degrees: bool = False

        euler: Arg

    class Davenport(pydantic.BaseModel, extra="forbid"):
        class Arg(pydantic.BaseModel, extra="forbid"):
            axes: ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., slice(1, 4), 3))]
            order: ty.Literal["e", "extrinsic", "i", "intrinsic"]
            angles: (
                float
                | ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., slice(1, 4)))]
            )
            degrees: bool = False

        davenport: Arg

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
            ty.Discriminator(_disc),
        ]
    )


@functools.lru_cache(maxsize=1)
def _ndarray_adaptor() -> pydantic.TypeAdapter:
    import numpy as np

    return pydantic.TypeAdapter(
        ty.Annotated[np.ndarray, NDArrayAdapter(shape=(..., 4))]
    )


def _validate_rotation(value: ty.Any) -> Rotation:
    from scipy.spatial.transform import Rotation

    if isinstance(value, Rotation):
        return value

    if isinstance(value, Mapping):
        return _mapping_validator().validate_python(value)

    if isinstance(value, Iterable):
        return _ndarray_adaptor(value)

    err_t = "invalid_rotation_type"
    msg = (
        f"Cannot convert {type(value)!r} to scipy.spatial.transform.Rotation. "
        "Expected a Rotation, a quaternion array, or a mapping with one of: "
        '"quat", "matrix", "rotvec", "mrp", "euler" or "davenport".'
    )
    raise PydanticCustomError(err_t, msg)
