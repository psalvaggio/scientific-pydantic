"""Pydantic adapters for numpy."""

import typing as ty
from collections.abc import Sequence

import pydantic
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

if ty.TYPE_CHECKING:
    import numpy as np


class NDArrayAdapter:
    """Pydantic type adapter for numpy ndarrays with validation constraints.

    Usage:
        class MyModel(BaseModel):
            field: Annotated[
                np.ndarray,
                NDArrayAdapter(shape=(3, None), dtype=float)
            ]

    Parameters
    ----------
    dtype: "type | np.dtype | str | None" = None,
    ndim: int | None = None,
    shape: Sequence[int | range | slice | None] | None = None,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    clip: tuple[float | None, float | None] = (None, None),
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dtype: "type | np.dtype | str | None" = None,
        ndim: int | None = None,
        shape: Sequence[int | range | slice | None] | None = None,
        gt: float | None = None,
        ge: float | None = None,
        lt: float | None = None,
        le: float | None = None,
        clip: tuple[float | None, float | None] = (None, None),
    ) -> None:
        from .validators import NDArrayValidator

        self._validator = NDArrayValidator.from_kwargs(
            dtype=dtype,
            ndim=ndim,
            shape=shape,
            gt=gt,
            ge=ge,
            lt=lt,
            le=le,
            clip=clip,
        )

    def __get_pydantic_core_schema__(
        self,
        _source_type: ty.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for an NDArray"""
        import numpy as np

        def validate(value: ty.Any) -> np.typing.NDArray:
            return self._validator(value)

        def serialize(value: np.typing.NDArray) -> list:
            """Serialize ndarray to nested lists for JSON"""
            return value.tolist()

        python_schema = core_schema.no_info_after_validator_function(
            validate,
            core_schema.any_schema(),
        )

        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema(
                [
                    core_schema.any_schema(),
                    core_schema.no_info_after_validator_function(
                        validate,
                        core_schema.any_schema(),
                    ),
                ],
            ),
            python_schema=python_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize,
                info_arg=False,
                return_schema=core_schema.any_schema(),
            ),
        )

    def __get_pydantic_json_schema__(  # noqa: C901
        self,
        _core_schema: core_schema.CoreSchema,
        _handler: pydantic.GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Generate JSON schema for the ndarray field"""
        json_schema: dict[str, ty.Any] = {"type": "array"}

        # Add description of constraints
        constraints = []
        if self._validator.dtype is not None:
            constraints.append(f"dtype: {self._validator.dtype.dtype}")

        ndim = self._validator.ndim.ndim if self._validator.ndim is not None else None
        if ndim is not None:
            constraints.append(f"{ndim}D array")
        if self._validator.shape is not None:
            shape_desc = "x".join(
                str(s) if s is not None else "?" for s in self._validator.shape.shape
            )
            constraints.append(f"shape: ({shape_desc})")
        if self._validator.ge is not None:
            constraints.append(f"values >= {self._validator.ge.ge}")
        if self._validator.le is not None:
            constraints.append(f"values <= {self._validator.le.le}")
        if self._validator.gt is not None:
            constraints.append(f"values > {self._validator.gt.gt}")
        if self._validator.lt is not None:
            constraints.append(f"values < {self._validator.lt.lt}")
        if self._validator.clip is not None:
            constraints.append(
                f"clipped to [{self._validator.clip.clip[0]}, "
                f"{self._validator.clip.clip[1]}]",
            )

        if constraints:
            json_schema["description"] = "NumPy array: " + ", ".join(constraints)

        # Add items constraint based on ndim
        if ndim is not None and ndim <= 1:
            json_schema["items"] = {"type": "number"}
        elif ndim is not None and ndim > 1:
            # Nested arrays
            items_schema = {"type": "number"}
            for _ in range(ndim - 1):
                items_schema = {"type": "array", "items": items_schema}
            json_schema["items"] = items_schema

        return json_schema
