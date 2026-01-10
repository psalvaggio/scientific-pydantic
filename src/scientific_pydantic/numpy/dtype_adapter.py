"""Pydantic adapters for numpy data types."""

import typing as ty

import pydantic
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

if ty.TYPE_CHECKING:
    import numpy as np


class DTypeAdapter:
    """Pydantic adapter for numpy.dtype

    Runtime value: numpy.dtype
    JSON representation: dtype.str (e.g. '<f8', '|i4')
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: type[ty.Any],
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
            ),
        )

    @staticmethod
    def _validate(value: ty.Any) -> "np.dtype":
        import numpy as np

        if isinstance(value, np.dtype):
            return value

        try:
            return np.dtype(value)
        except Exception as exc:
            msg = f"Invalid numpy dtype: {value!r}"
            raise ValueError(msg) from exc

    @staticmethod
    def _serialize(value: "np.dtype") -> str:
        return value.str

    def __get_pydantic_json_schema__(
        self,
        _core_schema: core_schema.CoreSchema,
        _handler: pydantic.GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Generate JSON schema for the ndarray field"""
        json_schema = core_schema.str_schema()
        json_schema["description"] = "NumPy dtype"
        return json_schema
