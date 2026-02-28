"""Adaptor for range"""

import typing as ty

import pydantic
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from .slice_syntax import (
    SliceSyntaxError,
    format_slice_syntax,
    parse_slice_syntax,
)


class RangeAdapter:
    """Pydantic adapter for Python ``range`` using slice syntax."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: ty.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""

        def _validate(value: ty.Any) -> range:
            if isinstance(value, range):
                return value

            if isinstance(value, str):
                try:
                    start, stop, step = parse_slice_syntax(
                        value,
                        converter=int,
                        require_start=False,
                        require_stop=True,
                    )
                except SliceSyntaxError as exc:
                    raise ValueError(str(exc)) from exc

                return range(
                    start if start is not None else 0,
                    stop,
                    step if step is not None else 1,
                )

            msg = "Expected range or slice-syntax string"
            raise ValueError(msg)

        def _serialize(value: range) -> str:
            step = None if value.step == 1 else value.step
            return format_slice_syntax(value.start, value.stop, step)

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize,
                when_used="json",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: core_schema.CoreSchema,
        _handler: pydantic.GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Get the JSON schema for this type"""
        return {
            "type": "string",
            "title": "range",
            "description": "Python range syntax: start:stop[:step]",
            "pattern": r"^\s*-?\d+\s*:\s*-?\d+\s*(?::\s*-?\d+\s*)?$",
        }
