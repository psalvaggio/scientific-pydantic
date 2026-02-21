"""Pydantic adapter for slice's"""

import numbers
import typing as ty
from collections.abc import Mapping, Sequence

import pydantic
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from scientific_pydantic.slice_syntax import (
    SliceSyntaxError,
    format_slice_syntax,
    parse_slice_syntax,
)


class SliceAdapter:
    """Pydantic adapter for Python's built-in `slice`.

    JSON representation: `"[start]:[stop][:step]"`
    """

    def __init__(
        self,
        default_type: type = str,
        *,
        start_type: type | None = None,
        stop_type: type | None = None,
        step_type: type | None = None,
    ) -> None:
        distinct_types = {default_type, start_type, stop_type, step_type}
        adapters = {
            t: pydantic.TypeAdapter(t) for t in distinct_types if t not in (None, str)
        }
        self._default_adapter = (
            adapters[default_type] if default_type is not str else None
        )

        self._start_adapter = (
            adapters[start_type]
            if start_type not in (str, None)
            else self._default_adapter
        )
        self._stop_adapter = (
            adapters[stop_type]
            if stop_type not in (str, None)
            else self._default_adapter
        )
        self._step_adapter = (
            adapters[step_type]
            if step_type not in (str, None)
            else self._default_adapter
        )

    def __get_pydantic_core_schema__(  # noqa: C901
        self,
        _source_type: ty.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""

        def _validate(value: ty.Any) -> slice:
            match value:
                case slice():
                    return value
                case Mapping():
                    start, stop, step = _from_mapping(value)
                case str():
                    start, stop, step = _from_str(value)
                case Sequence():
                    start, stop, step = _from_sequence(value)
                case _:
                    msg = "Expected a slice, sequence, mapping or str"
                    raise ValueError(msg)

            if start is not None and self._start_adapter is not None:
                start = self._start_adapter.validate_python(start)
            if stop is not None and self._stop_adapter is not None:
                stop = self._stop_adapter.validate_python(stop)
            if step is not None and self._step_adapter is not None:
                step = self._step_adapter.validate_python(step)
            return slice(start, stop, step)

        def _serialize(value: slice) -> str | dict[str, ty.Any]:
            if all(
                x is None or isinstance(x, numbers.Number)
                for x in (value.start, value.stop, value.step)
            ):
                return format_slice_syntax(value.start, value.stop, value.step)

            return {
                "start": value.start,
                "stop": value.stop,
                "step": value.step,
            }

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize,
                when_used="json",
            ),
        )

    def __get_pydantic_json_schema__(
        self,
        _core_schema: core_schema.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Get the JSON schema for this object"""
        return handler(
            core_schema.union_schema(
                [
                    core_schema.str_schema(),
                    core_schema.list_schema(min_length=1, max_length=3),
                    core_schema.typed_dict_schema(
                        {
                            "start": core_schema.typed_dict_field(
                                self._start_adapter.core_schema
                                if self._start_adapter is not None
                                else core_schema.any_schema(),
                            ),
                            "stop": core_schema.typed_dict_field(
                                self._stop_adapter.core_schema
                                if self._stop_adapter is not None
                                else core_schema.any_schema(),
                            ),
                            "step": core_schema.typed_dict_field(
                                self._step_adapter.core_schema
                                if self._step_adapter is not None
                                else core_schema.any_schema(),
                            ),
                        },
                        total=False,
                    ),
                ],
            ),
        )


IntSliceAdapter = SliceAdapter(int)


def _from_mapping(value: Mapping[str, ty.Any]) -> tuple[ty.Any, ty.Any, ty.Any]:
    if any(x not in ("start", "stop", "step") for x in value):
        msg = 'Invalid key for slice, can only accept "start"/"stop"/"step"'
        raise ValueError(msg)
    return (value.get("start"), value.get("stop"), value.get("step"))


def _from_str(value: str) -> tuple[ty.Any, ty.Any, ty.Any]:
    try:
        start, stop, step = parse_slice_syntax(
            value,
            converter=str,
            require_start=False,
            require_stop=True,
        )
    except SliceSyntaxError as exc:
        raise ValueError(str(exc)) from exc

    return (start, stop, step)


def _from_sequence(value: Sequence[ty.Any]) -> tuple[ty.Any, ty.Any, ty.Any]:
    if 1 <= len(value) <= 3:  # noqa: PLR2004
        return tuple(value)
    msg = "A sequence input to slice must have 1-3 elements"
    raise ValueError(msg)
