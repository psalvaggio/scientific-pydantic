"""Pydantic adapter for astropy.units.PhysicalType."""

from __future__ import annotations

import typing as ty

import pydantic
from pydantic_core import core_schema


class PhysicalTypeAdapter:
    """A pydantic adapter for astropy.units.PhysicalType"""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: ty.Any,
        handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""
        import astropy.units as u

        from .validators import validate_physical_type

        del handler

        if source_type is not u.PhysicalType:
            msg = (
                "PhysicalTypeAdapter is only usable with "
                f"astropy.units.PhysicalType, not {source_type}."
            )
            raise pydantic.PydanticSchemaGenerationError(msg)

        validator = core_schema.no_info_plain_validator_function(validate_physical_type)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema([core_schema.str_schema(), validator]),
            python_schema=validator,
            serialization=core_schema.to_string_ser_schema(
                when_used="json-unless-none"
            ),
        )

    def __get_pydantic_json_schema__(
        self,
        core_schema_: core_schema.CoreSchema,
        handler: pydantic.json_schema.GetJsonSchemaHandler,
    ) -> pydantic.json_schema.JsonSchemaValue:
        """Get the JSON schema for this type"""
        del core_schema_

        desc = "An astropy PhysicalType expressed as a string."
        return handler(core_schema.str_schema()) | {
            "description": desc,
            "examples": ["length", "area"],
        }
