"""Pydantic adapter for astropy.units.Unit."""

from __future__ import annotations

import dataclasses
import typing as ty

if ty.TYPE_CHECKING:
    import astropy.units as u

import pydantic
import pydantic_core
from pydantic_core import core_schema


def _validate_unit(value: ty.Any) -> u.UnitBase:
    """Validate and coerce a value to an astropy Unit."""
    import astropy.units as u

    if isinstance(value, u.UnitBase):
        return value
    if isinstance(value, str):
        try:
            return u.Unit(value)
        except ValueError as exc:
            err_t = "astropy_unit_parse_error"
            msg = "Could not parse {value} as an astropy unit: {error}"
            raise pydantic_core.PydanticCustomError(
                err_t, msg, {"value": value, "error": str(exc)}
            ) from exc

    err_t = "astropy_unit_type_error"
    msg = "Expected a string or astropy UnitBase instance, got {type_name}"
    raise pydantic_core.PydanticCustomError(
        err_t, msg, {"type_name": type(value).__name__}
    )


class UnitAdapter:
    """A pydantic adapter for astropy units

    Parameters
    ----------
    equivalent_unit : astropy.units.UnitBase | str | None
        If given, validated values must be equivalent to this unit.
    equivalencies:
        Optional list of astropy equivalency pairs (as returned by e.g.
        ``astropy.units.spectral()``).  Passed verbatim to
        ``UnitBase.is_equivalent``.
    physical_type : u.PhysicalType | str | u.Quantity | u.UnitBase | None
        If given, the unit by have this physical type
    """

    def __init__(
        self,
        equivalent_unit: u.UnitBase | str | None = None,
        *,
        equivalencies: list[tuple] | None = None,
        physical_type: u.PhysicalType | str | u.Quantity | u.UnitBase | None = None,
    ) -> None:
        from .validators import (
            EquivalencyValidator,
            PhysicalTypeValidator,
            validate_physical_type,
        )

        @dataclasses.dataclass
        class Validators:
            equivalency: EquivalencyValidator | None = None
            physical_type: PhysicalTypeValidator | None = None

        validators: dict[str, ty.Any] = {}

        if equivalent_unit is not None:
            validators["equivalency"] = EquivalencyValidator(
                equivalent_unit, equivalencies=equivalencies
            )

        if physical_type is not None:
            validators["physical_type"] = PhysicalTypeValidator(
                validate_physical_type(physical_type)
            )
        self._validators = Validators(**validators)

    @property
    def equivalent_unit(self) -> u.UnitBase | None:
        """If non-None, validated units must be equivalent to this unit"""
        val = self._validators.equivalency
        return val.equivalent_unit if val is not None else None

    @property
    def equivalencies(self) -> list[tuple] | None:
        """Custom equivalencies for the equivalency check"""
        val = self._validators.equivalency
        return val.equivalencies if val is not None else None

    @property
    def physical_type(self) -> u.PhysicalType | None:
        """If non-None, validated unit must be of this physical type"""
        val = self._validators.physical_type
        return val.physical_type if val is not None else None

    def __get_pydantic_core_schema__(
        self,
        source_type: ty.Any,
        handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""
        import astropy.units as u

        del handler

        if source_type is not u.UnitBase:
            msg = (
                "UnitAdapter is only usable with "
                f"astropy.units.UnitBase, not {source_type}."
            )
            raise pydantic.PydanticSchemaGenerationError(msg)

        validators: list[core_schema.CoreSchema] = [
            core_schema.no_info_plain_validator_function(_validate_unit)
        ]
        if (equiv_val := self._validators.equivalency) is not None:
            validators.append(core_schema.no_info_plain_validator_function(equiv_val))
        if (pt_val := self._validators.physical_type) is not None:
            validators.append(core_schema.no_info_plain_validator_function(pt_val))

        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema(
                [core_schema.str_schema(), *validators]
            ),
            python_schema=core_schema.chain_schema(validators),
            serialization=core_schema.to_string_ser_schema(),
        )

    def __get_pydantic_json_schema__(
        self,
        core_schema_: core_schema.CoreSchema,
        handler: pydantic.json_schema.GetJsonSchemaHandler,
    ) -> pydantic.json_schema.JsonSchemaValue:
        """Get the JSON schema for this type"""
        del core_schema_

        desc = "An astropy unit expressed as a string."
        if self.equivalent_unit is not None:
            equiv_hint = ""
            if self.equivalencies is not None:
                equiv_hint = (
                    " (with custom equivalencies: "
                    + ", ".join(f"{x[0]} <-> {x[1]}" for x in self.equivalencies)
                    + ")"
                )
            desc += f' Must be equivalent to "{self.equivalent_unit}"{equiv_hint}.'
        if self.physical_type is not None:
            desc += f' Must be of type "{self.physical_type!s}".'

        return handler(core_schema.str_schema()) | {
            "description": desc,
            "examples": ["m / s", "km / h", "kg", "deg", "J / (kg K)"]
            if self.equivalent_unit is None
            else [str(self.equivalent_unit)],
        }
