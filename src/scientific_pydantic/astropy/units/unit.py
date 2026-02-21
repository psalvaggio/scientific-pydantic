"""Pydantic adapter for astropy.units.Unit."""

from __future__ import annotations

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


class EquivalencyValidator:
    """Validator for unit equivalency

    Parameters
    ----------
    equivalent_unit : astropy.units.UnitBase | str
        Validated values must be equivalent to this unit.
    equivalencies:
        Optional list of astropy equivalency pairs (as returned by e.g.
        ``astropy.units.spectral()``).  Passed verbatim to
        ``UnitBase.is_equivalent``.
    """

    def __init__(
        self,
        equivalent_unit: u.UnitBase | str,
        *,
        equivalencies: list[tuple] | None = None,
    ) -> None:
        if isinstance(equivalent_unit, str):
            import astropy.units as u

            equivalent_unit = u.Unit(equivalent_unit)

        self._equivalent_unit = equivalent_unit
        self._equivalencies = equivalencies

    @property
    def equivalent_unit(self) -> u.UnitBase:
        """If non-None, validated units must be equivalent to this unit"""
        return self._equivalent_unit

    @property
    def equivalencies(self) -> list[tuple] | None:
        """Custom equivalencies for the equivalency check"""
        return self._equivalencies

    def __call__(self, unit: u.UnitBase) -> u.UnitBase:
        """Validate the given unit for equivalency

        Parameters
        ----------
        unit : astropy.units.UnitBase
            The unit to validate

        Returns
        -------
        astropy.units.UnitBase
            The input ``unit`` (for validator chaining)
        """
        if self.equivalent_unit is None or unit.is_equivalent(
            self.equivalent_unit, equivalencies=self.equivalencies
        ):
            return unit

        equiv_hint = (
            f" (with equivalencies: {self.equivalencies})"
            if self.equivalencies is not None
            else ""
        )
        err_t = "astropy_unit_not_equivalent"
        msg = "Unit {unit} is not equivalent to {target}{hint}"
        raise pydantic_core.PydanticCustomError(
            err_t,
            msg,
            {
                "unit": unit.to_string(),
                "target": self.equivalent_unit.to_string(),
                "hint": equiv_hint,
            },
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
    """

    def __init__(
        self,
        equivalent_unit: u.UnitBase | str | None = None,
        *,
        equivalencies: list[tuple] | None = None,
    ) -> None:
        self._equivalency_validator = (
            EquivalencyValidator(equivalent_unit, equivalencies=equivalencies)
            if equivalent_unit is not None
            else None
        )

    @property
    def equivalent_unit(self) -> u.UnitBase | None:
        """If non-None, validated units must be equivalent to this unit"""
        return (
            self._equivalency_validator.equivalent_unit
            if self._equivalency_validator is not None
            else None
        )

    @property
    def equivalencies(self) -> list[tuple] | None:
        """Custom equivalencies for the equivalency check"""
        return (
            self._equivalency_validator.equivalencies
            if self._equivalency_validator is not None
            else None
        )

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

        validators = [core_schema.no_info_plain_validator_function(_validate_unit)]
        if self._equivalency_validator is not None:
            validators.append(
                core_schema.no_info_plain_validator_function(
                    self._equivalency_validator
                )
            )

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
        if self._equivalency_validator is not None:
            equiv_hint = ""
            if self.equivalencies is not None:
                equiv_hint = (
                    " (with custom equivalencies: "
                    + ", ".join(f"{x[0]} <-> {x[1]}" for x in self.equivalencies)
                    + ")"
                )
            desc += f' Must be equivalent to "{self.equivalent_unit}"{equiv_hint}.'

        return handler(core_schema.str_schema()) | {
            "description": desc,
            "examples": ["m / s", "km / h", "kg", "deg", "J / (kg K)"]
            if self.equivalent_unit is None
            else [str(self.equivalent_unit)],
        }
