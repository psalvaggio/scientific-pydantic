"""Pydantic adapter for astropy.units.Unit."""

from __future__ import annotations

import typing as ty

import pydantic_core

if ty.TYPE_CHECKING:
    import astropy.units as u


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


def validate_physical_type(value: ty.Any) -> u.PhysicalType:
    """Validate and coerce a value to an astropy Unit."""
    import astropy.units as u

    if isinstance(value, u.PhysicalType):
        return value
    if isinstance(value, (str, u.UnitBase, u.Quantity)):
        try:
            return u.get_physical_type(value)
        except ValueError as exc:
            err_t = "parse_error"
            msg = 'Could not parse "{value}" as an astropy PhysicalType: {error}'
            raise pydantic_core.PydanticCustomError(
                err_t, msg, {"value": value, "error": str(exc)}
            ) from exc

    err_t = "type_error"
    msg = (
        "Expected a string, astropy PhysicalType, or quantity-like object, "
        " got {type_name}"
    )
    raise pydantic_core.PydanticCustomError(
        err_t, msg, {"type_name": type(value).__name__}
    )


UnitOrQuantity = ty.TypeVar("UnitOrQuantity", "u.UnitBase", "u.Quantity")


class PhysicalTypeValidator:
    """Validator for the physical type of a unit/quantity

    Parameters
    ----------
    physical_type : astropy.units.PhysicalType
        The unit/quantity must have this physical type
    """

    def __init__(self, physical_type: u.PhysicalType) -> None:
        self._physical_type = physical_type

    @property
    def physical_type(self) -> u.PhysicalType:
        """The unit/quantity must have this physical type"""
        return self._physical_type

    def __call__(self, val: UnitOrQuantity) -> UnitOrQuantity:
        """Validate the given unit/quantity

        Parameters
        ----------
        val : astropy.units.UnitBase | astropy.units.Quantity
            The unit/quantity to validate

        Returns
        -------
        astropy.units.UnitBase | astropy.units.Quantity
            The input unit/quantity (for validator chaining)
        """
        import astropy.units as u

        unit = val.unit if isinstance(val, u.Quantity) else val
        if unit.physical_type != self._physical_type:  # type: ignore[missing-attribute]
            err_t = "wrong_physical_type"
            msg = "Unit ({unit}) must have physical type {req}, but it was {actual}"
            raise pydantic_core.PydanticCustomError(
                err_t,
                msg,
                {
                    "unit": str(unit),
                    "req": str(self._physical_type),
                    "actual": str(unit.physical_type),
                },
            )
        return val
