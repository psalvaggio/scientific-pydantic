"""Pydantic integration for `astropy.units.Quantity`."""

from __future__ import annotations

import dataclasses
import functools
import typing as ty

import pydantic
from pydantic import PydanticSchemaGenerationError
from pydantic_core import PydanticCustomError, core_schema

from scientific_pydantic.astropy.units.validators import validate_physical_type

if ty.TYPE_CHECKING:
    import types
    from collections.abc import Mapping, Sequence

    import astropy.units as u
    from numpy.typing import ArrayLike
    from pydantic import GetCoreSchemaHandler
    from pydantic.json_schema import JsonSchemaValue


class QuantityAdapter:
    """Pydantic type adapter for astropy.units.Quantity

    This type supports a similar API to the numpy NDArray validator, but omits
    dtype, as astropy.units.Quantity's are always floating point.

    Parameters
    ----------
    equivalent_unit : astropy.units.UnitBase | astropy.units.PhysicalType | None
        If given, then quantities must have units equivalent to this.
    equivalencies : list[tuple] | None
        Optional list of astropy equivalency pairs (as returned by e.g.
        ``astropy.units.spectral()``).  Passed verbatim to
        ``UnitBase.is_equivalent``.
    physical_type : u.PhysicalType | str | u.Quantity | u.UnitBase | None
        If given, the quantity by have this physical type.
    scalar : bool
        If True, only scalar quantities will be accepted. If False, only vector
        quantities will be accepted. If None, no scalar constraints are enforced,
        unless `ndim` or `shape` are provided.
    ndim : int | None
        If given, the dimensionality of the quantity must match this value. Must
        be >= 0.
    shape : Sequence[Ellipsis | int | range | slice | None] | None
        Shape specifier for the given array. See `NDArrayValidator` for a
        description of how this works.
    gt : ArrayLike | astropy.units.Quantity | None
        If given, all elements in the given quantity must be > this value. If no
        units are provided, then `equivalent_unit` is used (if provided).
    ge : ArrayLike | astropy.units.Quantity | None
        If given, all elements in the given quantity must be >= this value. If no
        units are provided, then `equivalent_unit` is used (if provided).
    lt : ArrayLike | astropy.units.Quantity | None
        If given, all elements in the given quantity must be < this value. If no
        units are provided, then `equivalent_unit` is used (if provided).
    le : ArrayLike | astropy.units.Quantity | None
        If given, all elements in the given quantity must be <= this value. If no
        units are provided, then `equivalent_unit` is used (if provided).
    clip : Sequence[ArrayLike | u.Quantity | None] | u.Quantity
        If given, a 2-element sequence of [min_clip, max_clip] to which to clip
        the values in the quantity. If no units are provided, then
        `equivalent_unit` is used (if provided).
    """

    def __init__(  # noqa: PLR0913, C901
        self,
        equivalent_unit: u.UnitBase | None = None,
        *,
        equivalencies: list[tuple] | None = None,
        physical_type: u.PhysicalType | str | u.Quantity | u.UnitBase | None = None,
        scalar: bool | None = None,
        ndim: int | None = None,
        shape: Sequence[types.EllipsisType | int | range | slice | None] | None = None,
        gt: ArrayLike | u.Quantity | None = None,
        ge: ArrayLike | u.Quantity | None = None,
        lt: ArrayLike | u.Quantity | None = None,
        le: ArrayLike | u.Quantity | None = None,
        clip: Sequence[ArrayLike | u.Quantity | None] | u.Quantity = (None, None),
        serialize_as_unit: u.UnitBase | None = None,
    ) -> None:
        import astropy.units as u
        import numpy as np

        from scientific_pydantic.numpy.validators import (
            NDimValidator,
            ShapeValidator,
            validate_all_ge,
            validate_all_gt,
            validate_all_le,
            validate_all_lt,
        )

        from .validators import (
            EquivalencyValidator,
            PhysicalTypeValidator,
        )

        self._serialize_as_unit = serialize_as_unit

        @dataclasses.dataclass
        class CtorValidators:
            equivalency: EquivalencyValidator | None = None
            physical_type: PhysicalTypeValidator | None = None
            scalar: ScalarValidator | None = None
            ndim: NDimValidator | None = None
            shape: ShapeValidator | None = None
            ge: ty.Callable[[u.Quantity], u.Quantity] | None = None
            gt: ty.Callable[[u.Quantity], u.Quantity] | None = None
            le: ty.Callable[[u.Quantity], u.Quantity] | None = None
            lt: ty.Callable[[u.Quantity], u.Quantity] | None = None
            clip: ty.Callable[[u.Quantity], u.Quantity] | None = None

        validators: dict[str, ty.Callable[[u.Quantity], u.Quantity]] = {}

        # Handle contradictions in the shape arguments
        if scalar is not None:
            if scalar:
                if ndim is not None and ndim != 0:
                    msg = f"scalar=True and ndim={ndim} contradict"
                    raise PydanticSchemaGenerationError(msg)
                ndim = None  # ndim = 0 is redundant
                if shape is not None and shape != ():
                    msg = f"scalar=True and shape={shape} contradict"
                    raise PydanticSchemaGenerationError(msg)
                shape = None  # shape = () is redundant
            else:
                if ndim == 0:
                    msg = "scalar=False and ndim=0 contradict"
                    raise PydanticSchemaGenerationError(msg)
                if shape == ():
                    msg = "scalar=False and shape=() contradict"
                    raise PydanticSchemaGenerationError(msg)
            validators["scalar"] = ScalarValidator(scalar=scalar)

        def apply_unit_to_bound(
            x: ArrayLike | u.Quantity | None, name: str
        ) -> u.Quantity | None:
            if x is None:
                return None
            if isinstance(x, u.Quantity):
                return x
            if equivalent_unit is not None:
                return x << equivalent_unit
            msg = (
                f'If equivalent_unit is not defined, then "{name}" must be '
                "a Quantity if given"
            )
            raise PydanticSchemaGenerationError(msg)

        for bound, name, val in (
            (gt, "gt", validate_all_gt),
            (ge, "ge", validate_all_ge),
            (lt, "lt", validate_all_lt),
            (le, "le", validate_all_le),
        ):
            bound_quant = apply_unit_to_bound(bound, name)
            if bound_quant is not None:
                validators[name] = functools.partial(val, bound=bound_quant)

        if len(clip) != 2:  # noqa: PLR2004
            msg = f"clip must be a sequence of size 2, was {len(clip)}"
            raise PydanticSchemaGenerationError(msg)

        clip = (
            apply_unit_to_bound(clip[0], "clip[0]"),
            apply_unit_to_bound(clip[1], "clip[1]"),
        )

        def clip_val(q: u.Quantity) -> u.Quantity:
            if clip[0] is not None or clip[1] is not None:
                return np.clip(q, clip[0], clip[1])  # type: ignore[bad-return]
            return q

        validators["clip"] = clip_val

        self._validators = CtorValidators(
            equivalency=EquivalencyValidator(
                equivalent_unit, equivalencies=equivalencies
            )
            if equivalent_unit is not None
            else None,
            physical_type=PhysicalTypeValidator(validate_physical_type(physical_type))
            if physical_type is not None
            else None,
            ndim=NDimValidator(ndim=ndim) if ndim is not None else None,
            shape=ShapeValidator(shape=shape) if shape is not None else None,
            **validators,  # type: ignore[bad-argument-type]
        )

    def __get_pydantic_core_schema__(
        self,
        _source_type: ty.Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for this type"""
        validators = [
            core_schema.no_info_plain_validator_function(func)
            for field in dataclasses.fields(self._validators)
            if (func := getattr(self._validators, field.name)) is not None
        ]

        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema(
                [
                    core_schema.no_info_plain_validator_function(_validate_quantity),
                    *validators,
                ],
            ),
            python_schema=core_schema.chain_schema(
                [
                    core_schema.no_info_plain_validator_function(_validate_quantity),
                    *validators,
                ],
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                functools.partial(
                    _serialize, serialize_as_unit=self._serialize_as_unit
                ),
                info_arg=True,
                when_used="unless-none",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: core_schema.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Generate JSON schema for the ndarray field"""
        schema = handler(core_schema.any_schema())
        schema["description"] = "An encoding of an astropy.units.Quantity"
        return schema


def _serialize(
    q: u.Quantity,
    info: core_schema.SerializationInfo,
    *,
    serialize_as_unit: u.UnitBase | None,
) -> ty.Any:
    """Serialize a Quantity"""
    if serialize_as_unit is not None:
        q = q.to(serialize_as_unit)

    if info.mode == "python":
        return q

    import numpy as np

    value = q.value

    # Convert numpy scalar to a plain Python scalar so that JSON
    # serialisers (and model_dump) don't emit ndarray objects.
    if (isinstance(value, np.ndarray) and value.ndim == 0) or isinstance(
        value, np.generic
    ):
        value = value.item()
    else:
        value = value.tolist()

    return {"value": value, "unit": str(q.unit)}


def _dict_to_quantity(data: Mapping[str, ty.Any]) -> u.Quantity:
    """Deserialise a dict produced by :func:`_quantity_to_dict`."""
    import astropy.units as u

    if (value := data.get("value")) is None:
        err_t = "missing_value"
        msg = 'Expected a "value" key in a quantity mapping'
        raise PydanticCustomError(err_t, msg)

    if (unit_val := data.get("unit")) is not None:
        from .validators import validate_unit

        unit = validate_unit(unit_val)
    else:
        unit = u.Unit("")

    try:
        return u.Quantity(value, unit=unit)
    except TypeError as e:
        err_t = "invalid_quantity"
        msg = "Cannot construct a quantity from the given value/unit: {e}"
        raise PydanticCustomError(err_t, msg, {"e": str(e)}) from e


def _validate_quantity(value: ty.Any) -> u.Quantity:
    """Core validation logic - accepts Quantity, dict, or bare numeric."""
    import astropy.units as u

    if isinstance(value, u.Quantity):
        return value

    if isinstance(value, dict):
        return _dict_to_quantity(value)

    # Bare scalar / array -> dimensionless
    try:
        return u.Quantity(value)
    except TypeError as e:
        err_t = "invalid_quantity"
        msg = "Cannot construct a quantity from the given value: {e}"
        raise PydanticCustomError(err_t, msg, {"e": str(e)}) from e


class ScalarValidator(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Validator for the scalar property on a Quantity"""

    scalar: bool

    def __call__(self, q: u.Quantity) -> u.Quantity:
        """Apply scalar validation"""
        if q.isscalar != self.scalar:
            err_t = "scalar_error"
            msg = "Expected isscalar to be {exp}, was {actual}"
            raise PydanticCustomError(
                err_t, msg, {"exp": self.scalar, "actual": q.isscalar}
            )
        return q
