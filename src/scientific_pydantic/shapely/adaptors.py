"""Pydantic adaptors for shapely types"""

import typing as ty
from collections.abc import Iterable

import pydantic
from numpy.typing import NDArray
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from shapely import from_wkt, get_coordinates
from shapely.errors import ShapelyError
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    base,
)

from ..numpy.validators import NDArrayValidator

T = ty.TypeVar("T", bound=base.BaseGeometry)

GEOMETRY_TYPE_MAP: dict[str, type[base.BaseGeometry]] = {
    "point": Point,
    "linestring": LineString,
    "linearring": LinearRing,
    "polygon": Polygon,
    "multipoint": MultiPoint,
    "multilinestring": MultiLineString,
    "multipolygon": MultiPolygon,
    "geometrycollection": GeometryCollection,
}


class ShapelyGeometryConstraints(pydantic.BaseModel):
    """Validation constraints that can be applied to shapely geometries"""

    class CoordinateBounds(pydantic.BaseModel):
        """Bounds checks for coordinates"""

        gt: float | None = pydantic.Field(
            default=None, description="All coordinates must be > this value"
        )
        ge: float | None = pydantic.Field(
            default=None, description="All coordinates must be >= this value"
        )
        lt: float | None = pydantic.Field(
            default=None, description="All coordinates must be < this value"
        )
        le: float | None = pydantic.Field(
            default=None, description="All coordinates must be <= this value"
        )

        def __call__(self, coordinates: NDArray) -> NDArray:
            """Validate the bounds on the given coordinates"""
            return NDArrayValidator(**self.model_dump())(coordinates)

    geometry_type: list[type[base.BaseGeometry]] | None = pydantic.Field(
        default=None,
        min_length=1,
        description="Geometry type whitelist. Accepts any type if None.",
    )

    dimensionality: ty.Literal[2, 3] | None = pydantic.Field(
        default=None,
        description="Dimensionality of the coordinates to accept. Accepts any if None.",
    )

    x_bounds: CoordinateBounds | None = pydantic.Field(
        default=None,
        description="Bounds for all of the x-coordinates in the geometry",
    )

    y_bounds: CoordinateBounds | None = pydantic.Field(
        default=None,
        description="Bounds for all of the y-coordinates in the geometry",
    )

    z_bounds: CoordinateBounds | None = pydantic.Field(
        default=None,
        description="Bounds for all of the z-coordinates in the geometry",
    )

    def __call__(self, geom: T) -> T:
        """Validate the given shapely geometry w.r.t the given constraints

        Parameters
        ----------
        geom : BaseGeometry
            The geometry to validate

        Returns
        -------
        BaseGeometry
            The geometry (if it passed validation)

        Raises
        ------
        ValueError
        If the geometry violated one of the user-provided constraints.
        """
        if not isinstance(geom, base.BaseGeometry):
            msg = f"the given object ({type(geom).__name__}) was not a shapely geometry"
            raise ValueError(msg)  # noqa: TRY004 (pydantic wants ValueError)

        if self.geometry_type is not None and not isinstance(
            geom, tuple(self.geometry_type)
        ):
            msg = (
                f"Geometry type {type(geom).__name__} not allowed. "
                f"Expected one of: {[t.__name__ for t in self.geometry_type]}"
            )
            raise ValueError(msg)

        has_z = getattr(geom, "has_z", False)
        if self.dimensionality == 2 and has_z:  # noqa: PLR2004
            msg = "Only 2D geometries are allowed."
            raise ValueError(msg)
        if self.dimensionality == 3 and not has_z:  # noqa: PLR2004
            msg = "Only 3D geometries are allowed."
            raise ValueError(msg)

        coords = None
        for idx, dim in enumerate("xyz"):
            bounds = getattr(self, f"{dim}_bounds")
            if bounds is None or (dim == "z" and not has_z):
                continue
            if coords is None:
                coords = get_coordinates(geom, include_z=True)
            try:
                bounds(coords[:, idx])
            except ValueError as e:
                msg = f"{dim} coordinates failed bounds check: {e}"
                raise ValueError(msg) from None

        return geom

    def summary(self) -> str:
        """Make a summary of the constraints"""
        constraints = []
        if self.geometry_type is not None:
            constraints.append(
                "one of the following types: "
                f"{', '.join(t.__name__ for t in self.geometry_type)}"
            )
        if self.dimensionality is not None:
            constraints.append(f"dimensionality = {self.dimensionality}")

        for dim in "xyz":
            bounds = getattr(self, f"{dim}_bounds")
            if bounds is None:
                continue
            for field, sign in (("le", "<="), ("lt", "<"), ("gt", ">"), ("ge", ">=")):
                if (val := getattr(bounds, field)) is not None:
                    constraints.append(f"{dim} {sign} {val}")

        return (
            " ".join(f"{i + 1}. {c}" for i, c in enumerate(constraints))
            if len(constraints) > 0
            else "N/A"
        )

    @pydantic.field_validator("geometry_type", mode="before")
    @classmethod
    def _normalize_geometry_types(
        cls,
        value: ty.Any,
    ) -> tuple[type[base.BaseGeometry], ...]:
        """Supports string literals for types"""
        if isinstance(value, (str, type)):
            value = [value]

        if not isinstance(value, Iterable):
            return value  # this will raise a better error later

        out = []
        for val in value:
            if (
                isinstance(val, str)
                and (x := GEOMETRY_TYPE_MAP.get(val.lower())) is not None
            ):
                out.append(x)
            else:
                out.append(val)
        return out


def shapely_geometry(**kwargs) -> ty.Any:
    """Generate a field annotation for a shapely geometry"""
    validator = ShapelyGeometryConstraints(**kwargs)
    if validator.geometry_type is not None:
        t = (
            validator.geometry_type[0]
            if len(validator.geometry_type) == 0
            else ty.Union[tuple(validator.geometry_type)]  # noqa: UP007
        )
    else:
        t = base.BaseGeometry

    class ShapelyGeometryField:
        """Pydantic adapter for a shapely geometry"""

        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source_type: ty.Any,
            _handler: pydantic.GetCoreSchemaHandler,
        ) -> core_schema.CoreSchema:
            def validate(value: ty.Any) -> ty.Any:
                if isinstance(value, str):
                    try:
                        value = from_wkt(value)
                    except ShapelyError as e:
                        raise ValueError(str(e)) from e
                return validator(value)

            def serialize(geom: base.BaseGeometry) -> str:
                return geom.wkt

            schema = core_schema.no_info_plain_validator_function(validate)
            return core_schema.json_or_python_schema(
                json_schema=core_schema.chain_schema(
                    [core_schema.str_schema(), schema]
                ),
                python_schema=schema,
                serialization=core_schema.plain_serializer_function_ser_schema(
                    serialize,
                    return_schema=core_schema.str_schema(),
                ),
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls,
            core_schema: core_schema.CoreSchema,
            handler: pydantic.GetJsonSchemaHandler,
        ) -> JsonSchemaValue:
            json_schema = handler(core_schema)
            json_schema = handler.resolve_ref_schema(json_schema)
            json_schema["description"] = json_schema.get(
                "description", "No user description"
            ) + (f" (WKT string with the following constraints: {validator.summary()})")
            return json_schema

    return ty.Annotated[t, ShapelyGeometryField()]
