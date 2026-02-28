"""Pydantic adapters for shapely types"""

import types
import typing as ty
from collections.abc import Mapping

import pydantic
from numpy.typing import ArrayLike, NDArray
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError, core_schema

from ..numpy.validators import NDArrayValidator

T = ty.TypeVar("T")


class GeometryConstraints(pydantic.BaseModel):
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

        def __call__(self, coordinates: ArrayLike) -> NDArray:
            """Validate the bounds on the given coordinates"""
            return NDArrayValidator.from_kwargs(**self.model_dump())(coordinates)

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
        import shapely

        if not isinstance(geom, shapely.geometry.base.BaseGeometry):
            msg = f"the given object ({type(geom).__name__}) was not a shapely geometry"
            raise ValueError(msg)  # noqa: TRY004 (pydantic wants ValueError)

        has_z = getattr(geom, "has_z", False)
        if self.dimensionality == 2 and has_z:  # noqa: PLR2004
            err_t = "dimensionality"
            msg = "Only 2D geometries are allowed."
            raise PydanticCustomError(err_t, msg)
        if self.dimensionality == 3 and not has_z:  # noqa: PLR2004
            err_t = "dimensionality"
            msg = "Only 3D geometries are allowed."
            raise PydanticCustomError(err_t, msg)

        coords: NDArray | None = None
        for idx, dim in enumerate("xyz"):
            bounds = getattr(self, f"{dim}_bounds")
            if bounds is None or (dim == "z" and not has_z):
                continue
            if coords is None:
                coords = shapely.get_coordinates(geom, include_z=True)
            try:
                bounds(coords[:, idx])
            except ValueError as e:
                err_t = "out_of_bounds"
                msg = "{dim} coordinates failed bounds check: {e}"
                raise PydanticCustomError(err_t, msg, {"dim": dim, "e": e}) from None

        return geom

    def summary(self) -> str:
        """Make a summary of the constraints"""
        constraints = []
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


class GeometryAdapter:
    """A pydantic adapter for shapely geometry"""

    CoordinateBounds: ty.ClassVar[type] = GeometryConstraints.CoordinateBounds

    def __init__(
        self,
        *,
        dimensionality: ty.Literal[2, 3] | None = None,
        x_bounds: CoordinateBounds | None = None,
        y_bounds: CoordinateBounds | None = None,
        z_bounds: CoordinateBounds | None = None,
    ) -> None:
        self._validator = GeometryConstraints(
            dimensionality=dimensionality,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
        )

    def __get_pydantic_core_schema__(
        self,
        source_type: ty.Any,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Get the pydantic schema for the shapely geometry"""
        import shapely

        allowable_types = _get_allowable_types(source_type)

        def validate(value: ty.Any) -> ty.Any:
            if isinstance(value, shapely.geometry.base.BaseGeometry):
                pass
            elif isinstance(value, str):
                value = _parse_str(value)
            elif (is_mapping := isinstance(value, Mapping)) or hasattr(
                value, "__geo_interface__"
            ):
                # shapely raises an AttributeError in this case
                if is_mapping and "type" not in value:
                    msg = 'Invalid GeoJSON mapping, missing "type"'
                    err_t = "invalid_geojson"
                    raise PydanticCustomError(err_t, msg)
                try:
                    value = shapely.geometry.shape(value)  # type: ignore[bad-argument-type]
                except (KeyError, ValueError, shapely.errors.ShapelyError) as e:
                    msg = "Invalid GeoJSON mapping ({e})"
                    err_t = "invalid_geojson"
                    raise PydanticCustomError(err_t, msg, {"e": e}) from e

            if not isinstance(value, allowable_types):
                msg = "Value was of incorrect type: {t}. {exp}"
                subs = {"t": type(value).__name__}
                if len(allowable_types) == 1:
                    subs["exp"] = f"Expected {allowable_types[0].__name__}."
                else:
                    subs["exp"] = (
                        "Expected one of: "
                        f"{', '.join(t.__name__ for t in allowable_types)}."
                    )
                err_t = "geometry_type"
                raise PydanticCustomError(err_t, msg, subs)

            return self._validator(value)

        def serialize(geom: shapely.geometry.base.BaseGeometry) -> dict[str, ty.Any]:
            return geom.__geo_interface__

        schema = core_schema.no_info_plain_validator_function(validate)
        return core_schema.json_or_python_schema(
            json_schema=core_schema.chain_schema(
                [
                    core_schema.union_schema(
                        [core_schema.str_schema(), core_schema.dict_schema()]
                    ),
                    schema,
                ]
            ),
            python_schema=schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize,
                return_schema=core_schema.dict_schema(),
            ),
        )

    def __get_pydantic_json_schema__(
        self,
        core_schema: core_schema.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        """Get the JSON schema for this field"""
        json_schema = handler(core_schema)
        json_schema = handler.resolve_ref_schema(json_schema)
        json_schema["description"] = json_schema.get(
            "description", "No user description"
        ) + (
            " (WKT string or GeoJSON object with the following constraints: "
            f"{self._validator.summary()})"
        )
        return json_schema


def _get_allowable_types(source_type: ty.Any) -> tuple[type, ...]:
    """Get the allowable geometry types from the field annotation"""
    import shapely

    if isinstance(source_type, type):
        allowable_types = (source_type,)
    else:
        origin = ty.get_origin(source_type)
        if origin is types.UnionType or origin is ty.Union:
            allowable_types = ty.get_args(source_type)
        else:
            msg = (
                "GeometryAdapter can only be used on a shapely geometry type "
                f"or a union of shapely geometry types, not {source_type}."
            )
            raise pydantic.PydanticSchemaGenerationError(msg)

    if (
        len(
            bad_types := [
                t
                for t in allowable_types
                if not isinstance(t, type)
                or not issubclass(t, shapely.geometry.base.BaseGeometry)
            ]
        )
        > 0
    ):
        msg = (
            "GeometryAdapter can only be used on a shapely geometry type or a "
            "union of shapely geometry types. Found the following invalid "
            f"arguments: {', '.join(str(x) for x in bad_types)}."
        )
        raise pydantic.PydanticSchemaGenerationError(msg)

    return allowable_types


def _parse_str(val: str) -> ty.Any:  # actually shapely
    """Parse a geometry from a string"""
    import shapely

    fails = []

    try:
        return shapely.from_geojson(val)
    except shapely.errors.ShapelyError as e:
        fails.append(f"GeoJSON error: {e}")
    try:
        return shapely.from_wkt(val)
    except shapely.errors.ShapelyError as e:
        fails.append(f"WKT error: {e}")

    err_t = "invalid_geometry"
    msg = "invalid geometry string ({errs})"
    subs = {"errs": ", ".join(fails)}
    raise PydanticCustomError(err_t, msg, subs)
