"""Unit tests for shapely geometry adaptors"""

import typing as ty
from collections.abc import Mapping

import numpy.testing as npt
import pydantic
import pytest
import shapely
import shapely.testing
from numpy.typing import ArrayLike

from scientific_pydantic.shapely.adapters import (
    ShapelyGeometryAdapter,
    ShapelyGeometryConstraints,
)

AnyShapelyGeometry = ty.Annotated[
    shapely.geometry.base.BaseGeometry, ShapelyGeometryAdapter()
]


@pytest.mark.parametrize(
    ("params", "data"),
    [
        pytest.param({"gt": 0.0}, [1.0, 2.0, 3.0], id=">0"),
        pytest.param({"ge": 0.0}, [0.0, 1.0, 2.0], id=">=0"),
        pytest.param({"lt": 10.0}, [1.0, 2.0, 3.0], id="<10"),
        pytest.param({"le": 10.0}, [1.0, 10.0, 3.0], id="<=10"),
        pytest.param({"ge": 0.0, "le": 10.0}, [[[0.0, 5.0, 10.0]]], id="<=10,>=0"),
    ],
)
def test_coordinate_bounds_valid(params: dict[str, ty.Any], data: ArrayLike) -> None:
    """Test passing cases for coordinate bounds"""
    bounds = ShapelyGeometryConstraints.CoordinateBounds(**params)
    result = bounds(data)
    npt.assert_array_equal(result, data)


@pytest.mark.parametrize(
    ("params", "data", "match"),
    [
        pytest.param(
            {"gt": 0},
            [-1.0, 2.0, 3.0],
            "Not all elements were greater than 0",
            id=">0",
        ),
        pytest.param(
            {"lt": 10},
            [-1.0, 11.0, 3.0],
            "Not all elements were less than 10",
            id="<10",
        ),
        pytest.param(
            {"le": 10, "ge": 0},
            [0.0, 5.0, 11.0],
            "Not all elements were less than or equal to 10",
            id=">=0,<=10",
        ),
    ],
)
def test_coordinate_bounds_invalid(
    params: dict[str, ty.Any], data: ArrayLike, match: str
) -> None:
    """Test failing cases for coordinate bounds"""
    bounds = ShapelyGeometryConstraints.CoordinateBounds(**params)
    with pytest.raises(ValueError, match=match):
        bounds(data)


@pytest.mark.parametrize(
    ("data", "must_haves"),
    [
        pytest.param({}, ["N/A"], id="none"),
        pytest.param({"dimensionality": 2}, ["dimensionality = 2"], id="dim"),
        pytest.param(
            {"x_bounds": {"ge": 0.0, "le": 10.0}},
            ["x >=", "x <=", "0.0", "10.0"],
            id="x_bounds",
        ),
        pytest.param(
            {
                "x_bounds": {"ge": -180.0, "le": 180.0},
                "y_bounds": {"ge": -90.0},
                "z_bounds": {"le": 1000.0},
            },
            ["x >=", "x <=", "y >=", "z <="],
            id="all_bounds",
        ),
        pytest.param(
            {"y_bounds": {"gt": 0.0, "lt": 10.0}},
            ["y > ", "y < ", "0.0", "10.0"],
            id="y_bounds",
        ),
    ],
)
def test_summary(data: dict[str, ty.Any], must_haves: list[str]) -> None:
    """Test the summary method"""
    constraints = ShapelyGeometryConstraints(**data)
    summary = constraints.summary()
    for x in must_haves:
        assert x in summary


class GeometryModel(pydantic.BaseModel, frozen=True, extra="forbid"):
    """Test model"""

    base: AnyShapelyGeometry | None = None

    any_point: ty.Annotated[shapely.Point, ShapelyGeometryAdapter()] | None = None

    point3d: (
        ty.Annotated[shapely.Point, ShapelyGeometryAdapter(dimensionality=3)] | None
    ) = None

    point2d_01: (
        ty.Annotated[
            shapely.Point,
            ShapelyGeometryAdapter(
                dimensionality=2,
                x_bounds={"ge": 0, "le": 1},
                y_bounds={"ge": 0, "le": 1},
            ),
        ]
        | None
    ) = None

    linear_ring2d: (
        ty.Annotated[shapely.LinearRing, ShapelyGeometryAdapter(dimensionality=2)]
        | None
    ) = None

    pt_or_multipt: (
        ty.Annotated[shapely.Point | shapely.MultiPoint, ShapelyGeometryAdapter()]
        | None
    ) = None

    any_polygon: ty.Annotated[shapely.Polygon, ShapelyGeometryAdapter()] | None = None


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"base": shapely.Point(0, 1)}, id="base"),
        pytest.param({"any_point": shapely.Point(2, 3)}, id="any_point"),
        pytest.param({"any_point": "POINT (1 2)"}, id="any_point-wkt"),
        pytest.param(
            {"any_point": {"type": "Point", "coordinates": [1, 2]}},
            id="any_point-geojson",
        ),
        pytest.param({"point3d": shapely.Point(2, 3, 4)}, id="point3d"),
        pytest.param({"point2d_01": shapely.Point(0, 1)}, id="point2d_01"),
        pytest.param(
            {"linear_ring2d": shapely.LinearRing([(0, 1), (1, 0), (1, 1)])},
            id="linear_ring2d",
        ),
        pytest.param(
            {"pt_or_multipt": shapely.Point(1.2345e10, -2.3456e-10)},
            id="pt_or_multipt-pt",
        ),
        pytest.param(
            {"pt_or_multipt": shapely.MultiPoint([(0, 1), (-1, -2)])},
            id="pt_or_multipt-multipt",
        ),
        pytest.param(
            {"pt_or_multipt": shapely.MultiPoint([])}, id="pt_or_multipt-multipt-empty"
        ),
        pytest.param(
            {"any_polygon": "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"}, id="polygon-wkt"
        ),
        pytest.param(
            {
                "any_polygon": shapely.Polygon(
                    [(0, 0), (10, 0), (10, 10), (0, 10)],
                    [[(2, 2), (8, 2), (8, 8), (2, 8)]],
                )
            },
            id="any_polygon-with_holes",
        ),
    ],
)
def test_valid_models(data: dict[str, ty.Any]) -> None:
    """Test valid use cases of shapely fields"""
    model = GeometryModel(**data)
    for field in GeometryModel.model_fields:
        val = getattr(model, field)
        if (truth := data.get(field)) is not None:
            if isinstance(truth, str):
                shapely.testing.assert_geometries_equal(val, shapely.from_wkt(truth))
            elif isinstance(truth, Mapping):
                shapely.testing.assert_geometries_equal(
                    val, shapely.geometry.shape(truth)
                )
            else:
                shapely.testing.assert_geometries_equal(val, truth)
        else:
            assert val is None


@pytest.mark.parametrize(
    ("data", "match"),
    [
        pytest.param({"base": "foo"}, "invalid_geometry", id="invalid-geo-str"),
        pytest.param(
            {"base": '{"type": "Point", "coordinates": [1, 2, 3, 4]}'},
            "invalid_geometry",
            id="invalid-geojson",
        ),
        pytest.param({"base": 10}, "geometry_type", id="int-for-base"),
        pytest.param(
            {"any_point": "MULTIPOINT (1 2, 3 4)"},
            "geometry_type",
            id="multipoint-for-any_point",
        ),
        pytest.param(
            {"pt_or_multipt": "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"},
            "geometry_type",
            id="polygon-for-pt_or_multipt",
        ),
        pytest.param(
            {"any_point": {"a": 4}}, "invalid_geojson", id="any_point-bad_geojson-1"
        ),
        pytest.param(
            {"any_point": {"type": "Point", "coordinats": [1, 2]}},
            "invalid_geojson",
            id="any_point-bad_geojson-2",
        ),
        pytest.param(
            {"any_point": {"type": "Point", "coordinates": [1, 2, 3, 4]}},
            "invalid_geojson",
            id="any_point-bad_geojson-3",
        ),
        pytest.param(
            {"point3d": shapely.Point(1, 2)}, "dimensionality", id="point3d_2d"
        ),
        pytest.param(
            {"point2d_01": shapely.Point(0.5, 0.5, 0.5)},
            "dimensionality",
            id="point2d_3d",
        ),
        pytest.param(
            {"point2d_01": shapely.Point(-0.5, 0.5)},
            "out_of_bounds",
            id="point2d_01_x_out",
        ),
        pytest.param(
            {"point2d_01": shapely.Point(0.5, 2)},
            "out_of_bounds",
            id="point2d_01_y_out",
        ),
    ],
)
def test_invalid_fields(data: dict[str, ty.Any], match: str) -> None:
    """Test invalid values for shapely fields"""
    with pytest.raises(pydantic.ValidationError, match=match):
        GeometryModel(**data)


@pytest.mark.parametrize(
    ("annotation", "match"),
    [
        pytest.param(
            ty.Annotated[5, ShapelyGeometryAdapter()],
            "ShapelyGeometryAdapter can only be used on a shapely "
            "geometry type or a union of shapely geometry types, not 5",
            id="5",
        ),
        pytest.param(
            ty.Annotated[int, ShapelyGeometryAdapter()],
            "ShapelyGeometryAdapter can only be used on a shapely "
            "geometry type or a union of shapely geometry types.*int",
            id="int",
        ),
    ],
)
def test_bad_models(annotation: ty.Any, match: str) -> None:
    """Test models with bad fields"""
    with pytest.raises(pydantic.PydanticSchemaGenerationError, match=match):

        class Model(pydantic.BaseModel):
            field: annotation


def test_round_trip_serialization() -> None:
    """Test round-trip serialization/deserialization"""

    class Model(pydantic.BaseModel):
        location: AnyShapelyGeometry

    original = Model(location=shapely.Point(1.5, 2.5))
    json_str = original.model_dump_json()
    restored = Model.model_validate_json(json_str)
    assert original.location.x == restored.location.x
    assert original.location.y == restored.location.y


def test_wkt_list_deserialization() -> None:
    """Test deserializing list of WKT strings"""

    class Model(pydantic.BaseModel):
        waypoints: list[ty.Annotated[shapely.Point, ShapelyGeometryAdapter()]]

    model = Model(waypoints=["POINT (0 1)", "POINT (2 3)", "POINT (4 5)"])
    assert len(model.waypoints) == 3
    assert all(isinstance(p, shapely.Point) for p in model.waypoints)
    npt.assert_array_equal(
        [shapely.get_coordinates(x).squeeze() for x in model.waypoints],
        [[0, 1], [2, 3], [4, 5]],
    )


def test_constaints_bad_type() -> None:
    """Test ShapelyGeometryConstraints raises on an invalid type"""
    with pytest.raises(ValueError, match="was not a shapely geometry"):
        ShapelyGeometryConstraints()(5)


def test_json_schema() -> None:
    """Test the JSON schema"""
    # This test should be improved, for now just make sure we don't raise
    GeometryModel.model_json_schema()
