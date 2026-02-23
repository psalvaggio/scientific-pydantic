"""Unit test for rotation.py"""

from __future__ import annotations

import typing as ty
from unittest import mock

import numpy as np
import numpy.testing as npt
import pydantic
import pytest
import scipy
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from scientific_pydantic.scipy.spatial.transform import RotationAdapter
from scientific_pydantic.version_check import version_lt

IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]
IDENTITY_ROT = Rotation.from_quat(IDENTITY_QUAT)
SCIPY_LT_1_17 = version_lt(scipy, (1, 17, 0))


def assert_rotations_close(a: Rotation, b: Rotation) -> None:
    """Return True when two rotations are approximately equal."""
    # Compare via quaternions (both sign conventions map to same rotation)
    qa, qb = a.as_quat(canonical=True), b.as_quat(canonical=True)
    npt.assert_allclose(qa, qb, atol=1e-9)


class Basic(BaseModel):
    """Basic unconstrained rotation"""

    rotation: ty.Annotated[Rotation, RotationAdapter()]


@pytest.mark.parametrize(
    "rot",
    [
        pytest.param(IDENTITY_ROT, id="identity"),
        pytest.param(Rotation.from_euler("z", 90, degrees=True), id="non-identity"),
    ],
)
def test_rotation_passthrough(rot: Rotation) -> None:
    """Tests that a Rotation instance is passed through"""
    pose = Basic(rotation=rot)
    assert isinstance(pose.rotation, Rotation)
    assert_rotations_close(pose.rotation, rot)


@pytest.mark.skipif(SCIPY_LT_1_17, reason="N-D Rotations are scipy 1.17.0 and up")
def test_stacked_rotations_passthrough() -> None:
    """Tests passthrough on a multiple rotation"""
    rot = Rotation.from_euler("z", [[0], [90], [180]], degrees=True)
    pose = Basic(rotation=rot)
    assert isinstance(pose.rotation, Rotation)
    assert len(pose.rotation) == 3
    assert_rotations_close(pose.rotation[0, 0], rot[0, 0])
    assert_rotations_close(pose.rotation[1, 0], rot[1, 0])
    assert_rotations_close(pose.rotation[2, 0], rot[2, 0])


@pytest.mark.parametrize(
    ("payload", "expected_quat"),
    [
        pytest.param(
            {"quat": [0, 0, 0, 1]},
            [0.0, 0.0, 0.0, 1.0],
            id="quat_identity_list",
        ),
        pytest.param(
            {"quat": np.asarray([0, 0, 0, 1], dtype=float)},
            [0.0, 0.0, 0.0, 1.0],
            id="quat_identity_ndarray",
        ),
        pytest.param(
            {"quat": [1, 0, 0, 0], "scalar_first": True},
            [0.0, 0.0, 0.0, 1.0],
            id="quat_scalar_first",
        ),
        pytest.param(
            {"matrix": np.eye(3).tolist()},
            [0.0, 0.0, 0.0, 1.0],
            id="matrix_identity",
        ),
        pytest.param(
            {"matrix": np.eye(3).tolist(), "assume_valid": True},
            [0.0, 0.0, 0.0, 1.0],
            id="matrix_identity_assume_valid",
            marks=[
                pytest.mark.skipif(
                    SCIPY_LT_1_17, reason="assume_valid requires scipy >= 1.17.0"
                )
            ],
        ),
        pytest.param(
            {"rotvec": [0, 0, 0]},
            [0.0, 0.0, 0.0, 1.0],
            id="rotvec_zero_identity",
        ),
        pytest.param(
            {"rotvec": [0, 0, 180], "degrees": True},
            [0, 0, 1, 0],
            id="rotvec_180deg_around_z",
        ),
        pytest.param(
            {"mrp": [0, 0, 0]},
            [0.0, 0.0, 0.0, 1.0],
            id="mrp_zero_identity",
        ),
        pytest.param(
            {"euler": {"seq": "z", "angles": 0.0, "degrees": True}},
            [0.0, 0.0, 0.0, 1.0],
            id="euler_zero_z_degrees",
        ),
        pytest.param(
            {"euler": {"seq": "xyz", "angles": [0.0, 0.0, 0.0], "degrees": True}},
            [0.0, 0.0, 0.0, 1.0],
            id="euler_xyz_zeros_degrees",
        ),
        pytest.param(
            {"euler": {"seq": "z", "angles": 0.0}},
            [0.0, 0.0, 0.0, 1.0],
            id="euler_zero_z_radians_default",
        ),
        pytest.param(
            {"euler": {"seq": "z", "angles": 90, "degrees": True}},
            [0.0, 0.0, np.sqrt(2), np.sqrt(2)],
            id="euler_90deg_around_z",
        ),
        pytest.param(
            {
                "davenport": {
                    "axes": [[1, 0, 0]],
                    "order": "extrinsic",
                    "angles": 0.0,
                    "degrees": True,
                }
            },
            IDENTITY_QUAT,
            id="davenport_identity",
        ),
        pytest.param(
            {
                "davenport": {
                    "axes": [[1, 0, 0], [0, 1, 0]],
                    "order": "i",
                    "angles": [30, 60],
                    "degrees": True,
                }
            },
            [0.22414387, 0.48296291, 0.12940952, 0.8365163],
            id="davenport_i",
        ),
    ],
)
def test_valid_mapping_inputs(
    payload: dict,
    expected_quat: list[float],
) -> None:
    """Test valid mapping inputs for scalar rotations"""
    m = Basic(rotation=payload)
    assert isinstance(m.rotation, Rotation)
    assert_rotations_close(m.rotation, Rotation(expected_quat))


def test_model_dump_contains_quat_key() -> None:
    """Test model dumping behavior"""
    m = Basic(rotation=IDENTITY_ROT)
    assert m.model_dump()["rotation"] is IDENTITY_ROT

    d = m.model_dump(mode="json")
    assert "quat" in d["rotation"]
    npt.assert_allclose(d["rotation"]["quat"], [0, 0, 0, 1])


def test_model_dump_json_roundtrip() -> None:
    """Test round-tripping through JSON"""
    m = Basic(rotation={"quat": [0.123456, 0.234567, 0.345678, 0.456789]})
    json_str = m.model_dump_json()
    m2 = Basic.model_validate_json(json_str)
    assert_rotations_close(m.rotation, m2.rotation)


def test_serialised_quat_is_scalar_last() -> None:
    """Serialised quaternion must follow xyzw (scalar-last) convention."""
    rot = Rotation.from_euler("z", 90, degrees=True)
    m = Basic(rotation=rot)
    quat = m.model_dump(mode="json")["rotation"]["quat"]
    reconstructed = Rotation.from_quat(quat)  # scipy expects scalar-last
    assert_rotations_close(rot, reconstructed)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        pytest.param(42, "Cannot convert", id="invalid_type_int"),
        pytest.param("not_a_rotation", "Cannot convert", id="invalid_type_string"),
        pytest.param({}, "no valid keys found in mapping", id="empty_mapping_no_keys"),
        pytest.param(
            {"unknown_key": [1, 2, 3]},
            "no valid keys found in mapping",
            id="mapping_unknown_key",
        ),
        pytest.param(
            {"quat": [0, 0, 1]},
            "Array shape .* does not match spec",
            id="quat_wrong_length_3",
        ),
        pytest.param(
            {"quat": [0, 0, 0, 0, 1]},
            "Array shape .* does not match spec",
            id="quat_wrong_length_5",
        ),
        pytest.param(
            {"matrix": [[1, 0], [0, 1]]},
            "Array shape .* does not match spec",
            id="matrix_wrong_shape_2x2",
        ),
        pytest.param(
            {"rotvec": [0, 0]},
            "Array shape .* does not match spec",
            id="rotvec_wrong_length_2",
        ),
        pytest.param(
            {"mrp": [0, 0]},
            "Array shape .* does not match spec",
            id="mrp_wrong_length_2",
        ),
        pytest.param(
            {"euler": {"seq": "z", "angles": 0.0, "extra_field": True}},
            "Extra inputs are not permitted",
            id="euler_extra_forbidden_field",
        ),
        pytest.param(
            {"euler": {"seq": "zzz1", "angles": 0.0}},
            "String should match pattern",
            id="euler_invalid_seq_pattern",
        ),
        pytest.param(
            {"davenport": {"axes": [[1, 0, 0]], "order": "bad", "angles": 0.0}},
            "Input should be 'e', 'extrinsic', 'i' or 'intrinsic'",
            id="davenport_invalid_order_literal",
        ),
        pytest.param(
            {"quat": [0, 0, 0, 1], "matrix": np.eye(3).tolist()},
            "Extra inputs are not permitted",
            id="mapping_two_keys_conflict",
        ),
    ],
)
def test_invalid_inputs_unconstrained(payload: ty.Any, match: str) -> None:
    """Tests invalid inputs"""
    with pytest.raises(pydantic.ValidationError, match=match):
        Basic(rotation=payload)


def test_single_rotation_constraint() -> None:
    """Tests behavior of the single constraint"""

    class Single(pydantic.BaseModel):
        rotation: ty.Annotated[Rotation, RotationAdapter(single=True)]

    m = Single(rotation={"quat": IDENTITY_QUAT})
    assert isinstance(m.rotation, Rotation)

    with pytest.raises(pydantic.ValidationError, match="invalid_rotation_shape"):
        Single(rotation={"quat": [[0, 0, 0, 1], [0, 0, 0, 1]]})


def test_ndim1_stacked_accepted() -> None:
    """Test behavior of ndim constraint"""

    class Ndim1(pydantic.BaseModel):
        rotation: ty.Annotated[Rotation, RotationAdapter(ndim=1)]

    pose = Ndim1(rotation={"quat": [[0, 0, 0, 1], [0, 0, 0, 1]]})
    assert np.asarray(pose.rotation).shape == (2,)

    with pytest.raises(pydantic.ValidationError, match="invalid_rotation_shape"):
        Ndim1(rotation={"quat": IDENTITY_QUAT})


@pytest.mark.skipif(SCIPY_LT_1_17, reason="N-D Rotations are scipy 1.17.0 and up")
@pytest.mark.parametrize(
    "quats",
    [
        pytest.param([[0, 0, 0, 1], [0, 0, 0, 1]], id="batch_of_2_identity"),
        pytest.param(
            [[[0, 0, 0, 1]], [[0, 0, 1, 0]], [[0, 1, 0, 0]]],
            id="batch_of_3_different",
        ),
    ],
)
def test_batch_quat_accepted(quats: list[list[float]]) -> None:
    """Test that we accept batch rotation unconstrained"""
    m = Basic(rotation={"quat": quats})
    assert isinstance(m.rotation, Rotation)
    assert np.asarray(m.rotation).shape == np.asarray(quats).shape[:-1]


@pytest.mark.skipif(SCIPY_LT_1_17, reason="N-D Rotations are scipy 1.17.0 and up")
@pytest.mark.parametrize(
    ("shape_constraint", "data", "truth_shape", "error"),
    [
        pytest.param(
            (1, 2), [[IDENTITY_QUAT, IDENTITY_QUAT]], (1, 2), None, id="pass-1x2"
        ),
        pytest.param(
            (..., range(2, 4)),
            [[[[IDENTITY_QUAT, IDENTITY_QUAT]]]],
            (1, 1, 1, 2),
            None,
            id="pass-1x2",
        ),
    ],
)
def test_shape_constraint(
    shape_constraint: tuple[ty.Any, ...],
    data: ty.Any,
    truth_shape: tuple[int, ...],
    error: str | None,
) -> None:
    """Test the shape constraint"""

    class Model(pydantic.BaseModel):
        rotation: ty.Annotated[Rotation, RotationAdapter(shape=shape_constraint)]

    if error is None:
        m = Model(rotation=data)
        assert isinstance(m.rotation, Rotation)
        assert np.asarray(m.rotation).shape == truth_shape
    else:
        with pytest.raises(pydantic.ValidationError, match=error):
            Model(rotation=data)


@pytest.mark.skipif(SCIPY_LT_1_17, reason="N-D Rotations are scipy 1.17.0 and up")
@pytest.mark.parametrize(
    ("data", "truth_shape"),
    [
        pytest.param({"rotvec": [[0, 0, 0], [0, 0, np.pi]]}, (2,), id="rotvec-2"),
        pytest.param(
            {
                "euler": {
                    "seq": "z",
                    "angles": [[[0.0]], [[90.0]], [[180.0]]],
                    "degrees": True,
                }
            },
            (3, 1),
            id="euler-3x1",
        ),
    ],
)
def test_batch_rotvec_accepted(
    data: dict[str, ty.Any], truth_shape: tuple[int, ...]
) -> None:
    """Test that we accept mapping rotation unconstrained"""
    m = Basic(rotation=data)
    assert isinstance(m.rotation, Rotation)
    assert np.asarray(m.rotation).shape == truth_shape


def test_wrong_source_type_raises() -> None:
    """Test that a bad input type results in an error"""
    with pytest.raises(pydantic.PydanticSchemaGenerationError):

        class BadModel(BaseModel):
            rotation: ty.Annotated[int, RotationAdapter()]


def test_scipy_without_nd_support() -> None:
    """Test that scipy without N-D shape support only allows scalar Rotation"""
    with mock.patch(
        "scientific_pydantic.scipy.spatial.transform.rotation._supports_shape",
        return_value=False,
    ):

        class Valid(pydantic.BaseModel):
            f0: ty.Annotated[Rotation, RotationAdapter()]
            f1: ty.Annotated[Rotation, RotationAdapter(single=True)]
            f2: ty.Annotated[Rotation, RotationAdapter(ndim=0)]
            f3: ty.Annotated[Rotation, RotationAdapter(shape=())]
            f4: ty.Annotated[Rotation, RotationAdapter(single=True, ndim=0, shape=())]

        with pytest.raises(pydantic.PydanticSchemaGenerationError):
            RotationAdapter(ndim=2)


def test_scipy_without_assume_valid_support() -> None:
    """Test that we can only pass assume_valid=False if scipy doesn't support it"""
    with mock.patch(
        "scientific_pydantic.scipy.spatial.transform.rotation._matrix_supports_assume_valid",
        return_value=False,
    ):
        Basic(
            rotation={
                "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "assume_valid": False,
            }
        )
        with pytest.raises(
            pydantic.ValidationError,
            match=r"assume_valid=True is only supported in scipy >= 1\.17\.0",
        ):
            Basic(
                rotation={
                    "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "assume_valid": True,
                }
            )
