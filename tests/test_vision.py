"""Unit tests for the generic vision geometry (no Isaac / GPU needed).

Pin down foreground segmentation, blob grouping and the pixel→table back-projection
so the perception math is trustworthy even though the full path only runs on the
robot machine with the camera.
"""

from __future__ import annotations

import math

import numpy as np

from ssr_robotics import vision as V


def test_foreground_mask_keys_on_chroma_not_a_specific_colour():
    img = np.full((10, 10, 3), 120, dtype=np.float32)  # uniform gray table
    img[2:4, 2:4] = (220, 20, 20)   # a red object
    img[6:8, 6:8] = (20, 60, 220)   # a blue object — different hue, still detected
    mask = V.foreground_mask(img, chroma_thresh=45.0)
    assert mask[2:4, 2:4].all() and mask[6:8, 6:8].all()
    assert not mask[0, 0]  # gray background is not foreground
    # White/black (robot) are achromatic → never foreground regardless of brightness.
    white = np.full((4, 4, 3), 250, dtype=np.float32)
    assert not V.foreground_mask(white).any()


def test_connected_blobs_separates_two_objects():
    img = np.zeros((20, 40, 3), dtype=np.float32)
    img[5:9, 5:9] = (220, 20, 20)      # blob A (16 px)
    img[10:14, 30:36] = (20, 200, 30)  # blob B (24 px), larger
    blobs = V.connected_blobs(V.foreground_mask(img), min_pixels=5)
    assert len(blobs) == 2
    # Largest first.
    assert blobs[0][2] == 24 and blobs[1][2] == 16
    ub, vb, _ = blobs[0]
    assert math.isclose(ub, 32.5, abs_tol=0.01) and math.isclose(vb, 11.5, abs_tol=0.01)
    ua, va, _ = blobs[1]
    assert math.isclose(ua, 6.5, abs_tol=0.01) and math.isclose(va, 6.5, abs_tol=0.01)


def test_connected_blobs_drops_sub_threshold_noise():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[0, 0] = (220, 20, 20)  # single stray pixel
    assert V.connected_blobs(V.foreground_mask(img), min_pixels=5) == []


def test_pixel_to_table_point_straight_down_principal_ray():
    K = np.array([[200.0, 0, 160.0], [0, 200.0, 120.0], [0, 0, 1.0]])
    cam_pos = np.array([0.4, 0.1, 1.0])
    cam_rot = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # world<-optical, looking down
    p = V.pixel_to_table_point(160.0, 120.0, K, cam_pos, cam_rot, table_z=0.05)
    assert p is not None and np.allclose(p, [0.4, 0.1, 0.05], atol=1e-6)


def test_pixel_to_table_point_offset_pixel_has_correct_sign():
    K = np.array([[200.0, 0, 160.0], [0, 200.0, 120.0], [0, 0, 1.0]])
    cam_pos = np.array([0.0, 0.0, 1.0])
    cam_rot = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    p = V.pixel_to_table_point(360.0, 120.0, K, cam_pos, cam_rot, table_z=0.0)
    assert p is not None
    assert math.isclose(p[0], 1.0, abs_tol=1e-6) and math.isclose(p[1], 0.0, abs_tol=1e-6)


def test_pixel_to_point_with_depth_matches_table_plane_when_consistent():
    # Camera 1 m above the plane looking straight down; a pixel whose depth is the
    # full 1 m must land on the table plane at the same point the table-plane method
    # gives — depth and plane agree when the object sits on the table.
    K = np.array([[200.0, 0, 160.0], [0, 200.0, 120.0], [0, 0, 1.0]])
    cam_pos = np.array([0.0, 0.0, 1.0])
    cam_rot = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    u, v = 360.0, 120.0
    plane = V.pixel_to_table_point(u, v, K, cam_pos, cam_rot, table_z=0.0)
    # Depth to image plane for this pixel: the optical +Z component to reach z=0.
    # ray optical dir z-component maps to world -z; at world z=0 the planar depth is
    # cam height / |dir_world_z| with dir_cam z = 1 → here 1.0 m straight component.
    depth = 1.0
    pt = V.pixel_to_point_with_depth(u, v, depth, K, cam_pos, cam_rot)
    assert pt is not None
    # Both should agree on x (1.0) and z (0.0); plane method pins z exactly.
    assert math.isclose(pt[0], plane[0], abs_tol=1e-6)
    assert math.isclose(pt[2], 0.0, abs_tol=1e-6)


def test_pixel_to_point_with_depth_rejects_invalid_depth():
    K = np.array([[200.0, 0, 160.0], [0, 200.0, 120.0], [0, 0, 1.0]])
    cam_pos = np.array([0.0, 0.0, 1.0])
    cam_rot = np.eye(3)
    for bad in (None, 0.0, -0.5, float("inf"), float("nan")):
        assert V.pixel_to_point_with_depth(160.0, 120.0, bad, K, cam_pos, cam_rot) is None


def test_pixel_to_table_point_ray_parallel_returns_none():
    K = np.array([[200.0, 0, 160.0], [0, 200.0, 120.0], [0, 0, 1.0]])
    cam_pos = np.array([0.0, 0.0, 1.0])
    cam_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # principal ray horizontal
    assert V.pixel_to_table_point(160.0, 120.0, K, cam_pos, cam_rot, table_z=0.0) is None


def test_look_at_quat_ros_straight_down_matches_overhead_convention():
    # A camera looking straight down should reproduce the canonical overhead
    # rotation used throughout these tests: x->x, y->-y, z->-z.
    q = V.look_at_quat_ros(eye=[0.4, 0.1, 1.0], target=[0.4, 0.1, 0.0], up=(0, 1, 0))
    R = V.quat_wxyz_to_matrix(q)
    assert np.allclose(R @ [0, 0, 1], [0, 0, -1], atol=1e-9)  # forward = world -z


def test_look_at_quat_ros_points_optical_z_at_target():
    eye = np.array([1.0, 0.8, 0.9])
    target = np.array([0.4, 0.0, 0.05])
    q = V.look_at_quat_ros(eye, target)
    R = V.quat_wxyz_to_matrix(q)
    fwd = R @ [0, 0, 1]  # optical +Z in world
    expect = (target - eye) / np.linalg.norm(target - eye)
    assert np.allclose(fwd, expect, atol=1e-9)
    # Zero roll: optical +X (image right) is horizontal (no world-z component).
    right = R @ [1, 0, 0]
    assert math.isclose(right[2], 0.0, abs_tol=1e-9)
    # Optical +Y (image down) has a downward world component (elevated camera).
    down = R @ [0, 1, 0]
    assert down[2] < 0


def test_look_at_round_trips_through_pixel_back_projection():
    # A pixel through the principal point of a look-at camera must back-project
    # exactly onto the look-at target — the two functions are inverses.
    eye, target = [1.0, 0.8, 0.9], [0.4, 0.0, 0.05]
    q = V.look_at_quat_ros(eye, target)
    R = V.quat_wxyz_to_matrix(q)
    K = np.array([[200.0, 0, 160.0], [0, 200.0, 120.0], [0, 0, 1.0]])
    p = V.pixel_to_table_point(160.0, 120.0, K, eye, R, table_z=0.05)
    assert p is not None and np.allclose(p, target, atol=1e-9)


def test_look_at_quat_ros_rejects_coincident_eye_and_target():
    import pytest

    with pytest.raises(ValueError):
        V.look_at_quat_ros([0, 0, 1], [0, 0, 1])          # eye == target


def test_look_at_quat_ros_straight_down_with_default_up_falls_back():
    # View parallel to the default up (a straight-down camera — a legitimate
    # SSR_ARM_CAM_POS config) must not raise: the roll reference falls back to
    # world +Y, giving the canonical overhead convention (x->x, y->-y, z->-z),
    # so _position_camera applies the pose instead of silently keeping the old one.
    q = V.look_at_quat_ros([0.4, 0.1, 1.0], [0.4, 0.1, 0.0])  # default up=(0,0,1)
    R = V.quat_wxyz_to_matrix(q)
    assert np.allclose(R, [[1, 0, 0], [0, -1, 0], [0, 0, -1]], atol=1e-9)
    # And straight UP (parallel the other way) also resolves without raising.
    q_up = V.look_at_quat_ros([0, 0, 0], [0, 0, 1])
    Ru = V.quat_wxyz_to_matrix(q_up)
    assert np.allclose(Ru @ [0, 0, 1], [0, 0, 1], atol=1e-9)
