"""Generic vision for the OpenArm bridge.

Locate graspable objects from the overhead camera WITHOUT any privileged
ground-truth scene state and WITHOUT hardcoding any object's colour or identity.

Pipeline:
  1. ``foreground_mask`` — segment pixels more chromatic than the (≈gray) table
     surface. This keys on "differs from the background", not a specific colour,
     so it generalizes to arbitrary coloured objects.
  2. ``connected_blobs`` — group foreground pixels into connected components and
     return each blob's centroid pixel.
  3. ``pixel_to_table_point`` — back-project a pixel through the pinhole camera
     onto the table plane to recover a world position.

Everything here is pure NumPy (no Isaac / torch) so the geometry is unit-testable
without a GPU; :mod:`ssr_robotics.isaac_env` feeds in the live camera RGB,
intrinsics and pose. The chroma threshold and table height are heuristics expected
to need tuning against real frames — detections are logged on the bridge console.
"""

from __future__ import annotations

import numpy as np


def foreground_mask(rgb, chroma_thresh: float = 45.0):
    """Boolean ``HxW`` mask of pixels more colourful than the background.

    ``chroma`` = max channel − min channel: ≈0 for gray/white/black (table, robot),
    large for saturated colours (objects). No specific hue is assumed.
    """
    rgb = np.asarray(rgb, dtype=float)[..., :3]
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    return chroma >= chroma_thresh


def connected_blobs(mask, min_pixels: int = 15):
    """4-connected components of ``mask``, as ``(u, v, count)`` centroids.

    ``u`` = column (x), ``v`` = row (y). Blobs smaller than ``min_pixels`` are
    dropped as noise. Returned largest-first.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    blobs = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        sx = sy = cnt = 0
        while stack:
            y, x = stack.pop()
            sx += x
            sy += y
            cnt += 1
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if cnt >= min_pixels:
            blobs.append((sx / cnt, sy / cnt, cnt))
    blobs.sort(key=lambda b: -b[2])
    return blobs


def pixel_to_point_with_depth(u, v, depth, K, cam_pos, cam_rot):
    """Unproject pixel ``(u, v)`` at planar ``depth`` to a world point.

    ``depth`` is the distance to the image plane (optical +Z), i.e. Isaac Lab's
    ``distance_to_image_plane`` output. Returns world ``[x, y, z]`` (NumPy array),
    or ``None`` if the depth is missing/non-finite/non-positive (e.g. background).

    Args mirror :func:`pixel_to_table_point` (``cam_rot`` is world ``<-`` camera
    optical, the matrix of the camera's ``quat_w_ros``). Unlike the table-plane
    method this needs no surface-height assumption, so objects of differing height
    localize correctly.
    """
    if depth is None:
        return None
    d = float(depth)
    if not np.isfinite(d) or d <= 0.0:
        return None
    K = np.asarray(K, dtype=float)
    cam_pos = np.asarray(cam_pos, dtype=float)
    cam_rot = np.asarray(cam_rot, dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p_cam = np.array([(u - cx) / fx * d, (v - cy) / fy * d, d])  # optical frame
    return cam_pos + cam_rot @ p_cam


def look_at_quat_ros(eye, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (wxyz) of a camera at ``eye`` looking at ``target``, ROS/optical
    convention (+X right, +Y down, +Z forward = view direction).

    The inverse companion of ``cam_rot`` in :func:`pixel_to_table_point`: the
    returned quaternion is what Isaac Lab's ``Camera.set_world_poses(...,
    convention="ros")`` expects, and its rotation matrix maps camera-optical to
    world. ``up`` is the world up used to level the image (zero roll). When the
    view direction is parallel to ``up`` — e.g. a perfectly straight-down camera,
    a legitimate config — the roll reference falls back to world +Y, reproducing
    the canonical overhead convention (x→x, y→-y, z→-z) instead of failing.
    Raises only if ``eye`` and ``target`` coincide (no view direction at all).
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    fwd = target - eye
    n = np.linalg.norm(fwd)
    if n < 1e-9:
        raise ValueError("look_at: eye and target coincide")
    z = fwd / n                       # optical +Z: view direction
    x = np.cross(z, up)               # optical +X: right = forward x up
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        # Vertical view: level the image against world +Y instead (and +X as the
        # last resort if the caller's up itself was +Y).
        for alt in ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)):
            x = np.cross(z, np.asarray(alt))
            nx = np.linalg.norm(x)
            if nx >= 1e-9:
                break
    x = x / nx
    y = np.cross(z, x)                # optical +Y: down (right-handed: z x x = y)
    r = np.stack([x, y, z], axis=1)   # world <- optical, columns are the axes
    # Rotation matrix -> quaternion (wxyz), standard Shepperd branch selection.
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w, qx, qy, qz = (0.25 * s, (r[2, 1] - r[1, 2]) / s,
                         (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w, qx, qy, qz = ((r[2, 1] - r[1, 2]) / s, 0.25 * s,
                         (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w, qx, qy, qz = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                         0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w, qx, qy, qz = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                         (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    q = np.array([w, qx, qy, qz])
    return q / np.linalg.norm(q)


def quat_wxyz_to_matrix(q):
    """Rotation matrix (world <- optical) of a wxyz quaternion. Pure NumPy."""
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pixel_to_table_point(u, v, K, cam_pos, cam_rot, table_z):
    """Back-project pixel ``(u, v)`` onto the world plane ``z = table_z``.

    Args:
        u, v: pixel column/row.
        K: ``3x3`` pinhole intrinsics (ROS/optical convention).
        cam_pos: camera position in world, length-3.
        cam_rot: ``3x3`` rotation world ``<-`` camera optical frame (the matrix of
            the camera's ``quat_w_ros``).
        table_z: height of the plane to intersect, in world.

    Returns world ``[x, y, z]`` (NumPy array), or ``None`` if the ray is parallel
    to / points away from the plane.
    """
    K = np.asarray(K, dtype=float)
    cam_pos = np.asarray(cam_pos, dtype=float)
    cam_rot = np.asarray(cam_rot, dtype=float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])  # optical: x right, y down, z fwd
    dir_world = cam_rot @ dir_cam
    if abs(dir_world[2]) < 1e-9:
        return None
    s = (table_z - cam_pos[2]) / dir_world[2]
    if s <= 0:
        return None
    return cam_pos + s * dir_world
