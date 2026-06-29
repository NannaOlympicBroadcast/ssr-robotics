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
