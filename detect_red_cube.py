"""Detect a red cube in a MuJoCo eye-in-hand camera and localise it in 3D.

Two steps:
  1. segment the red pixels -> centroid (u, v) + bounding-box size;
  2. back-project that ray into the world using a pinhole camera model, and
     estimate the cube-centre depth from the apparent size of the known cube.

The returned point is in *world* coordinates, ready to feed the arm controller.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import mujoco


#: Default edge length of the red box (m); the grasp scene re-exports this so the
#: visual detector and the simulated cube can never disagree about its size.
DEFAULT_CUBE_SIDE = 0.05

#: A pixel counts as "red" when it is bright and clearly dominated by red.
RED_MIN = 90.0
RED_DOMINANCE = 1.4


def red_mask(img: np.ndarray) -> np.ndarray:
    """Boolean mask of the red pixels in an RGB(A) image."""
    r = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    b = img[..., 2].astype(np.float32)
    return (r > RED_MIN) & (r > RED_DOMINANCE * (g + 1.0)) & (
        r > RED_DOMINANCE * (b + 1.0)
    )


def render_rgbd(renderer, data, cam_name: str = "cam_hand"):
    """Render ``(colour, depth)`` with one persistent renderer.

    Depth rendering is a sticky setting on a ``mujoco.Renderer``, so the colour
    pass must always come first; centralising the sequence removes an easy way
    to corrupt the colour image.
    """
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    depth = renderer.render()
    return rgb, depth


def intrinsics(fovy_deg: float, h: int, w: int):
    """Return (fx, fy, cx, cy) for a square-pixel pinhole camera."""
    fy = (h / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    fx = fy
    cx = w / 2.0
    cy = h / 2.0
    return fx, fy, cx, cy


#: Backwards-compatible private alias.
_intrinsics = intrinsics


def _ray_world(model, data, cam_id, u, v, fx, fy, cx, cy):
    """Unit world-direction of the ray through pixel (u, v) for the camera.

    Note: MuJoCo image row ``v`` increases downward, but its camera local ``y``
    axis points upward, so the y-component uses ``(cy - v)``.
    """
    xr = (u - cx) / fx
    yr = (cy - v) / fy
    R = data.cam_xmat[cam_id].reshape(3, 3)
    # x -> image right, y -> image up, forward = local -z.
    d = R[:, 0] * xr + R[:, 1] * yr - R[:, 2]
    return d / np.linalg.norm(d)


def project_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    point,
    cam_name: str = "cam_hand",
    width: int = 320,
    height: int = 240,
) -> Optional[tuple[float, float]]:
    """Project a world point into ``(u, v)`` pixels of the named camera.

    This is the exact inverse of :func:`_ray_world`: image rows grow downward
    while the camera local ``y`` axis points up, so the vertical pixel is
    ``cy - fy * y``.  Returns ``None`` for a point behind the camera.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"camera {cam_name!r} not found")

    rotation = data.cam_xmat[cam_id].reshape(3, 3)
    local = rotation.T @ (np.asarray(point, dtype=float) - data.cam_xpos[cam_id])
    depth = -local[2]  # forward is the camera's local -z
    if depth <= 1e-9:
        return None

    fx, fy, cx, cy = intrinsics(float(model.cam_fovy[cam_id]), height, width)
    return fx * local[0] / depth + cx, cy - fy * local[1] / depth


def segment_red(img: np.ndarray) -> Optional[tuple[float, float, float, float, float]]:
    """Return (u, v, min_dim_px, w_px, h_px) of the red blob, or None."""
    mask = red_mask(img)
    try:
        from scipy import ndimage

        lab, nlab = ndimage.label(mask)
        if nlab == 0:
            return None
        sizes = ndimage.sum(mask, lab, range(1, nlab + 1))
        best = int(np.argmax(sizes)) + 1
        ys, xs = np.nonzero(lab == best)
    except Exception:
        ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    u = float(xs.mean())
    v = float(ys.mean())
    w_px = float(xs.max() - xs.min() + 1)
    h_px = float(ys.max() - ys.min() + 1)
    min_dim = min(w_px, h_px)
    return u, v, min_dim, w_px, h_px


def estimate_cube_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_name: str = "cam_hand",
    img: Optional[np.ndarray] = None,
    cube_side: float = DEFAULT_CUBE_SIDE,
    width: int = 640,
    height: int = 480,
) -> Optional[np.ndarray]:
    """Estimate the red cube's world position from the named camera image.

    Returns a 3-vector (world XYZ) or ``None`` if no red blob is visible.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"camera {cam_name!r} not found")

    if img is None:
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera=cam_name)
        img = renderer.render()
        renderer.close()
    h, w = img.shape[:2]

    blob = segment_red(img)
    if blob is None:
        return None
    u, v, min_dim, _w_px, _h_px = blob
    if min_dim <= 0:
        return None

    fovy = float(model.cam_fovy[cam_id])
    fx, fy, cx, cy = _intrinsics(fovy, h, w)

    # Depth of the *near* cube face from its apparent size, then add half the
    # cube side to get the centre distance along the optical axis.
    near_depth = fy * cube_side / min_dim
    depth = near_depth + cube_side / 2.0

    cam_pos = data.cam_xpos[cam_id].copy()
    dir_world = _ray_world(model, data, cam_id, u, v, fx, fy, cx, cy)
    return cam_pos + dir_world * depth


def estimate_cube_world_planar(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_name: str = "cam_hand",
    plane_z: float = 0.45,
    cube_side: float = DEFAULT_CUBE_SIDE,
    img: Optional[np.ndarray] = None,
    width: int = 640,
    height: int = 480,
) -> Optional[np.ndarray]:
    """Localise the red cube via ray---known-plane intersection (monocular).

    Casts the ray through the detected blob centroid and intersects it with the
    horizontal plane ``z = plane_z`` (the table top).  The cube centre is that
    intersection lifted by half a cube side.  This is robust to the cube's
    orientation, which size-based depth is not.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"camera {cam_name!r} not found")
    if img is None:
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera=cam_name)
        img = renderer.render()
        renderer.close()
    h, w = img.shape[:2]

    blob = segment_red(img)
    if blob is None:
        return None
    u, v, _min_dim, _w_px, _h_px = blob

    fovy = float(model.cam_fovy[cam_id])
    fx, fy, cx, cy = _intrinsics(fovy, h, w)
    cam_pos = data.cam_xpos[cam_id].copy()
    dir_world = _ray_world(model, data, cam_id, u, v, fx, fy, cx, cy)

    if abs(dir_world[2]) < 1e-6:
        return None
    t = (plane_z - cam_pos[2]) / dir_world[2]
    if t <= 0:
        return None
    hit = cam_pos + dir_world * t
    return hit + np.array([0.0, 0.0, cube_side / 2.0])


def estimate_cube_world_rgbd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_name: str = "cam_hand",
    cube_side: float = DEFAULT_CUBE_SIDE,
    width: int = 640,
    height: int = 480,
    plane_z: Optional[float] = None,
    img: Optional[np.ndarray] = None,
    depth_img: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Localise the red cube with RGB-D: per-pixel depth back-projection.

    Renders the colour image for red segmentation and a depth image, then
    back-projects every red pixel to a 3D point using its own depth.  The cube
    centre is the *median* of those points (robust to silhouette-edge noise).
    If ``plane_z`` is given, the returned point is snapped to that plane with the
    cube lifted half a side, which removes the monocular scale error along the
    view axis.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"camera {cam_name!r} not found")

    if img is not None and depth_img is not None:
        rgb = img
        depth = depth_img
    else:
        # Fresh renderer per call: enabling depth rendering is a persistent state
        # on the Renderer object, so reusing one would corrupt subsequent reads.
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera=cam_name)
        rgb = renderer.render()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam_name)
        depth = renderer.render()
        renderer.close()

    h, w = rgb.shape[:2]
    blob = segment_red(rgb)
    if blob is None:
        return None
    u, v, _min_dim, _w_px, _h_px = blob

    fovy = float(model.cam_fovy[cam_id])
    fx, fy, cx, cy = _intrinsics(fovy, h, w)

    ys, xs = np.nonzero(red_mask(rgb))
    if xs.size == 0:
        return None
    zdepth = depth[ys, xs].astype(np.float64)

    xr = (xs - cx) / fx
    yr = (cy - ys) / fy
    R = data.cam_xmat[cam_id].reshape(3, 3)
    cam_pos = data.cam_xpos[cam_id].copy()
    world = (
        cam_pos
        + np.outer(xr * zdepth, R[:, 0])
        + np.outer(yr * zdepth, R[:, 1])
        - np.outer(zdepth, R[:, 2])
    )

    # Depth band of the cube, with robust outlier rejection.  A single stray
    # "red" pixel (an anti-aliased edge against the dark fingers reads as
    # (134, 11, 11)) sits right next to the camera, and taking ``min`` there
    # threw away the whole cube and returned a point 28 cm too close.  Reject by
    # median absolute deviation instead: the cube owns most of the pixels, so
    # the median is its depth no matter what a few edges do.
    median = np.median(zdepth)
    spread = np.median(np.abs(zdepth - median))
    keep = np.abs(zdepth - median) <= max(3.0 * spread, 0.05)
    zband = zdepth[keep]
    if zband.size == 0:
        return None
    if plane_z is None:
        return np.median(world[keep], axis=0)
    # Use the silhouette-centroid ray at the cube's *middle* depth for x/y, and
    # snap z to the known plane.  This removes the visible-surface-centroid bias
    # that the simple 3D average suffers from.
    centre_depth = (zband.min() + zband.max()) / 2.0
    dir_world = _ray_world(model, data, cam_id, u, v, fx, fy, cx, cy)
    # MuJoCo's depth buffer is measured along the optical axis, while
    # ``dir_world`` is a unit ray: scale by the ray's inclination or the target
    # is pulled towards the camera by (1 - cos) -- 10 cm at the edge of frame,
    # which is exactly where the cube sits when the hand looks at the bench.
    centre_depth /= float(-(dir_world @ R[:, 2]))
    p = data.cam_xpos[cam_id].copy() + dir_world * centre_depth
    p[2] = plane_z + cube_side / 2.0
    return p
