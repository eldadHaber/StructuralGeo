"""Extract fault polylines from a GeoModel + build the fault distance volume.

GeoGen's :class:`geogen.model.geoprocess.Fault` carries the fault plane
parameters (``strike``, ``dip``, ``rake``, ``amplitude``, ``origin``)
directly. We walk ``GeoModel.history_unpacked`` (or recursively
``model.history``) to enumerate every Fault instance, then derive a
surface-trace polyline by intersecting the fault plane with z=0 (top of
the spec frame), clipped to the model's (x, y) bbox.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Avoid hard import of geogen.model at module-import time; lazy below.


def _walk_history(history) -> List:
    """Yield leaf processes from a possibly-nested history."""
    out = []
    for ev in history:
        if hasattr(ev, "history") and isinstance(ev.history, list) and ev.history:
            out.extend(_walk_history(ev.history))
        else:
            out.append(ev)
    return out


def _resolve_origin(origin_obj, model) -> Tuple[float, float, float]:
    """``Fault.origin`` may be a tuple or a BacktrackedPoint. Resolve to floats."""
    if hasattr(origin_obj, "point"):
        try:
            p = origin_obj.point
        except Exception:
            p = (0.0, 0.0, 0.0)
    else:
        p = origin_obj
    try:
        x, y, z = (float(p[0]), float(p[1]), float(p[2]))
    except Exception:
        x = y = z = 0.0
    return (x, y, z)


def _clip_line_to_bbox(
    p_xy: Tuple[float, float],
    direction: Tuple[float, float],
    bbox: Tuple[float, float, float, float],
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Clip an infinite 2D line through ``p`` along ``direction`` to a rect.

    Returns ((x0, y0), (x1, y1)) or None if the line misses the box.
    """
    x_min, y_min, x_max, y_max = bbox
    px, py = p_xy
    dx, dy = direction
    # Parametrise as p + t * d. Find t-bounds where line is in bbox.
    ts = []
    if dx != 0:
        ts.extend([(x_min - px) / dx, (x_max - px) / dx])
    if dy != 0:
        ts.extend([(y_min - py) / dy, (y_max - py) / dy])
    if len(ts) < 2:
        return None
    # For each candidate t, check that the corresponding point is in bbox.
    pts_in_bbox = []
    for t in ts:
        x = px + t * dx
        y = py + t * dy
        if x_min - 1e-6 <= x <= x_max + 1e-6 and y_min - 1e-6 <= y <= y_max + 1e-6:
            pts_in_bbox.append((float(x), float(y)))
    if len(pts_in_bbox) < 2:
        return None
    # Pick the two points farthest apart
    import itertools
    best = None
    best_d = -1.0
    for a, b in itertools.combinations(pts_in_bbox, 2):
        d = np.hypot(a[0] - b[0], a[1] - b[1])
        if d > best_d:
            best_d = d
            best = (a, b)
    if best is None or best_d < 1e-3:
        return None
    return best


def extract_faults(
    model,
    spec_bbox: Tuple[float, float, float, float],
) -> List[Dict]:
    """Return a list of fault dicts in spec's fault_traces.json schema.

    The polyline is the intersection of the fault plane with z=0 (top of
    the spec frame), clipped to ``spec_bbox = (x_min, y_min, x_max, y_max)``.

    For sub-vertical faults the trace barely depends on z, so this is a
    good representation. For shallow-dipping faults the trace at depth
    differs; downstream consumers can use ``dip`` + ``dip_direction`` to
    extrapolate as needed.
    """
    from geogen.model.geoprocess import Fault

    leaves = _walk_history(model.history_unpacked or model.history)
    out: List[Dict] = []
    for ev in leaves:
        if not isinstance(ev, Fault):
            continue
        # NOTE: Slip/Fault converts strike/dip/rake to *radians* in __init__,
        # so ev.strike / ev.dip / ev.rake are stored in radians.
        strike = float(np.degrees(ev.strike)) % 360.0
        dip = float(np.degrees(ev.dip))
        rake = float(np.degrees(getattr(ev, "rake", 0.0)))
        amplitude = float(getattr(ev, "amplitude", 0.0))
        origin = _resolve_origin(ev.origin, model)

        # Strike vector in (x, y), measured clockwise from +y (north).
        theta = np.deg2rad(strike)
        strike_dir = (np.sin(theta), np.cos(theta))
        seg = _clip_line_to_bbox((origin[0], origin[1]), strike_dir, spec_bbox)
        if seg is None:
            continue
        (x0, y0), (x1, y1) = seg
        # Throw approximation: vertical component of slip = amplitude * sin(dip)
        # (good for normal/reverse faults; underestimates strike-slip).
        throw = float(abs(amplitude * np.sin(np.deg2rad(dip))))
        dip_dir = (strike + 90.0) % 360.0  # right-hand rule down-dip
        out.append({
            "id": f"fault_gt_{len(out) + 1:03d}",
            "polyline_xy_m": [[round(x0, 2), round(y0, 2)],
                              [round(x1, 2), round(y1, 2)]],
            "dip_degrees": round(dip, 2),
            "dip_direction_degrees": round(dip_dir, 2),
            "rake_degrees": round(rake, 2),
            "throw_m": round(throw, 2),
            "amplitude_m": round(amplitude, 2),
            "origin_xyz_m": [round(origin[0], 2), round(origin[1], 2), round(origin[2], 2)],
            "source": "ground_truth",
        })
    return out


def fault_distance_volume(
    fault_traces: List[Dict],
    nz: int, ny: int, nx: int,
    dx_m: float, dy_m: float, dz_m: float,
    z_top_m: float = 0.0,
) -> np.ndarray:
    """Distance from each voxel to the nearest extruded fault plane.

    Each fault is extruded down-dip from its surface trace through the full
    z-extent and approximated as a discrete plane. The returned volume
    minimum should be ~0 on voxels intersecting fault surfaces (QC #8).
    """
    if not fault_traces:
        # No faults: return a constant max-distance volume so downstream
        # acceptance criteria don't crash; caller should also flag this.
        big = float(max(nz * dz_m, ny * dy_m, nx * dx_m))
        return np.full((nz, ny, nx), big, dtype=np.float32)

    # Voxel center coords
    z_centers = z_top_m - (np.arange(nz) + 0.5) * dz_m  # (nz,)
    y_centers = (np.arange(ny) + 0.5) * dy_m            # (ny,)
    x_centers = (np.arange(nx) + 0.5) * dx_m            # (nx,)

    # Build XX, YY, ZZ broadcastable
    ZZ = z_centers[:, None, None]
    YY = y_centers[None, :, None]
    XX = x_centers[None, None, :]

    # For each fault: signed distance from voxel center to the fault plane.
    # Plane normal n = (cos(strike) * sin(dip), -sin(strike) * sin(dip), cos(dip)).
    # Strike measured CW from +y (north). The plane passes through (origin_xy, z_top).
    # Distance from point P to plane = |n . (P - origin)|.
    out = np.full((nz, ny, nx), np.inf, dtype=np.float32)
    for ft in fault_traces:
        # Use the polyline midpoint as a point on the plane (intersection
        # with z = z_top). Strike direction defines an in-plane vector.
        (x0, y0), (x1, y1) = ft["polyline_xy_m"]
        px = 0.5 * (x0 + x1)
        py = 0.5 * (y0 + y1)
        pz = z_top_m
        strike = float(ft["dip_direction_degrees"]) - 90.0  # invert dip_dir back
        dip = float(ft["dip_degrees"])
        theta = np.deg2rad(strike % 360.0)
        delta = np.deg2rad(dip)
        nxh = np.cos(theta) * np.sin(delta)
        nyh = -np.sin(theta) * np.sin(delta)
        nzh = np.cos(delta)
        # Plane: n . (P - p0) = 0; distance = |n . (P - p0)|
        d = np.abs(nxh * (XX - px) + nyh * (YY - py) + nzh * (ZZ - pz))
        np.minimum(out, d.astype(np.float32), out=out)
    return out


def split_user_known_faults(
    fault_traces: List[Dict],
    fraction: float,
    rng: np.random.Generator,
) -> Tuple[List[Dict], List[Dict]]:
    """Partition into (user_known, hidden) for the inputs/ground_truth split."""
    if not fault_traces:
        return [], []
    n_known = int(round(fraction * len(fault_traces)))
    n_known = max(0, min(len(fault_traces), n_known))
    indices = list(range(len(fault_traces)))
    rng.shuffle(indices)
    known_idx = set(indices[:n_known])
    user = []
    hidden = []
    for i, ft in enumerate(fault_traces):
        copy_ft = dict(ft)
        copy_ft["source"] = "user_input" if i in known_idx else "ground_truth"
        if i in known_idx:
            user.append(copy_ft)
        else:
            hidden.append(copy_ft)
    return user, hidden
