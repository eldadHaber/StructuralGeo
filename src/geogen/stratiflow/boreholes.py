"""Borehole generation: Poisson-disk collars, trajectory walk, sample grouping."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from geogen.stratiflow.config import GenerationConfig
from geogen.stratiflow.lithology import AIR_CODE, is_major_oxide


# ---------------------------------------------------------------------------
# Poisson-disk sampling


def _poisson_disk_2d(
    bbox: Tuple[float, float, float, float],
    min_dist: float,
    n_target: int,
    rng: np.random.Generator,
    max_attempts: int = 30,
    must_include: Optional[List[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    """Bridson's-ish Poisson-disk sampling. Returns up to ``n_target`` points
    with pairwise distance >= ``min_dist``. ``must_include`` points are placed
    first if they pass the spacing constraint with each other."""
    x_min, y_min, x_max, y_max = bbox
    points: List[Tuple[float, float]] = []
    if must_include:
        for p in must_include:
            if not points or all(np.hypot(p[0] - q[0], p[1] - q[1]) >= min_dist
                                 for q in points):
                points.append(tuple(p))
    while len(points) < n_target:
        ok = False
        for _ in range(max_attempts):
            cand = (
                float(rng.uniform(x_min, x_max)),
                float(rng.uniform(y_min, y_max)),
            )
            if all(np.hypot(cand[0] - q[0], cand[1] - q[1]) >= min_dist for q in points):
                points.append(cand)
                ok = True
                break
        if not ok:
            break  # could not place more; return what we have
    return points


# ---------------------------------------------------------------------------
# Borehole orientation + trajectory


def _vertical_dip_azimuth(rng: np.random.Generator, inclined: bool) -> Tuple[float, float]:
    """Return (dip_deg, azimuth_deg). Borehole convention: dip 90 = straight
    down, azimuth = clockwise from north (+y)."""
    if not inclined:
        return (90.0, 0.0)
    dip = float(rng.uniform(60.0, 80.0))
    az = float(rng.uniform(0.0, 360.0))
    return (dip, az)


def _trajectory_step_dxyz(dip_deg: float, az_deg: float, step_m: float
                          ) -> Tuple[float, float, float]:
    """Per-step horizontal/vertical increments. dip 90 = down (dz=-step)."""
    dip = np.deg2rad(dip_deg)
    az = np.deg2rad(az_deg)
    dz = -step_m * np.sin(dip)
    horiz = step_m * np.cos(dip)
    dx = horiz * np.sin(az)
    dy = horiz * np.cos(az)
    return float(dx), float(dy), float(dz)


def _voxel_lookup(
    lithology_zyx: np.ndarray,
    x_m: float, y_m: float, z_m: float,
    dx_m: float, dy_m: float, dz_m: float,
    z_top_m: float,
) -> int:
    """Return the lithology code at world coords (x, y, z)."""
    nz, ny, nx = lithology_zyx.shape
    i = int(x_m // dx_m)
    j = int(y_m // dy_m)
    k = int((z_top_m - z_m) // dz_m)
    if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
        return AIR_CODE
    return int(lithology_zyx[k, j, i])


# ---------------------------------------------------------------------------
# Sample group + chemistry


def _sample_chemistry(
    template: Dict[str, float],
    rng: np.random.Generator,
    rel_noise_major: float,
    rel_noise_trace: float,
) -> Dict[str, float]:
    out = {}
    for k, mu in template.items():
        rel = rel_noise_major if is_major_oxide(k) else rel_noise_trace
        sigma = abs(mu) * rel
        out[k] = float(np.maximum(0.0, rng.normal(mu, sigma)))
    return out


def _build_intervals(
    codes_along_trace: List[int],
    depths_along_trace: List[float],
    cfg: GenerationConfig,
) -> List[Tuple[int, float, float]]:
    """Group consecutive same-code samples into (code, start_depth, length)
    intervals, splitting any interval longer than ``sample_max_length_m``."""
    out: List[Tuple[int, float, float]] = []
    if not codes_along_trace:
        return out
    cur_code = codes_along_trace[0]
    cur_start = depths_along_trace[0]
    last_depth = depths_along_trace[0]
    for code, depth in zip(codes_along_trace[1:], depths_along_trace[1:]):
        if code != cur_code:
            length = last_depth - cur_start
            if length >= cfg.sample_min_length_m:
                _split_long_intervals(out, cur_code, cur_start, length, cfg)
            cur_code = code
            cur_start = depth
        last_depth = depth
    # Trailing interval
    length = last_depth - cur_start
    if length >= cfg.sample_min_length_m:
        _split_long_intervals(out, cur_code, cur_start, length, cfg)
    return out


def _split_long_intervals(
    accum: List[Tuple[int, float, float]],
    code: int,
    start: float,
    length: float,
    cfg: GenerationConfig,
) -> None:
    while length > cfg.sample_max_length_m:
        accum.append((code, start, cfg.sample_max_length_m))
        start += cfg.sample_max_length_m
        length -= cfg.sample_max_length_m
    if length > 0:
        accum.append((code, start, length))


# ---------------------------------------------------------------------------
# Top-level generator


def generate_boreholes(
    lithology_zyx: np.ndarray,
    dem_yx: np.ndarray,
    lithology_codes: Dict[int, dict],
    fault_traces: List[Dict],
    cfg: GenerationConfig,
    rng: np.random.Generator,
) -> List[Dict]:
    """Build a list of borehole dicts matching the spec's boreholes.json schema."""
    nz, ny, nx = lithology_zyx.shape
    bbox_xy = (
        cfg.dx_m * 0.5, cfg.dy_m * 0.5,
        cfg.nx * cfg.dx_m - cfg.dx_m * 0.5,
        cfg.ny * cfg.dy_m - cfg.dy_m * 0.5,
    )

    # Seed with required collars: one near centre + one per major fault midpoint.
    must_include: List[Tuple[float, float]] = [
        (cfg.nx * cfg.dx_m * 0.5, cfg.ny * cfg.dy_m * 0.5)
    ]
    for ft in fault_traces[: max(1, len(fault_traces))]:
        (x0, y0), (x1, y1) = ft["polyline_xy_m"]
        # Move slightly off-trace so the collar isn't sitting on the fault but
        # the borehole crosses it as it descends.
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        # Perpendicular offset of 1 grid spacing along the dip direction
        ang = np.deg2rad(ft["dip_direction_degrees"])
        cx += -50.0 * np.sin(ang)
        cy += -50.0 * np.cos(ang)
        if (bbox_xy[0] <= cx <= bbox_xy[2]) and (bbox_xy[1] <= cy <= bbox_xy[3]):
            must_include.append((cx, cy))

    collars = _poisson_disk_2d(
        bbox_xy, min_dist=cfg.borehole_min_spacing_m,
        n_target=max(cfg.n_boreholes, len(must_include)),
        rng=rng, must_include=must_include,
    )
    collars = collars[: cfg.n_boreholes]

    out: List[Dict] = []
    n_inclined_target = int(round(cfg.inclined_fraction * len(collars)))
    inclined_flags = [True] * n_inclined_target + [False] * (len(collars) - n_inclined_target)
    rng.shuffle(inclined_flags)

    for k, ((cx, cy), inclined) in enumerate(zip(collars, inclined_flags)):
        # Collar elevation = surface
        # Convert cell-space lookup
        ix = int(cx // cfg.dx_m); iy = int(cy // cfg.dy_m)
        ix = max(0, min(nx - 1, ix))
        iy = max(0, min(ny - 1, iy))
        collar_z = float(dem_yx[iy, ix])

        dip, az = _vertical_dip_azimuth(rng, inclined)
        length = float(rng.uniform(cfg.borehole_length_m_min, cfg.borehole_length_m_max))

        # Walk down the trajectory in 5-m increments.
        codes: List[int] = []
        depths: List[float] = []
        x = cx; y = cy; z = collar_z
        depth = 0.0
        dx_step, dy_step, dz_step = _trajectory_step_dxyz(dip, az, cfg.sample_step_m)
        while depth <= length:
            code = _voxel_lookup(
                lithology_zyx, x, y, z,
                cfg.dx_m, cfg.dy_m, cfg.dz_m, z_top_m=0.0,
            )
            codes.append(code)
            depths.append(depth)
            x += dx_step; y += dy_step; z += dz_step
            depth += cfg.sample_step_m

        intervals = _build_intervals(codes, depths, cfg)

        samples = []
        for s_code, s_start, s_len in intervals:
            if s_code == AIR_CODE or s_code not in lithology_codes:
                continue
            unit = lithology_codes[s_code]
            density_mu = float(unit["density_kg_m3"])
            density = float(rng.normal(density_mu, cfg.density_sample_noise_kg_m3))
            chem = _sample_chemistry(
                unit["chemistry_template"], rng,
                cfg.chemistry_relative_noise_major,
                cfg.chemistry_relative_noise_trace,
            )
            samples.append({
                "start_depth_m": round(float(s_start), 2),
                "length_m": round(float(s_len), 2),
                "density_kg_m3": round(density, 2),
                "uncertainty_kg_m3": round(float(cfg.density_sample_noise_kg_m3), 2),
                "lithology": unit["lithology"],
                "stratigraphic_unit": unit["name"],
                "chemistry": {kk: round(vv, 4) for kk, vv in chem.items()},
                "lithology_code": int(s_code),
                "source": "synthetic_core",
            })
        out.append({
            "id": f"BH-SYNTH-{k + 1:03d}",
            "x_m": round(float(cx), 2),
            "y_m": round(float(cy), 2),
            "elevation_m": round(collar_z, 2),
            "azimuth_deg": round(az, 2),
            "dip_deg": round(dip, 2),
            "total_depth_m": round(length, 2),
            "samples": samples,
        })
    return out
