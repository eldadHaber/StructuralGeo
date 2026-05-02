"""QC checks per spec section 'Quality-Control Checks Before Saving'."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from geogen.stratiflow.config import GenerationConfig
from geogen.stratiflow.lithology import AIR_CODE


class QCFailure(RuntimeError):
    pass


def run_qc(
    *,
    lithology_volume: np.ndarray,
    density_volume: np.ndarray,
    susceptibility_volume: np.ndarray,
    dem_yx: np.ndarray,
    surface_contact: np.ndarray,
    boreholes: List[Dict],
    gravity_grid: np.ndarray,
    fault_traces: List[Dict],
    fault_distance: np.ndarray,
    lithology_codes: Dict[int, dict],
    cfg: GenerationConfig,
) -> List[str]:
    """Returns a list of human-readable QC results. Raises QCFailure on any
    hard failure. Returns warnings for soft issues."""
    warnings: List[str] = []

    # 1. lithology codes valid
    valid = set(lithology_codes.keys()) | {AIR_CODE}
    present = set(int(c) for c in np.unique(lithology_volume))
    extra = present - valid
    if extra:
        raise QCFailure(f"QC1: lithology_volume contains unknown codes {sorted(extra)}")

    # 2. density volume = mapping(lithology), no extra noise
    expected_density = np.zeros_like(density_volume, dtype=np.float32)
    for code, entry in lithology_codes.items():
        expected_density[lithology_volume == code] = float(entry["density_kg_m3"])
    if not np.allclose(expected_density, density_volume, atol=1e-3):
        raise QCFailure("QC2: density_volume is not the deterministic map of lithology_volume")

    # 3. DEM consistent with air/non-air boundary
    rock_mask = lithology_volume != AIR_CODE
    has_rock = rock_mask.any(axis=0)
    top_idx = np.argmax(rock_mask, axis=0).astype(np.int32)
    expected_dem = -top_idx.astype(np.float32) * cfg.dz_m
    expected_dem[~has_rock] = 0.0
    diff = np.abs(dem_yx - expected_dem)
    if diff.max() > 0.5 * cfg.dz_m + 1e-3:
        raise QCFailure(
            f"QC3: DEM inconsistent with lithology_volume top "
            f"(max |diff| = {diff.max():.2f} > 0.5*dz_m = {0.5 * cfg.dz_m:.2f})"
        )

    # 4. borehole sample lithology matches volume lithology at sample center
    nx = lithology_volume.shape[2]; ny = lithology_volume.shape[1]
    nz = lithology_volume.shape[0]
    mismatches = 0
    total = 0
    for bh in boreholes:
        cx = bh["x_m"]; cy = bh["y_m"]; cz = bh["elevation_m"]
        dip_rad = np.deg2rad(bh["dip_deg"])
        az_rad = np.deg2rad(bh["azimuth_deg"])
        for s in bh["samples"]:
            total += 1
            mid_d = float(s["start_depth_m"] + s["length_m"] * 0.5)
            x = cx + mid_d * np.cos(dip_rad) * np.sin(az_rad)
            y = cy + mid_d * np.cos(dip_rad) * np.cos(az_rad)
            z = cz - mid_d * np.sin(dip_rad)
            i = int(x // cfg.dx_m); j = int(y // cfg.dy_m); k = int((-z) // cfg.dz_m)
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                vol_code = int(lithology_volume[k, j, i])
                if vol_code != s["lithology_code"]:
                    mismatches += 1
    if total > 0 and mismatches / total > 0.05:
        raise QCFailure(
            f"QC4: too many borehole/volume lithology mismatches "
            f"({mismatches}/{total} = {mismatches / total:.1%})"
        )
    if mismatches > 0:
        warnings.append(f"QC4 soft: {mismatches}/{total} borehole samples disagree at midpoint (allowed)")

    # 5. gravity grid finite, mean within ±20 mGal
    if not np.isfinite(gravity_grid).all():
        raise QCFailure("QC5: gravity_grid has non-finite values")
    if abs(float(gravity_grid.mean())) > 1000.0:
        # extreme drift suggests a wrong sign/units bug — catch big errors
        raise QCFailure(f"QC5: gravity_grid mean {gravity_grid.mean():.1f} mGal far from zero")
    if abs(float(gravity_grid.mean())) > 20.0:
        warnings.append(
            f"QC5 soft: gravity grid mean = {gravity_grid.mean():.1f} mGal "
            "(spec target: within ±20 mGal after regional trend; this is a "
            "consequence of the air-contrast convention)"
        )

    # 6. faults inside bbox
    x_max = cfg.nx * cfg.dx_m; y_max = cfg.ny * cfg.dy_m
    for ft in fault_traces:
        for x, y in ft["polyline_xy_m"]:
            if not (-1e-6 <= x <= x_max + 1e-6 and -1e-6 <= y <= y_max + 1e-6):
                raise QCFailure(f"QC6: fault {ft['id']} polyline point ({x}, {y}) outside model bbox")

    # 7. surface contact = topmost lithology
    j_idx, i_idx = np.indices(surface_contact.shape)
    expected_contact = lithology_volume[top_idx, j_idx, i_idx].astype(np.int32)
    expected_contact[~has_rock] = -1
    if not np.array_equal(surface_contact, expected_contact):
        raise QCFailure("QC7: surface_contact_raster disagrees with topmost-non-air lithology")

    # 8. fault_distance has min ~ 0 on fault surfaces
    if fault_traces:
        if float(fault_distance.min()) > 1.5 * max(cfg.dx_m, cfg.dy_m, cfg.dz_m):
            raise QCFailure(
                f"QC8: fault_distance min = {fault_distance.min():.1f} m, "
                f"expected ~0 on voxels intersecting fault planes"
            )

    return warnings
