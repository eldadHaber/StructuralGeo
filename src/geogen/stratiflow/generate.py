"""End-to-end sample generation.

generate_sample(seed, config, out_dir) writes one full spec-compliant sample
to disk: ground_truth/, inputs/, expected_outputs/, README.md, config.yaml.

The script that drives 5 samples lives in code_examples/.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from geogen.generation import MarkovGeostoryGenerator
from geogen.stratiflow.boreholes import generate_boreholes
from geogen.stratiflow.config import GenerationConfig
from geogen.stratiflow.faults import (
    extract_faults,
    fault_distance_volume,
    split_user_known_faults,
)
from geogen.stratiflow.gravity import forward_gravity_grid, write_gravity_stations_csv
from geogen.stratiflow.lithology import (
    AIR_CODE,
    BASEMENT_CODE,
    build_density_volume,
    build_susceptibility_volume,
    build_vocabulary,
    reorder_by_median_depth,
)
from geogen.stratiflow.qc import QCFailure, run_qc
from geogen.stratiflow.surface import (
    apply_topographic_perturbation,
    degrade_surface_map,
    extract_dem_from_volume,
    surface_contact_raster,
    write_dem_geotiff,
)


# ---------------------------------------------------------------------------
# GeoGen <-> spec axis conversion


def _geogen_to_spec_axes(litho_xyz: np.ndarray) -> np.ndarray:
    """GeoGen returns ``[nx, ny, nz]`` with z=0 at bottom. Spec wants
    ``[nz, ny, nx]`` with z=0 at top."""
    return np.transpose(litho_xyz, (2, 1, 0))[::-1, :, :].astype(np.int32).copy()


# ---------------------------------------------------------------------------
# Acceptance: re-roll until at least min_faults + min_units present


def _generate_acceptable_geomodel(cfg: GenerationConfig, rng: np.random.Generator):
    """Sample geomodels until one passes spec acceptance constraints.

    Returns (model, lithology_volume_zyx, fault_traces) where lithology is in
    spec axis convention. Raises RuntimeError if too many attempts fail.
    """
    last_summary = ""
    for attempt in range(1, cfg.max_geomodel_attempts + 1):
        # Per-attempt seed so the rolls are reproducible from cfg.seed
        attempt_seed = int(rng.integers(0, 2**31 - 1))
        np.random.seed(attempt_seed)

        gen = MarkovGeostoryGenerator(
            model_bounds=cfg.model_bounds,
            model_resolution=cfg.model_resolution,
        )
        model = gen.generate_model()
        model.fill_nans()
        litho_xyz = model.get_data_grid()
        litho_zyx = _geogen_to_spec_axes(litho_xyz)

        # Surface relief check (acceptance constraint)
        z_top_temp = 0.0
        rock_mask = litho_zyx != AIR_CODE
        has_rock = rock_mask.any(axis=0)
        if not has_rock.any():
            last_summary = "no rock voxels"
            continue

        # Faults from history
        bbox = (0.0, 0.0, cfg.nx * cfg.dx_m, cfg.ny * cfg.dy_m)
        faults = extract_faults(model, bbox)

        present_codes = sorted(int(c) for c in np.unique(litho_zyx) if int(c) != AIR_CODE)
        n_units = len(present_codes)
        n_faults = len(faults)

        if n_faults < cfg.min_faults:
            last_summary = f"only {n_faults} faults"; continue
        if n_units < cfg.min_units:
            last_summary = f"only {n_units} units"; continue
        if n_units > cfg.max_units:
            last_summary = f"{n_units} units (>max {cfg.max_units})"; continue

        return model, litho_zyx, faults, attempt
    raise RuntimeError(
        f"Could not produce an acceptable geomodel in "
        f"{cfg.max_geomodel_attempts} attempts (last reject: {last_summary}). "
        f"Try a different seed or relax acceptance constraints."
    )


# ---------------------------------------------------------------------------
# Sample writer


def generate_sample(
    cfg: GenerationConfig,
    out_dir: Path,
    sample_id: str = "sample_001",
) -> Dict:
    """Build one spec-compliant sample under ``out_dir``."""
    out_dir = Path(out_dir)
    (out_dir / "ground_truth").mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "expected_outputs").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    # ---- Stage A: geomodel + acceptance ----
    model, litho_zyx, faults, attempts = _generate_acceptable_geomodel(cfg, rng)

    # If geomodel is flat-topped, apply 2D Fourier perturbation to imprint relief.
    dem_initial = extract_dem_from_volume(litho_zyx, cfg.dz_m)
    if float(dem_initial.max() - dem_initial.min()) < cfg.surface_relief_min_m:
        litho_zyx, dem_yx = apply_topographic_perturbation(
            litho_zyx, cfg.dz_m, cfg.surface_relief_amplitude_m, rng,
        )
    else:
        dem_yx = dem_initial

    return _finish_sample_writeout(
        cfg=cfg, out_dir=out_dir, sample_id=sample_id,
        rng=rng, model=model, litho_zyx=litho_zyx, dem_yx=dem_yx,
        faults=faults, attempts=attempts, extra_metadata=None,
        crs=None,                                    # synthetic frame
        origin_xy_absolute_m=(0.0, 0.0),
        z_top_absolute_m=None,
    )


def _finish_sample_writeout(
    *,
    cfg: GenerationConfig,
    out_dir: Path,
    sample_id: str,
    rng: np.random.Generator,
    model,
    litho_zyx: np.ndarray,
    dem_yx: np.ndarray,
    faults: list,
    attempts: int,
    extra_metadata: Optional[Dict] = None,
    crs: Optional[str] = None,
    origin_xy_absolute_m: Tuple[float, float] = (0.0, 0.0),
    z_top_absolute_m: Optional[float] = None,
) -> Dict:
    """Shared sample-writeout pipeline (everything from vocab onward).

    Both ``generate_sample`` and ``generate_sample_from_nz_tile`` call this
    once they have a geomodel + DEM.
    """
    # ---- Stage B: vocabulary + density / susceptibility volumes ----
    present_codes = [int(c) for c in np.unique(litho_zyx) if int(c) != AIR_CODE]
    ordered_codes = reorder_by_median_depth(litho_zyx, present_codes)
    lithology_codes, stratigraphic_codes = build_vocabulary(ordered_codes, rng)
    density_vol = build_density_volume(litho_zyx, lithology_codes)
    susceptibility_vol = build_susceptibility_volume(litho_zyx, lithology_codes)

    # ---- Stage D: surface contact raster ----
    contact = surface_contact_raster(litho_zyx)
    contact_degraded = degrade_surface_map(
        contact,
        cfg.surface_mapping_unmapped_fraction,
        cfg.surface_mapping_error_fraction,
        rng,
    )

    # ---- Stage E: boreholes ----
    boreholes = generate_boreholes(
        litho_zyx, dem_yx, lithology_codes, faults, cfg, rng,
    )

    # ---- Stage F: gravity ----
    gravity_grid, obs_z, gravity_meta = forward_gravity_grid(
        density_vol, dem_yx, cfg, rng,
    )
    stations_csv = out_dir / "inputs" / "gravity_stations.csv"
    write_gravity_stations_csv(
        gravity_grid, dem_yx, cfg, rng, stations_csv,
        crs=crs, origin_xy_absolute_m=origin_xy_absolute_m,
    )

    # ---- Stage G: split user-known vs hidden faults ----
    user_faults, hidden_faults = split_user_known_faults(
        faults, cfg.user_known_fault_fraction, rng,
    )

    # ---- Stage H: expected outputs ----
    fault_dist = fault_distance_volume(
        faults, cfg.nz, cfg.ny, cfg.nx, cfg.dx_m, cfg.dy_m, cfg.dz_m,
    )

    # ---- QC ----
    warnings = run_qc(
        lithology_volume=litho_zyx,
        density_volume=density_vol,
        susceptibility_volume=susceptibility_vol,
        dem_yx=dem_yx,
        surface_contact=contact,
        boreholes=boreholes,
        gravity_grid=gravity_grid,
        fault_traces=faults,
        fault_distance=fault_dist,
        lithology_codes=lithology_codes,
        cfg=cfg,
    )

    # ---- Persist ----
    np.save(out_dir / "ground_truth" / "lithology_volume.npy", litho_zyx.astype(np.int32))
    np.save(out_dir / "ground_truth" / "density_volume.npy", density_vol)
    np.save(out_dir / "ground_truth" / "susceptibility_volume.npy", susceptibility_vol)

    with open(out_dir / "ground_truth" / "lithology_codes.json", "w") as f:
        json.dump({str(k): v for k, v in lithology_codes.items()}, f, indent=2)
    with open(out_dir / "ground_truth" / "stratigraphic_codes.json", "w") as f:
        json.dump({str(k): v for k, v in stratigraphic_codes.items()}, f, indent=2)
    with open(out_dir / "ground_truth" / "fault_traces.json", "w") as f:
        json.dump({"faults": faults}, f, indent=2)

    x_max_abs = origin_xy_absolute_m[0] + cfg.nx * cfg.dx_m
    y_max_abs = origin_xy_absolute_m[1] + cfg.ny * cfg.dy_m
    metadata = {
        "schema_version": "stratiflow-sample-v1",
        "sample_id": sample_id,
        "seed": cfg.seed,
        "geomodel_attempts": attempts,
        "structuralgeo_axis_note": (
            "GeoGen returns [nx, ny, nz] with z_index=0 at the bottom. "
            "Volumes here are converted to spec [nz, ny, nx] with z_index=0 at the top "
            "via transpose((2,1,0))[::-1, :, :]."
        ),
        "mesh": {
            "nz": cfg.nz, "ny": cfg.ny, "nx": cfg.nx,
            "dx_m": cfg.dx_m, "dy_m": cfg.dy_m, "dz_m": cfg.dz_m,
            "z_top_m": 0.0,
        },
        "geospatial": {
            "crs": crs,                                  # e.g. "EPSG:2193" or null for local frame
            "horizontal_units": "meters",
            "origin_xy_absolute_m": [float(origin_xy_absolute_m[0]),
                                     float(origin_xy_absolute_m[1])],
            "bbox_xy_absolute_m": [float(origin_xy_absolute_m[0]),
                                   float(origin_xy_absolute_m[1]),
                                   float(x_max_abs), float(y_max_abs)],
            "z_top_m_local": 0.0,
            "z_top_m_absolute": (float(z_top_absolute_m)
                                 if z_top_absolute_m is not None else None),
            "axis_to_absolute_xy_note": (
                "Local x_m -> absolute x = origin_xy_absolute_m[0] + x_m. "
                "Local y_m -> absolute y = origin_xy_absolute_m[1] + y_m. "
                "Local z (k=0 top, increasing downward) -> absolute elevation "
                "= z_top_m_absolute - (k + 0.5) * dz_m   (when z_top_m_absolute is set)."
            ),
        },
        "physics": {
            "reference_density_kg_m3": cfg.reference_density_kg_m3,
            **gravity_meta,
        },
        "summary": {
            "n_units": len(lithology_codes),
            "n_faults": len(faults),
            "n_user_faults": len(user_faults),
            "n_boreholes": len(boreholes),
            "surface_relief_m": float(dem_yx.max() - dem_yx.min()),
            "qc_warnings": warnings,
        },
        "geomodel_event_history": [type(ev).__name__ for ev in (model.history_unpacked or model.history)],
    }
    if extra_metadata:
        metadata["provenance"] = extra_metadata
    with open(out_dir / "ground_truth" / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    np.save(out_dir / "inputs" / "dem.npy", dem_yx)
    # Use the declared CRS (NZ tile path) or a synthetic local-CRS fallback.
    geotiff_crs_epsg = _epsg_from_crs(crs) if crs else 32760
    write_dem_geotiff(
        dem_yx, out_dir / "inputs" / "dem.tif",
        dx_m=cfg.dx_m, dy_m=cfg.dy_m,
        origin_xy=tuple(origin_xy_absolute_m),
        crs_epsg=geotiff_crs_epsg,
    )
    np.save(out_dir / "inputs" / "obs_z.npy", obs_z)
    np.save(out_dir / "inputs" / "gravity_grid.npy", gravity_grid)
    np.save(out_dir / "inputs" / "surface_contact_raster.npy", contact_degraded)
    with open(out_dir / "inputs" / "boreholes.json", "w") as f:
        json.dump({"boreholes": boreholes}, f, indent=2)
    with open(out_dir / "inputs" / "user_fault_traces.json", "w") as f:
        json.dump({"faults": user_faults}, f, indent=2)

    np.save(out_dir / "expected_outputs" / "stratigraphic_label.npy",
            litho_zyx.astype(np.int32))
    np.save(out_dir / "expected_outputs" / "lithology_label.npy",
            litho_zyx.astype(np.int32))
    np.save(out_dir / "expected_outputs" / "fault_distance.npy", fault_dist.astype(np.float32))

    _write_notes(out_dir / "expected_outputs" / "notes.md")
    _write_readme(out_dir / "README.md", metadata, cfg, lithology_codes)
    _write_pipeline_config(
        out_dir / "config.yaml", cfg, lithology_codes, stratigraphic_codes,
        crs=crs, origin_xy_absolute_m=origin_xy_absolute_m,
        z_top_absolute_m=z_top_absolute_m,
    )

    return {
        "out_dir": str(out_dir),
        "n_units": len(lithology_codes),
        "n_faults": len(faults),
        "n_boreholes": len(boreholes),
        "surface_relief_m": float(dem_yx.max() - dem_yx.min()),
        "qc_warnings": warnings,
    }


# ---------------------------------------------------------------------------
# NZ-tile path (real DEM from Microsoft Planetary Computer)


def _stitch_geogen_to_real_dem(
    litho_zyx: np.ndarray,
    real_dem_absolute_yx: np.ndarray,
    cfg: GenerationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-anchor a GeoGen-derived volume to a real DEM.

    Steps:
      1. Forward-fill GeoGen's internal air with the topmost rock per column
         (so the column has a continuous lithology stack).
      2. Shift the real DEM so its max equals 0 (matches stratiflow's frame
         convention z_top = 0, increasing z is downward).
      3. For each (j, i), mask voxels with elevation > shifted_dem[j, i]
         as AIR.

    Returns ``(new_lithology_zyx, shifted_dem_yx, dem_offset_m)``.
    """
    nz, ny, nx = litho_zyx.shape

    # Forward-fill GeoGen internal air with topmost-rock per column
    out = litho_zyx.copy()
    for j in range(ny):
        for i in range(nx):
            col = out[:, j, i]
            non_air = np.where(col != AIR_CODE)[0]
            if non_air.size == 0:
                col[:] = BASEMENT_CODE
                continue
            top_rock_idx = non_air.min()
            top_rock_val = col[top_rock_idx]
            col[:top_rock_idx] = top_rock_val   # spec: z=0 is top, so fill upward

    dem_offset_m = float(real_dem_absolute_yx.max())
    dem_shifted = real_dem_absolute_yx - dem_offset_m  # max -> 0, others negative

    # Clip the shifted DEM to the frame floor so no column ends up entirely
    # above the deepest voxel center -- which would mask the column to all-air
    # and break the "DEM matches top-of-rock" QC invariant for high-relief
    # tiles (relief > nz * dz_m). The clip elevation is one quarter-voxel
    # above the deepest cell center so the deepest cell is guaranteed rock.
    deepest_center_m = -(nz - 0.5) * cfg.dz_m
    dem_floor_m = deepest_center_m + 0.25 * cfg.dz_m
    dem_shifted = np.maximum(dem_shifted, dem_floor_m).astype(np.float32)

    # Voxel-center elevations in spec frame (z_top = 0, k=0 at top):
    z_centers = -(np.arange(nz) + 0.5) * cfg.dz_m
    above_dem = z_centers[:, None, None] > dem_shifted[None, :, :]
    out[above_dem] = AIR_CODE
    return out, dem_shifted, dem_offset_m


def generate_sample_from_nz_tile(
    region,
    cfg: GenerationConfig,
    out_dir: Path,
    sample_id: str = "sample_001",
    *,
    max_tile_attempts: int = 30,
    max_built_up_fraction: float = 0.02,
    max_water_fraction: float = 0.30,
) -> Dict:
    """Generate a stratiflow sample using a real NZ DEM as the surface.

    Pulls Copernicus DEM 30m via MPC for a randomly-sampled tile in the
    given tectonic region, rejects tiles with > ``max_built_up_fraction``
    built-up land cover (ESA WorldCover 10m), then runs the standard
    stratiflow pipeline with that DEM stitched in as the top boundary of
    the GeoGen subsurface volume.
    """
    from geogen.gis.mpc import PlanetaryComputerClient
    from geogen.stratiflow.nz_tile import (
        fetch_s2_rgb_at_grid,
        pick_nz_tile_for_stratiflow,
    )

    out_dir = Path(out_dir)
    (out_dir / "ground_truth").mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "expected_outputs").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    # ---- Real NZ tile via MPC ----
    client = PlanetaryComputerClient()
    nz_result = pick_nz_tile_for_stratiflow(
        region, cfg, rng,
        max_attempts=max_tile_attempts,
        max_built_up=max_built_up_fraction,
        max_water=max_water_fraction,
        client=client,
    )

    # Fetch a Sentinel-2 RGB composite at the stratiflow grid (best-effort).
    # Stored at inputs/satellite_rgb.npy as float32 [3, ny, nx] in [0, 1].
    s2_rgb = fetch_s2_rgb_at_grid(
        client, nz_result.tile.bbox_nztm, ny=cfg.ny, nx=cfg.nx,
    )
    if s2_rgb is not None:
        np.save(out_dir / "inputs" / "satellite_rgb.npy", s2_rgb.astype(np.float32))

    # ---- GeoGen acceptance loop ----
    model, litho_zyx, faults, attempts = _generate_acceptable_geomodel(cfg, rng)

    # ---- Stitch GeoGen to real DEM ----
    litho_zyx, dem_shifted_yx, dem_offset_m = _stitch_geogen_to_real_dem(
        litho_zyx, nz_result.dem_yx, cfg,
    )

    # Provenance metadata
    extra = {
        "source": "microsoft_planetary_computer",
        "dem_collection": "cop-dem-glo-30",
        "landcover_collection": "esa-worldcover",
        "region_name": region.name,
        "center_lonlat": list(nz_result.tile.center_lonlat),
        "bbox_nztm": list(nz_result.tile.bbox_nztm),
        "absolute_dem_offset_m": dem_offset_m,
        "absolute_dem_range_m": [
            float(nz_result.dem_yx.min()), float(nz_result.dem_yx.max()),
        ],
        "landcover_stats": {
            "built_up_fraction": nz_result.landcover_stats["built_up"],
            "water_fraction": nz_result.landcover_stats["water"],
            "snow_ice_fraction": nz_result.landcover_stats["snow_ice"],
        },
        "tile_picker_attempts": nz_result.attempts,
    }

    xmin, ymin, _xmax, _ymax = nz_result.tile.bbox_nztm
    return _finish_sample_writeout(
        cfg=cfg, out_dir=out_dir, sample_id=sample_id,
        rng=rng, model=model, litho_zyx=litho_zyx, dem_yx=dem_shifted_yx,
        faults=faults, attempts=attempts, extra_metadata=extra,
        crs="EPSG:2193",                             # NZGD2000 / NZTM2000
        origin_xy_absolute_m=(float(xmin), float(ymin)),
        z_top_absolute_m=dem_offset_m,
    )


# ---------------------------------------------------------------------------
# Plain-text writers


def _write_notes(path: Path):
    path.write_text(
        "# Expected-output notes\n\n"
        "- `lithology_label.npy` mirrors `ground_truth/lithology_volume.npy`.\n"
        "  The pipeline's lithology head should reproduce this within its\n"
        "  acceptance criterion (label-recall on rock voxels).\n"
        "- `stratigraphic_label.npy` is identical to lithology_label here\n"
        "  (lithology = stratigraphy mapping in v1 per spec).\n"
        "- `fault_distance.npy` is the per-voxel Euclidean distance to the\n"
        "  nearest ground-truth fault plane (extruded down-dip). The pipeline\n"
        "  output should agree within ~2 mesh cells (acceptance criterion 9).\n"
    )


def _write_readme(path: Path, metadata, cfg: GenerationConfig, lithology_codes):
    geo = metadata.get("geospatial", {})
    crs_line = (f"- CRS: `{geo['crs']}`  (origin xy abs = "
                f"{geo['origin_xy_absolute_m'][0]:.1f}, {geo['origin_xy_absolute_m'][1]:.1f} m"
                + (f"; z_top abs = {geo['z_top_m_absolute']:.1f} m)"
                   if geo.get("z_top_m_absolute") is not None else ")")
                ) if geo.get("crs") else "- CRS: local meters (no projection)"
    prov = metadata.get("provenance")
    prov_lines: list = []
    if prov:
        prov_lines = [
            "",
            "## Provenance",
            "",
            f"- Source: {prov.get('source', '?')}",
            f"- Region: {prov.get('region_name', '?')}",
            f"- Center (lon, lat): {prov.get('center_lonlat', '?')}",
            f"- DEM: {prov.get('dem_collection', '?')}",
            f"- Land cover: {prov.get('landcover_collection', '?')}  "
            f"(built-up {100 * prov.get('landcover_stats', {}).get('built_up_fraction', 0):.1f}%, "
            f"water {100 * prov.get('landcover_stats', {}).get('water_fraction', 0):.1f}%)",
            f"- Tile picker attempts: {prov.get('tile_picker_attempts', '?')}",
        ]
    lines = [
        f"# Stratiflow synthetic sample `{metadata['sample_id']}`",
        "",
        "Generated by `geogen.stratiflow.generate.generate_sample` per the",
        "synthetic-test-dataset spec (StructuralGeo + derived channels).",
        "",
        f"- Seed: `{cfg.seed}`",
        f"- Mesh: nz={cfg.nz}, ny={cfg.ny}, nx={cfg.nx}, dz=dy=dx={cfg.dx_m} m",
        f"- Domain extent: {cfg.nx * cfg.dx_m / 1000:.1f} x {cfg.ny * cfg.dy_m / 1000:.1f} x {cfg.nz * cfg.dz_m / 1000:.1f} km",
        f"- Reference density: {cfg.reference_density_kg_m3} kg/m^3",
        crs_line,
        f"- Units: {metadata['summary']['n_units']}",
        f"- Faults (ground truth / user-known): {metadata['summary']['n_faults']} / {metadata['summary']['n_user_faults']}",
        f"- Boreholes: {metadata['summary']['n_boreholes']}",
        f"- Surface relief: {metadata['summary']['surface_relief_m']:.0f} m",
        *prov_lines,
        "",
        "## Lithology vocabulary",
        "",
        "| Code | Unit | Lithology | Density (kg/m^3) | Susceptibility (SI) |",
        "|------|------|-----------|------------------|---------------------|",
    ]
    for code, e in sorted(lithology_codes.items()):
        lines.append(
            f"| {code} | {e['name']} | {e['lithology']} | "
            f"{e['density_kg_m3']:.0f} | {e['susceptibility_si']:.1e} |"
        )
    lines.extend([
        "",
        "## Regenerate",
        "",
        "```python",
        "from geogen.stratiflow import GenerationConfig, generate_sample",
        f"cfg = GenerationConfig(seed={cfg.seed})",
        "summary = generate_sample(cfg, out_dir='stratiflow_samples/<name>')",
        "```",
        "",
        "## Acceptance criteria mapping",
        "",
        "- `expected_outputs/lithology_label.npy` -> per-voxel lithology recall",
        "- `expected_outputs/stratigraphic_label.npy` -> per-voxel stratigraphy recall",
        "- `expected_outputs/fault_distance.npy` -> distance-MAE within ~2 cells",
        "- `inputs/user_fault_traces.json` -> hidden faults must be rediscovered",
        "",
        "## QC warnings",
        "",
    ])
    if metadata["summary"]["qc_warnings"]:
        for w in metadata["summary"]["qc_warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("- (none)")
    path.write_text("\n".join(lines) + "\n")


def _epsg_from_crs(crs: Optional[str]) -> int:
    """Extract integer EPSG code from a string like 'EPSG:2193'."""
    if not crs:
        return 32760
    return int(str(crs).split(":")[-1])


def _write_pipeline_config(
    path: Path, cfg: GenerationConfig,
    lithology_codes, stratigraphic_codes,
    *,
    crs: Optional[str] = None,
    origin_xy_absolute_m: Tuple[float, float] = (0.0, 0.0),
    z_top_absolute_m: Optional[float] = None,
):
    """Write a minimal pipeline config.yaml referencing the inputs/ files."""
    lithology_yaml = "\n".join(
        f"  {c}: {{name: {e['name']}, lithology: {e['lithology']}, "
        f"density_kg_m3: {e['density_kg_m3']:.1f}, susceptibility_si: {e['susceptibility_si']:.6e}}}"
        for c, e in sorted(lithology_codes.items())
    )
    strat_yaml = "\n".join(
        f"  {c}: {{name: {e['name']}, age_order: {e['age_order']}}}"
        for c, e in sorted(stratigraphic_codes.items())
    )
    crs_yaml_value = f'"{crs}"' if crs else "null"
    z_top_abs_yaml = (f"{z_top_absolute_m:.2f}"
                      if z_top_absolute_m is not None else "null")
    text = f"""# Pipeline config generated for this synthetic sample
mesh:
  nz: {cfg.nz}
  ny: {cfg.ny}
  nx: {cfg.nx}
  dx_m: {cfg.dx_m}
  dy_m: {cfg.dy_m}
  dz_m: {cfg.dz_m}
  observation_height_m: {cfg.obs_height_m}

geospatial:
  crs: {crs_yaml_value}
  horizontal_units: meters
  origin_xy_absolute_m: [{origin_xy_absolute_m[0]:.3f}, {origin_xy_absolute_m[1]:.3f}]
  z_top_m_local: 0.0
  z_top_m_absolute: {z_top_abs_yaml}

reference_density_kg_m3: {cfg.reference_density_kg_m3}

inputs:
  dem: inputs/dem.npy
  dem_geotiff: inputs/dem.tif
  obs_z: inputs/obs_z.npy
  gravity_grid: inputs/gravity_grid.npy
  gravity_stations: inputs/gravity_stations.csv
  boreholes: inputs/boreholes.json
  surface_contact_raster: inputs/surface_contact_raster.npy
  user_fault_traces: inputs/user_fault_traces.json

expected_outputs:
  stratigraphic_label: expected_outputs/stratigraphic_label.npy
  lithology_label: expected_outputs/lithology_label.npy
  fault_distance: expected_outputs/fault_distance.npy

lithology_codes:
{lithology_yaml}

stratigraphic_codes:
{strat_yaml}
"""
    path.write_text(text)
