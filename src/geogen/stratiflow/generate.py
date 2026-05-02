"""End-to-end sample generation.

generate_sample(seed, config, out_dir) writes one full spec-compliant sample
to disk: ground_truth/, inputs/, expected_outputs/, README.md, config.yaml.

The script that drives 5 samples lives in code_examples/.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

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
    """Build one spec-compliant sample under ``out_dir``.

    Returns a small summary dict (counts, paths) for logging / aggregation.
    """
    out_dir = Path(out_dir)
    (out_dir / "ground_truth").mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "expected_outputs").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    # ---- Stage A: geomodel + acceptance ----
    model, litho_zyx, faults, attempts = _generate_acceptable_geomodel(cfg, rng)

    # Surface relief constraint: if synthetic relief is below threshold,
    # apply a 2D perturbation to deform the top voxels into air.
    dem_initial = extract_dem_from_volume(litho_zyx, cfg.dz_m)
    if float(dem_initial.ptp() if False else (dem_initial.max() - dem_initial.min())) < cfg.surface_relief_min_m:
        litho_zyx, dem_yx = apply_topographic_perturbation(
            litho_zyx, cfg.dz_m, cfg.surface_relief_amplitude_m, rng,
        )
    else:
        dem_yx = dem_initial

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
    write_gravity_stations_csv(gravity_grid, dem_yx, cfg, rng, stations_csv)

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
    with open(out_dir / "ground_truth" / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    np.save(out_dir / "inputs" / "dem.npy", dem_yx)
    write_dem_geotiff(dem_yx, out_dir / "inputs" / "dem.tif",
                      dx_m=cfg.dx_m, dy_m=cfg.dy_m)
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
    _write_pipeline_config(out_dir / "config.yaml", cfg, lithology_codes, stratigraphic_codes)

    return {
        "out_dir": str(out_dir),
        "n_units": len(lithology_codes),
        "n_faults": len(faults),
        "n_boreholes": len(boreholes),
        "surface_relief_m": float(dem_yx.max() - dem_yx.min()),
        "qc_warnings": warnings,
    }


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
        f"- Units: {metadata['summary']['n_units']}",
        f"- Faults (ground truth / user-known): {metadata['summary']['n_faults']} / {metadata['summary']['n_user_faults']}",
        f"- Boreholes: {metadata['summary']['n_boreholes']}",
        f"- Surface relief: {metadata['summary']['surface_relief_m']:.0f} m",
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


def _write_pipeline_config(
    path: Path, cfg: GenerationConfig,
    lithology_codes, stratigraphic_codes,
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
    text = f"""# Pipeline config generated for this synthetic sample
mesh:
  nz: {cfg.nz}
  ny: {cfg.ny}
  nx: {cfg.nx}
  dx_m: {cfg.dx_m}
  dy_m: {cfg.dy_m}
  dz_m: {cfg.dz_m}
  observation_height_m: {cfg.obs_height_m}

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
