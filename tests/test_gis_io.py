"""Round-trip and spec-conformance tests for the sample tile IO layer."""

import json
from pathlib import Path

import numpy as np

from geogen.gis.io import (
    AIR_LITHOLOGY,
    DEFAULT_BACKGROUND_DENSITY_KG_M3,
    SCHEMA_VERSION,
    build_sample_tile,
    load_sample_tile,
    save_sample_tile,
)
from geogen.gis.mpc import TILE_SIZE_M, Tile
from geogen.gis.regions import NZ_REGIONS


def _toy_tile_and_lithology(nz_voxels: int = 32):
    """A small synthetic tile + GeoGen-style lithology grid for fast tests."""
    region = NZ_REGIONS[0]
    nx = ny = 32
    yy, xx = np.mgrid[0:ny, 0:nx]
    # Smooth DEM with mild relief
    dem = (200.0 + 30.0 * (xx + yy) / nx).astype(np.float32)
    tile = Tile(
        region=region,
        center_lonlat=(170.0, -43.5),
        bbox_nztm=(0.0, 0.0, TILE_SIZE_M, TILE_SIZE_M),
        dem=dem,
    )
    # GeoGen layout: [nx, ny, nz], z=0 at bottom, -1 = air
    # Build a column: basement (0) on bottom, sediment (3) on top, air (-1) above
    litho_xyz = np.zeros((nx, ny, nz_voxels), dtype=np.int8)
    litho_xyz[:, :, : nz_voxels // 2] = 0          # basement (bottom half)
    litho_xyz[:, :, nz_voxels // 2 : 3 * nz_voxels // 4] = 3  # sediment
    litho_xyz[:, :, 3 * nz_voxels // 4 :] = AIR_LITHOLOGY     # air on top
    return tile, litho_xyz


def test_build_sample_tile_shapes_and_types():
    tile, litho = _toy_tile_and_lithology(nz_voxels=16)
    sample = build_sample_tile(
        tile, litho,
        dz_m=30.0,
        skip_geographic=True,  # bbox is synthetic
        density_seed=0,
    )
    ny, nx = tile.dem.shape
    nz = litho.shape[2]
    assert sample.surface.shape == (ny, nx)
    assert sample.obs_z.shape == (ny, nx)
    assert sample.target.shape == (nz, ny, nx)
    assert sample.lithology.shape == (nz, ny, nx)
    assert sample.gravity.shape == (ny, nx)
    assert sample.target.dtype == np.float32
    assert sample.lithology.dtype == np.int8
    assert sample.metadata["schema_version"] == SCHEMA_VERSION


def test_air_voxels_are_negative_background_contrast():
    tile, litho = _toy_tile_and_lithology(nz_voxels=16)
    sample = build_sample_tile(tile, litho, dz_m=30.0, skip_geographic=True)
    air_mask = sample.lithology == AIR_LITHOLOGY
    assert air_mask.any()
    np.testing.assert_allclose(
        sample.target[air_mask], -DEFAULT_BACKGROUND_DENSITY_KG_M3
    )


def test_obs_z_equals_surface_plus_observation_height():
    tile, litho = _toy_tile_and_lithology(nz_voxels=8)
    sample = build_sample_tile(
        tile, litho, dz_m=30.0, observation_height_m=50.0,
        skip_geographic=True,
    )
    np.testing.assert_allclose(sample.obs_z - sample.surface, 50.0)


def test_topography_anchors_to_max_dem():
    tile, litho = _toy_tile_and_lithology(nz_voxels=16)
    sample = build_sample_tile(tile, litho, dz_m=30.0, skip_geographic=True)
    # Voxels above the local DEM should be air.
    z_top = sample.metadata["mesh"]["z_top_m"]
    # Top voxel center elevation:
    z_top_center = z_top - 0.5 * sample.metadata["mesh"]["dz_m"]
    # If z_top_center > local DEM, then top voxel must be air at that pixel
    surface = sample.surface
    expected_air = z_top_center > surface
    actual_air = sample.lithology[0] == AIR_LITHOLOGY
    np.testing.assert_array_equal(actual_air, expected_air)


def test_gravity_is_finite_and_non_zero():
    tile, litho = _toy_tile_and_lithology(nz_voxels=16)
    sample = build_sample_tile(tile, litho, dz_m=30.0, skip_geographic=True)
    assert np.isfinite(sample.gravity).all()
    # Air contrast (-2670) above DEM should pull observation upward, so gz < 0 somewhere
    assert (sample.gravity < 0).any() or (sample.gravity > 0).any()


def test_save_load_round_trip(tmp_path: Path):
    tile, litho = _toy_tile_and_lithology(nz_voxels=8)
    sample = build_sample_tile(tile, litho, dz_m=30.0, skip_geographic=True,
                               density_seed=7)
    out = tmp_path / "tile"
    save_sample_tile(sample, out)
    assert out.with_suffix(".npz").exists()
    assert out.with_suffix(".json").exists()
    loaded = load_sample_tile(out)
    np.testing.assert_array_equal(loaded.surface, sample.surface)
    np.testing.assert_array_equal(loaded.target, sample.target)
    np.testing.assert_array_equal(loaded.gravity, sample.gravity)
    assert loaded.metadata["schema_version"] == SCHEMA_VERSION


def test_sidecar_json_contains_required_mesh_fields(tmp_path: Path):
    tile, litho = _toy_tile_and_lithology(nz_voxels=8)
    sample = build_sample_tile(tile, litho, dz_m=30.0, skip_geographic=True)
    out = tmp_path / "tile"
    save_sample_tile(sample, out)
    meta = json.loads((out.with_suffix(".json")).read_text())
    for key in ("schema_version", "terrain_id", "crs_horizontal", "mesh", "physics"):
        assert key in meta
    for k in ("nx", "ny", "nz", "dx_m", "dy_m", "dz_m"):
        assert k in meta["mesh"]
