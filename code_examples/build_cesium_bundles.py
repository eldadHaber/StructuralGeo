"""Pre-process stratiflow NZ samples into Cesium-friendly bundles.

For each ``stratiflow_samples_nz/sample_NNN/``, writes a self-contained
bundle under ``cesium_viewer/samples/sample_NNN/`` containing:

  * ``meta.json``           sample summary + WGS84 bbox + center
  * ``imagery.png``         Sentinel-2 RGB true-color
  * ``dem_color.png``       DEM colored for surface display
  * ``litho_xz.png``        lithology cross-section at y=center (categorical)
  * ``density_xz.png``      density-contrast cross-section at y=center
  * ``litho_yz.png``        lithology cross-section at x=center
  * ``density_yz.png``      density-contrast cross-section at x=center
  * ``faults.geojson``      surface fault polylines in EPSG:4326 with
                            dip/throw/source attributes
  * ``boreholes.json``      collars in WGS84 + total_depth_m and dip/azimuth

The viewer (cesium_viewer/index.html) reads these directly. NZTM coords
are converted to WGS84 lon/lat via pyproj.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np


def _save_rgb_png(arr_3hw: np.ndarray, path: Path) -> None:
    """Save (3, H, W) float [0,1] array as 8-bit RGB PNG."""
    from PIL import Image
    rgb = np.transpose(arr_3hw, (1, 2, 0))
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path, optimize=True)


def _stretch_per_band(rgb_3hw: np.ndarray, p_lo=2.0, p_hi=98.0) -> np.ndarray:
    out = np.empty_like(rgb_3hw)
    for c in range(rgb_3hw.shape[0]):
        b = rgb_3hw[c]
        finite = b[np.isfinite(b)]
        if finite.size == 0:
            out[c] = 0.0; continue
        lo, hi = np.percentile(finite, [p_lo, p_hi])
        if hi - lo < 1e-6:
            out[c] = 0.0; continue
        out[c] = np.clip((b - lo) / (hi - lo), 0, 1)
    return out


def _categorical_palette(codes):
    """Return a (len(codes), 3) palette for unique integer codes (incl -1=air)."""
    import matplotlib.pyplot as plt
    base = np.array(plt.get_cmap("tab10").colors + plt.get_cmap("tab20").colors)
    out = np.zeros((len(codes), 3), dtype=np.float32)
    for i, c in enumerate(codes):
        if c == -1:
            out[i] = (0.85, 0.85, 0.85)
        else:
            out[i] = base[i % len(base)]
    return out


def _save_categorical_png(slice2d: np.ndarray, path: Path) -> None:
    """Save a 2D categorical array as colored PNG."""
    from PIL import Image
    codes = sorted(np.unique(slice2d).tolist())
    palette = _categorical_palette(codes)
    code_to_idx = {c: i for i, c in enumerate(codes)}
    img = np.zeros((*slice2d.shape, 3), dtype=np.float32)
    for c, i in code_to_idx.items():
        img[slice2d == c] = palette[i]
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(path, optimize=True)


def _save_diverging_png(arr2d: np.ndarray, path: Path,
                        vmin=None, vmax=None, cmap_name="RdBu_r") -> None:
    from PIL import Image
    import matplotlib.cm as cm
    if vmin is None:
        v = float(np.max(np.abs(arr2d))) or 1.0
        vmin, vmax = -v, v
    norm = (arr2d - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path, optimize=True)


def _save_terrain_png(dem: np.ndarray, path: Path) -> None:
    from PIL import Image
    import matplotlib.cm as cm
    dmin, dmax = float(dem.min()), float(dem.max())
    rng = max(dmax - dmin, 1e-6)
    norm = (dem - dmin) / rng
    cmap = cm.get_cmap("terrain")
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path, optimize=True)


def _exposed_pos(mask: np.ndarray, axis: int) -> np.ndarray:
    """True where voxel is in mask AND its +axis neighbour is False / out of bounds."""
    nbr = np.roll(mask, -1, axis=axis)
    if axis == 0:
        nbr[-1, :, :] = False
    elif axis == 1:
        nbr[:, -1, :] = False
    elif axis == 2:
        nbr[:, :, -1] = False
    return mask & ~nbr


def _exposed_neg(mask: np.ndarray, axis: int) -> np.ndarray:
    nbr = np.roll(mask, 1, axis=axis)
    if axis == 0:
        nbr[0, :, :] = False
    elif axis == 1:
        nbr[:, 0, :] = False
    elif axis == 2:
        nbr[:, :, 0] = False
    return mask & ~nbr


def _extract_unit_boundary_mesh(
    volume_zyx: np.ndarray, code: int,
    dx: float, dy: float, dz: float,
):
    """Return (positions_local, indices) for the boundary mesh of one unit.

    positions_local: float32 (N, 3) in spec local frame (x_local in
        [0, nx*dx], y_local in [0, ny*dy], z_local <= 0 with elevation
        decreasing downward). Returns ``(None, None)`` if the unit has no
        voxels.
    """
    mask = (volume_zyx == code)
    if not mask.any():
        return None, None

    # Six face directions: +z (axis 0 wraps "downward in spec"), etc.
    axes = [
        ("+x", 2, _exposed_pos),
        ("-x", 2, _exposed_neg),
        ("+y", 1, _exposed_pos),
        ("-y", 1, _exposed_neg),
        ("+z", 0, _exposed_pos),     # face below voxel (deeper)
        ("-z", 0, _exposed_neg),     # face above voxel (shallower)
    ]

    pos_chunks = []
    idx_chunks = []
    base = 0
    for name, axis, fn in axes:
        exposed = fn(mask, axis)
        idx = np.argwhere(exposed)
        n = idx.shape[0]
        if n == 0:
            continue
        ks, js, is_ = idx.T.astype(np.float32)
        x0 = is_ * dx
        x1 = (is_ + 1) * dx
        y0 = js * dy
        y1 = (js + 1) * dy
        # Spec frame: z=0 at top, increasing downward. Voxel k has elevation
        # range [-(k+1)*dz, -k*dz].
        z_high = -ks * dz             # top of voxel
        z_low = -(ks + 1) * dz        # bottom of voxel

        if name == "+x":
            v0 = np.column_stack([x1, y0, z_low])
            v1 = np.column_stack([x1, y1, z_low])
            v2 = np.column_stack([x1, y1, z_high])
            v3 = np.column_stack([x1, y0, z_high])
        elif name == "-x":
            v0 = np.column_stack([x0, y1, z_low])
            v1 = np.column_stack([x0, y0, z_low])
            v2 = np.column_stack([x0, y0, z_high])
            v3 = np.column_stack([x0, y1, z_high])
        elif name == "+y":
            v0 = np.column_stack([x1, y1, z_low])
            v1 = np.column_stack([x0, y1, z_low])
            v2 = np.column_stack([x0, y1, z_high])
            v3 = np.column_stack([x1, y1, z_high])
        elif name == "-y":
            v0 = np.column_stack([x0, y0, z_low])
            v1 = np.column_stack([x1, y0, z_low])
            v2 = np.column_stack([x1, y0, z_high])
            v3 = np.column_stack([x0, y0, z_high])
        elif name == "+z":     # bottom face (lower elevation = z_low)
            v0 = np.column_stack([x0, y0, z_low])
            v1 = np.column_stack([x1, y0, z_low])
            v2 = np.column_stack([x1, y1, z_low])
            v3 = np.column_stack([x0, y1, z_low])
        elif name == "-z":     # top face (higher elevation = z_high)
            v0 = np.column_stack([x0, y0, z_high])
            v1 = np.column_stack([x0, y1, z_high])
            v2 = np.column_stack([x1, y1, z_high])
            v3 = np.column_stack([x1, y0, z_high])

        pos = np.empty((n * 4, 3), dtype=np.float32)
        pos[0::4] = v0; pos[1::4] = v1; pos[2::4] = v2; pos[3::4] = v3

        idx_arr = np.empty((n, 6), dtype=np.uint32)
        face_base = (np.arange(n, dtype=np.uint32) * 4) + np.uint32(base)
        idx_arr[:, 0] = face_base
        idx_arr[:, 1] = face_base + 1
        idx_arr[:, 2] = face_base + 2
        idx_arr[:, 3] = face_base
        idx_arr[:, 4] = face_base + 2
        idx_arr[:, 5] = face_base + 3

        pos_chunks.append(pos)
        idx_chunks.append(idx_arr.reshape(-1, 3))
        base += n * 4

    if not pos_chunks:
        return None, None
    positions = np.concatenate(pos_chunks, axis=0)
    indices = np.concatenate(idx_chunks, axis=0)
    return positions, indices


def _dedupe_vertices(positions: np.ndarray, indices: np.ndarray,
                     quantum_m: float = 0.05):
    """Merge coincident vertices (boundary extraction emits 4 per face corner).

    Vertices are quantised to ``quantum_m`` metres to handle float jitter,
    then ``np.unique`` collapses duplicates. The triangle indices are
    remapped through the inverse mapping. Returns ``(positions, indices)``
    with deduplicated vertex set.
    """
    rounded = np.round(positions / quantum_m).astype(np.int64)
    _, first_idx, inverse = np.unique(
        rounded, axis=0, return_index=True, return_inverse=True,
    )
    new_positions = positions[first_idx]
    new_indices = inverse[indices.ravel()].reshape(indices.shape).astype(np.uint32)
    return new_positions.astype(positions.dtype), new_indices


def _taubin_smooth(positions: np.ndarray, indices: np.ndarray,
                   n_iters: int = 10, lam: float = 0.5, mu: float = -0.53):
    """Taubin volume-preserving Laplacian smoothing.

    Alternates one Laplacian pass with positive ``lam`` (smooth/shrink) and
    one with slightly larger negative ``mu`` (re-inflate). The standard
    parameters (lam=0.5, mu=-0.53) keep the mesh's bulk near-constant.
    """
    from scipy.sparse import coo_matrix
    n = positions.shape[0]
    if n == 0 or indices.shape[0] == 0 or n_iters <= 0:
        return positions
    # Symmetric adjacency from triangle edges
    e = np.concatenate([
        indices[:, [0, 1]], indices[:, [1, 2]], indices[:, [2, 0]],
        indices[:, [1, 0]], indices[:, [2, 1]], indices[:, [0, 2]],
    ], axis=0)
    e = np.unique(e, axis=0)
    rows = e[:, 0]; cols = e[:, 1]
    A = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.float32), (rows, cols)),
        shape=(n, n),
    ).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    pos = positions.astype(np.float32, copy=True)
    for i in range(n_iters):
        delta = (A @ pos) / deg[:, None] - pos
        pos = pos + (lam if i % 2 == 0 else mu) * delta
    return pos


def _extract_subsurface_units(
    litho_zyx: np.ndarray,
    palette: dict,
    transformer,
    ox: float, oy: float,
    dx: float, dy: float, dz: float,
    z_top_abs_m: float,
    smoothing_iters: int = 30,
):
    """Run boundary extraction for every present non-air unit and convert each
    mesh's vertices to (lon, lat, abs_elevation) for direct ingestion in JS."""
    payload = []
    present = sorted(int(c) for c in np.unique(litho_zyx) if int(c) >= 0)
    for code in present:
        info = palette.get(str(code))
        if info is None:
            continue
        positions_local, indices = _extract_unit_boundary_mesh(
            litho_zyx, code, dx, dy, dz,
        )
        if positions_local is None:
            continue
        # Dedupe coincident vertices (4-per-corner from face emission), then
        # apply Taubin smoothing in the local x/y/z metric frame.
        positions_local, indices = _dedupe_vertices(positions_local, indices)
        if smoothing_iters > 0:
            positions_local = _taubin_smooth(
                positions_local, indices, n_iters=smoothing_iters,
            )
        # local x, y -> absolute NZTM -> WGS84 lon/lat
        x_abs = ox + positions_local[:, 0]
        y_abs = oy + positions_local[:, 1]
        lon, lat = transformer.transform(x_abs, y_abs)
        h_abs = z_top_abs_m + positions_local[:, 2]
        positions_lonlatH = np.column_stack([
            np.asarray(lon, dtype=np.float32),
            np.asarray(lat, dtype=np.float32),
            np.asarray(h_abs, dtype=np.float32),
        ])
        payload.append({
            "code": code,
            "name": info["name"],
            "lithology": info["lithology"],
            "color_hex": info["color_hex"],
            "density_kg_m3": info["density_kg_m3"],
            "positions_lonlatH": positions_lonlatH,
            "indices": indices,
        })
    return payload


def _nztm_to_wgs84(transformer, x_m: float, y_m: float):
    """NZTM (EPSG:2193) -> WGS84 (EPSG:4326). Returns (lon, lat)."""
    return transformer.transform(x_m, y_m)


def _bbox_wgs84_corners(transformer, xmin, ymin, xmax, ymax):
    """Return polygon ring (lon, lat) for the bbox corners (NZTM rectangle is
    not a perfect rectangle in lon/lat, so emit four corners)."""
    return [
        list(_nztm_to_wgs84(transformer, xmin, ymin)),
        list(_nztm_to_wgs84(transformer, xmax, ymin)),
        list(_nztm_to_wgs84(transformer, xmax, ymax)),
        list(_nztm_to_wgs84(transformer, xmin, ymax)),
    ]


def process_sample(sample_dir: Path, out_dir: Path) -> dict:
    """Build a Cesium bundle from one stratiflow sample."""
    import pyproj

    sample_dir = Path(sample_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((sample_dir / "ground_truth" / "metadata.json").read_text())
    geo = meta["geospatial"]
    mesh = meta["mesh"]
    if not geo.get("crs"):
        raise ValueError(f"{sample_dir}: no CRS in geospatial metadata")
    crs_src = geo["crs"]
    transformer = pyproj.Transformer.from_crs(crs_src, "EPSG:4326", always_xy=True)

    nx, ny, nz = mesh["nx"], mesh["ny"], mesh["nz"]
    dx, dy, dz = mesh["dx_m"], mesh["dy_m"], mesh["dz_m"]
    ox, oy = geo["origin_xy_absolute_m"]
    xmin, ymin, xmax, ymax = geo["bbox_xy_absolute_m"]
    z_top_abs = float(geo.get("z_top_m_absolute") or 0.0)

    bbox_polygon = _bbox_wgs84_corners(transformer, xmin, ymin, xmax, ymax)
    center_lon, center_lat = _nztm_to_wgs84(
        transformer, 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    )

    # ---- Imagery ----
    s2_path = sample_dir / "inputs" / "satellite_rgb.npy"
    has_imagery = s2_path.exists()
    if has_imagery:
        rgb = np.load(s2_path)             # (3, ny, nx)
        rgb_disp = _stretch_per_band(rgb)
        # PNG image origin is top-left; raster row 0 should be the *northern*
        # edge of the bbox so it aligns when Cesium maps it via SingleTileImagery.
        _save_rgb_png(rgb_disp[:, ::-1, :], out_dir / "imagery.png")

    # ---- DEM colored ----
    dem = np.load(sample_dir / "inputs" / "dem.npy")        # (ny, nx) shifted (max=0)
    _save_terrain_png(dem[::-1], out_dir / "dem_color.png")

    # ---- Cross-sections ----
    litho = np.load(sample_dir / "ground_truth" / "lithology_volume.npy")  # (nz, ny, nx)
    density = np.load(sample_dir / "ground_truth" / "density_volume.npy")
    j_mid = ny // 2
    i_mid = nx // 2
    _save_categorical_png(litho[:, j_mid, :], out_dir / "litho_xz.png")
    _save_diverging_png(density[:, j_mid, :], out_dir / "density_xz.png")
    _save_categorical_png(litho[:, :, i_mid], out_dir / "litho_yz.png")
    _save_diverging_png(density[:, :, i_mid], out_dir / "density_yz.png")

    # ---- Faults as GeoJSON (in WGS84, surface clamped) ----
    faults_payload = json.loads(
        (sample_dir / "ground_truth" / "fault_traces.json").read_text()
    )["faults"]
    user_faults = json.loads(
        (sample_dir / "inputs" / "user_fault_traces.json").read_text()
    )["faults"]
    user_ids = {f["id"] for f in user_faults}
    geo_features = []
    for ft in faults_payload:
        coords = []
        for x_local, y_local in ft["polyline_xy_m"]:
            lon, lat = _nztm_to_wgs84(transformer, ox + x_local, oy + y_local)
            coords.append([lon, lat])
        geo_features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "id": ft["id"],
                "dip_deg": ft["dip_degrees"],
                "dip_dir_deg": ft["dip_direction_degrees"],
                "throw_m": ft["throw_m"],
                "amplitude_m": ft["amplitude_m"],
                "is_user_known": ft["id"] in user_ids,
            },
        })
    (out_dir / "faults.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "features": geo_features,
    }, indent=2))

    # ---- Boreholes (with per-sample lithology so the viewer can colour sticks) ----
    bhs = json.loads(
        (sample_dir / "inputs" / "boreholes.json").read_text()
    )["boreholes"]
    bh_out = []
    for bh in bhs:
        lon, lat = _nztm_to_wgs84(
            transformer, ox + bh["x_m"], oy + bh["y_m"]
        )
        # Forward only the fields the viewer needs from each downhole sample.
        samples_lite = [
            {
                "start_depth_m": float(s["start_depth_m"]),
                "length_m": float(s["length_m"]),
                "lithology_code": int(s["lithology_code"]),
            }
            for s in bh.get("samples", [])
        ]
        bh_out.append({
            "id": bh["id"],
            "lon": float(lon), "lat": float(lat),
            "collar_elevation_m": z_top_abs + bh["elevation_m"],
            "azimuth_deg": bh["azimuth_deg"],
            "dip_deg": bh["dip_deg"],
            "total_depth_m": bh["total_depth_m"],
            "n_samples": len(bh["samples"]),
            "samples": samples_lite,
        })
    (out_dir / "boreholes.json").write_text(json.dumps({"boreholes": bh_out}, indent=2))

    # ---- Lithology palette (code -> {name, color hex, density, susceptibility}) ----
    litho_codes_meta = json.loads(
        (sample_dir / "ground_truth" / "lithology_codes.json").read_text()
    )
    palette_codes = sorted(int(k) for k in litho_codes_meta.keys())
    palette_arr = _categorical_palette(palette_codes)
    palette = {}
    for c, color in zip(palette_codes, palette_arr):
        info = litho_codes_meta[str(c)]
        palette[str(c)] = {
            "name": info["name"],
            "lithology": info["lithology"],
            "density_kg_m3": info["density_kg_m3"],
            "susceptibility_si": info["susceptibility_si"],
            "color_hex": "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255),
            ),
        }
    palette["-1"] = {"name": "air", "lithology": "(air)",
                    "color_hex": "#dadada", "density_kg_m3": 0.0, "susceptibility_si": 0.0}
    (out_dir / "lithology_palette.json").write_text(json.dumps(palette, indent=2))

    # ---- Per-unit subsurface mesh (boundary-face extraction) ----
    units_payload = _extract_subsurface_units(
        litho, palette, transformer, ox, oy, dx, dy, dz, z_top_abs,
    )
    if units_payload:
        # Pack into one binary blob + JSON header so the viewer does one fetch.
        offsets = []
        chunks = []
        cursor = 0
        for u in units_payload:
            pos = u["positions_lonlatH"]   # float32 [N, 3]
            idx = u["indices"]             # uint32 [M, 3]
            offsets.append({
                "code": int(u["code"]),
                "name": u["name"],
                "lithology": u["lithology"],
                "color_hex": u["color_hex"],
                "density_kg_m3": u["density_kg_m3"],
                "position_offset_bytes": int(cursor),
                "vertex_count": int(pos.shape[0]),
                "index_offset_bytes": int(cursor + pos.nbytes),
                "triangle_count": int(idx.shape[0]),
            })
            chunks.append(pos.tobytes(order="C"))
            chunks.append(idx.tobytes(order="C"))
            cursor += pos.nbytes + idx.nbytes
        (out_dir / "units.bin").write_bytes(b"".join(chunks))
        (out_dir / "units_meta.json").write_text(json.dumps({
            "units": offsets,
            "byte_order": "little",
        }, indent=2))

    # ---- Raw DEM as a binary float32 (for building a topography mesh in JS) ----
    # Stratiflow DEM is in *local* metres relative to z_top=0 (max elevation = 0).
    # We add the absolute offset so the viewer can use absolute heights directly.
    dem_abs = (dem.astype(np.float32) + z_top_abs).astype(np.float32)
    (out_dir / "dem.bin").write_bytes(dem_abs.tobytes(order="C"))
    (out_dir / "dem_meta.json").write_text(json.dumps({
        "ny": int(dem.shape[0]),
        "nx": int(dem.shape[1]),
        "dtype": "float32",
        "byte_order": "little",
        "row_major": True,
        "row0_is_north_edge": True,
        "min_m_absolute": float(dem_abs.min()),
        "max_m_absolute": float(dem_abs.max()),
        "z_top_m_absolute": z_top_abs,
    }, indent=2))

    # ---- Summary meta ----
    summary_meta = {
        "sample_id": meta["sample_id"],
        "seed": meta["seed"],
        "region": meta.get("provenance", {}).get("region_name"),
        "center_lonlat": [float(center_lon), float(center_lat)],
        "bbox_polygon_lonlat": bbox_polygon,
        "mesh": {"nx": nx, "ny": ny, "nz": nz, "dx_m": dx, "dy_m": dy, "dz_m": dz},
        "z_top_m_absolute": z_top_abs,
        "absolute_dem_offset_m": meta.get("provenance", {}).get("absolute_dem_offset_m"),
        "summary": meta["summary"],
        "has_imagery": has_imagery,
    }
    (out_dir / "meta.json").write_text(json.dumps(summary_meta, indent=2))
    return summary_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", default="stratiflow_samples_nz")
    parser.add_argument("--out", default="cesium_viewer/samples")
    args = parser.parse_args()

    in_root = Path(args.in_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(p for p in in_root.iterdir()
                         if p.is_dir() and (p / "ground_truth").is_dir())
    index = []
    for sd in sample_dirs:
        try:
            summary = process_sample(sd, out_root / sd.name)
            index.append({
                "sample_id": summary["sample_id"],
                "region": summary.get("region"),
                "center_lonlat": summary["center_lonlat"],
                "summary": summary["summary"],
                "has_imagery": summary["has_imagery"],
            })
            print(f"  built {sd.name}")
        except Exception as e:
            print(f"  FAILED {sd.name}: {type(e).__name__}: {e}")
    (out_root / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nWrote index: {out_root / 'index.json'} ({len(index)} samples)")


if __name__ == "__main__":
    main()
