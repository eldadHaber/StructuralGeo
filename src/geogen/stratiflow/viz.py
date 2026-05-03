"""Quick visual-validation viewer for stratiflow samples.

For each sample, builds a 2x3 matplotlib panel:

    Row 1:  DEM (faults+boreholes overlaid)  | Surface contact map  | Gravity grid
    Row 2:  Lithology xz cross-section       | Density xz cross-section  | Fault distance (top slice)

Saves to ``<sample_dir>/preview.png`` so the user can scroll through all
samples in a single image viewer. The CLI also assembles an
``index_montage.png`` showing every sample in one grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


def _load_sample(sample_dir: Path) -> Dict:
    """Load all arrays + json metadata into a dict."""
    sd = Path(sample_dir)
    out = {
        "dir": sd,
        "lithology": np.load(sd / "ground_truth" / "lithology_volume.npy"),
        "density": np.load(sd / "ground_truth" / "density_volume.npy"),
        "dem": np.load(sd / "inputs" / "dem.npy"),
        "obs_z": np.load(sd / "inputs" / "obs_z.npy"),
        "gravity": np.load(sd / "inputs" / "gravity_grid.npy"),
        "contact": np.load(sd / "inputs" / "surface_contact_raster.npy"),
        "fault_dist": np.load(sd / "expected_outputs" / "fault_distance.npy"),
        "metadata": json.loads((sd / "ground_truth" / "metadata.json").read_text()),
        "lithology_codes": json.loads(
            (sd / "ground_truth" / "lithology_codes.json").read_text()
        ),
        "fault_traces": json.loads(
            (sd / "ground_truth" / "fault_traces.json").read_text()
        )["faults"],
        "user_faults": json.loads(
            (sd / "inputs" / "user_fault_traces.json").read_text()
        )["faults"],
        "boreholes": json.loads(
            (sd / "inputs" / "boreholes.json").read_text()
        )["boreholes"],
    }
    s2_path = sd / "inputs" / "satellite_rgb.npy"
    out["satellite_rgb"] = np.load(s2_path) if s2_path.exists() else None
    return out


def _hillshade(dem: np.ndarray, az_deg: float = 315.0, alt_deg: float = 45.0,
               pixel_m: float = 50.0) -> np.ndarray:
    az = np.deg2rad(360.0 - az_deg + 90.0)
    alt = np.deg2rad(alt_deg)
    gy, gx = np.gradient(dem.astype(np.float32), pixel_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    return np.clip(
        np.sin(alt) * np.cos(slope)
        + np.cos(alt) * np.sin(slope) * np.cos(az - aspect),
        0.0, 1.0,
    )


def _stretch_rgb(rgb: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """Per-band percentile stretch to [0, 1] for display, NaN-safe."""
    out = np.empty_like(rgb)
    for c in range(rgb.shape[-1]):
        band = rgb[..., c]
        if not np.isfinite(band).any():
            out[..., c] = 0.0
            continue
        lo, hi = np.nanpercentile(band, [p_lo, p_hi])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
            out[..., c] = 0.0
            continue
        scaled = (band - lo) / (hi - lo)
        scaled = np.where(np.isfinite(scaled), scaled, 0.0)
        out[..., c] = np.clip(scaled, 0.0, 1.0)
    return out


def _categorical_cmap_norm(codes: List[int]):
    """Build a discrete colormap + norm for a list of integer lithology codes.

    Air (-1) is rendered light gray. Other codes get tab10/tab20 colors.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    sorted_codes = sorted(set(codes))
    base = plt.get_cmap("tab10").colors + plt.get_cmap("tab20").colors
    palette = []
    for c in sorted_codes:
        if c == -1:
            palette.append((0.85, 0.85, 0.85))
        else:
            idx = sorted_codes.index(c) % len(base)
            palette.append(base[idx])
    cmap = ListedColormap(palette)
    boundaries = [c - 0.5 for c in sorted_codes] + [sorted_codes[-1] + 0.5]
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, sorted_codes


def plot_sample(sample_dir: Path, out_path: Optional[Path] = None,
                figsize=(20, 10)) -> Path:
    """Render the per-sample preview panel (2x4 with satellite when available)."""
    import matplotlib.pyplot as plt

    sd = Path(sample_dir)
    s = _load_sample(sd)
    meta = s["metadata"]
    mesh = meta["mesh"]
    nx, ny, nz = mesh["nx"], mesh["ny"], mesh["nz"]
    dx, dy, dz = mesh["dx_m"], mesh["dy_m"], mesh["dz_m"]
    extent_xy = [0, nx * dx, 0, ny * dy]
    extent_xz = [0, nx * dx, -nz * dz, 0]
    summary = meta["summary"]
    label_map = {int(k): v["lithology"] for k, v in s["lithology_codes"].items()}
    label_map[-1] = "(air)"

    fig, axes = plt.subplots(2, 4, figsize=figsize)

    # --- Panel 1: Sentinel-2 RGB (or hillshade fallback) -----------------
    ax = axes[0, 0]
    if s["satellite_rgb"] is not None:
        rgb_arr = np.transpose(s["satellite_rgb"], (1, 2, 0))  # (ny, nx, 3)
        rgb_disp = _stretch_rgb(rgb_arr)
        nodata = ~np.all(np.isfinite(rgb_arr), axis=-1)
        coverage = 100.0 * (1.0 - nodata.mean())
        # Hillshade undercoat so missing pixels still show terrain
        ax.imshow(_hillshade(s["dem"], pixel_m=dx), cmap="gray",
                  extent=extent_xy, origin="lower", alpha=0.5)
        # The image is row-major; origin="lower" + flip vertically so geographic
        # north is up (matches DEM panel orientation).
        ax.imshow(np.flipud(rgb_disp), extent=extent_xy, origin="lower",
                  alpha=np.where(np.flipud(nodata), 0.0, 1.0))
        title = "Sentinel-2 RGB (true-color)"
        if coverage < 99.0:
            title += f"  [{coverage:.0f}% cov]"
        ax.set_title(title)
    else:
        ax.imshow(_hillshade(s["dem"], pixel_m=dx), cmap="gray",
                  extent=extent_xy, origin="lower")
        ax.set_title("DEM hillshade (no satellite imagery)")
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 2: DEM color (terrain) ------------------------------------
    ax = axes[0, 1]
    im = ax.imshow(s["dem"], cmap="terrain", extent=extent_xy, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Elevation (m)")
    ax.set_title(f"Topography (relief={float(np.ptp(s['dem'])):.0f} m)")
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 3: DEM hillshade + faults + boreholes ---------------------
    ax = axes[0, 2]
    ax.imshow(_hillshade(s["dem"], pixel_m=dx), cmap="gray",
              extent=extent_xy, origin="lower")
    for ft in s["fault_traces"]:
        (x0, y0), (x1, y1) = ft["polyline_xy_m"]
        is_user = any(uf["id"] == ft["id"] for uf in s["user_faults"])
        ax.plot([x0, x1], [y0, y1],
                color="red" if not is_user else "blue",
                lw=2 if not is_user else 1.5,
                ls="-" if not is_user else "--")
    for bh in s["boreholes"]:
        ax.plot(bh["x_m"], bh["y_m"], "o", color="yellow", mec="black", ms=5)
    ax.set_xlim(0, nx * dx); ax.set_ylim(0, ny * dy)
    ax.set_title("Hillshade + faults (red=hidden, blue dashed=known) + BHs")
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 4: Gravity grid -------------------------------------------
    ax = axes[0, 3]
    g = s["gravity"]
    vmax = float(np.max(np.abs(g - g.mean())))
    im = ax.imshow(g, cmap="seismic", extent=extent_xy, origin="lower",
                   vmin=g.mean() - vmax, vmax=g.mean() + vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="gz (mGal)")
    ax.set_title(f"Gravity  [{g.min():.1f}, {g.max():.1f}] mGal")
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 5: Surface contact raster ---------------------------------
    ax = axes[1, 0]
    contact = s["contact"]
    cmap, norm, codes = _categorical_cmap_norm([int(c) for c in np.unique(contact)])
    im = ax.imshow(contact, cmap=cmap, norm=norm, extent=extent_xy, origin="lower",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=codes)
    cbar.ax.set_yticklabels([label_map.get(c, str(c)) for c in codes])
    ax.set_title("Surface contact (top unit)")
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 6: Lithology xz cross-section -----------------------------
    ax = axes[1, 1]
    j_mid = ny // 2
    litho_xz = s["lithology"][:, j_mid, :]
    cmap, norm, codes = _categorical_cmap_norm(
        [int(c) for c in np.unique(s["lithology"])]
    )
    im = ax.imshow(litho_xz, cmap=cmap, norm=norm, extent=extent_xz,
                   origin="upper", interpolation="nearest", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=codes)
    cbar.ax.set_yticklabels([label_map.get(c, str(c)) for c in codes])
    ax.set_title(f"Lithology xz @ y={j_mid * dy:.0f} m")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")

    # --- Panel 7: Density xz cross-section -------------------------------
    ax = axes[1, 2]
    dens_xz = s["density"][:, j_mid, :]
    vmax = float(np.max(np.abs(dens_xz)))
    if vmax < 1e-6:
        vmax = 1.0
    im = ax.imshow(dens_xz, cmap="RdBu_r", extent=extent_xz,
                   origin="upper", vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="density contrast (kg/m^3)")
    ax.set_title(f"Density xz @ y={j_mid * dy:.0f} m")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")

    # --- Panel 8: Fault distance (top slice) -----------------------------
    ax = axes[1, 3]
    fd_top = s["fault_dist"][0]
    im = ax.imshow(fd_top, cmap="magma_r", extent=extent_xy, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="dist to fault (m)")
    for ft in s["fault_traces"]:
        (x0, y0), (x1, y1) = ft["polyline_xy_m"]
        ax.plot([x0, x1], [y0, y1], color="cyan", lw=1.5)
    ax.set_title(f"Fault distance (gt: {len(s['fault_traces'])})")
    ax.set_xticks([]); ax.set_yticks([])

    region = ""
    prov = meta.get("provenance")
    if prov:
        region = f"  |  {prov['region_name']}  ({prov['center_lonlat'][0]:.3f}, {prov['center_lonlat'][1]:.3f})"
    title = (
        f"{meta['sample_id']}  seed={meta['seed']}{region}  |  "
        f"{summary['n_units']} units, "
        f"{summary['n_faults']} faults ({summary['n_user_faults']} known), "
        f"{summary['n_boreholes']} BHs, "
        f"relief {summary['surface_relief_m']:.0f} m"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if out_path is None:
        out_path = sd / "preview.png"
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all(samples_root: Path, montage_path: Optional[Path] = None) -> Dict:
    """Render previews for all sample subdirectories under ``samples_root``.

    Also writes a small montage image stitching every per-sample PNG in a
    grid for at-a-glance review. Returns a dict of paths.
    """
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    root = Path(samples_root)
    sample_dirs = sorted(p for p in root.iterdir()
                         if p.is_dir() and (p / "ground_truth").is_dir())
    if not sample_dirs:
        raise FileNotFoundError(f"No sample subdirs under {root}")

    paths = {}
    for sd in sample_dirs:
        try:
            paths[sd.name] = plot_sample(sd)
            print(f"  rendered {sd.name} -> {paths[sd.name]}")
        except Exception as e:
            print(f"  FAILED {sd.name}: {type(e).__name__}: {e}")

    # Build montage: vertical stack of per-sample previews
    if montage_path is None:
        montage_path = root / "montage.png"
    pngs = [paths[k] for k in sorted(paths)]
    if not pngs:
        return {"per_sample": paths, "montage": None}
    imgs = [imread(p) for p in pngs]
    fig, axes = plt.subplots(len(imgs), 1, figsize=(16, 9 * len(imgs)))
    if len(imgs) == 1:
        axes = [axes]
    for ax, img, p in zip(axes, imgs, pngs):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(Path(p).parent.name, fontsize=14, loc="left")
    fig.tight_layout()
    fig.savefig(montage_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return {"per_sample": paths, "montage": montage_path}
