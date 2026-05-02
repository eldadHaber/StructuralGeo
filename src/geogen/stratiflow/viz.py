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
                figsize=(16, 9)) -> Path:
    """Render the per-sample preview panel and return its file path."""
    import matplotlib.pyplot as plt

    sd = Path(sample_dir)
    s = _load_sample(sd)
    meta = s["metadata"]
    mesh = meta["mesh"]
    nx, ny, nz = mesh["nx"], mesh["ny"], mesh["nz"]
    dx, dy, dz = mesh["dx_m"], mesh["dy_m"], mesh["dz_m"]
    extent_xy = [0, nx * dx, 0, ny * dy]                 # imshow uses (left, right, bottom, top)
    extent_xz = [0, nx * dx, -nz * dz, 0]                # x-z cross-section
    summary = meta["summary"]

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # --- Panel 1: DEM with faults + boreholes overlaid -------------------
    ax = axes[0, 0]
    im = ax.imshow(s["dem"], cmap="terrain", extent=extent_xy, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Elevation (m)")
    for ft in s["fault_traces"]:
        (x0, y0), (x1, y1) = ft["polyline_xy_m"]
        is_user = any(uf["id"] == ft["id"] for uf in s["user_faults"])
        ax.plot([x0, x1], [y0, y1],
                color="red" if not is_user else "blue",
                lw=2 if not is_user else 1.5,
                ls="-" if not is_user else "--",
                label=("hidden" if not is_user else "user") if ft is s["fault_traces"][0] else None)
    for bh in s["boreholes"]:
        ax.plot(bh["x_m"], bh["y_m"], "o", color="white",
                mec="black", ms=5)
    ax.set_xlim(0, nx * dx); ax.set_ylim(0, ny * dy)
    ax.set_title(f"DEM + faults (red=hidden, blue dashed=user-known)\n"
                 f"+ {len(s['boreholes'])} borehole collars")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # --- Panel 2: Surface contact raster ---------------------------------
    ax = axes[0, 1]
    contact = s["contact"]
    cmap, norm, codes = _categorical_cmap_norm([int(c) for c in np.unique(contact)])
    im = ax.imshow(contact, cmap=cmap, norm=norm, extent=extent_xy, origin="lower",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=codes)
    label_map = {int(k): v["lithology"] for k, v in s["lithology_codes"].items()}
    label_map[-1] = "(air)"
    cbar.ax.set_yticklabels([label_map.get(c, str(c)) for c in codes])
    ax.set_title("Surface contact (top stratigraphic unit)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # --- Panel 3: Gravity grid -------------------------------------------
    ax = axes[0, 2]
    g = s["gravity"]
    vmax = float(np.max(np.abs(g - g.mean())))
    im = ax.imshow(g, cmap="seismic", extent=extent_xy, origin="lower",
                   vmin=g.mean() - vmax, vmax=g.mean() + vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="gz (mGal)")
    ax.set_title(f"Forward gravity  range=[{g.min():.1f}, {g.max():.1f}] mGal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # --- Panel 4: Lithology xz cross-section -----------------------------
    ax = axes[1, 0]
    j_mid = ny // 2
    litho_xz = s["lithology"][:, j_mid, :]   # (nz, nx) z=0 top
    cmap, norm, codes = _categorical_cmap_norm(
        [int(c) for c in np.unique(s["lithology"])]
    )
    im = ax.imshow(litho_xz, cmap=cmap, norm=norm, extent=extent_xz,
                   origin="upper", interpolation="nearest", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=codes)
    cbar.ax.set_yticklabels([label_map.get(c, str(c)) for c in codes])
    ax.set_title(f"Lithology cross-section at y = {j_mid * dy:.0f} m")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")

    # --- Panel 5: Density xz cross-section -------------------------------
    ax = axes[1, 1]
    dens_xz = s["density"][:, j_mid, :]
    vmax = float(np.max(np.abs(dens_xz)))
    im = ax.imshow(dens_xz, cmap="RdBu_r", extent=extent_xz,
                   origin="upper", vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="density contrast (kg/m^3)")
    ax.set_title(f"Density cross-section at y = {j_mid * dy:.0f} m")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")

    # --- Panel 6: Fault distance (top slice) -----------------------------
    ax = axes[1, 2]
    fd_top = s["fault_dist"][0]    # near-surface slice
    im = ax.imshow(fd_top, cmap="magma_r", extent=extent_xy, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="distance to nearest fault (m)")
    for ft in s["fault_traces"]:
        (x0, y0), (x1, y1) = ft["polyline_xy_m"]
        ax.plot([x0, x1], [y0, y1], color="cyan", lw=1.5)
    ax.set_title(f"Fault distance @ top slice  (gt: {len(s['fault_traces'])})")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    title = (
        f"{meta['sample_id']}  seed={meta['seed']}  |  "
        f"{summary['n_units']} units, "
        f"{summary['n_faults']} faults ({summary['n_user_faults']} user-known), "
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
