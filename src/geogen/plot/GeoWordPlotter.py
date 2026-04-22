"""
A multi-plotter PyVista window offering viewing of GeoWord sentences in 3D.

Allows for rapid sampling and viewing of histories generated from GeoWords.
Simply pass in a sentence (list of GeoWords) and the plotter will generate and plot samples.

Shortcut keys:
- 'r': Refresh the samples.
- '1': View the samples in volume mode.
- '2': View the samples in orthogonal slice mode.
- '3': View the samples in n-slice mode.
- '4': View the samples in one-slice mode.

Plotter Parameters:
- sentence (list): A list of GeoWords (a sentence or Geostory) to generate histories from.
- bounds (tuple): The bounds of the model in the form ((xmin, xmax), (ymin, ymax), (zmin, zmax))
- res (tuple): The resolution of the model in the form (nx, ny, nz)
- n_samples (int): The number of samples to generate and plot. Plotter defaults to square grid layout.
"""

import numpy as np
from pyvista import Box
from pyvistaqt import BackgroundPlotter

import geogen.plot as geovis
from geogen.generation import InfiniteBasement, InfiniteSedimentUniform, generate_history
from geogen.model import GeoModel


class GeoWordPlotter:
    def __init__(self, sentence, bounds, res, n_samples=16, clim=(0, 12)):
        self.sentence = sentence
        self.bounds = bounds
        self.res = res
        self.n_samples = n_samples
        self.models_cache = []
        self.current_view_mode = self.volview
        self.plotter = None
        self.clim = clim
        self.cmap = "gist_ncar"
        self.initialize_plotter()

    def initialize_plotter(self):
        rows, cols = self.calculate_grid_dims(self.n_samples)
        self.plotter = BackgroundPlotter(shape=(rows, cols))
        self.plotter.raise_()  # Bring plotter to front focus window
        self.update_samples()

        # Bind keys
        self.plotter.add_key_event("r", self.refresh_samples)
        self.plotter.add_key_event("1", lambda: self.set_view_mode(self.volview))
        self.plotter.add_key_event("2", lambda: self.set_view_mode(self.orthsliceview))
        self.plotter.add_key_event("3", lambda: self.set_view_mode(self.nsliceview))
        self.plotter.add_key_event("4", lambda: self.set_view_mode(self.onesliceview))

        # Start the Qt application event loop
        self.plotter.app.exec_()

    def calculate_grid_dims(self, n):
        """Calculate grid dimensions that are as square as possible."""
        sqrt_n = np.sqrt(n)
        rows = np.ceil(sqrt_n)
        cols = rows
        return int(rows), int(cols)

    def refresh_samples(self):
        """Refresh the samples using the current view mode."""
        self.models_cache.clear()  # Clear cache so new models are generated
        self.update_samples()
        print("Updated samples.")

    def set_view_mode(self, view_mode):
        """Set the current view mode and refresh samples."""
        self.current_view_mode = view_mode
        self.update_samples(use_cache=True)  # Use cached models to avoid recomputation

    def update_samples(self, use_cache=False):
        """Update and plot the samples with the current view mode."""
        if not use_cache or not self.models_cache:

            for _ in range(self.n_samples):
                hist = generate_history(self.sentence)
                model = GeoModel(bounds=self.bounds, resolution=self.res)
                model.add_history(hist)
                model.compute_model(normalize=True, keep_snapshots=False)
                self.models_cache.append(model)

        self.plotter.clear_actors()
        self.plotter.clear_plane_widgets()

        for i, model in enumerate(self.models_cache):
            row, col = divmod(i, self.plotter.shape[1])
            self.plotter.subplot(row, col)
            self.current_view_mode(model, plotter=self.plotter)
            self.add_bounding_box(model, plotter=self.plotter)

        self.plotter.link_views()
        self.plotter.add_scalar_bar(title="Scalar Bar", n_labels=4, vertical=True, fmt="%.0f")

        self.plotter.render()

    def volview(self, model, plotter):
        mesh = geovis.get_voxel_grid_from_model(model)
        plotter.add_mesh(
            mesh,
            scalars="values",
            show_scalar_bar=False,
            cmap=self.cmap,
            clim=self.clim,
        )

    def orthsliceview(self, model, plotter=None):
        mesh = geovis.get_voxel_grid_from_model(model)
        plotter.add_mesh_slice_orthogonal(mesh, scalars="values", show_scalar_bar=False, cmap=self.cmap)

    def nsliceview(self, model, n=5, axis="x", plotter=None):
        mesh = geovis.get_voxel_grid_from_model(model)
        slices = mesh.slice_along_axis(n=n, axis=axis)
        plotter.add_mesh(slices, scalars="values", show_scalar_bar=False, cmap=self.cmap)
        plotter.add_axes(line_width=5)

    def onesliceview(self, model, plotter=None):
        mesh = geovis.get_voxel_grid_from_model(model)
        skin = mesh.extract_surface()
        plotter.add_mesh_slice(mesh, scalars="values", show_scalar_bar=False, cmap=self.cmap)
        plotter.add_mesh(skin, scalars="values", show_scalar_bar=False, cmap=self.cmap, opacity=0.1)

    def add_bounding_box(self, model, plotter=None):
        flat_bounds = [item for sublist in model.bounds for item in sublist]
        bounding_box = Box(flat_bounds)
        plotter.add_mesh(bounding_box, color="black", style="wireframe", line_width=2)


def main():
    sentence = [InfiniteBasement(), InfiniteSedimentUniform()]
    bounds = ((-3840, 3840), (-3840, 3840), (-1920, 1920))
    res = (64, 64, 32)
    GeoWordPlotter(sentence, bounds, res)


if __name__ == "__main__":
    main()
    print("BackgroundPlotter closed. Exiting script.")
