""" A multi-plotter PyVista window offering viewing of GeoWord stories in 3D.

Shortcut keys:
- 'r': Refresh the samples.
- '1': View the samples in volume mode.
- '2': View the samples in orthogonal slice mode.
- '3': View the samples in n-slice mode.
- '4': View the samples in one-slice mode.  

Plotter Parameters:
- sentence (list): A list of GeoWords to generate histories from
- bounds (tuple): The bounds of the model in the form ((xmin, xmax), (ymin, ymax), (zmin, zmax))
- res (tuple): The resolution of the model in the form (nx, ny, nz)
- n_samples (int): The number of samples to generate and plot. Plotter defaults to square grid layout.
"""

from pyvistaqt import BackgroundPlotter
from structgeo.generation import *
import structgeo.plot as geovis

class GeoWordPlotter:
    def __init__(self, sentence, bounds, res, n_samples=16):
        self.sentence = sentence
        self.bounds = bounds
        self.res = res
        self.n_samples = n_samples
        self.current_view_mode = self.volview
        self.plotter = None
        self.initialize_plotter()

    def initialize_plotter(self):
        rows, cols = self.calculate_grid_dims(self.n_samples)
        self.plotter = BackgroundPlotter(shape=(rows, cols))
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
        self.update_samples()
        print("Updated samples.")

    def set_view_mode(self, view_mode):
        """Set the current view mode and refresh samples."""
        self.current_view_mode = view_mode
        self.update_samples()

    def update_samples(self):
        """Update and plot the samples with the current view mode."""
        histories = [generate_history(self.sentence) for _ in range(self.n_samples)]
        self.plotter.clear_actors()
        for i, hist in enumerate(histories):
            row, col = divmod(i, self.plotter.shape[1])
            self.plotter.subplot(row, col)
            model = generate_normalized_model(hist, self.bounds, self.res)
            self.current_view_mode(model, plotter=self.plotter)
        
        self.plotter.link_views()
        self.plotter.render()

    def volview(self, model, plotter):
        mesh = geovis.get_voxel_grid_from_model(model)
        plotter.add_mesh(mesh, scalars="values", show_scalar_bar=False, cmap='gist_ncar')

    def orthsliceview(self, model, plotter=None):
        mesh = geovis.get_voxel_grid_from_model(model)
        plotter.add_mesh_slice_orthogonal(mesh, scalars="values", show_scalar_bar=False, cmap='gist_ncar')

    def nsliceview(self, model, n=5, axis="x", plotter=None):
        mesh = geovis.get_voxel_grid_from_model(model)
        slices = mesh.slice_along_axis(n=n, axis=axis)
        plotter.add_mesh(slices, scalars="values", show_scalar_bar=False, cmap='gist_ncar')
        plotter.add_axes(line_width=5)

    def onesliceview(self, model, plotter=None):
        mesh = geovis.get_voxel_grid_from_model(model)
        skin = mesh.extract_surface()
        plotter.add_mesh_slice(mesh, scalars="values", show_scalar_bar=False, cmap='gist_ncar')
        plotter.add_mesh(skin, scalars='values', show_scalar_bar=False, cmap='gist_ncar', opacity=0.1)

def main():
    sentence = [InfiniteBasement(), InfiniteSedimentUniform()]
    bounds = ((-3840, 3840), (-3840, 3840), (-1920, 1920))
    res = (64, 64, 32)
    GeoWordPlotter(sentence, bounds, res)

if __name__ == "__main__":
    main()
    print('BackgroundPlotter closed. Exiting script.')