import numpy as np
import pyvista as pv

import geogen.model as geo


def add_snapshots_to_plotter(plotter, model, cmap):
    resolution = model.resolution
    # Calculate the offset to separate each snapshot
    # The offset is chosen based on the overall size of the model
    x_offset = model.bounds[0][1] - model.bounds[0][0]  # Width of the model along x

    # Remove first data time entry which is empty, add the final data time entry
    data_snapshots = np.concatenate((model.data_snapshots[1:], model.data.reshape(1, -1)), axis=0)

    # Reverse the snapshots
    mesh_snapshots = model.mesh_snapshots[::-1]
    data_snapshots = data_snapshots[::-1]

    actors = []
    for i, (mesh_snapshot, data_snapshot) in enumerate(zip(mesh_snapshots, data_snapshots)):
        # Assuming snapshots are stored as Nx3 arrays
        deformed_points = mesh_snapshot.reshape(resolution + (3,))
        grid = pv.StructuredGrid(
            deformed_points[..., 0] + (i + 1) * x_offset * 1.3,  # Shift along x
            deformed_points[..., 1],
            deformed_points[..., 2],
        )
        # Set the same values to the new grid
        grid["values"] = data_snapshot.reshape(model.X.shape).flatten(
            order="F"
        )  # Assigning scalar values to the grid
        # Add grid to plotter with a unique color and using the same scalar values
        a = plotter.add_mesh(
            grid,
            style="wireframe",
            scalars="values",
            cmap=cmap,
            line_width=1,
            show_scalar_bar=False,
        )
        actors.append(a)

    return actors


""" Plot and viewing parameters"""
# Create a plotter object
plotter = pv.Plotter(shape=(2, 2), window_size=[1200, 1200], off_screen=True)  # type: ignore
view_vector = [1, 3, 1]
zoom = 0.8
colormap = "viridis"
bounds = (-1, 1)
resolution = 4

""" Initial simple grid"""
plotter.subplot(0, 1)

model = geo.GeoModel(bounds=bounds, resolution=resolution)
hist = []
bedrock = geo.Bedrock(base=2, value=1)
hist.append(bedrock)
model.add_history(hist)
model.clear_data()
model.compute_model()

sample_points = pv.PolyData(model.xyz)
# Point cloud representation
mesh_points = model.xyz
# Hack to get null data values
data_vals = model.data_snapshots[0]

resolution = model.resolution
points = (mesh_points).reshape(resolution + (3,))
grid = pv.StructuredGrid(points[..., 0], points[..., 1], points[..., 2])
grid["values"] = data_vals.reshape(model.X.shape).flatten(order="F")
sample_points["values"] = data_vals
plotter.add_mesh(
    grid,
    scalars="values",
    cmap=colormap,
    style="wireframe",
    show_edges=True,
    show_scalar_bar=False,
    point_size=25,
)
plotter.add_mesh(
    sample_points,
    scalars="values",
    cmap=colormap,
    style="points",
    show_edges=True,
    show_scalar_bar=False,
    point_size=25,
)
# Add axes
_ = plotter.add_axes(line_width=6)

# Show bounds
plotter.show_bounds(
    grid=False,
    location="outer",
    ticks="outside",
    n_xlabels=4,
    n_ylabels=4,
    n_zlabels=4,
    font_size=12,
    # show_xlabels=False, show_ylabels=False, show_zlabels=False,
    xtitle="Easting",
    ytitle="",
    ztitle="Elevation",
)
# plotter.add_title("Initial Meshgrid", font_size=12, color="black")
# Set the view vector
plotter.view_vector(view_vector)
# Increase the camera distance to zoom out
plotter.camera.zoom(zoom)

""" Add a fault to history and pull the mesh snapshot"""
plotter.subplot(0, 0)

model = geo.GeoModel(bounds=bounds, resolution=resolution)
hist = []
bedrock = geo.Bedrock(base=2, value=1)
hist.append(bedrock)
fault = geo.Fault(10, 90, 90, 0.4)
hist.append(fault)

model.add_history(hist)
model.clear_data()
model.compute_model()

# Point cloud representation
mesh_points = model.mesh_snapshots[0]
data_vals = model.data_snapshots[0]

resolution = model.resolution
points = (mesh_points).reshape(resolution + (3,))
grid = pv.StructuredGrid(points[..., 0], points[..., 1], points[..., 2])
grid["values"] = data_vals.reshape(model.X.shape).flatten(order="F")
plotter.add_mesh(
    grid,
    scalars="values",
    cmap=colormap,
    style="wireframe",
    show_edges=True,
    show_scalar_bar=False,
    point_size=25,
)
plotter.add_mesh(
    grid,
    scalars="values",
    cmap=colormap,
    style="points",
    show_edges=True,
    show_scalar_bar=False,
    point_size=25,
)
# Add axes
_ = plotter.add_axes(line_width=6)

# Show bounds
plotter.show_bounds(
    grid=False,
    location="outer",
    ticks="outside",
    n_xlabels=4,
    n_ylabels=4,
    n_zlabels=4,
    font_size=12,
    # show_xlabels=False, show_ylabels=False, show_zlabels=False,
    xtitle="Easting",
    ytitle="",
    ztitle="Elevation",
)
# plotter.add_title("Inverse Fault Transform", font_size=12, color="black")
# Set the view vector
plotter.view_vector(view_vector)
# Increase the camera distance to zoom out
plotter.camera.zoom(zoom)

""" Fill from sediment"""
plotter.subplot(1, 0)
model = geo.GeoModel(bounds=bounds, resolution=resolution)
hist = []
sediment = geo.Sedimentation([1, 3, 4], [1, 0.8, 0.5])
hist.append(sediment)
fault = geo.Fault(10, 90, 90, 0.3)
hist.append(fault)

model.add_history(hist)
model.clear_data()
model.compute_model()

# Point cloud representation
mesh_points = model.mesh_snapshots[0]
data_vals = model.data

resolution = model.resolution
points = (mesh_points).reshape(resolution + (3,))
grid = pv.StructuredGrid(points[..., 0], points[..., 1], points[..., 2])
grid["values"] = data_vals.reshape(model.X.shape).flatten(order="F")
plotter.add_mesh(
    grid,
    scalars="values",
    cmap=colormap,
    style="wireframe",
    show_edges=True,
    show_scalar_bar=False,
    point_size=25,
)
plotter.add_mesh(
    grid,
    scalars="values",
    cmap=colormap,
    style="points",
    show_edges=True,
    show_scalar_bar=False,
    point_size=25,
)
# Add axes
_ = plotter.add_axes(line_width=6)

# Add in high resolution underlying points
model = geo.GeoModel(bounds=bounds, resolution=128)
bounds_edit = model.bounds
model.bounds = (bounds_edit[0], bounds_edit[1], (-1.3, 1.3))
hist = []
sediment = geo.Sedimentation([1, 3, 4], [1, 0.8, 0.4], base=-1)
hist.append(sediment)

model.add_history(hist)
model.clear_data()
model.compute_model()

mesh_points = model.xyz
data_vals = model.data

res = model.resolution
points = (mesh_points).reshape(res + (3,))
grid = pv.StructuredGrid(points[..., 0], points[..., 1], points[..., 2])
# Set data to the grid
values = model.data.reshape(model.X.shape)
grid["values"] = values.flatten(order="F")  # Flatten the data in Fortran order
# Create mesh thresholding to exclude np.nan values or sentinel values
mesh = grid.threshold(-0.5, all_scalars=True)

plotter.add_mesh(mesh, scalars="values", cmap=colormap, show_scalar_bar=False)

# Show bounds
plotter.show_bounds(
    grid=False,
    location="outer",
    ticks="outside",
    n_xlabels=4,
    n_ylabels=4,
    n_zlabels=4,
    font_size=12,
    # show_xlabels=False, show_ylabels=False, show_zlabels=False,
    xtitle="Easting",
    ytitle="",
    ztitle="Elevation",
)
# plotter.add_title("Sampling of Sediment Layers", font_size=12, color="black")
# Set the view vector
plotter.view_vector(view_vector)
# Increase the camera distance to zoom out
plotter.camera.zoom(zoom)

""" Final model"""
plotter.subplot(1, 1)

model = geo.GeoModel(bounds=bounds, resolution=resolution)
hist = []
sediment = geo.Sedimentation([1, 3, 4], [1, 0.8, 0.4], base=-1)
hist.append(sediment)
fault = geo.Fault(10, 90, 90, 0.3)
hist.append(fault)

model.add_history(hist)
model.clear_data()
model.compute_model()

sample_points = pv.PolyData(model.xyz)
# Point cloud representation
mesh_points = model.xyz
# Hack to get null data values
data_vals = model.data

resolution = model.resolution
points = (mesh_points).reshape(resolution + (3,))
grid = pv.StructuredGrid(points[..., 0], points[..., 1], points[..., 2])
grid["values"] = data_vals.reshape(model.X.shape).flatten(order="F")
sample_points["values"] = data_vals
plotter.add_mesh(
    grid,
    scalars="values",
    cmap=colormap,
    style="wireframe",
    show_edges=True,
    show_scalar_bar=False,
    point_size=25,
)
plotter.add_mesh(
    sample_points,
    scalars="values",
    cmap=colormap,
    style="points",
    show_edges=True,
    show_scalar_bar=False,
    point_size=25,
)
# Add axes
_ = plotter.add_axes(line_width=6)

# high res
model = geo.GeoModel(bounds=bounds, resolution=128)
hist = []
sediment = geo.Sedimentation([1, 3, 4], [1, 0.8, 0.4], base=-1)
hist.append(sediment)
fault = geo.Fault(10, 90, 90, 0.3)
hist.append(fault)

model.add_history(hist)
model.clear_data()
model.compute_model()

mesh_points = model.xyz
data_vals = model.data

res = model.resolution
points = (mesh_points).reshape(res + (3,))
grid = pv.StructuredGrid(points[..., 0], points[..., 1], points[..., 2])
# Set data to the grid
values = model.data.reshape(model.X.shape)
grid["values"] = values.flatten(order="F")  # Flatten the data in Fortran order
# Create mesh thresholding to exclude np.nan values or sentinel values
mesh = grid.threshold(-0.5, all_scalars=True)

plotter.add_mesh(mesh, scalars="values", cmap=colormap, show_scalar_bar=False)

# Show bounds
plotter.show_bounds(
    grid=False,
    location="outer",
    ticks="outside",
    n_xlabels=4,
    n_ylabels=4,
    n_zlabels=4,
    font_size=12,
    # show_xlabels=False, show_ylabels=False, show_zlabels=False,
    xtitle="Easting",
    ytitle="",
    ztitle="Elevation",
)
# plotter.add_title("Final Sample Grid", font_size=12, color="black")
# Set the view vector
plotter.view_vector(view_vector)
# Increase the camera distance to zoom out
plotter.camera.zoom(zoom)


plotter.show()
# save the plot to disk
plotter.screenshot("fig2_transformations.png")
