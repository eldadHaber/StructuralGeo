import pickle as pkl

import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt

import geogen.model as geo
import geogen.plot as geovis

# Load geomodel from pickle file

model = geo.GeoModel
with open(
    "C:\\Users\\sghys\\ComputationalResearch\\StructuralGeo\\docs\\paper-image-generators\\geological-sequence\\sequence_model.pkl",
    "rb",
) as f:
    model = pkl.load(f)

print(model.get_history_string())
model.compute_model(normalize=True)


mesh = geovis.get_mesh_from_model(model)

""" Plot the model with the snapshots of the transformation history."""
# Define the shape of the plotter as 1 row and 4 columns
shape = (1, 2)

# Create the plotter with the defined shape and groups
p = pv.Plotter(shape=shape, window_size=(1200, 450), border=True, off_screen=True)

p.subplot(0, 0) 

time_arrow = pv.Arrow(
    start=(22000, 0, 4000),
    direction=(-1, 0, 0),
    scale=25000,
    tip_length=0.20,
    tip_radius=0.01,
    shaft_radius=0.003,
)
p.add_mesh(time_arrow, color="black", label="Time Arrow")

# Get final present-day mesh of model
final_mesh = geovis.get_mesh_from_model(model)
cmap = plt.get_cmap("gist_ncar", 17)

plot_config = {
    # Color map for the rock types
    "cmap": cmap,  # Vibrant color map to differentiate rock types
    # Scalar bar settings
    "scalar_bar_args": {
        "title": "Rock Type",
        "title_font_size": 16,
        "label_font_size": 10,
        "shadow": True,
        "italic": True,
        "font_family": "arial",
        "n_labels": 2,  # Reducing the number of labels for clarity
        "vertical": False,
        "n_colors": 17,
    },
}

# change to horizontal colorbar

if np.all(np.isnan(model.data)):
    p.add_text("No data to show, all values are NaN.", font_size=20)
else:
    # Add the mesh to the plotter
    final_actor = p.add_mesh(
        final_mesh,
        scalars="values",
        **plot_config,
        clim = [0,17]
    )
_ = p.add_axes(line_width=5)

clim = final_actor.mapper.scalar_range
cmap = plot_config["cmap"]

resolution = model.resolution
# Calculate the offset to separate each snapshot
# The offset is chosen based on the overall size of the model
x_offset = model.bounds[0][1] - model.bounds[0][0]  # Width of the model along x

# Remove first data time entry which is empty, add the final data time entry
data_snapshots = np.concatenate((model.data_snapshots[1:], model.data.reshape(1, -1)), axis=0)

# Reverse the snapshots for proper plotting
mesh_snapshots = model.mesh_snapshots[::-1]
data_snapshots = data_snapshots[::-1]

actors = []
for i, (mesh_snapshot, data_snapshot) in enumerate(zip(mesh_snapshots, data_snapshots)):
    # Assuming snapshots are stored as Nx3 arrays
    # Reshape to 3D grid of points-- i.e. 4x4x4 grid of (x,y,z) points
    deformed_points = mesh_snapshot.reshape(resolution + (3,))
    grid = pv.StructuredGrid(
        deformed_points[..., 0] + (i + 1) * x_offset * 1.4,  # Shift along x
        deformed_points[..., 1],
        deformed_points[..., 2],
    )
    # Set the same values to the new grid
    grid["values"] = data_snapshot.reshape(model.X.shape).flatten(
        order="F"
    )  # Assigning scalar values to the grid
    # Add grid to plotter using the same colormap and scalar range as the final mesh
    a = p.add_mesh(
        grid,
        style="wireframe",
        scalars="values",
        cmap=cmap,  # Use the colormap from the final mesh
        clim=clim,  # Use the scalar range from the final mesh
        line_width=1,
        show_scalar_bar=False,
    )
    actors.append(a)

# Create a plotter object
p.subplot(0, 1)  # Right subplot (1/4 width)
plot_config = geovis.get_plot_config()

if np.all(np.isnan(model.data)):
    p.add_text("No data to show, all values are NaN.", font_size=20)
else:
    # Add the mesh to the plotter
    p.add_mesh(
        mesh,
        scalars="values",
        **plot_config,
        interpolate_before_map=False,
    )
_ = p.add_axes(line_width=5)

# add a bounding box
flat_bounds = [item for sublist in model.bounds for item in sublist]
bounding_box = pv.Box(flat_bounds)
p.add_mesh(bounding_box, color="black", style="wireframe", line_width=1)


# Adjust subplot sizes
p.subplot(0, 0)
p.view_vector([0.1, 1, 0.5])
p.camera.roll = 190
p.add_title("Depositions on a Deformed Mesh", font_size=12, color="black")
p.camera.zoom(1.4)

p.subplot(0, 1)
p.camera_position = "iso"
p.camera.zoom(0.8)
p.add_title("Final Model", font_size=12, color="black")

p.show()
p.screenshot("complex-transforms.png")
