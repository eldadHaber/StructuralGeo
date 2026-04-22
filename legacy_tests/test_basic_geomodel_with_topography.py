import os

import numpy as np
import pyvista as pv

import geogen.model as geo
import geogen.plot as geovis
import geogen.probability as rv

# Generators
# Create a list of numbers from 4 to 6
sediment_rock_types = list(range(4, 7))
# Shuffle the list in place
np.random.shuffle(sediment_rock_types)

# Depostions
bedrock = geo.Bedrock(base=-5, value=0)
dike = geo.DikePlane(strike=45, dip=75, width=3, origin=[0, 0, 0], value=7)

sediment0 = geo.Sedimentation(
    value_list=range(1, 5),
    thickness_list=[2, 3, 2, 1],
)
sediment1 = geo.Sedimentation(value_list=[4, 5, 6], thickness_list=[1, 2, 1])
erosion_process0 = geo.UnconformityDepth(depth=2)
erosion_process1 = geo.UnconformityBase(base=0)

# Transformations
tilt = geo.Tilt(strike=45, dip=20, origin=(-1, -1, 0))
tilt2 = geo.Tilt(strike=10, dip=20, origin=(3, 0, 0))
upright_fold = geo.Fold(strike=65, dip=80, rake=20, period=20, amplitude=1)
upright_fold2 = geo.Fold(
    strike=110,
    dip=90,
    rake=80,
    period=40,
    amplitude=2,
    shape=1,
    origin=(0, 0, 0),
    periodic_func=rv.noisy_sine_wave(frequency=3, smoothing=10, noise_scale=0.1),
)
slip0 = geo.Fault(strike=80, dip=70, rake=35, amplitude=3, origin=(0, 0, 0))

# Histories
test_history0 = [
    bedrock,
    sediment0,
    tilt,
    sediment1,
    dike,
    upright_fold,
    upright_fold2,
    erosion_process0,
]
test_history1 = [sediment0, sediment1, upright_fold, upright_fold2]
test_history2 = [bedrock, tilt2, sediment0]
test_history3 = [bedrock, sediment0, dike, slip0]

bounds = ((-20, 20), (-20, 20), (-10, 10))
model = geo.GeoModel(bounds=bounds, resolution=128, height_tracking=True)
model.add_history(test_history0)
model.compute_model()

# Get directory of this file
dir_path = os.path.dirname(os.path.realpath(__file__))
elevation_data = os.path.join(dir_path, "elevation_data.npy")
mesh = np.load(elevation_data)
mesh_scaled = mesh * 6
model.add_topography(mesh_scaled)
p = geovis.transformationview(model, threshold=-0.5)
p.show()
# Three types of visualization
# p = geovis.volview(model, threshold= -.5)
# p.show()
# p = geovis.orthsliceview(model, threshold= -.5)
# p.show()
# p = geovis.nsliceview(model, n=6, axis="x", threshold= -.5)
# p.show()
