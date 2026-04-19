import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt

import geogen.model as geo
import geogen.plot as geovis

model = geo.GeoModel(bounds=(-1920, 1920), resolution=128)
model.add_history(geo.Sedimentation([1, 3, 4, 5], [500, 200, 300, 200, 500]))

plug = geo.DikePlug(
    origin=geo.BacktrackedPoint((0, 0, 0)),
    diam=100,
    minor_axis_scale=0.4,
    rotation=30,
    shape=2.5,
    value=5,
    clip=False,
)
model.add_history(plug)


model.add_history(
    geo.PushHemisphere(
        origin=geo.BacktrackedPoint((0, 0, -1000)),
        diam=2000,
        height=350,
        minor_axis_scale=0.8,
        rotation=10,
    )
)
model.add_history(
    geo.DikeHemisphere(
        origin=geo.BacktrackedPoint((0, 0, -1000)),
        diam=2000,
        height=350,
        minor_axis_scale=0.8,
        rotation=10,
        value=6,
        clip=False,
    )
)
model.add_history(
    geo.DikeColumn(
        origin=geo.BacktrackedPoint((0, 0, -1000)),
        diam=500,
        depth=-1600,
        minor_axis_scale=0.3,
        rotation=60,
        value=6,
        clip=False,
    )
)

model.add_history(
    geo.Laccolith(
        origin=geo.BacktrackedPoint((0, 0, -2000)),
        cap_diam=4000,
        stem_diam=300,
        height=600,
        minor_axis_scale=0.3,
        rotation=10,
        value=6,
    )
)

model.compute_model()

p = geovis.categorical_grid_view(model)
p.show()
