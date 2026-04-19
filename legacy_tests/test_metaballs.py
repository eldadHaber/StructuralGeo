import numpy as np
import pyvista as pv

import geogen.model as geo
import geogen.plot as geovis

p = pv.Plotter(shape=(2, 2))

# simple test with two balls
radius = 2
goo_factor = 0.4
ball1 = geo.Ball(origin=[0, 0, 0], radius=radius)
ball2 = geo.Ball(origin=[5, 0, 0], radius=3)
ball3 = geo.Ball(origin=[0, 2, 0], radius=3)
balls = [ball1, ball2, ball3]

metaball = geo.MetaBall(balls=balls, threshold=1.5, value=1, clip=False)
model = geo.GeoModel(bounds=(-10, 10), resolution=128)
model.add_history(metaball)
model.compute_model()

p.subplot(0, 0)
geovis.volview(model, threshold=-0.5, plotter=p)
p.add_title("Simple MetaBall")

# -- Generative Balls Test -- #
# Make a starting list of balls generatively and create a deterministic MetaBall object
ballgen = geo.BallListGenerator(step_range=[2, 3], rad_range=[1, 1], goo_range=[0.8, 1])
balls = ballgen.generate(n_balls=12, origin=[0, 0, 0])

# Threshold 1 MetaBall
metaball = geo.MetaBall(balls=balls, threshold=1, value=1, clip=False)
model = geo.GeoModel(bounds=(-10, 10), resolution=128)
model.add_history(metaball)
model.compute_model()

p.subplot(1, 0)
geovis.volview(model, threshold=-0.5, plotter=p)
p.add_title("Threshold 1 MetaBall")

# Redo with a higher threshold
metaball = geo.MetaBall(balls=balls, threshold=1.5, value=1, clip=False)
model = geo.GeoModel(bounds=(-10, 10), resolution=128)
model.add_history(metaball)
model.compute_model()

p.subplot(1, 1)
geovis.volview(model, threshold=-0.5, plotter=p)
p.add_title("Threshold 1.5 MetaBall")
p.show()
