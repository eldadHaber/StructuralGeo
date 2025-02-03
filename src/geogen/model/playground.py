import numpy as np
import pyvista as pv

import geogen.model as geo
WS = (600, 400)  # Set a custom window size to be reused in most plots
import geogen.plot as geovis
from geogen.probability import SedimentBuilder, MarkovSedimentHelper
pv.set_jupyter_backend("static")

bedrock = geo.Bedrock(base=-5, value=1)
sediment0 = geo.Sedimentation(value_list=[2, 3, 4, 5], thickness_list=[1, 2])

p = pv.Plotter(shape=(1, 2))

model = geo.GeoModel(bounds=(-10, 10), resolution=128)
model.clear_history()
# Add a base layer of bedrock
model.add_history(bedrock)
model.compute_model()
p.subplot(0, 0)
geovis.volview(model, plotter=p)
p.add_title(title="Bedrock Model", font_size=8)

model.add_history(sediment0)
model.compute_model()
p.subplot(0, 1)
geovis.volview(model, plotter=p)

p.add_title(title="Sedimentation Ontop of Bedrock Model", font_size=6)
p.window_size = WS
p.show()

# display(model)