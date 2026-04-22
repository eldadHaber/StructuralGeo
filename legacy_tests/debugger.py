import time as clock

import numpy as np

import geogen as gg
import geogen.plot as geovis
from geogen.generation.geowords import *
from geogen.generation.model_generators import MarkovGeostoryGenerator

# Generate a geostory
# Model resolution and bounds
z = 32
res = (2 * z, 2 * z, z)

reduction_factor = 4
x = tuple([i / reduction_factor for i in BOUNDS_X])
y = tuple([i / reduction_factor for i in BOUNDS_Y])
z = tuple([i / reduction_factor for i in BOUNDS_Z])
bounds = (x, y, z)

gen = MarkovGeostoryGenerator(model_bounds=bounds, model_resolution=res)
sentence = gen.build_sentence()
test_viewer = geovis.GeoWordPlotter(sentence, bounds, res, 9, clim=(0, 13))
