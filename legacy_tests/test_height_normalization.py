import time

import numpy as np

import geogen.generation as gen
import geogen.model as geo
import geogen.plot as geovis

sentence = [
    gen.InfiniteBasement(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
    gen.FineRepeatSediment(),
]

# Save directory for models
DEFAULT_BASE_DIR = "../saved_models"

# Model resolution and bounds
res = (256, 256, 128)
bounds = ((-3840, 3840), (-3840, 3840), (-1920, 1920))
hist = gen.generate_history(sentence)

model = geo.GeoModel(bounds=bounds, resolution=res)
model.add_history(hist)
start = time.time()
model.compute_model(normalize=False)
stop = time.time()
print(f"Model computed in {stop-start:.2f} seconds.")
model.clear_data()

start = time.time()
model.compute_model(normalize=True)
stop = time.time()
print(f"Normalization computation time: {stop-start:.2f}")

# Generate the final model with the correct resolution and normalized height
p = geovis.volview(model)
p.show()

# Equivalently from library
print(model.xyz[np.argmax(model.data)])
geovis.volview(model).show()
