import copy
import time as clock

import numpy as np
import pyvista as pv

import geogen.model as geo
import geogen.plot as geovis
from geogen.generation import *


def single_plotter():
    # List of geological words to generate
    sentence = [BaseStrata(), SingleDikeWarped()]
    # Model resolution and boundse
    z = 64

    res = (2 * z, 2 * z, z)
    bounds = (
        BOUNDS_X,
        BOUNDS_Y,
        BOUNDS_Z,
    )  # Bounds imported from generation (geowords file)ee

    hist = generate_history(sentence)
    start = clock.time()
    model = geo.GeoModel(bounds=bounds, resolution=res)
    model.add_history(hist)
    model.compute_model(normalize=True)
    finish = clock.time()
    print(f"Model computed in {finish-start:.2f} seconds.")
    print(model.get_history_string())

    # geovis.transformationview(model).show()e
    geovis.categorical_grid_view(model).show()


def process_plotter():
    bed = geo.Bedrock(0, 1)
    sediment_word = Sediment()
    hist = [bed, sediment_word.generate(), sediment_word.generate()]

    blob = BlobWord(origin=geo.BacktrackedPoint((0, 0, 0))).generate()
    print(blob)
    hist.append(blob)

    # Model resolution and boundse
    z = 128
    res = (2 * z, 2 * z, z)
    bounds = (
        BOUNDS_X,
        BOUNDS_Y,
        BOUNDS_Z,
    )  # Bounds imported from generation (geowords file)

    res = (64, 64, 64)
    bounds = (
        tuple([x / 4 for x in BOUNDS_X]),
        tuple([x / 4 for x in BOUNDS_Y]),
        tuple([x / 2 for x in BOUNDS_Z]),
    )

    model = geo.GeoModel(bounds=bounds, resolution=res)
    model.add_history(hist)
    start = clock.time()
    model.compute_model(normalize=True)
    stop = clock.time()
    print(f"Model computed in {stop-start:.2f} seconds.")
    geovis.categorical_grid_view(model).show()


if __name__ == "__main__":
    single_plotter()
