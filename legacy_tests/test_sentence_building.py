import random

import numpy as np
import pyvista as pv

import geogen.model as geo
import geogen.plot as geovis
from geogen.generation import *


# ---- Single Sentence Testing ---- #
def single_sentence_test():
    sentence = [
        InfiniteBasement(),
        FineRepeatSediment(),
        FineRepeatSediment(),
        MicroNoise(),
        NullWord(),
    ]
    histories = [generate_history(sentence) for _ in range(16)]
    # Select a random set of 16 histories
    selected_histories = random.sample(histories, 16)
    p = pv.Plotter(shape=(4, 4))
    for i, hist in enumerate(selected_histories):
        p.subplot(i // 4, i % 4)
        model = geo.GeoModel(bounds=(-1920, 1920), resolution=128)
        model.add_history(hist)
        model.compute_model(normalize=True)
        geovis.volview(model, plotter=p)
    p.show()


single_sentence_test()
