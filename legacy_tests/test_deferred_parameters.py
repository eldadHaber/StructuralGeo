import time as clock

import numpy as np
import pyvista as pv

import geogen.generation as gen
import geogen.model as geo
import geogen.plot as geovis


def main():
    # Create a simple geological model with a few fold transformsations
    model = geo.GeoModel(bounds=(-10, 10), resolution=128)

    bed = geo.Bedrock(base=-5, value=0)
    sed = geo.Sedimentation(value_list=[1, 2, 3], thickness_list=[2, 2, 2])
    fold = geo.Fold(strike=0, dip=90, rake=0, period=10, amplitude=5, origin=(0, 0, 0))
    ball = geo.Ball(origin=(0, 0, 0), radius=1)
    metaball = geo.MetaBall(balls=[ball], threshold=0.5, value=5, clip=False)

    model.add_history(
        [
            bed,
            sed,
            metaball,
            fold,
        ]
    )

    start = clock.time()
    model.compute_model()
    stop = clock.time()
    print(f"Model computed in {stop-start:.2f} seconds.")

    start = clock.time()
    model.compute_model()
    stop = clock.time()
    print(f"Model computed in {stop-start:.2f} seconds.")

    model_deferred = geo.GeoModel(bounds=(-10, 10), resolution=128)
    defer_origin = geo.BacktrackedPoint(point=(0, 0, 0))
    metaball = geo.MetaBall(balls=[ball], threshold=0.5, value=5, reference_origin=defer_origin, clip=False)
    model_deferred.add_history([bed, sed, metaball, fold])
    print(model_deferred.history[-2].reference_origin)
    start = clock.time()
    model_deferred.compute_model()
    stop = clock.time()
    print(f"Deferred model computed in {stop-start:.2f} seconds.")
    print(model_deferred.history[-2].reference_origin)
    p = pv.Plotter(shape=(1, 2))
    p.subplot(0, 0)
    geovis.volview(model, plotter=p)
    p.subplot(0, 1)
    geovis.volview(model_deferred, plotter=p)

    p.show()

    dike = geo.DikePlane(strike=0, dip=45, width=2, origin=geo.BacktrackedPoint((0, 0, 0)), value=5)

    model.clear_history()
    model.add_history(
        [
            bed,
            sed,
            metaball,
            dike,
            fold,
        ]
    )
    model.compute_model()
    geovis.volview(model).show()


if __name__ == "__main__":
    main()
