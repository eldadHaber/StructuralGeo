"""A simple demonstration sandbox for generating and visualizing geomodels"""

import time as clock

import pyvista as pv
from torch.utils.data import DataLoader

import geogen.model as geo  # Model contains the GeoModel and GeoProcess classes, the mechanics of the modeling
import geogen.plot as geovis  # Plot contains all the tools related to visualizing the geomodels
import geogen.probability as rv  # markov sediment builders, random variable helpers, etc.
from geogen.dataset import GeoData3DStreamingDataset
from geogen.generation import *  # Generation module contains all of the GeoWords and history generating functions


def main():
    # Set of demonstraition functions to run and/or follow
    dataloader_test()  # <-- Testing the DataLoader
    sampling_summary()  # <-- Summary of using GeoWords
    direct_model_generation_demo()

def visualize_model_demo(model):
    """Demonstration of visualizing a geomodel."""
    # Now we can visualize the model, there are a few options:
    # geovis.volview(model, show_bounds=True)
    # geovis.orthsliceview(model)
    # geovis.nsliceview(model, n=10)
    # geovis.onesliceview(model)

    # All of these return a PyVista plotter object, or optionally a plotter can be passed in
    # As a 'plotter = ' argument. This allows for modes like subplotting, etc.
    p = geovis.volview(model, show_bounds=True)
    p.show()

    # Now using passed plotter to view them all:
    p = pv.Plotter(shape=(2, 2))
    p.subplot(0, 0)
    geovis.volview(model, plotter=p)
    p.subplot(0, 1)
    geovis.orthsliceview(model, plotter=p)
    p.subplot(1, 0)
    geovis.nsliceview(model, n=10, plotter=p)
    p.subplot(1, 1)
    geovis.onesliceview(model, plotter=p)

    p.show()

    # Some additional views that are available:
    geovis.transformationview(model).show()
    geovis.categorical_grid_view(model).show()


def dataloader_test():
    # Decide on bounds and resolution for the model
    bounds = ((-3840, 3840), (-3840, 3840), (-1920, 1920))
    resolution = (128, 128, 64)

    # Dataset, Loader and Batch
    dataset = GeoData3DStreamingDataset(
        model_bounds=bounds, model_resolution=resolution
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    batch = next(iter(loader))

    batch_shape = batch.shape
    print(f"Datlaloader yields a sample of shape: {batch_shape}")

    # The torch tensors can be converted back into a geomodel to gain access to all
    # of the geovis visualization tools

    # If bounds are provded then measurement units are applied to plotting.
    # Else, the measurement units are in voxel units by default

    model = geo.GeoModel.from_tensor(bounds=bounds, data_tensor=batch[0])
    geovis.volview(model, show_bounds=True).show()


def get_empty_model(z_axis_res=64):
    bounds = (
        BOUNDS_X,
        BOUNDS_Y,
        BOUNDS_Z,
    )  # The geowords import from geogen.generation import * contains
    # the intended bounds for the geowords
    x = z_axis_res
    resolution = (
        2 * x,
        2 * x,
        x,
    )  # Intended shape with bounds is 2x2x1 ratio, cubic voxels
    model = geo.GeoModel(bounds, resolution)  # Initialize the model
    return model


def sampling_summary():
    # To summarize generation and testing via GeoWords:

    # 1. A reference guide to GeoProcesses to work with is in /icons/docs/process-icons.drawio-poster.pdf

    # 2. Premade GeoWords are found in /generation/geowords.py

    # 3. GeoWords can be chained together in a list:
    geosentence = [
        BaseStrata(),
        Fold(),
        SingleDikeWarped(),
        Sediment(),
    ]

    # 4. The sentence can be sampled into a concrete history:
    history = [geoword.generate() for geoword in geosentence]
    # Alternatively use the function from generation module:
    history = generate_history(geosentence)

    # 5. The history is added to a model and computed with height normalization to generate raw data:
    model = get_empty_model(64)
    model.add_history(history)
    model.compute_model(normalize=True)

    # 6. Various visualization options are available in plot module:
    geovis.volview(model, show_bounds=True).show()

    # 7. Optionally a batch sampling can be done directly on a geosentence with keyboard commands:
    geovis.GeoWordPlotter(geosentence, model.bounds, model.resolution, n_samples=16)


def direct_model_generation_demo():
    """Demonstration of direct model building using GeoProcess class."""

    # All GeoWords have a generate() method that returns a packaged snippet of history
    sample = InfiniteSedimentMarkov().generate()

    # Hint: The infinite base-strata use 'Infinite' in their name, they provide a base foundation for the model
    geosentence = [
        InfiniteSedimentMarkov(),
        FourierFold(),
        SingleDikeWarped(),
        CoarseRepeatSediment(),
    ]

    # We can inspect each one to get an idea of what they do
    for geoword in geosentence:
        print(f"Generating Sample from {geoword.__class__.__name__}")
        print(
            geoword.generate()
        )  # GeoProcess classes generated mostly have descriptive __str__ methods
        print("\n")

    # The most general classes are categorical events, which sample from a subset of GeoWords
    geosentence = [
        BaseStrata(),
        Fold(),
        Dike(),
        Erosion(),
        Sediment(),
    ]

    # Now we can chain them together to form a history
    history = [geoword.generate() for geoword in geosentence]

    # The history is the instruction set for the GeoModel, fetch a model with a helper
    # function. It is convenient to fix the resolution at 2x2x1 ratio, cubic voxels
    # Defining only the z-axis resolution, the x and y axes are derived from this
    model = get_empty_model(z_axis_res=64)

    # Now with a model and a history, we can build the model
    model.add_history(history)
    model.compute_model()
    print(model.data)
    visualize_model_demo(model)

    # There is an issue to be addressed, the model is not filled in well.
    # An automatic normalization scheme has been implemented to address this.
    model.compute_model(normalize=True)
    visualize_model_demo(model)

    # There is also a visualization to show categorical chunks of the model
    p = geovis.categorical_grid_view(model)
    p.show()


if __name__ == "__main__":
    main()
