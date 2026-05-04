from setuptools import find_packages, setup

setup(
    name="GeoGen",
    version="0.1.0",
    description="A package for creating, visualizing, and exporting 3D structural geology models. \
    Allows either user specified, or randomized generation of models.",
    packages=find_packages(where="src"),  # Look for packages in the 'src' directory
    package_dir={"": "src"},  # Root package directory is 'src'
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pyvista[all]",
        "pyvistaqt",
        "ipywidgets",
        "ipykernel",
        "trame",
        "trame-vuetify",
        "trame-vtk",
        "tqdm",
        "PyDTMC",
        "geoflow-contracts>=0.2,<0.3",
    ],
    extras_require={
        # Microsoft Planetary Computer + raster ingestion for geogen.gis
        "gis": [
            "pystac-client>=0.7",
            "planetary-computer>=1.0",
            "rioxarray>=0.15",
            "rasterio>=1.3",
            "xarray>=2023.1",
            "pyproj>=3.5",
            "shapely>=2.0",
            "affine>=2.4",
        ],
    },
    package_data={
        "geogen.generation.markov_matrix": ["default_markov_matrix.csv"],
    },
    include_package_data=True,
)
