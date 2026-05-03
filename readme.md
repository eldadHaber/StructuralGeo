# StructuralGeo

**StructuralGeo** is a Python package designed for creating, visualizing, and randomizing synthetic structural geology data. The package leverages NumPy for efficient data handling and PyVista for powerful 3D visualization. It includes a randomization scheme within the generation and dataset modules, enabling the creation of an extensive synthetic dataset for use with PyTorch or similar frameworks.

## Shared Contracts

This repository includes `contracts/` as a git submodule pointing at
`geoflow-contracts`. That submodule is the source of truth for cross-repo schema
IDs, schema versions, feature-set IDs, service run envelopes, and validation
fixtures.

When changing synthetic data outputs consumed by Stratiflow, Geoflow, or the ML
training pipelines, update `contracts/` first, validate the fixtures there, then
update this repository against the new contracts pin.

Initialize submodules with:

```bash
git submodule update --init --recursive
```

### Project Installation
To install StructuralGeo to your Python environment, first clone this repo into a working directory and change directory into the cloned folder.

```bash
cd path/to/my/project
git clone https://github.com/eldadHaber/StructuralGeo
cd StructuralGeo
```

>[!NOTE]
> Before continuing with installation you may want to create a virtual environment to install the package into, using a tool such as `venv` or `conda`.

The package and its dependencies can be installed using the `setup.py` file and the `pip install` command. The `-e` flag installs the package as editable, allowing changes to the package to be reflected immedeately in the Python environment, and is recommended for development.

```bash
cd path/to/my/project/StructuralGeo
pip install -e .
```

>[!IMPORTANT]
>The package has a dependency on PyTorch for using the dataset module which is not automatically installed. PyTorch can be installed using the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/). 

After installation, the package can be imported into your Python environment with the following command:

```python
import geogen
```
### Dataset Quick Start

The streaming dataset is initialized with:
- `model_bounds`: ((xmin, xmax), (ymin, ymax), (zmin, zmax)) in meters. Note the randomization scheme of Geowords is designed for models that are ((-3840, 3840), (-3840, 3840), (-1920, 1920)) in size.
- `model_resolution`: Resolution of the meshgrid, along with bounds determines the voxel size.
- `generator_config`: Path pointing to a configuration file or CSV with the markov matrix weights, if None, uses a default set.
- `dataset_size`: Artifical number of samples in one epoch. The dataset does not reuse samples across epochs and always streams new ones.
- `device`: Device to load model tensors

```python
from geogen.dataset import GeoData3DStreamingDataset

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
```
A quickstart guide is found in `code_examples/quickstart.py`
___
### Project Structure

The project is structured with the following directories:

| Directory          | Description                                                        |
|--------------------|--------------------------------------------------------------------|
| `code_examples/`   | Example code to demonstrate package features with explanations     |
| `gui/`             | A GUI interface for viewing saved .pkl GeoModels, run using main.py|
| `docs/`            | Project documentation and source materials                         |
| `tests/`           | Contains some basic tests for the package, mostly human inspected  |
| `src/geogen/`      | Contains the main package source code                              |

### Project Source

#### `dataset`
- **Description**: Contains data management functionality.
- **Components**:
  - **GeoData3DStreamingDataSEt**: Provides a PyTorch dataset interface for streaming 3D data randomly generated at runtime.

#### `filemanagement`
- **Description**: Contains data management functionality.
- **Components**:
  - **FileManager**: Manages loading and saving of pickled models from a base directory. Supports recursive operations and model updates.

#### `generation`
- **Description**: Module overlay for automated randomized generation of geological models.
- **Components**:
  - **geowords.py**: A collection of parameter randomization schemes for GeoProcess events.
  - **categorical_events.py**: A collection of basic geological event categories-- i.e. Erosion, Fold, Fault, etc.
  - **model_generators.py**: Categorical event chain constructors for generating geological models. Markov chain generation implemented.
  - **geohistgen.py**: Helper functions related to automated geological history generation.

#### `model`
- **Description**: Core framework for creating parametrized geological models. 
- **Components**:
  - **GeoModel**: Main class for creating a blank model, can be updated with geological history and visualized with PyVista.
  - **GeoProcess**: A collection of parameterized geological events that can be applied to a GeoModel. i.e. Fault, Unconformity, Deposition, etc.
  - **MetaBall**: A collection of code related to a blob-like GeoProcess deposition event.
  - **DeferredParameter**: A collection of objects that allow for conditional parameterization of GeoProcess events at runtime.
  - **util.py**: Contains helper functions for the model module.

#### `plot`
- **Description**: Visualization toolbox for GeoModels.
- **Components**:
  - **plot.py**: PyVista plotting library with basic plotting functions and standardized plotter setup.
  - **ModelReviewerJupyter**: GUI designed for Jupyter Notebook as a widget to view, save, or reject generated models and renormalize model height.
  - **GeoWordPlotter**: A multi-plotter designed to ingest a list of GeoWords and show an array of sampled generated models, useful for testing parameter distributions.

#### `probability`
- **Description**: Collection of functions and classes related to generating probability parameters of interest.
- **Components**:
  - **random_varibles.py**: Helper functions related to generating random variables related to GeoWords.
  - **sedimentbuilders.py**: Helper functions related to generating sedimentation events.
  - **wavegenerators.py**: Helper functions related to generating wave form functions for functional parameterization of GeoWords.

___

#### Jupyter Notebook Viewing

The visualization is handled with Pyvista which may require additional configuration for Jupyter Notebook to view the model iteractively. 

The type of visualization backend for Jupyter can be set to `static`, `html` or `trame`.

- `static` will render non-interactive plots quickly, recommended for quick testing.
- `html` will render interactive plots in the notebook but is slower to build the viewer than static. The model can be rotated and zoomed in the notebook.
- `trame` will render interactive plots in the notebook through a viewer with menu options. It takes a while to load and is slower to render than the other options.

The backend can be specified for all Pyvista plots at the top of the notebook with the following command:

```python
pyvista.set_jupyter_backend('static') # or 'html' or 'trame'
```

On an individual Pyvista plot that is returned by geovis package, the backend can be overridden with the following example command:

```python
p = geovis.volview(model)
p.window_size = window_size
p.add_title(title='Sedimentation Ontop of Bedrock Model', font_size=8)
p.show(jupyter_backend='static') 
```

To install the trame framework, use the following command:

```bash
pip install 'jupyterlab>=3' ipywidgets 'pyvista[all,trame]'   
```

If `trame` is not installed the jupyter backend should be set to static which will render non-interactive plots. See examples folder for implementation, or read more at the [Pyvista documentation](https://tutorial.pyvista.org/tutorial/00_jupyter/index.html).

More information can be found in the code_examples folder.

## Acknowledgements

StructuralGeo was developed by Simon Ghyselincks and Eldad Haber.  
This software was used in collaborative research funded in part by the King Abdullah University of Science and Technology (KAUST).

## Citation

If you have found this software useful in your work, please consider citing it:

```
@software{ghyselincks_structuralgeo_2025,
  author       = {Simon Ghyselincks and Eldad Haber},
  title        = {GeoGen: Synthetic Data for Structural Geology},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.15244035},
  url          = {https://doi.org/10.5281/zenodo.15244035}
}
```


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15244035.svg)](https://doi.org/10.5281/zenodo.15244035)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
