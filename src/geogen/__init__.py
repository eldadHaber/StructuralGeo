# -*- coding: utf-8 -*-

"""
GeoGen Geological Modeling Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GeoGen is a Python library for generating and manipulating synthetic geological
models. Basic usage:

   >>> import geogen
   >>> model = geogen.model.GeoModel(
   ...     bounds=((0, 100), (0, 100), (0, 100)),
   ...     resolution=(100, 100, 50),
   ... )

The streaming PyTorch dataset requires the optional ``[torch]`` extra:

   >>> from geogen.dataset import GeoData3DStreamingDataset

If torch is not installed, ``geogen`` itself still imports cleanly; only the
``geogen.dataset`` module will raise a helpful ``ImportError``.

:copyright: (c) 2024 by Simon Ghyselincks.
:license: MIT, see LICENSE for more details.
"""

from importlib.metadata import PackageNotFoundError, version

from geogen import generation as gen
from geogen import model, plot

__title__ = "GeoGen"

try:
    __version__ = version("geogen")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["model", "plot", "gen", "__version__", "__title__"]
