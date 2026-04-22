from .plot import *

# Optional widgets — require pyvistaqt / ipywidgets from the [all] extra.
# Import lazily so `import geogen.plot` always succeeds on the core install.
try:
    from .GeoWordPlotter import GeoWordPlotter
except ImportError:
    GeoWordPlotter = None  # requires: pip install "geogen[all]" (pyvistaqt)

try:
    from .ModelReviewerJupyter import *  # noqa: F401,F403
except ImportError:
    pass  # requires: pip install "geogen[all]" (ipywidgets)
