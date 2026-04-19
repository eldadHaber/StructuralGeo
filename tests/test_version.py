"""Version-sanity test."""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — only exercised on 3.10
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_version_matches_pyproject():
    import geogen

    data = tomllib.loads(PYPROJECT.read_text())
    pyproject_version = data["project"]["version"]
    assert geogen.__version__ == pyproject_version, (
        f"geogen.__version__={geogen.__version__!r} "
        f"does not match pyproject version={pyproject_version!r}"
    )


def test_version_is_pep440_parseable():
    import geogen

    # Lightweight PEP 440 check: the importlib-resolved version must at least
    # start with a numeric component. We don't pull `packaging` just for this.
    head = geogen.__version__.split("+", 1)[0].split(".", 1)[0]
    assert head.isdigit(), f"Non-PEP440 version: {geogen.__version__!r}"
