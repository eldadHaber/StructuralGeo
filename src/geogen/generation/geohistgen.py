"""Collection of functions to generate histories and models using GeoWords and GeoProcesses."""

from typing import List

import geogen.model as geo

from .geowords import GeoWord


def generate_history(sentence: List[GeoWord]) -> List[geo.GeoProcess]:
    """Generate a geological history from a sentence."""
    h = [word.generate() for word in sentence]
    return h
