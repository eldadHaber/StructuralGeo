""" Categorical definitions for sampling of broad categories of GeoWords."""

import warnings
from collections import namedtuple
from typing import List
from .geowords import *
from structgeo.model import GeoProcess
import numpy as np

class EventSampler(GeoWord):
    """
    A special case of GeoWord that selects from a set of cases with associated probabilities.
    This class is used to form more general categories of events that can be sampled from.
    
    The cases include a name, probability of selection, and a sequence of actions (GeoWords or GeoProcesses) 
    that form the history of the event.
    
    Parameters
    ----------
    cases : List[Case]
        A list of Cases to sample from. Each case should have a name, probability, and a sequence of GeoWords or GeoProcesses.
    rng : Optional[np.random.Generator]
        A random number generator for reproducibility.   
    """
    Event = namedtuple('Case', ['name', 'p', 'processes'])

    def __init__(self, cases: List[Event], rng=None):
        super().__init__(rng)
        self.cases = cases
        self.selected_case = None
        self.probabilities = None
        self._validate_cases()
        self._validate_probabilities()

    def generate(self):
        """
        Generate the geological history by selecting a case and building the corresponding history.

        Returns
        -------
        geo.CompoundProcess
            A sampled geological history snippet with a CompoundProcess wrapper.
        """
        self.hist.clear()
        self.build_history()
        name = f"{self.__class__.__name__}: {self.selected_case.name}"
        geoprocess = geo.CompoundProcess(self.hist.copy(), name=name)
        return geoprocess

    def build_history(self):
        """
        Randomly select a case based on probabilities and build the corresponding history.
        """
        selected_index = self.rng.choice(len(self.cases), p=self.probabilities)
        self.selected_case = self.cases[selected_index]
        self.add_process(self.selected_case.processes)

    def _validate_cases(self):
        """
        Ensure that the case list is correctly defined with valid types.
        """
        if not self.cases:
            raise ValueError("Cases are not defined.")
        
        for case in self.cases:
            if not isinstance(case.name, str):
                raise TypeError(f"Case name must be a string, got {type(case.name).__name__}.")
            if not isinstance(case.p, float):
                raise TypeError(f"Case probability must be a float, got {type(case.p).__name__}.")
            if not isinstance(case.processes, list):
                raise TypeError(f"Case processes must be a list, got {type(case.processes).__name__}.")
            for process in case.processes:
                if not isinstance(process, (GeoProcess, GeoWord)):
                    raise TypeError(f"Processes must be instances of GeoProcess or GeoWord, got {type(process).__name__}.")

    def _validate_probabilities(self):
        """
        Ensure that the probabilities sum to 1 and renormalize if necessary.
        """
        probabilities = np.array([case.p for case in self.cases])
        sum_prob = np.sum(probabilities)
        
        if not np.isclose(sum_prob, 1.0):
            warnings.warn(
                f"Probabilities sum to {sum_prob:.4f}, but should sum to 1.0. Renormalizing.",
                RuntimeWarning
            )
            probabilities = np.array(probabilities) / sum_prob
        
        self.probabilities 

class EventBaseStrata(EventSampler):
    """
    A sampling regime for base strata.
    """

    def __init__(self, rng=None):
        cases = [
            self.Event(name="Basement",                p=0.4, processes=[InfiniteBasement(), EventSediment()]),
            self.Event(name="Sediment: Markov",        p=0.2, processes=[InfiniteSedimentMarkov()]),
            self.Event(name="Sediment: Uniform",       p=0.2, processes=[InfiniteSedimentUniform()]),
            self.Event(name="Sediment: Tilted Markov", p=0.2, processes=[InfiniteSedimentTilted()])
        ]
        super().__init__(cases=cases, rng=rng)

class EventSediment(EventSampler):
    """
    A sampling regime for sediment events.
    """

    def __init__(self, rng=None):
        cases = [
            self.Event(name="Fine",   p=0.4, processes=[FineRepeatSediment()]),
            self.Event(name="Coarse", p=0.5, processes=[CoarseRepeatSediment()]),
            self.Event(name="Single", p=0.1, processes=[SingleRandSediment()])
        ]
        super().__init__(cases=cases, rng=rng)
        
class EventFold(EventSampler):
    """
    A sampling regime for folding events.
    """

    def __init__(self, rng=None):
        cases = [
            self.Event(name="Simple",  p=0.2, processes=[SimpleFold()]),
            self.Event(name="Shaped",  p=0.3, processes=[ShapedFold()]),
            self.Event(name="Fourier", p=0.5, processes=[FourierFold()]),
        ]
        super().__init__(cases=cases, rng=rng)

class EventErosion(EventSampler):
    """
    A sampling regime for erosion events.
    """

    def __init__(self, rng=None):
        cases = [
            self.Event(name="Flat",   p=0.2, processes=[FlatUnconformity()]),
            self.Event(name="Tilted", p=0.4, processes=[TiltedUnconformity()]),
            self.Event(name="Wave",   p=0.4, processes=[WaveUnconformity()]),
        ]
        super().__init__(cases=cases, rng=rng)