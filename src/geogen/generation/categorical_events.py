""" Categorical definitions for sampling of broad categories of GeoWords."""

__all__ = [
    "BaseStrata",
    "Sediment",
    "Erosion",
    "Dike",
    "Sills",
    "Pluton",
    "OreDeposit",
    "Fold",
    "Fault",
    "Slip",
    "Mountains",
    "End",
]

import warnings
from collections import namedtuple
from typing import List

import numpy as np

from geogen.generation.geowords import *
from geogen.model import CompoundProcess, GeoProcess


class _EventTemplateClass(GeoWord):
    """
    A special case of GeoWord that selects from a set of cases with associated probabilities.
    This class is used to form more general categories of events that can be sampled from.

    The Events include a name, probability of selection, and a sequence of actions (GeoWords or GeoProcesses)
    that form the history of the event.

    Parameters
    ----------
    cases : List[Event]
        A list of Events to sample from. Each event should have a name, probability, and a sequence of GeoWords or GeoProcesses.
    rng : Optional[np.random.Generator]
        A random number generator for reproducibility.
    """

    Event = namedtuple("Case", ["name", "p", "processes"])

    def __init__(self, cases: List[Event], seed=None):
        super().__init__(seed)
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
        geoprocess = CompoundProcess(self.hist.copy(), name=name)
        return geoprocess

    def build_history(self):
        """
        Randomly select a case based on probabilities and build the corresponding history.
        """
        assert self.probabilities is not None, "Probabilities are not defined."
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
                    raise TypeError(
                        f"Processes must be instances of GeoProcess or GeoWord, got {type(process).__name__}."
                    )

    def _validate_probabilities(self):
        """
        Ensure that the probabilities sum to 1 and renormalize if necessary.
        """
        probabilities = np.array([case.p for case in self.cases])
        sum_prob = np.sum(probabilities)

        if not np.isclose(sum_prob, 1.0):
            warnings.warn(
                f"{self.__class__.__name__}: Probabilities sum to {sum_prob:.4f}, but should sum to 1.0. Renormalizing.",
                RuntimeWarning,
            )
            probabilities = np.array(probabilities) / sum_prob

        self.probabilities = probabilities


class BaseStrata(_EventTemplateClass):
    """
    A sampling regime for base strata.
    """

    def __init__(self, seed=None):
        cases = [
            self.Event(name="Basement", p=0.27, processes=[InfiniteBasement(), Sediment()]),
            self.Event(name="Sediment: Markov", p=0.22, processes=[InfiniteSedimentMarkov()]),
            self.Event(name="Sediment: Uniform", p=0.22, processes=[InfiniteSedimentUniform()]),
            self.Event(
                name="Sediment: Tilted Markov",
                p=0.29,
                processes=[InfiniteSedimentTilted()],
            ),
        ]
        super().__init__(cases=cases, seed=seed)


class Sediment(_EventTemplateClass):
    """
    A sampling regime for sediment events.
    """

    def __init__(self, seed=None):
        cases = [
            self.Event(name="Fine", p=0.4, processes=[FineRepeatSediment()]),
            self.Event(name="Coarse", p=0.5, processes=[CoarseRepeatSediment()]),
            self.Event(name="Single", p=0.1, processes=[SingleRandSediment()]),
        ]
        super().__init__(cases=cases, seed=seed)


class Erosion(_EventTemplateClass):
    """
    A sampling regime for erosion events.
    """

    def __init__(self, seed=None):
        cases = [
            self.Event(name="Flat", p=0.20, processes=[FlatUnconformity()]),
            self.Event(name="Tilted", p=0.25, processes=[TiltedUnconformity()]),
            self.Event(name="TiltCutFill", p=0.20, processes=[TiltCutFill()]),
            self.Event(name="Wave", p=0.35, processes=[WaveUnconformity()]),
        ]
        super().__init__(cases=cases, seed=seed)


class Dike(_EventTemplateClass):
    """
    A sampling regime for intrusion events.
    """

    def __init__(self, seed=None):
        cases = [
            self.Event(name="Dike", p=0.4, processes=[DikePlaneWord()]),
            self.Event(name="WarpedDike", p=0.4, processes=[SingleDikeWarped()]),
            self.Event(name="DikeGroup", p=0.2, processes=[DikeGroup()]),
        ]
        super().__init__(cases=cases, seed=seed)


class Sills(_EventTemplateClass):
    """
    A sampling regime for sill events.
    """

    def __init__(self, seed=None):
        cases = [
            self.Event(name="SillSingle", p=0.2, processes=[SillWord()]),
            # Note the sill system places a large sediment deposit at same time for embedding
            self.Event(name="SillSystem", p=0.8, processes=[SillSystem()]),
        ]
        super().__init__(cases=cases, seed=seed)


class Pluton(_EventTemplateClass):
    """
    A sampling regime for volcanic events.
    """

    def __init__(self, seed=None):
        cases = [
            self.Event(name="Laccolith", p=0.4, processes=[Laccolith()]),
            self.Event(name="Lopolith", p=0.6, processes=[Lopolith()]),
        ]
        super().__init__(cases=cases, seed=seed)


class OreDeposit(_EventTemplateClass):
    """
    A sampling regime for ore deposit events.

    Can be expanded to include more types in the future
    """

    def __init__(self, seed=None):
        cases = [
            self.Event(name="BlobCluster", p=1.0, processes=[BlobCluster()]),
        ]
        super().__init__(cases=cases, seed=seed)


class Fold(_EventTemplateClass):
    """
    A sampling regime for folding events.
    """

    def __init__(self, seed=None):
        cases = [
            self.Event(name="Simple", p=0.2, processes=[SimpleFold()]),
            self.Event(name="Shaped", p=0.3, processes=[ShapedFold()]),
            self.Event(name="Fourier", p=0.5, processes=[FourierFold()]),
        ]
        super().__init__(cases=cases, seed=seed)


class Fault(_EventTemplateClass):
    """A sampling regime for fault events."""

    def __init__(self, seed=None):
        cases = [
            self.Event(name="Normal", p=0.1, processes=[FaultNormal()]),
            self.Event(name="Reverse", p=0.1, processes=[FaultReverse()]),
            self.Event(name="StrikeSlip", p=0.1, processes=[FaultStrikeSlip()]),
            self.Event(name="HorstGraben", p=0.1, processes=[FaultHorstGraben()]),
            self.Event(name="StrikeSlip", p=0.25, processes=[FaultStrikeSlip()]),
            self.Event(name="FullyRandom", p=0.2, processes=[FaultRandom()]),
            self.Event(name="Sequence", p=0.15, processes=[FaultSequence()]),
        ]
        super().__init__(cases=cases, seed=seed)
        
class Mountains(_EventTemplateClass):
    """A sampling regime for mountain events."""

    def __init__(self, seed=None):
        cases = [
            self.Event(name="TiltedMountains", p=1.0, processes=[TiltedMountains()]),
        ]
        super().__init__(cases=cases, seed=seed)


# TODO: Implement Slip events in GeoWords and add to the Slip class
class Slip(_EventTemplateClass):
    """A sampling regime for slip events."""

    def __init__(self, seed=None):
        cases = [self.Event(name="Null", p=1.0, processes=[NullWord()])]
        super().__init__(cases=cases, seed=seed)
        NotImplementedError()


class End(_EventTemplateClass):
    """An ending flag for the geostory."""

    def __init__(self, seed=None):
        cases = [self.Event(name="Termination of Sequence", p=1.0, processes=[NullWord()])]
        super().__init__(cases=cases, seed=seed)
