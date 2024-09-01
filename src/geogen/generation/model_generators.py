""" Sentence structures and generation processes for geo histories. """

import abc as _abc
import csv
from typing import List

import importlib.resources as resources
import os

import numpy as np
from pydtmc import MarkovChain

import geogen.generation.categorical_events as events
from geogen.model.geomodel import GeoModel, GeoProcess


class _GeostoryGenerator(_abc.ABC):
    """
    An interface for generating geological models, with common parameters and methods.

    Parameters
    ----------
    model_bounds : tuple, optional
        The bounds of the model in the form ((xmin, xmax), (ymin, ymax), (zmin, zmax)), by default ((-3840, 3840), (-3840, 3840), (-1920, 1920))
    model_resolution : tuple, optional
        The resolution of the model in the form (nx, ny, nz), by default (256, 256, 128)
    config : str, optional
        The path to a configuration file or object, if any.
    **kwargs : dict, optional
        Additional keyword arguments specific to the generator.
    """

    def __init__(self, model_bounds=None, model_resolution=None, config=None, **kwargs):

        self.model_bounds = model_bounds or (
            (-3840, 3840),
            (-3840, 3840),
            (-1920, 1920),
        )
        self.model_resolution = model_resolution or (256, 256, 128)
        self.config = config
        self.additional_params = kwargs

    def _history_to_model(self, hist: List[GeoProcess]) -> GeoModel:
        """Generate a model from a history and normalize the height."""
        model = GeoModel(bounds=self.model_bounds, resolution=self.model_resolution)
        model.add_history(hist)
        model.clear_data()
        model.compute_model(normalize=True)
        return model

    @_abc.abstractmethod
    def generate_models(self, n_samples: int = 1) -> List[GeoModel]:
        """Generate multiple geological models."""
        pass

    def generate_model(self) -> GeoModel:
        """Generate a single geological model."""
        return self.generate_models(1)[0]


class MarkovGeostoryGenerator(_GeostoryGenerator):
    """
    A class that generates geological models from a Markov process and general geoword.

    Inherits from:
    --------------
    _GeostoryGenerator

    Notes
    ----------
    This class has a strong dependency on the categorical_events module and the geowords module.
    Any changes to categorical_events object names may require changes to this class.

    Parameters
    ----------
    model_bounds : tuple, optional
        The bounds of the model in the form ((xmin, xmax), (ymin, ymax), (zmin, zmax)), by default ((-3840, 3840), (-3840, 3840), (-1920, 1920))
    model_resolution : tuple, optional
        The resolution of the model in the form (nx, ny, nz), by default (256, 256, 128)
    config : str, optional
        The path to a CSV file containing a labeled Markov transition matrix, a default matrix is provided.
        See the MarkovMatrixParser class for more information on format and requirements.
    """

    _START_STATE = "BaseStrata" # Name of the Markov chain start state, must reference valid events class
    _END_STATE = "End" # Name of the Markov chain termination event, must reference valid events class
    _MAX_STEPS = 20

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.markov_matrix_parser = MarkovMatrixParser(self.config)
        self.mc: MarkovChain = self.markov_matrix_parser.get_markov_chain()
        self.event_dictionary = self.markov_matrix_parser.get_event_dictionary()

    def build_geostory(self):
        """Build a geological history from a Markov chain."""
        sequence = self._build_markov_sequence()
        # Instantiate the event classes from the sequence
        sentence = [self.event_dictionary[state]() for state in sequence]
        # Generate the history from the instantiated events
        history = [word.generate() for word in sentence]
        return history

    def _build_markov_sequence(self) -> List[str]:
        """Generate a list of dictionary keys using the Markov chain."""
        sequence = self.mc.simulate(
            steps=self._MAX_STEPS,
            initial_state=self._START_STATE,
            final_state=self._END_STATE,
        )
        return sequence

    def generate_models(self, n_samples: int = 1) -> List[GeoModel]:
        """Generate multiple geological models."""
        models = []
        for _ in range(n_samples):
            history = self.build_geostory()
            model = self._history_to_model(history)
            models.append(model)
        return models

    def generate_model(self) -> GeoModel:
        """Generate a single geological model."""
        return self.generate_models(1)[0]


class MarkovMatrixParser:
    """
    A helper class to parse valid events and a transition matrix from a CSV file.

    Notes
    ----------
    The CSV file should be formatted as follows:
    - The first row contains state names, and the first column contains matching row labels.
    - The matrix should occupy the upper left portion of the CSV, with any additional columns or rows ignored.
    - The row and column labels should match in order, forming a square matrix. All the labels should
        correspond to valid event dictionary keys/ categorical event classes.

    - Sample spreadsheet to generate csv from:
    https://docs.google.com/spreadsheets/d/1OzP1ewVcsB4IKpeLPMQyVwLWbeFcTm4OtPxi-n7J5Ng/edit?gid=0#gid=0

    Parameters
    ----------
    path : str, optional
        The path to the CSV file containing the transition matrix. If None, the default path is used.

    Attributes
    ----------
    event_dictionary : dict
        A dictionary mapping event names to valid categorical event classes.
    states : list
        A list of state names corresponding to the rows and columns of the transition matrix.
    transition_matrix : np.ndarray
        A square matrix representing the transition probabilities between states.
    """

    def __init__(self, path=None):
        # Get a path
        if path is None:
            self.path = self._get_default_path()
        else:
            self.path = path
        self._validate_path(self.path)

        # Build a dictionary of string keys mapping to valid event classes
        self.event_dictionary = self._build_event_dictionary()
        # Load the transition matrix from the CSV file
        self.markov_states, self.transition_matrix = self._load_transition_matrix_from_csv(
            self.path
        )
        # Run a check on the matrix that it is valid markov matrix
        self._validate_transition_matrix()

    def _get_default_path(self):
        # Using importlib.resources to access the default Markov matrix file
        return resources.files('geogen.generation.markov_matrix').joinpath('default_markov_matrix.csv')

    def _validate_path(self, path):
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}")
        return path

    def _build_event_dictionary(self):
        """
        Build a mapping from strings to event classes in an automated fashion.

        Only include classes that are explicitly listed in the `__all__` attribute of the `categorical_events` module.
        """
        try:
            # Get the list of valid event names from __all__ in the events module
            valid_event_names = getattr(events, "__all__", [])

            # The categorical_events module contains all of the Markov events to be used.
            event_dictionary = {
                name: cls
                for name, cls in events.__dict__.items()
                if name in valid_event_names and isinstance(cls, type)
            }
        except Exception as e:
            raise ValueError(
                f"Failed to build event dictionary from the categorical events module: {e}"
            )

        return event_dictionary

    def _load_transition_matrix_from_csv(self, path):
        """
        Load a transition matrix from a CSV file.

        Format:
        The first row contains state names, and the first column contains matching row labels.
        The matrix should occupy the upper left portion of the CSV, with any additional columns or rows ignored.

        The row and column labels should match in order, forming a square matrix. All the labels should
        correspond to valid event dictionary keys/ categorical event classes.

        Example:

        ----| BaseStrata | Fold | End |----|----| This cell is ignored
        BaseStrata | 0.1 | 0.2 | 0.7  |----|----| It is not contigous with the upper left
        Fold       | 0.3 | 0.4 | 0.3 |
        End        | 0.1 | 0.1 | 0.8 |
        |          |     |     |     |
        These cells also ignored, as they are not part of the matrix

        """
        with open(path, mode="r") as file:
            reader = csv.reader(file)

            # Skip comment lines to get to the true header row
            for header_row in reader:
                if not any(
                    "#" in col for col in header_row
                ):  # Skip if any column contains '#'
                    break

            states = []
            for header in header_row[1:]:
                header = header.strip()
                if (
                    not header
                ):  # Stop if we hit an empty column, signaling end of contiguous block
                    break
                states.append(header)

            # Ensure that all headers are valid event dictionary keys with corresponding event classes
            for state in states:
                if state not in self.event_dictionary:
                    raise ValueError(
                        f"Invalid state name '{state}' in the CSV. Must be one of: {list(self.event_dictionary.keys())}"
                    )

            # Initialize an empty transition matrix
            matrix_size = len(states)
            matrix = np.zeros((matrix_size, matrix_size))

            # Populate the transition matrix from the remaining rows
            for i, row in enumerate(reader):
                row_label = row[0].strip()
                if (
                    not row_label
                ):  # Stop if we hit an empty row, signaling end of contiguous block
                    break

                # Validate the row label
                if row_label not in self.event_dictionary:
                    raise ValueError(
                        f"Invalid row label '{row_label}' in the CSV. Must be one of: {list(self.event_dictionary.keys())}"
                    )

                # Ensure that the row label matches the corresponding state
                if row_label != states[i]:
                    raise ValueError(
                        f"Row label '{row_label}' does not match the corresponding state '{states[i]}' in the header."
                    )

                # Extract and convert the row's probabilities, ignoring anything after the contiguous block
                row_data = row[1 : 1 + matrix_size]
                matrix[i] = list(map(float, row_data))
            return states, matrix

    def _validate_transition_matrix(self):
        """
        Ensure that the transition matrix is square and that the probabilities sum to 1.
        Provide detailed error messages including row sums and state names.
        """

        # Check if the matrix is square
        if not self.transition_matrix.shape[0] == self.transition_matrix.shape[1]:
            raise ValueError("Transition matrix must be square.")

        # Calculate the sum of each row
        row_sums = np.sum(self.transition_matrix, axis=1)

        # Check each row sum and provide detailed error messages
        for i, row_sum in enumerate(row_sums):
            if not np.isclose(row_sum, 1.0):
                state_name = self.markov_states[i]
                raise ValueError(
                    f"Row {i+1} ({state_name}) sum is {row_sum:.4f}, but it must sum to 1.0."
                )

    def get_event_dictionary(self):
        return self.event_dictionary

    def get_states(self):
        return self.markov_states

    def get_transition_matrix(self):
        return self.transition_matrix

    def get_markov_chain(self) -> MarkovChain:
        # Create the MarkovChain library object
        mc = MarkovChain(self.transition_matrix, self.markov_states)
        return mc