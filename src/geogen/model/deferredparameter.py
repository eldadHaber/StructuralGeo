import numpy as np

from geogen.model.geoprocess import DeferredParameter, Sedimentation, Transformation


class BacktrackedPoint(DeferredParameter):
    """
    A deferred parameter used to specify a point in the final coordinate frame, which is then backtracked
    through the mesh deformations to determine its original position in the initial coordinate frame.

    This class is useful in geological modeling when a point of interest is defined in the final (present-day)
    configuration of the model, but needs to be traced back through all previous transformations to find its
    corresponding position at an earlier stage of geological history.

    Parameters
    ----------
    point : tuple
        A 3D coordinate tuple (x, y, z) representing the point in the final coordinate frame.

    Methods
    -------
    compute_func(xyz, data, history, index)
        Resolves the deferred parameter by applying the inverse of all transformations that occurred
        after the current process in the model's history, effectively backtracking the point to its original
        position in the initial coordinate frame.
    """

    def __init__(self, point: tuple):
        super().__init__()
        self.point = point

    def __str__(self):
        return str(self.point)

    def compute_func(self, xyz, data, history, index) -> tuple:
        backtracked_point = np.atleast_2d(
            np.array(self.point, dtype=np.float32)
        )  # Cast tuple into 2D array for processing
        data = np.array([np.nan])  # Dummy data to go with point

        # Reverse through the history events applying transformations
        future = history[index + 1 :]
        for event in reversed(future):
            if isinstance(event, Transformation):
                backtracked_point, _ = event.run(xyz=backtracked_point, data=data)

        backtracked_point = tuple(backtracked_point.flatten())  # Cast back to 3d tuple

        return backtracked_point  # Sets the backtracked point as the resolved value


class SedimentConditionedOrigin(DeferredParameter):
    """
    Takes an x and a y coordinate and conditions the z coordinate based on the sedimentation process
    that occurred at the same location. The z coordinate is set to the specified boundary of previous
    sedimentation process.

    Parameters
    ----------
    x : float
        The x coordinate of the point.
    y : float
        The y coordinate of the point.
    boundary_index : int
        The index of the boundary to retrieve from the previous sedimentation process.
    """

    def __init__(self, x: float, y: float, boundary_index: int):
        super().__init__()
        self.x = x
        self.y = y
        self.boundary_index = boundary_index

    def __str__(self):
        return f"({self.x}, {self.y}, boundary {self.boundary_index})"

    def compute_func(self, xyz, data, history, index) -> tuple:
        # Find the most recent sedimentation process
        for event in reversed(history[:index]):
            if isinstance(event, Sedimentation):
                sedimentation = event
                break
        else:
            raise ValueError(
                "SedimentConditionedSillOrigin: No sedimentation process found in the history."
            )

        # Find the boundary at the specified index
        boundaries = sedimentation.boundaries
        if self.boundary_index >= len(boundaries):
            raise IndexError("SedimentConditionedSillOrigin: Boundary index out of range.")

        boundary = boundaries[self.boundary_index]

        # Set the z coordinate to the boundary value
        return (self.x, self.y, boundary)


class LookBackParameter(DeferredParameter):
    """
    A deferred parameter that looks back a specified number of processes in the history and
    retrieves an attribute. If the attribute is a list, it can optionally return an element at a specified index.

    Parameters
    ----------
    steps_back : int
        The number of steps to look back in the history.
    attr_name : str
        The name of the attribute to retrieve from the targeted process.
    list_index : int, optional
        The index to retrieve from the list attribute. If not provided, the entire attribute is returned.
    """

    def __init__(self, steps_back, attr_name, list_index=None):
        super().__init__()
        self.steps_back = steps_back
        self.attr_name = attr_name
        self.list_index = list_index

    def compute_func(self, xyz, data, history, index):
        target_index = index - self.steps_back
        if target_index < 0:
            raise IndexError("LookBackParameter: Cannot look back beyond the start of the history.")

        target_process = history[target_index]
        if not hasattr(target_process, self.attr_name):
            raise AttributeError(
                f"LookBackParameter: The attribute '{self.attr_name}' does not exist in the target process."
            )

        attr_value = getattr(target_process, self.attr_name)

        # If list_index is provided, return the specific element
        if self.list_index is not None:
            if isinstance(attr_value, list):
                return attr_value[self.list_index]
            else:
                raise TypeError(
                    f"LookBackParameter: The attribute '{self.attr_name}' is not a list, so cannot index it."
                )

        # Otherwise, return the whole attribute
        return attr_value
