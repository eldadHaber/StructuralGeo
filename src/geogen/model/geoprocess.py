""" Base classes for implementing parametric geological processes."""

import warnings
from typing import List

import numpy as np

from geogen.model.util import rotate, slip_normal_vectors


class GeoProcess:
    """
    Base class template for all geological processes. Includes handling for cases where process
    parameters that are conditional such as origin are not known until runtime or model generation time.
    These are deferred parameters that are resolved at runtime.
    """

    def apply_process(self, xyz, data, history, index):
        """
        This method should not be overridden by subclasses. The subclass should implement the `run` method.

        Parameters
        ----------
        xyz : np.ndarray
            The coordinates of the model points.
        data : np.ndarray
            The geological data to modify.
        history : list
            The history of geological processes applied to the model.
        index : int
            The index of this process in the history list for context.
        """
        try:
            # Ensure deferred parameters are resolved (conditioned on model state and history)
            self.resolve_deferred_parameters(xyz, data, history, index)
            # Delegate actual processing to the subclass's `run` method (mutates xyz and data)
            return self.run(xyz, data)

        except Exception as e:
            # Handle the exception, log the issue, and potentially skip the process if cant be resolved
            warnings.warn(
                f"Process {str(self)} at index {index} failed: {e}. Skipping process."
            )
            return (
                xyz,
                data,
            )  # Return unmodified data or handle differently based on the case

    def resolve_deferred_parameters(self, xyz, data, history, index):
        """
        Resolves all deferred parameters in the process by iterating over attributes.
        A deferred parameter is a special class that allows for conditioning the parameter based
        on the model state and history and run time, instead of being fixed at initialization.

        For example: self.origin = BacktrackedPoint((0, 0, 0)) will trace the point (0,0,0) back
        through history from the present state at runtime.

        Parameters
        ----------
        xyz : np.ndarray
            The coordinates of the model points.
        data : np.ndarray
            The geological data.
        history : list
            The history of geological processes applied to the model.
        index : int
            The index of this process in the history list for context.
        """
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, DeferredParameter):
                try:
                    # Attempt to resolve the deferred parameter
                    resolved_value = attr_value.resolve(xyz, data, history, index)
                    setattr(self, attr_name, resolved_value)
                except Exception as e:
                    # Raise an error to be caught in apply_process
                    raise RuntimeError(
                        f"Error resolving deferred parameter '{attr_name}': {e}"
                    )

    def run(self, xyz, data):
        """
        The method to be implemented by subclasses for the actual geological process logic.

        This is where the specific logic for each subclass should go.

        Parameters
        ----------
        xyz : np.ndarray
            The coordinates of the model points.
        data : np.ndarray
            The geological data to modify.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class DeferredParameter:
    """
    Base class template to allow option for a GeoProcess parameter to be deferred until runtime.
    This allows for conditioning based on the model history.

    Attributes:
        compute_func (callable): A function that computes the parameter's value.
    """

    def __init__(self):
        self.value = None  # Holds the resolved value once computed

    def resolve(self, xyz, data, history, index):
        """
        Resolve the deferred parameter using the provided computation function.

        Parameters
        ----------
        xyz : np.ndarray
            The coordinates of the model points.
        data : np.ndarray
            The geological data.
        history : list
            The history of geological processes applied to the model.
        index : int
            The index of this process in the history list.

        Returns
        -------
        The resolved value.
        """
        # Avoid recomputing the value if it has already been resolved
        if self.value is None:
            self.value = self.compute_func(xyz, data, history, index)
        return self.value


class Deposition(GeoProcess):
    """
    Base class for all deposition processes, such as layers and dikes.

    Depositions modify the geological data (e.g., rock types) associated with a mesh point
    without altering or transforming the mesh.
    """

    pass


class Transformation(GeoProcess):
    """
    Base class for all transformation processes, such as tilting and folding.

    Transformations modify the mesh coordinates without altering the data contained at each point.
    Frame convention is that north is 0 strike on positive y-axis
    """

    pass


class CompoundProcess(GeoProcess):
    """
    A compound geological process that consists of multiple sequential sub-processes.

    Can include deposition, transformation, or other compound processes. Used to group
    together multiple processes into a single entity.

    Parameters
    ----------
    processes : List[GeoProcess]
        A list of sub-processes to apply sequentially.
    name : str
        An optional name for the compound process, will display in the model history.
    """

    def __init__(self, processes: List[GeoProcess] = None, name: str = None):
        self.name = name
        self.history = processes if processes is not None else []
        self._check_history()

    def __str__(self) -> str:
        name_str = f" ({self.name})" if self.name else ""
        history_str = f"{self.__class__.__name__}{name_str} with {len(self.history)} sub-processes:\n"

        for index, sub_process in enumerate(self.history):
            history_str += f"    {index + 1}. {str(sub_process)}\n"

        return history_str.strip()  # Remove the trailing newline

    def _check_history(self):
        if not self.history:
            warnings.warn(
                f"{self.__class__.__name__} initialized with an empty history. Ensure to add processes before computation.",
                UserWarning,
            )

    def unpack(self):
        if not self.history:
            raise ValueError(f"{self.__class__.__name__} has no history to unpack.")
        unpacked_history = []
        for process in self.history:
            if isinstance(process, CompoundProcess):
                unpacked_history.extend(process.unpack())
            else:
                unpacked_history.append(process)
        return unpacked_history


class NullProcess(GeoProcess):
    """A null process that does not modify the model."""

    def __str__(self):
        return "NullProcess: no compute action."

    def run(self, xyz, data):
        return xyz, data


class Layer(Deposition):
    """
    Fill the model with a layer of rock.

    Parameters
    ----------
    base : float
        The z-coordinate of the bottom of the layer.
    width : float
        The thickness of the layer in the z-direction.
    value : int
        The rock type value to assign to the layer.
    """

    def __init__(self, base, width, value):
        self.base = base
        self.width = width
        self.value = value

    def run(self, xyz, data):
        # Extract z-coordinates
        z_coords = xyz[:, 2]

        # Create a mask where z is within the specified range and data is None
        mask = (
            (self.base <= z_coords)
            & (z_coords <= self.base + self.width)
            & (np.isnan(data))
        )

        # Apply the mask and update data where condition is met
        data[mask] = self.value

        # Return the unchanged xyz and the potentially modified data
        return xyz, data


class Shift(Transformation):
    """Shift the model by a given vector."""

    def __init__(self, vector):
        self.vector = np.array(vector)

    def __str__(self):
        return f"Shift: vector {self.vector}"

    def run(self, xyz, data):
        # Apply the shift to the xyz points (inverse operation is negative vector)
        xyz_transformed = xyz - self.vector
        return xyz_transformed, data


class Rotate(Transformation):
    """
    Rotate the model by a given angle about an axis.

    Parameters
    ----------
    axis : tuple
        The axis of rotation in x, y, z coordinates.
    angle : float
        The angle of rotation in degrees, using left-hand rule or clockwise rotation.
    """

    def __init__(self, axis, angle):
        self.angle = np.radians(angle)
        self.axis = np.array(axis)

    def __str__(self):
        return f"Rotation: angle {np.degrees(self.angle):.1f}°, axis {self.axis}"

    def run(self, xyz, data):
        R = rotate(self.axis, self.angle)
        # Apply the rotation to the xyz points
        xyz = xyz @ R.T
        return xyz, data


class Bedrock(Deposition):
    """
    Fill the model with a base level of bedrock.

    Parameters
    ----------
    base : float
        The z-coordinate of the top of the limitless bedrock.
    value : int
        The rock type value to assign to the bedrock.
    """

    def __init__(self, base, value):
        self.base = base
        self.value = value

    def __str__(self):
        return f"Bedrock: with z <= {self.base:.1f} and value {self.value:.1f}"

    def run(self, xyz, data):
        # Extract z-coordinates
        z_coords = xyz[:, 2]

        # Create a mask where z is below the base level
        mask = z_coords <= self.base

        # Apply the mask and update data where condition is met
        data[mask] = self.value

        # Return the unchanged xyz and the potentially modified data
        return xyz, data

    def last_value(self):
        return self.value


class Sedimentation(Deposition):
    """
    Fill with layers of sediment, with rock values and thickness of layers given as lists.
    The base elevation from which to fill can either be specified or can be deduced at generation time from
    the lowest unfilled value in the mesh. Will not overwrite existing rock.

    Parameters
    ----------
    value_list : list
        A list of rock type values for the layers.
    thickness_list : list
        A list of thicknesses for the layers.
    base : float, optional
        The base elevation from which to start filling. If not specified, the base will be calculated at runtime.

    Attributes
    ----------
    value_list : list
        A list of rock type values for the layers.
    thickness_list : list
        A list of thicknesses for the layers.
    base : float
        The base elevation from which to start filling.
    boundaries : np.ndarray
        The calculated z boundary value for each layer. There are n+1 boundaries for n layers.
    """

    def __init__(self, value_list, thickness_list, base=np.nan):
        self.value_list = list(value_list)
        self.thickness_list = thickness_list
        self.base = base  # If the base is np.nan, it will be calculated at runtime
        self.boundaries = None  # Calculated at runtime

    def __str__(self):
        values_summary = ", ".join(
            f"{v:.1f}" if isinstance(v, float) else str(v) for v in self.value_list[:3]
        )
        values_summary += "..." if len(self.value_list) > 3 else ""
        thicknesses = ", ".join(f"{t:.3f}" for t in self.thickness_list[:3])
        thicknesses += "..." if len(self.thickness_list) > 3 else ""
        return (
            f"Sedimentation: rock type values [{values_summary}], "
            f"and thicknesses {thicknesses}, with base = {self.base}"
        )

    def run(self, xyz, data):
        z_values, nan_idxs = self.get_nan_z_values(xyz, data)
        current_base = self.calculate_base(z_values)
        thicknesses = self.extend_thicknesses()
        self.boundaries = self.calculate_boundaries(current_base, thicknesses)
        self.assign_values_to_data(z_values, nan_idxs, self.boundaries, data)
        return xyz, data

    def get_nan_z_values(self, xyz, data):
        """Extract z values where data is NaN and return them along with their mask."""
        nan_idxs = np.isnan(data)
        z_values = xyz[nan_idxs, 2] if np.any(nan_idxs) else np.array([])
        return z_values, nan_idxs

    def calculate_base(self, z_values):
        """Calculate the base elevation, using the lowest z value where data is NaN if base is not set."""
        if z_values.size > 0:
            return np.min(z_values) if np.isnan(self.base) else self.base
        return self.base if not np.isnan(self.base) else float("Inf")

    def extend_thicknesses(self):
        """Extend the thickness list to match the number of layers."""
        num_layers = len(self.value_list)
        if len(self.thickness_list) < num_layers:
            repetitions = (num_layers + len(self.thickness_list) - 1) // len(
                self.thickness_list
            )
            return (self.thickness_list * repetitions)[:num_layers]
        return self.thickness_list[:num_layers]

    def calculate_boundaries(self, current_base, thicknesses):
        """Calculate the boundaries for each layer based on the base and thicknesses."""
        return current_base + np.concatenate(([0], np.cumsum(thicknesses)))

    def assign_values_to_data(self, z_values, nan_idxs, boundaries, data):
        """Assign the sediment values to the data array based on the layer boundaries."""
        if z_values.size > 0:
            # Bin the z values into the layer (which layer they belong to)
            layer_indices = np.digitize(z_values, boundaries)
            # Map the layer indices to the corresponding value, 0 bin and last bin are out of bounds (no layer)
            extended_value_list = np.array([np.nan] + self.value_list + [np.nan])
            data[nan_idxs] = extended_value_list[layer_indices]

    def last_value(self):
        return self.value_list[-1]


class DikePlane(Deposition):
    """
    A base planar dike intrusion with variable thickness.

    Parameters
    ----------
    strike : float
        Strike angle in clockwise degrees (center-line of the dike) from north (y-axis).
    dip : float
        Dip angle in degrees.
    width : float
        Net width of the dike.
    origin : tuple of float
        Origin point of the local coordinate frame (x, y, z).
    value : int
        Value of the rock type to assign to the dike.
    thickness_func : callable, optional
        Function that determines thickness multiplier as a function of (x, y) in the local dike frame.
        x corresponds to the vertical (dip) direction, and y corresponds to the lateral (strike)
        direction. The default thickness function is a constant unity function giving constant width.

    Notes
    -----
    The default thickness function assumes a constant thickness, but you can pass a custom
    function to vary the thickness across the dike plane.
    """

    def __init__(
        self,
        strike=0.0,
        dip=90.0,
        width=1.0,
        origin=(0, 0, 0),
        value=0.0,
        thickness_func=None,
    ):
        self.strike = np.radians(strike)
        self.dip = np.radians(dip)
        self.width = width
        self.origin = origin
        self.value = value
        self.thickness_func = (
            thickness_func if thickness_func else self.default_thickness_func
        )

    def __str__(self):
        # Convert radians back to degrees for more intuitive understanding
        strike_deg = np.degrees(self.strike)
        dip_deg = np.degrees(self.dip)
        if isinstance(self.origin, DeferredParameter):
            origin_str = str(self.origin)  # Use DeferredParameter's __str__ method
        else:
            # Format the tuple to limit decimal points
            origin_str = (
                f"({self.origin[0]:.2f},{self.origin[1]:.2f},{self.origin[2]:.2f})"
            )

        return (
            f"Dike: strike {strike_deg:.1f}°, dip {dip_deg:.1f}°, width {self.width:.1f}, "
            f"origin ({origin_str}), value {self.value:.1f}."
        )

    def run(self, xyz, data):
        self.origin = np.array(self.origin)
        # Calculate rotation matrices to align coordinates with the dike plane
        M1 = rotate([0, 0, 1.0], self.strike)  # Rotation around z-axis for strike
        M2 = rotate(
            [0.0, 1.0, 0], -self.dip
        )  # Rotation around y-axis in strike frame for dip

        # Combine rotations and apply to the coordinates
        xyz_local = (xyz - self.origin) @ M1.T @ M2.T

        # Calculate distances from the dike plane in the local frame
        x_dist = xyz_local[:, 0]  # Dipped direction distance
        y_dist = xyz_local[:, 1]  # Strike direction distance
        z_dist = xyz_local[:, 2]  # Normal direction distance

        # Determine the thickness based on the thickness function and width
        local_thickness = self.thickness_func(x_dist, y_dist) * self.width

        # Create mask for points within the dike thickness
        mask = (np.abs(z_dist) <= local_thickness / 2.0) & (~np.isnan(data))

        # Update data where the mask is true
        data[mask] = self.value

        return xyz, data

    def default_thickness_func(self, x, y):
        """Default thickness function: constant thickness across the dike plane."""
        return 1  # Constant unity function

    def last_value(self):
        return self.value


Dike = DikePlane  # Backward compatibility for Dike class in earlier versions


class DikeColumn(Deposition):
    """
    Columnar dike intrusion.

    Parameters
    ----------
    origin : tuple, optional
        Origin point of the dike in the model reference frame; column propagates downward in a column.
    diam : float, optional
        Diameter of the dike.
    depth : float, optional
        Stopping depth of the dike; -infinity by default goes down through the entire model.
    minor_axis_scale : float, optional
        Scaling factor for the x-axis of the dike.
    rotation : float, optional
        Rotation of the dike in the xy plane.
    value : int, optional
        Value of rock-type to assign to the dike.
    clip : bool, optional
        Clip the dike to not protrude above the surface.
    end_point : tuple, optional
        Optional end point of the dike; if provided, the dike will extend to this point from origin.
    """

    def __init__(
        self,
        origin=(0, 0, 0),
        diam=100,
        depth=np.inf,
        minor_axis_scale=1.0,
        rotation=0.0,
        value=0.0,
        clip=False,
        end_point=None,
    ):
        self.origin = origin
        self.diam = diam
        self.depth = depth
        self.minor_scale = minor_axis_scale
        self.rotation = rotation
        self.value = value
        self.clip = clip
        self.end_point = end_point

    def __str__(self):
        if isinstance(self.origin, DeferredParameter):
            origin_str = str(self.origin)  # Use DeferredParameter's __str__ method
        else:
            # Format the tuple to limit decimal points
            origin_str = (
                f"({self.origin[0]:.2f},{self.origin[1]:.2f},{self.origin[2]:.2f})"
            )
        return (
            f"DikeColumn: origin ({origin_str}), diam {self.diam:.1f}, depth {self.depth:.1f}, "
            f"minor_axis_scale {self.minor_scale:.2f}, rotation {self.rotation:.1f}, value {self.value:.1f}."
        )

    def run(self, xyz, data):
        self.origin = np.array(self.origin)
        self.end_point = (
            np.array(self.end_point) if self.end_point is not None else None
        )
        # Translate points to origin coordinate frame
        v0 = xyz - self.origin

        # Rotate the frame to align with endpoint if provided, calculate new downward depth
        if self.end_point is not None:
            # Calculate the direction vector from origin to end point
            direction = self.end_point - self.origin
            self.depth = np.linalg.norm(direction)
            # Calculate the rotation matrix to align the direction vector with the z-axis
            R = self.align_vector_with_axis(direction)
            v0 = -v0 @ R.T

        # Rotate the points in the xy plane of the plug formation ccw
        R = rotate([0, 0, 1], np.deg2rad(self.rotation))
        v0 = v0 @ R.T
        # Scale the points along the x-axis (minor axis)
        v0[:, 0] /= self.minor_scale

        x, y, z = v0[:, 0], v0[:, 1], v0[:, 2]
        r = np.sqrt(x**2 + y**2)

        mask = (r <= self.diam / 2.0) & (z <= 0) & (z >= (self.origin[2] - self.depth))
        if self.clip:
            mask &= ~np.isnan(data)
        data[mask] = self.value

        return xyz, data

    def align_vector_with_axis(self, v):
        """Calculate the rotation matrix to align a vector with z axis."""
        v = np.array(v)
        z_axis = np.array([0, 0, 1])
        # Calculate the cross product between the vector and z-axis
        axis = np.cross(v, z_axis)
        # Calculate the angle between the vector and z-axis
        angle = np.arccos(
            np.dot(v, z_axis) / (np.linalg.norm(v) * np.linalg.norm(z_axis))
        )
        # Calculate the rotation matrix
        return rotate(axis, angle)


class DikeHemisphere(Deposition):
    """
    A lenticular dike intrusion.

    This class represents a dike intrusion with a hemispherical or lenticular shape. The dike can be rotated, scaled, and
    positioned within a 3D geological model.

    Parameters
    ----------
    origin : tuple, optional
        The origin point of the dike in the model reference frame; defaults to (0, 0, 0).
    diam : float, optional
        The diameter of the dike; defaults to 500.
    height : float, optional
        The height of the dike; defaults to 100.
    minor_axis_scale : float, optional
        Scaling factor for the x-axis of the dike; defaults to 1.0.
    rotation : float, optional
        Rotation of the dike in the xy plane, in degrees; defaults to 0.0.
    value : float, optional
        Value of rock-type to assign to the dike; defaults to 0.0.
    upper : bool, optional
        If True, the upper hemisphere of the dike is used; if False, the lower hemisphere is used; defaults to True.
    clip : bool, optional
        If True, clips the dike to not protrude above the surface; defaults to False.
    z_function : callable, optional
        A function defining the z-coordinate for the dike's shape; defaults to an elliptical hemisphere function if None is provided.
    """

    def __init__(
        self,
        origin=(0, 0, 0),
        diam=500,
        height=100,
        minor_axis_scale=1.0,
        rotation=0.0,
        value=0.0,
        upper=True,
        clip=False,
        z_function=None,
    ):
        self.origin = origin
        self.diam = diam
        self.height = height
        self.minor_scale = minor_axis_scale
        self.rotation = rotation
        self.value = value
        self.upper = upper
        self.clip = clip
        self.z_function = self.default_z_function if z_function is None else z_function

    def __str__(self):
        if isinstance(self.origin, DeferredParameter):
            origin_str = str(self.origin)  # Use DeferredParameter's __str__ method
        else:
            # Format the tuple to limit decimal points
            origin_str = (
                f"({self.origin[0]:.2f},{self.origin[1]:.2f},{self.origin[2]:.2f})"
            )
        return (
            f"DikeHemisphere: origin ({origin_str}), diam {self.diam:.1f}, height {self.height:.1f}, "
            f"minor_axis_scale {self.minor_scale:.1f}, rotation {self.rotation:.1f}, value {self.value:.1f}."
        )

    def default_z_function(self, x, y):
        """Default z function: elliptical hemisphere."""
        r = 1
        inner = r**2 - x**2 - y**2
        z_surf = np.sqrt(np.maximum(0, inner))
        return z_surf

    def run(self, xyz, data):
        self.origin = np.array(self.origin)
        # Translate points to origin coordinate frame (bottom center of the sill)
        v0 = xyz - self.origin
        # Rotate the points in the xy plane of the lenticle formation ccw
        R = rotate([0, 0, 1], np.deg2rad(self.rotation))
        v0 = v0 @ R.T

        x, y, z = v0[:, 0], v0[:, 1], v0[:, 2]

        # Apply scaling transforms to form a uniform hemisphere
        x /= self.minor_scale
        x /= self.diam / 2.0
        y /= self.diam / 2.0
        z /= self.height

        if self.upper:
            inside = z < self.z_function(x, y)
            mask = inside & (z > 0)
        else:
            inside = z > -self.z_function(x, y)
            mask = inside & (z < 0)
        if self.clip:
            mask &= ~np.isnan(data)
        data[mask] = self.value

        return xyz, data


class PushHemisphere(Transformation):
    """
    Push a hemisphere intrusion in the z-direction.

    This transformation models a hemispherical intrusion being pushed along the z-axis. The hemisphere can be rotated,
    scaled, and positioned within a 3D geological model.

    Parameters
    ----------
    origin : tuple, optional
        The origin point of the hemisphere in the model reference frame; defaults to (0, 0, 0).
    diam : float, optional
        The diameter of the hemisphere; defaults to 1.0.
    height : float, optional
        The height of the hemisphere; defaults to 1.0.
    minor_axis_scale : float, optional
        Scaling factor for the x-axis of the hemisphere; defaults to 1.0.
    rotation : float, optional
        Rotation of the hemisphere in the xy plane, in degrees; defaults to 0.0.
    upper : bool, optional
        If True, the upper hemisphere is used; if False, the lower hemisphere is used; defaults to True.
    z_function : callable, optional
        A function defining the z-coordinate for the hemisphere's shape; defaults to an elliptical hemisphere function if None is provided.
    """

    def __init__(
        self,
        origin=(0, 0, 0),
        diam=1.0,
        height=1.0,
        minor_axis_scale=1.0,
        rotation=0.0,
        upper=True,
        z_function=None,
    ):
        self.origin = origin
        self.diam = diam
        self.height = height
        self.minor_scale = minor_axis_scale
        self.rotation = rotation
        self.upper = upper
        self.z_function = self.default_z_function if z_function is None else z_function

    def __str__(self):
        if isinstance(self.origin, DeferredParameter):
            origin_str = str(self.origin)  # Use DeferredParameter's __str__ method
        else:
            # Format the tuple to limit decimal points
            origin_str = (
                f"({self.origin[0]:.2f},{self.origin[1]:.2f},{self.origin[2]:.2f})"
            )
        return (
            f"PushHemisphere: origin ({origin_str}), diam {self.diam:.1f}, height {self.height:.1f}, "
            f"minor_axis_scale {self.minor_scale:.1f}, rotation {self.rotation:.1f}."
        )

    def default_z_function(self, x, y):
        """Default z function: elliptical hemisphere."""
        r = 1
        inner = r**2 - x**2 - y**2
        z_surf = np.sqrt(np.maximum(0, inner))
        return z_surf

    def run(self, xyz, data):
        self.origin = np.array(self.origin)
        # Step 1: Translate points to origin coordinate frame (bottom center of hemisphere)
        v0 = xyz - self.origin
        # Step 2: Rotate the points in the xy plane of the lenticle formation ccw
        R = rotate([0, 0, 1], np.deg2rad(self.rotation))
        v0 = v0 @ R.T

        x, y, z = v0[:, 0], v0[:, 1], v0[:, 2]

        # Step 3: Apply scaling transforms to normalize all directions to 1
        x /= self.minor_scale
        x /= self.diam / 2.0
        y /= self.diam / 2.0
        z /= self.height

        # Step 4: Calculate radial and rho distances of all points in xyz' local coords
        r = np.sqrt(x**2 + y**2 + z**2)
        rho = np.sqrt(x**2 + y**2)
        xyz_prime = np.column_stack((x, y, z))

        # Step 5: Use projected unit vector onto z as basis of deflection
        norms = np.linalg.norm(xyz_prime, axis=1)
        unit_vectors = xyz_prime / norms[:, np.newaxis]
        z_proj = unit_vectors[:, 2]

        # Apply to outer points first
        z_surf = self.z_function(x, y)
        # far points are full deflected vertically by height projection
        if self.upper:
            outside = z > z_surf
            mask = outside & (z > 0)
        else:
            outside = z < -z_surf
            mask = outside & (z < 0)

        scaling = np.ones(z.shape)
        # Scale based on lateral distance from origin
        scaling = 1 / (1 + np.exp(8 * (rho - 1)))
        # also scale by overall distance away from origin
        scaling *= 1 / (r + 1e-6) ** 1.5
        z[mask] -= scaling[mask] * z_proj[mask] * (1 - np.exp(-10 * np.abs(z[mask])))

        # inside points interpolate towards origin
        inside = ~outside
        if self.upper:
            mask = inside & (z > 0)
        else:
            mask = inside & (z < 0)
        z[mask] -= r[mask] * z_proj[mask] * (1 - np.exp(-10 * np.abs(z[mask])))

        # Update the original xyz z-coordinates without inverting back the rest
        xyz[:, 2] = z * self.height + self.origin[2]

        return xyz, data


class DikeHemispherePushed(CompoundProcess):
    """Creates a hemisphere with a pushed curved boundary.

    This compound process combines a hemispherical dike intrusion with a deformation that pushes its boundary,
    creating a unique geological feature with a curved and displaced boundary.

    Parameters
    ----------
    diam : float
        The diameter of the hemisphere.
    height : float
        The height of the hemisphere.
    origin : tuple, optional
        The origin point of the hemisphere in the model reference frame; defaults to (0, 0, 0).
    minor_axis_scale : float, optional
        Scaling factor for the x-axis of the hemisphere; defaults to 1.0.
    rotation : float, optional
        Rotation of the hemisphere in the xy plane, in degrees; defaults to 0.0.
    upper : bool, optional
        If True, the upper hemisphere is used; if False, the lower hemisphere is used; defaults to True.
    clip : bool, optional
        If True, clips the hemisphere to not protrude above the surface; defaults to False.
    value : float, optional
        The value of rock-type to assign to the dike; defaults to 0.0.
    z_function : callable, optional
        A function defining the z-coordinate for the hemisphere's shape; defaults to an elliptical hemisphere function if None is provided.
    """

    def __init__(
        self,
        diam,
        height,
        origin=(0, 0, 0),
        minor_axis_scale=1.0,
        rotation=0.0,
        upper=True,
        clip=False,
        value=0.0,
        z_function=None,
    ):
        self.deposition = DikeHemisphere(
            diam=diam,
            height=height,
            origin=origin,
            minor_axis_scale=minor_axis_scale,
            rotation=rotation,
            value=value,
            upper=upper,
            clip=clip,
            z_function=z_function,
        )
        self.transformation = PushHemisphere(
            diam=diam,
            height=height,
            origin=origin,
            minor_axis_scale=minor_axis_scale,
            rotation=rotation,
            upper=upper,
            z_function=z_function,
        )
        self.history = [self.transformation, self.deposition]

    def __str__(self):
        if isinstance(self.deposition.origin, DeferredParameter):
            origin_str = str(
                self.deposition.origin
            )  # Use DeferredParameter's __str__ method
        else:
            # Format the tuple to limit decimal points
            origin_str = f"({self.deposition.origin[0]:.2f},{self.deposition.origin[1]:.2f},{self.deposition.origin[2]:.2f})"
        return (
            f"DikeHemispherePushed: origin ({origin_str})), diam {self.deposition.diam:.1f}, height {self.deposition.height:.1f}, "
            f"minor_axis_scale {self.deposition.minor_scale:.1f}, rotation {self.deposition.rotation:.1f}, value {self.deposition.value:.1f}."
        )


class Laccolith(CompoundProcess):
    """Creates a Laccolith or a Lopolith.

    This compound process models a laccolith or lopolith, which are dome-shaped intrusions with a flat base
    that result from the injection of magma between sedimentary layers, forming a cap and stem structure.

    Parameters
    ----------
    origin : tuple, optional
        The origin point of the laccolith in the model reference frame; defaults to (0, 0, 0).
    cap_diam : float, optional
        The diameter of the laccolith's cap; defaults to 500.
    stem_diam : float, optional
        The diameter of the laccolith's stem; defaults to 80.
    height : float, optional
        The height of the laccolith cap; defaults to 100.
    minor_axis_scale : float, optional
        Scaling factor for the x-axis of the laccolith; defaults to 1.0.
    rotation : float, optional
        Rotation of the laccolith in the xy plane, in degrees; defaults to 0.0.
    value : float, optional
        The value of rock-type to assign to the laccolith; defaults to 0.0.
    upper : bool, optional
        If True, the upper hemisphere is used; if False, the lower hemisphere is used (Lopolith); defaults to True.
    clip : bool, optional
        If True, clips the laccolith to not protrude above the surface; defaults to False.
    z_function : callable, optional
        A function defining the z-coordinate for the laccolith's shape; defaults to an elliptical hemisphere function if None is provided.
    """

    def __init__(
        self,
        origin=(0, 0, 0),
        cap_diam=500,
        stem_diam=80,
        height=100,
        minor_axis_scale=1.0,
        rotation=0.0,
        value=0.0,
        upper=True,
        clip=False,
        z_function=None,
    ):
        col = DikeColumn(
            origin, stem_diam, np.inf, minor_axis_scale, rotation, value, clip
        )
        cap = DikeHemisphere(
            origin,
            cap_diam,
            height,
            minor_axis_scale,
            rotation,
            value,
            upper,
            clip,
            z_function=z_function,
        )
        push = PushHemisphere(
            origin, cap_diam, height, minor_axis_scale, rotation, upper
        )
        self.history = [push, cap, col]

    def __str__(self):
        origin_str = ", ".join(f"{coord:.1f}" for coord in self.col.origin)
        return f"Laccolith: origin ({origin_str}), cap diam {self.cap.diam:.1f}, stem diam {self.col.diam:.1f}, height {self.cap.height:.1f} "


class DikePlug(Deposition):
    """An intrusion formed as a parabolic or elliptical plug.

    This class models a dike intrusion that takes the shape of a parabolic or elliptical plug,
    with customizable parameters such as diameter, rotation, and shape. The plug can be clipped
    to ensure it does not protrude above the surface.

    Parameters
    ----------
    origin : tuple
        Origin point of the plug tip in the model reference frame.
    diam : float
        Diameter of the plug's major axis.
    minor_axis_scale : float
        Scaling factor for the minor axis of the plug.
    rotation : float
        Rotation of the plug's major axis clockwise from the y-axis, in degrees.
    shape : float
        Shape parameter for the plug, controlling the exponent of the rotated polynomial.
    value : int
        Value of rock-type to assign to the plug.
    clip : bool
        If True, clips the plug to prevent it from protruding above the surface.

    Shaping Function
    ----------------
    An elliptical cross-section has z <= -|d^shape|, where d is the distance from the plug axis in scaled ellipse space.
    """

    def __init__(
        self,
        diam=5,
        origin=(0, 0, 0),
        minor_axis_scale=1.0,
        rotation=0,
        shape=3.0,
        value=0.0,
        clip=True,
    ):
        self.origin = origin
        self.diameter = diam
        self.minor_scale = minor_axis_scale
        self.rotation = rotation
        self.shape = shape
        self.clip = clip
        self.value = value

    def __str__(self):
        if isinstance(self.origin, DeferredParameter):
            origin_str = str(self.origin)  # Use DeferredParameter's __str__ method
        else:
            # Format the tuple to limit decimal points
            origin_str = (
                f"({self.origin[0]:.2f},{self.origin[1]:.2f},{self.origin[2]:.2f})"
            )
        return (
            f"DikePlug: origin ({origin_str}), diameter {self.diameter:.1f}, "
            f"rotation {self.rotation:.1f}, value {self.value:.1f}."
        )

    def run(self, xyz, data):
        self.origin = np.array(self.origin)
        # Translate points to origin coordinate frame
        v0 = xyz - self.origin
        # Rotate the points in the xy plane of the plug formation ccw
        R = rotate([0, 0, 1], np.deg2rad(self.rotation))
        v0 = v0 @ R.T
        # Scale the points along the x-axis (minor axis)
        v0[:, 0] /= self.minor_scale
        # Calculate the radius from the plug axis in scaled ellipse space for z <= 0
        valid_indices = v0[:, 2] <= 0
        dists = np.full(v0.shape[0], np.nan)  # Initialize with NaN
        dists[valid_indices] = np.sqrt(
            v0[valid_indices, 0] ** 2 + v0[valid_indices, 1] ** 2
        )
        dists = dists / (self.diameter / 2.0)  # Normalize to the diameter of major axis

        # Condition to include in plug is using z<=-|d^shape|
        mask = v0[:, 2] <= -np.abs(dists**self.shape)

        if self.clip:
            # Clip the plug to not protrude above the surface
            mask &= ~np.isnan(data)
        data[mask] = self.value

        return xyz, data


class PushPlug(Transformation):
    """
    Applies a pushing transformation to a parabolic or elliptical plug.

    This class models the pushing of a geological plug intrusion, adjusting its shape based
    on a Gaussian displacement function. The transformation is applied by modifying the vertical
    displacement of the plug's surface points based on the distance from a reference surface.

    Parameters
    ----------
    origin : tuple
        The origin point of the plug in the model reference frame.
    diam : float
        The diameter of the plug's major axis.
    minor_axis_scale : float
        Scaling factor for the minor axis of the plug.
    rotation : float
        The rotation of the plug's major axis clockwise from the y-axis, in degrees.
    shape : float
        The shape parameter for the plug, controlling the exponent of the rotated polynomial.
    push : float
        The magnitude of the pushing displacement applied to the plug.

    """

    def __init__(self, origin, diam, minor_axis_scale, rotation, shape, push):
        self.origin = origin
        self.diameter = diam
        self.minor_scale = minor_axis_scale
        self.rotation = rotation
        self.shape = shape
        self.push = push

    def __str__(self):
        return f"DikePush: vector {self.vector}"

    def run(self, xyz, data):
        self.origin = np.array(self.origin)
        # Translate points to origin coordinate frame
        v0 = xyz - self.origin
        # Rotate the points in the xy plane of the plug formation ccw
        R = rotate([0, 0, 1], np.deg2rad(self.rotation))
        v0 = v0 @ R.T
        # Scale the points along the x-axis (minor axis)
        v0[:, 0] /= self.minor_scale

        x, y, z = v0[:, 0], v0[:, 1], v0[:, 2]

        r = np.sqrt(x**2 + y**2)
        r_scaled = r / (self.diameter / 2.0)
        z_surf = -np.abs(r_scaled**self.shape)
        dists = z - z_surf
        # Gaussian push function based on vertical distance from the surface
        displacement = self.push * np.exp(-0.05 * dists**2)

        xyz[:, 2] -= displacement

        return xyz, data


class DikePlugPushed(CompoundProcess):
    """
    A compound process representing a dike intrusion with a push deformation.

    This class models a geological dike plug that undergoes both deposition and
    a subsequent pushing deformation, resulting in a complex shape that combines
    the characteristics of both processes.

    Parameters
    ----------
    origin : tuple
        The origin point of the plug in the model reference frame.
    diam : float
        The diameter of the plug's major axis.
    minor_axis_scale : float
        Scaling factor for the minor axis of the plug.
    rotation : float
        The rotation of the plug's major axis clockwise from the y-axis, in degrees.
    shape : float
        The shape parameter for the plug, controlling the exponent of the rotated polynomial.
    push : float
        The magnitude of the pushing deformation applied to the plug.
    value : float
        The value of the rock type assigned to the plug.
    """

    def __init__(
        self,
        origin=(0, 0, 0),
        diam=3,
        minor_axis_scale=1.0,
        rotation=0,
        shape=3.0,
        push=1.0,
        value=0.0,
    ):
        self.origin = origin
        self.diameter = diam
        self.minor_scale = minor_axis_scale
        self.rotation = rotation
        self.shape = shape
        self.value = value
        deposition = DikePlug(
            diam=diam,
            origin=origin,
            minor_axis_scale=minor_axis_scale,
            rotation=rotation,
            shape=shape,
            value=value,
            clip=False,
        )
        transformation = PushPlug(
            diam=diam,
            origin=origin,
            minor_axis_scale=minor_axis_scale,
            rotation=rotation,
            shape=shape,
            push=push,
        )
        self.history = [transformation, deposition]

    def __str__(self):
        if isinstance(self.origin, DeferredParameter):
            origin_str = str(self.origin)  # Use DeferredParameter's __str__ method
        else:
            # Format the tuple to limit decimal points
            origin_str = (
                f"({self.origin[0]:.2f},{self.origin[1]:.2f},{self.origin[2]:.2f})"
            )
        return (
            f"DikePlugPushed: origin ({origin_str}), diam {self.diameter:.1f}, minor scaling {self.minor_scale:.1f}, "
            f"rotation {self.rotation:.1f}, shape {self.shape:.1f}, value {self.value:.1f}."
        )


class UnconformityBase(Deposition):
    """
    Erode the model from a given base level upwards.

    This class models an erosion process that removes material from the model
    above a specified base level. The areas above this base are filled with the
    specified value, typically an "air" value to signify rock removal

    Parameters
    ----------
    base : float
        The base level above which the erosion occurs.
    value : int or float, optional
        The value used to fill the eroded areas. Defaults to np.nan for air or voids.
    """

    def __init__(self, base, value=np.nan):
        self.base = base
        self.value = value

    def __str__(self):
        return f"UnconformityBase: base {self.base:.1f}, value {self.value:.1f}"

    def run(self, xyz, data):
        # Mask for points above the base level
        mask = xyz[:, 2] > self.base
        # Apply the mask and update data where condition is met
        data[mask] = self.value

        # Return the unchanged xyz and the potentially modified data
        return xyz, data


class UnconformityDepth(Deposition):
    """
    Erode the model from the highest point downwards by a specified thickness.

    This class models an erosion process that removes material from the model
    starting at the highest elevation and continuing downward until a specified
    thickness is eroded. The areas affected by the erosion are filled with the
    specified value, typically used to simulate unconformities or other geological
    processes that remove material from the top of the surface.

    Parameters
    ----------
    depth : float
        The thickness of the erosion layer, measured from the highest point
        on the surface.
    value : int or float, optional
        The value used to fill the eroded areas. Defaults to np.nan for air or voids.
    """

    def __init__(self, depth, value=np.nan):
        self.depth = depth
        self.value = value
        self.peak = None

    def __str__(self):
        return f"UnconformityDepth: depth {self.depth:.1f}, value {self.value:.1f}"

    def run(self, xyz, data):
        # Find the peak of non-NaN data
        self.peak = np.max(xyz[:, 2][~np.isnan(data)])

        # Erode down
        mask = xyz[:, 2] > self.peak - self.depth
        data[mask] = self.value

        # Return the unchanged xyz and the potentially modified data
        return xyz, data


class Tilt(Transformation):
    """Tilt the model by a given strike and dip and an origin point.

    Parameters:
        strike (float): Strike angle in CW degrees (center-line of the dike) from north (y-axis)
        dip    (float): Dip of the tilt in degrees (CW from the strike axis)
        origin (tuple): Origin point for the tilt (x,z,y)
    """

    def __init__(self, strike, dip, origin=(0, 0, 0)):
        self.strike = np.radians(strike)  # Convert degrees to radians
        self.dip = np.radians(dip)  # Convert degrees to radians
        self.origin = origin

    def __str__(self):
        # Convert radians back to degrees for more intuitive understanding
        strike_deg = np.degrees(self.strike)
        dip_deg = np.degrees(self.dip)
        if isinstance(self.origin, DeferredParameter):
            origin_str = str(self.origin)  # Use DeferredParameter's __str__ method
        else:
            # Format the tuple to limit decimal points
            origin_str = (
                f"({self.origin[0]:.2f},{self.origin[1]:.2f},{self.origin[2]:.2f})"
            )

        return (
            f"Tilt: strike {strike_deg:.1f}°, dip {dip_deg:.1f}°,"
            f"origin ({origin_str})"
        )

    def run(self, xyz, data):
        self.origin = np.array(self.origin)
        # Calculate rotation axis from strike (rotation around z-axis)
        axis = rotate([0, 0, 1], -self.strike) @ [0, 1.0, 0]
        # Calculate rotation matrix from dip (tilt)
        R = rotate(axis, -self.dip)

        # Apply rotation about origin -> translate to origin, rotate, translate back
        # Y = R * (X - O) + O
        xyz = xyz @ R.T + (-self.origin @ R.T + self.origin)

        # Apply rotation to xyz points
        return xyz, data  # Assuming xyz is an Nx3 numpy array


class Fold(Transformation):
    """
    Tilt the model by a given strike and dip about an origin point.

    This transformation tilts the geological model by rotating it around a specified
    origin point, using given strike and dip angles. The strike determines the direction
    of the tilt's axis relative to the north (y-axis), and the dip defines the angle
    of the tilt along this axis.

    Parameters
    ----------
    strike : float
        The strike angle in degrees clockwise (CW) from the north (y-axis).
    dip : float
        The dip angle in degrees CW from the strike axis.
    origin : tuple
        The origin point for the tilt, given as a tuple (x, y, z).
    """

    def __init__(
        self,
        strike=0.0,
        dip=90.0,
        rake=0.0,
        period=50.0,
        amplitude=10.0,
        phase=0.0,
        shape=0.0,
        origin=(0, 0, 0),
        periodic_func=None,
    ):
        self.strike = np.radians(strike)
        self.dip = np.radians(dip)
        self.rake = np.radians(rake)
        self.period = period
        self.amplitude = amplitude
        self.phase = phase
        self.shape = shape
        self.origin = origin
        # Accept a custom periodic function or use the default otherwise
        self.periodic_func = (
            periodic_func if periodic_func else self.periodic_func_default
        )

    def __str__(self):
        # Convert radians back to degrees for more intuitive understanding
        strike_deg = np.degrees(self.strike)
        dip_deg = np.degrees(self.dip)
        rake_deg = np.degrees(self.rake)
        if isinstance(self.origin, DeferredParameter):
            origin_str = str(self.origin)  # Use DeferredParameter's __str__ method
        else:
            # Format the tuple to limit decimal points
            origin_str = (
                f"({self.origin[0]:.2f},{self.origin[1]:.2f},{self.origin[2]:.2f})"
            )

        return (
            f"Fold: strike {strike_deg:.1f}°, dip {dip_deg:.1f}°, rake {rake_deg:.1f}°, period {self.period:.1f},"
            f"amplitude {self.amplitude:.1f}, origin ({origin_str})."
        )

    def run(self, xyz, data):
        self.origin = np.array(self.origin)
        # Adjust the rake to have slip vector (fold amplitude) perpendicular to the strike
        slip_vector, normal_vector = slip_normal_vectors(
            self.rake + np.pi / 2, self.dip, self.strike
        )

        # Translate points to origin coordinate frame
        v0 = xyz - self.origin
        # Orthogonal distance from origin along U
        fU = np.dot(v0, normal_vector)
        # Calculate the number of cycles orthogonal distance
        n_cycles = fU / self.period
        # Get the displacement as a function for n_cycles
        displacement_distance = -self.amplitude * self.periodic_func(n_cycles)
        # Calculate total displacement for each point, recast off as a column vector
        displacement_vector = slip_vector * displacement_distance[:, np.newaxis]
        # Return to global coordinates
        xyz_transformed = xyz + displacement_vector

        return xyz_transformed, data

    def periodic_func_default(self, n_cycles):
        """Default periodic function for the fold transformation. uses shaping from 3rd harmonic."""
        # Normalize to amplitude of 1
        norm = (1 + self.shape**2) ** 0.5
        func = np.cos(2 * np.pi * (n_cycles + self.phase)) + self.shape * np.cos(
            3 * 2 * np.pi * n_cycles
        )
        return func / norm


class Slip(Transformation):
    """
    Generalized slip transformation for modeling displacement along a fault plane.

    This class provides a customizable slip transformation where displacement is
    applied according to a user-defined function. The transformation allows for
    complex slip behaviors, including those involving variable displacement
    magnitudes depending on the distance from the slip plane.

    Parameters
    ----------
    displacement_func : callable
        A custom displacement function that maps a distance from the slip plane to
        a displacement value.
    strike : float
        The strike angle in degrees, representing the orientation of the fault line
        relative to the north (y-axis).
    dip : float
        The dip angle in degrees, representing the angle of the fault plane relative
        to the horizontal plane.
    rake : float
        The rake angle in degrees, representing the direction of slip along the fault plane.
    amplitude : float
        The amplitude of the slip, representing the maximum displacement along the slip vector.
    origin : tuple
        The origin point for the slip in the local coordinate frame, specified as (x, y, z).
    """

    def __init__(
        self,
        displacement_func,
        strike=0.0,
        dip=90.0,
        rake=0.0,
        amplitude=2.0,
        origin=(0, 0, 0),
    ):
        self.strike = np.radians(strike)
        self.dip = np.radians(dip)
        self.rake = np.radians(rake)
        self.amplitude = amplitude
        self.origin = origin
        self.displacement_func = displacement_func

    def __str__(self):
        strike_deg = np.degrees(self.strike)
        dip_deg = np.degrees(self.dip)
        rake_deg = np.degrees(self.rake)
        if isinstance(self.origin, DeferredParameter):
            origin_str = str(self.origin)  # Use DeferredParameter's __str__ method
        else:
            # Format the tuple to limit decimal points
            origin_str = (
                f"({self.origin[0]:.2f},{self.origin[1]:.2f},{self.origin[2]:.2f})"
            )
        return (
            f"{self.__class__.__name__} with strike {strike_deg:.1f}°, dip {dip_deg:.1f}°, rake {rake_deg:.1f}°, "
            f"amplitude {self.amplitude:.1f}, origin ({origin_str})."
        )

    def default_displacement_func(self, distances):
        # A simple linear displacement function as an example
        return np.zeros(
            np.shape(distances)
        )  # Displaces positively where the distance is positive

    def run(self, xyz, data):
        self.origin = np.array(self.origin)
        # Slip is measured from dip vector, while the slip_normal convention is from strike vector, add 90 degrees
        slip_vector, normal_vector = slip_normal_vectors(
            self.rake, self.dip, self.strike
        )

        # Translate points to origin coordinate frame
        v0 = xyz - self.origin
        # Orthogonal distance from origin along U
        distance_to_slip = np.dot(v0, normal_vector)
        # Apply the displacement function to the distances along normal
        displacements = -self.amplitude * self.displacement_func(distance_to_slip)
        # Calculate the movement vector along slip direction
        displacement_vectors = displacements[:, np.newaxis] * slip_vector
        # Return to global coordinates and apply the displacement
        xyz_transformed = xyz + displacement_vectors
        return xyz_transformed, data


class Fault(Slip):
    """
    Brittle fault transformation with a sharp step function across the fault plane.

    This class models a fault where displacement occurs abruptly across the fault plane,
    causing a sharp discontinuity in the geological strata. It is ideal for simulating
    brittle faults where a distinct separation between displaced and stationary sections is required.

    Parameters
    ----------
    strike : float
        The strike angle in degrees, representing the orientation of the fault line
        relative to the north (y-axis).
    dip : float
        The dip angle in degrees, representing the angle of the fault plane relative
        to the horizontal plane.
    rake : float
        The rake angle in degrees, representing the direction of slip along the fault plane.
    amplitude : float
        The maximum displacement magnitude along the slip vector.
    origin : tuple of float
        The x, y, z coordinates from which the fault originates within the local coordinate frame.

    Example
    -------
    Creating a Fault instance with specific geological parameters:

    fault = Fault(strike=30, dip=60, rake=90, amplitude=5, origin=(0, 0, 0))
    """

    def __init__(
        self,
        strike=0.0,
        dip=90.0,
        rake=0.0,
        amplitude=2.0,
        origin=(0, 0, 0),
    ):
        super().__init__(self.fault_displacement, strike, dip, rake, amplitude, origin)
        self.rotation = 0

    def fault_displacement(self, distances):
        return np.sign(distances) * 0.5  # Net peak to peak is 1


class Shear(Slip):
    """
    Shear transformation for modeling plastic deformation processes.

    This class models shear deformation, where displacement increases gradually
    with distance from the slip plane, following a sigmoid function. It is used
    to simulate more ductile behavior compared to brittle faults.

    Parameters
    ----------
    strike : float
        The strike angle in degrees, representing the orientation of the fault line
        relative to the north (y-axis).
    dip : float
        The dip angle in degrees, representing the angle of the fault plane relative
        to the horizontal plane.
    rake : float
        The rake angle in degrees, representing the direction of slip along the fault plane.
    amplitude : float
        The maximum displacement magnitude along the slip vector.
    origin : tuple of float
        The x, y, z coordinates from which the fault originates within the local coordinate frame.
    steepness : float
        The steepness of the sigmoid function, controlling the rate of change of displacement.
    """

    def __init__(
        self,
        strike=0.0,
        dip=90.0,
        rake=0.0,
        amplitude=2.0,
        steepness=1.0,
        origin=(0, 0, 0),
    ):
        self.steepness = steepness
        super().__init__(self.shear_displacement, strike, dip, rake, amplitude, origin)

    def shear_displacement(self, distances):
        # The sigmoid function will be centered around zero and will scale with amplitude
        return 1 / (1 + np.exp(-self.steepness * distances))  # Net peak to peak is 1

    def run(self, xyz, array):
        # Apply the shear transformation
        xyz_transformed, array = super().run(xyz, array)
        return xyz_transformed, array
