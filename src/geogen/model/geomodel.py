import copy
import logging

import numpy as np

from .geoprocess import *
from .util import resample_mesh

# Set up a simple logger
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("Geo")
log.setLevel(logging.DEBUG)
logging.disable()


class GeoModel:
    """
    A 3D geological model that can be built up from geological processes. The model is represented
    as a 3D meshgrid with data values at each point. A history of GeoProcesses can be added and
    computed to build the model.

    Parameters
    ----------
    bounds : tuple
        A tuple specifying the bounds of the model. It can be a single tuple of two values
        representing (min, max) for all dimensions, or a tuple of three tuples representing
        (xmin, xmax), (ymin, ymax), and (zmin, zmax).
    resolution : int or tuple
        The number of divisions in each dimension. If a single integer is provided, the same
        resolution is used for all dimensions. If a tuple of three integers is provided,
        they represent the resolution for x, y, and z dimensions, respectively.
    dtype : dtype, optional
        The data type for the model's data array. Default is np.float32.
    name : str, optional
        The name of the model. Default is "model".
    height_tracking : bool, optional
        Whether to track height above and below the model for renormalization. Default is True.
    """

    # fmt: off
    # Conversion value from NaN to integer for filling air/non-rock areas in the model
    EMPTY_VALUE = -1

    # Height tracking parameters
    HEIGHT_BAR_EXT_FACTOR = 3  # Factor of z-range to extend above and below model for depth measurement
    HEIGHT_BAR_RESOLUTION = 128  # Resolution of the extension bars (number of points computed above and below model)

    # Height normalization parameters
    HEIGHT_NORMALIZATION_FILL_TARGET = 0.85  # Target maximum height for model normalization    
    HEIGHT_NORMALIZATION_STD_DEV = 0.05  # Standard deviation for height normalization
    # fmt: on

    def __init__(
        self,
        bounds=(0, 16),
        resolution=128,
        dtype=np.float32,
        name="model",
        height_tracking=True,
    ):
        self.name = name
        self.dtype = dtype
        self.bounds = bounds
        self.resolution = resolution

        # Height tracking extensions
        self.height_tracking = height_tracking
        self.num_tracking_points = 0
        self.height_tracking_indices = []

        # A packed and cached unpacked history of geological processes
        self.history = []
        self.history_unpacked = []

        # Placeholders for mesh data
        self.data = np.empty(0)  # Vector of data values on mesh points
        self.xyz = np.empty((0, 0))  # nx3 matrix of mesh points (x, y, z)
        self.X = np.empty((0, 0, 0))  # 3D meshgrid for X coordinates
        self.Y = np.empty((0, 0, 0))  # 3D meshgrid for Y coordinates
        self.Z = np.empty((0, 0, 0))  # 3D meshgrid for Z coordinates
        self.mesh_snapshots = np.empty((0, 0, 0, 0))  # 4D array to store intermediate mesh states
        self.data_snapshots = np.empty((0, 0))  # 2D array to store intermediate data states

        self._validate_model_params()

    def _validate_model_params(self):
        """
        Validate the model parameters.

        Raises
        ------
        ValueError
            If the resolution or bounds are not properly formatted or invalid.
        """
        # Check and accept resolution as a single value or a tuple of 3 values
        if isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution, self.resolution)
        elif isinstance(self.resolution, tuple):
            assert len(self.resolution) == 3, "Resolution must be a single value or a tuple of 3 values."
        else:
            raise ValueError("Resolution must be a single value or a tuple of 3 values.")

        # Check and accept bounds as a single tuple of 2 values or a tuple of 3 tuples
        if isinstance(self.bounds[0], tuple):
            assert len(self.bounds) == 3, "Bounds must be a tuple of 3 tuples for x, y, and z dimensions."
        elif isinstance(self.bounds, tuple):
            assert len(self.bounds) == 2, "Bounds must be a tuple of 2 values for a single dimension."
            self.bounds = (self.bounds, self.bounds, self.bounds)
        else:
            raise ValueError("Bounds must be a tuple of 2 values or a tuple of 3 tuples.")

    def __repr__(self):
        """
        Provide a compact string representation of the GeoModel instance.

        Returns
        -------
        str
            A string that represents the GeoModel instance.
        """
        return f"GeoModel(name={self.name}, bounds={self.bounds}, resolution={self.resolution})"

    def __str__(self):
        """
        Provide a detailed string description of the GeoModel instance.

        Returns
        -------
        str
            A detailed string description of the GeoModel instance.
        """
        return f"GeoModel: {self.name}\nBounds: {self.bounds}\nResolution: {self.resolution}\nHistory: {self.get_history_string()}"

    def _repr_html_(self):
        """
        Provide an HTML representation of the GeoModel instance for Jupyter notebooks.

        Returns
        -------
        str
            An HTML-formatted string representation of the GeoModel instance.
        """
        # Generating the history column HTML
        if not self.history:
            history_html = "<p>No geological history to display.</p>"
        else:
            history_html = (
                "<div style='text-align: left;'><ol>"
                + "".join(f"<li>{process}</li>" for process in self.history)
                + "</ol></div>"
            )

        # Structuring the table with a dedicated history column
        table = f"""
        <table>
            <tr>
                <th style="text-align: left;">Parameter</th>
                <th style="text-align: left;">Value</th>
                <th style="text-align: left; vertical-align: top;" rowspan="5">History</th>
            </tr>
            <tr><td>Name</td><td>{self.name}</td><td rowspan="5">{history_html}</td></tr>
            <tr><td>Data Type</td><td>{self.dtype}</td></tr>
            <tr><td>Bounds</td><td>{self.bounds}</td></tr>
            <tr><td>Resolution</td><td>{self.resolution}</td></tr>
        </table>
        """
        return table

    def _setup_mesh(self):
        """
        Set up the 3D meshgrid and data based on the specified bounds and resolution.
        """
        # Unpack bounds and resolution
        try:
            x_bounds, y_bounds, z_bounds = self.bounds
        except ValueError:
            print("Bounds must be a tuple of 3 tuples for x, y, and z dimensions.")
            print("Bounds: ", self.bounds)
            print(f"Length: {len(self.bounds)}")
            raise
        x_res, y_res, z_res = self.resolution

        # Create linspace for x, y, z
        x = np.linspace(*x_bounds, num=x_res, dtype=self.dtype)
        y = np.linspace(*y_bounds, num=y_res, dtype=self.dtype)
        z = np.linspace(*z_bounds, num=z_res, dtype=self.dtype)

        # Init 3D meshgrid for X, Y, and Z coordinates of the view field
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")

        # Combine flattened arrays into a 2D numpy array where each row is an (x, y, z) coordinate
        self.xyz = np.column_stack((self.X.flatten(), self.Y.flatten(), self.Z.flatten()))

        # Initialize data array with NaNs
        self.data = np.full(self.xyz.shape[0], np.nan, dtype=self.dtype)

    def add_history(self, history):
        """
        Add one or more geological processes to the model's history with validation.

        Parameters
        ----------
        history : GeoProcess or list of GeoProcess
            A GeoProcess instance or a list of GeoProcess instances to be added to the model's history.

        Raises
        ------
        TypeError
            If any item in the history list is not an instance of GeoProcess.
        ValueError
            If a CompoundProcess in the history has no defined history.
        """
        # Check if the input is not a list, make it a list
        if not isinstance(history, list):
            history = [history]  # Convert a single GeoProcess instance into a list

        # Check if all elements in the list are instances of GeoProcess
        for event in history:
            if not isinstance(event, GeoProcess):
                msg = f"All items in the history list must be instances of the GeoProcess class. Found {type(event)} for {event}."
                raise TypeError(msg)

            # Check if it's a CompoundProcess and ensure it has a valid history
            if isinstance(event, CompoundProcess) and not event.history:
                msg = f"CompoundProcess {event} has no history defined."
                raise ValueError(msg)

        # Extend the existing history with the new history
        self.history.extend(history)
        # Clear the unpacked history cache
        self.history_unpacked = []

    def get_history_string(self, unpacked=False):
        """
        Get a string description of the complete geological history of the model.

        Returns
        -------
        str
            A string describing the geological history of the model.
        """
        if not self.history:
            return "No geological history to display."
        
        if not unpacked:
            history = self.history
        else:
            history = self.history_unpacked

        history_str = "Geological History:\n"

        for index, process in enumerate(history):
            history_str += f"{index + 1}: {str(process)}\n"

        return history_str.strip()  # Remove the trailing newline

    def clear_history(self):
        """
        Clear all geological processes from the model's history.
        """
        self.history = []
        self.history_unpacked = []

    def clear_data(self):
        """
        Clear the model data while retaining the build parameters.
        """
        self.mesh_snapshots = np.empty((0, 0, 0, 0))
        self.data_snapshots = np.empty((0, 0))
        self.data = np.empty(0)
        self.xyz = np.empty((0, 0))
        self.X = np.empty((0, 0, 0))
        self.Y = np.empty((0, 0, 0))
        self.Z = np.empty((0, 0, 0))

    def compute_model(self, keep_snapshots=True, normalize=False, low_res=(8, 8, 64)):
        """
        Compute the present-day model based on the geological history with an option to normalize the height.

        Parameters
        ----------
        keep_snapshots : bool, optional
            Whether to keep snapshots of the mesh during computation. Default is True.
        normalize : bool, optional
            Whether to auto-normalize the model's height to fit in the view field. Default is False.
        low_res : tuple, optional
            If normalize is True, the low-cost normalization model resolution used. Default is (8, 8, 64).
        """
        if normalize:
            # Run a preliminary low res model to normalize the height
            z_shift = self._get_lowres_z_shift_normalization(low_res=low_res)
            self.add_history(Shift([0, 0, z_shift]))

        # Run the actual model computation (whether normalized or not)
        self._apply_history_computation(keep_snapshots=keep_snapshots)

    def _apply_history_computation(self, keep_snapshots=True, remove_bars=True):
        """
        Compute the present-day model based on the geological history.

        Parameters
        ----------
        keep_snapshots : bool, optional
            Whether to keep snapshots of the mesh during computation. Default is True.
        remove_bars : bool, optional
            Whether to remove height tracking bars after computation. Default is True.

        Method Overview
        ---------------
        This method performs the following steps:

        - **Snapshots**:
        The xyz mesh sequence of deformations are saved to a preallocated array for use
        in the forward pass. The starting state [0] is always required; additional
        snapshots are needed at the start of any deposition process, since depositions are
        applied to the deformed mesh on the forward pass.

        - **Backward pass**:
        The xyz mesh is backtracked through history using the Transformation GeoProcesses
        that are stored in the history list. Snapshots of deformed mesh are stored to for the
        forward pass.

        - **Forward pass**:
        The Deposition GeoProcesses are applied to the xyz mesh in its intermediate transformed state.

        - **Conditional height tracking**:
        If height tracking is enabled, additional points are added to the model ro add
        extended low-resolution context above and below the model. This is crucial for tracking
        heights and stabilizing the model through shifting bounds, height, and sediments.
        The number of extra points added is tracked in the `num_tracking_points` attribute.

        - **Deferred Parameters**:
        GeoProcesses are allowed to use DeferredParameters, which are GeoProcess atrributes that
        are resolved at the time of computation since they depend on the context of the history.
        For example, it could involve tracking an origin point backward through history to get its
        equivalent position in the past. History and index are passed for this purpose.
        """
        if not self.history:
            raise ValueError("No geological history to compute.")

        # Clear the model data before recomputing
        self.clear_data()
        # Allocate memory for the mesh and data
        self._setup_mesh()

        # If height tracking is enabled, add bars
        self._add_height_tracking_bars() if self.height_tracking else 0

        # Unpack all compound events into atomic components
        self.history_unpacked = self._unpack_history()

        # Determine how many snapshots are needed for memory pre-allocation
        self._prepare_snapshots(self.history_unpacked)
        # Backward pass to reverse mesh grid of points
        self._backward_pass(self.history_unpacked)
        # Forward pass to apply deposition events
        self._forward_pass(self.history_unpacked)

        # Remove height tracking bars if required
        if remove_bars and self.num_tracking_points > 0:
            self._remove_tracking_points()

        # Clean up snapshots if not required
        if not keep_snapshots:
            self.mesh_snapshots = np.empty((0, 0, 0, 0))
            self.data_snapshots = np.empty((0, 0))

    def _add_height_tracking_bars(self):
        """
        Add height tracking bars that extend from the center and corners of the model's bounds
        above and below the model. One bar from each corner and one from the center, both the upper
        and lower face of the model for a total of 10 bars. EXT_FACTOR and RESOLUTION are used to control
        the factor of z-range to extend and the resolution of the bars.

        This method saves the indices of the added height tracking points in `self.height_tracking_indices`.
        """

        # Calculate x, y coords for center and corners
        x_center, y_center = np.mean(self.bounds[0]), np.mean(self.bounds[1])

        corners = [
            (self.bounds[0][0], self.bounds[1][0]),  # Bottom-left
            (self.bounds[0][0], self.bounds[1][1]),  # Top-left
            (self.bounds[0][1], self.bounds[1][0]),  # Bottom-right
            (self.bounds[0][1], self.bounds[1][1]),  # Top-right
        ]

        # Create upper and lower bars
        z_bounds = self.bounds[-1]
        z_range = z_bounds[1] - z_bounds[0]
        z_lower = np.linspace(
            z_bounds[0] - self.HEIGHT_BAR_EXT_FACTOR * z_range,
            z_bounds[0],
            num=self.HEIGHT_BAR_RESOLUTION,
            dtype=self.dtype,
        )
        z_upper = np.linspace(
            z_bounds[1],
            z_bounds[1] + self.HEIGHT_BAR_EXT_FACTOR * z_range,
            num=self.HEIGHT_BAR_RESOLUTION,
            dtype=self.dtype,
        )

        # Generate bars from center and corners (across all 5 (x,y) by 2 z sets)
        bars = [
            np.column_stack(
                [
                    np.full(self.HEIGHT_BAR_RESOLUTION, x),
                    np.full(self.HEIGHT_BAR_RESOLUTION, y),
                    z,
                ]
            )
            for x, y in [(x_center, y_center)] + corners
            for z in [z_lower, z_upper]
        ]

        # Stack all bars together into a single Mx3 array of (x, y, z) points
        all_bars = np.vstack(bars)
        M = all_bars.shape[0]

        # Append the new points to existing xyz and data arrays
        self.xyz = np.vstack((self.xyz, all_bars))
        self.data = np.concatenate((self.data, np.full(M, np.nan)))

        # Save the indices of the newly added points
        self.height_tracking_indices = np.arange(len(self.xyz) - M, len(self.xyz))

        # Update the number of tracking points
        self.num_tracking_points = M

    def _remove_tracking_points(self):
        """
        Remove the height tracking bars from the model.

        This method modifies the following attributes:
        - `xyz`: The meshgrid points array, with height tracking bars removed.
        - `data`: The data array corresponding to the meshgrid points.
        - `data_snapshots`: The array storing data snapshots, with tracking points removed.
        - `mesh_snapshots`: The array storing mesh snapshots, with tracking points removed.
        - `num_tracking_points`: Reset to zero after removing the height tracking bars.
        - `height_tracking_indices`: Cleared after removing the height tracking bars.
        """
        if self.num_tracking_points > 0 and hasattr(self, "height_tracking_indices"):
            mask = np.ones(len(self.xyz), dtype=bool)
            mask[self.height_tracking_indices] = False

            self.xyz = self.xyz[mask]
            self.data = self.data[mask]
            self.data_snapshots = self.data_snapshots[:, mask]
            self.mesh_snapshots = self.mesh_snapshots[:, mask]

            # Reset the tracking points information
            self.num_tracking_points = 0
            self.height_tracking_indices = []

    def _unpack_history(self):
        """
        Unpack all compound processes into atomic components and cache the result.

        This method resolves compound geological processes, breaking them into simpler, atomic
        components to ensure each process is individually applied during model computation.

        Returns
        -------
        list
            A list of unpacked (atomic) geological processes.
        """
        if not self.history_unpacked:
            self.history_unpacked = []
            history_copy = copy.deepcopy(self.history)
            for event in history_copy:
                if isinstance(event, CompoundProcess):
                    self.history_unpacked.extend(event.unpack())
                else:
                    self.history_unpacked.append(event)
        return self.history_unpacked

    def _prepare_snapshots(self, history):
        """
        Determine when to take snapshots of the mesh during the backward pass.

        Snapshots are taken at key points in the transformation sequence to capture
        the state of the mesh before deposition events are applied.

        Parameters
        ----------
        history : list
            The unpacked geological history of the model.

        Returns
        -------
        list
            A list of indices indicating when snapshots are to be taken.
        """
        # Always include the oldest time state of mesh
        snapshot_indices = [0]
        for i in range(1, len(history)):
            if isinstance(history[i], Deposition) and isinstance(history[i - 1], Transformation):
                snapshot_indices.append(i)

        self.snapshot_indices = snapshot_indices

        self.mesh_snapshots = np.empty((len(self.snapshot_indices), *self.xyz.shape))
        self.data_snapshots = np.empty((len(self.snapshot_indices), *self.data.shape))
        log.debug(f"Intermediate mesh states will be saved at {self.snapshot_indices}")
        log.debug(f"Total gigabytes of memory required: {self.mesh_snapshots.nbytes * 1e-9:.2f}")

        return snapshot_indices

    def _backward_pass(self, history):
        """
        Backtrack the xyz mesh through the geological history using transformations.

        This method assumes reverse transformations to revert the model's state
        back in time, storing snapshots at key intervals for the forward pass (inverse transform).

        Parameters
        ----------
        history : list
            The unpacked geological history of the model.
        """
        # Make a copy of the model xyz mesh to apply transformations
        current_xyz = self.xyz.copy()

        i = len(history) - 1
        for event in reversed(history):
            # Store snapshots of the mesh at required intervals
            if i in self.snapshot_indices:
                # The final state (index 0) uses the actual xyz since no further modifications are made,
                # avoiding unnecessary copying for efficiency.
                if i != 0:
                    self.mesh_snapshots[self.snapshot_indices.index(i)] = np.copy(current_xyz)
                else:
                    self.mesh_snapshots[0] = current_xyz

                log.debug(f"Snapshot taken at index {i}")
            # Apply transformation to the mesh (skipping depositon events that do not alter the mesh)
            if isinstance(event, Transformation):
                current_xyz, _ = event.apply_process(
                    xyz=current_xyz,
                    data=self.data,
                    history=history,  # Pass a copy of history for context
                    index=i,  # Pass the index of the event in the history
                )
            i -= 1

    def _forward_pass(self, history):
        """
        Apply deposition events to the mesh based on the geological history.

        This method applies deposition processes to the model, modifying the geological data
        according to the intermediate transformed state captured during the backward pass.

        Parameters
        ----------
        history : list
            The unpacked geological history of the model.
        """
        for i, event in enumerate(history):
            # Update mesh coordinates as required by fetching snapshot from the backward pass
            if i in self.snapshot_indices:
                snapshot_index = self.snapshot_indices.index(i)
                current_xyz = self.mesh_snapshots[snapshot_index, ...]
                self.data_snapshots[snapshot_index] = self.data.copy()
            if isinstance(event, Deposition):
                _, self.data = event.apply_process(
                    xyz=current_xyz,
                    data=self.data,
                    history=history,  # Pass a copy of history for context
                    index=i,  # Pass the index of the event in the history
                )

    def _get_lowres_z_shift_normalization(self, low_res=(8, 8, 64), max_iter=10):
        """
        Normalize the model to a new maximum height through iterative correction.

        This method generates a low-resolution version of the model to estimate the required
        vertical shift for normalizing the model's height to a desired range.

        Parameters
        ----------
        low_res : tuple, optional
            Resolution to use for the low-resolution model. Default is (8, 8, 64).
        max_iter : int, optional
            Maximum iterations for attempting to normalize the height. Default is 10.

        Returns
        -------
        float
            The total vertical shift needed to normalize the model's height.
        """
        total_z_shift = 0  # Accumulated total shift required to renormalize the model

        # Step 1: Generate a low-resolution model to estimate renormalization
        temp_model = self.__class__(self.bounds, resolution=low_res, dtype=self.dtype, height_tracking=True)
        temp_model.add_history(self.history)
        temp_model._apply_history_computation(keep_snapshots=False, remove_bars=False)

        # Step 2: Set a target to middle of model (50% filled z-range)
        model_z_min, model_z_max = temp_model.get_z_bounds()
        target_max_z = model_z_min + 0.5 * (model_z_max - model_z_min)
        model_max_filled_z = temp_model._get_max_filled_height()

        # Step 3: Iterate towards target, checking if model is in frame
        def is_in_frame(z):
            return model_z_min < z < model_z_max

        itr = 0
        while not is_in_frame(model_max_filled_z) and itr < max_iter:
            # Calculate the model shift required to shift to a desired maximum height
            shift_z = target_max_z - model_max_filled_z
            total_z_shift += shift_z
            # Add a shift transformation to the history and recompute
            temp_model.add_history(Shift([0, 0, shift_z]))
            temp_model.clear_data()
            temp_model._apply_history_computation(keep_snapshots=False, remove_bars=False)
            model_max_filled_z = temp_model._get_max_filled_height()
            itr += 1

            if is_in_frame(model_max_filled_z):
                log.debug(f"Model successfully normalized within {itr} iterations.")
                break
            elif itr == max_iter:
                log.warning(
                    f"Normalization loop reached maximum iterations ({max_iter}). Model may not be fully normalized."
                )

        # Step 4: Final adjustment to match the exact desired target height
        target_max_z = self.get_target_normalization()
        shift_z = target_max_z - model_max_filled_z
        total_z_shift += shift_z

        # Clean up the temporary model
        del temp_model

        return total_z_shift

    def _get_max_filled_height(self):
        """
        Get the maximum filled height of the model.

        This method returns the highest z-value among the non-NaN data points,
        representing the top of the geological structures within the model.

        Returns
        -------
        float
            The maximum height of the filled areas in the model.
        """
        valid_indices = ~np.isnan(self.data)
        valid_z_values = self.xyz[valid_indices, 2]
        try:
            max_z = np.max(valid_z_values)
        except ValueError:
            # traceback.print_exc()
            zmin, zmax = self.get_z_bounds()
            max_z = zmin
        return max_z

    def renormalize_height(self, new_max=0, auto=False, recompute=True):
        """
        Shift the model vertically so that the highest point in view field is at a new maximum height.

        This operation can be computationally expensive as it requires recomputing the model. This is
        a convenience method to be used for adjusting a model's height to a target value.

        Parameters
        ----------
        new_max : float, optional
            The new maximum height for the model. Default is 0.
        auto : bool, optional
            Automatically select a new maximum height based on the model's current height. Default is False.
        recompute : bool, optional
            Whether to recompute the model after renormalization. Default is True.

        Returns
        -------
        float
            The current maximum height of the model after renormalization.
        """
        current_max_z = self._get_max_filled_height()

        if auto:
            new_max = self.get_target_normalization()

        # Calculate the model shift required to shift to a desired maximum height
        shift_z = new_max - current_max_z

        # Add a shift transformation to the history and recompute
        self.add_history(Shift([0, 0, shift_z]))
        if recompute:
            self.clear_data()
            self._apply_history_computation()

        return current_max_z

    def get_target_normalization(
        self,
        target_max=HEIGHT_NORMALIZATION_FILL_TARGET,
        std_dev=HEIGHT_NORMALIZATION_STD_DEV,
    ):
        """
        Calculate the target normalization height for the model with some random variance.

        This method determines the target height to which the model should be normalized.
        The target height is computed as a fraction of the total z-range of the model, with
        an optional standard deviation to introduce variation.

        Parameters
        ----------
        target_max : float, optional
            The target maximum height for the model as a fraction of the total height.
            Default is HEIGHT_NORMALIZATION_FILL_TARGET.
        std_dev : float, optional
            The standard deviation for the normal distribution used to add variation to
            the target height. Default is HEIGHT_NORMALIZATION_STD_DEV.

        Returns
        -------
        float
            The calculated target height for normalization.
        """

        bounds = self.get_z_bounds()
        zmin, zmax = bounds
        z_range = zmax - zmin
        target_height = zmin + z_range * (target_max + np.abs(np.random.normal(0, std_dev)))
        log.debug(f"Normalization Target Height: {target_height}")
        return target_height

    def get_z_bounds(self):
        """
        Return the minimum and maximum z-coordinates of the model.

        This method retrieves the z-coordinate bounds of the model, which define the vertical
        extent of the model's space.

        Returns
        -------
        tuple
            A tuple containing the minimum and maximum z-coordinates of the model.
        """
        # Check if bounds is a tuple of tuples (multi-dimensional)
        if isinstance(self.bounds[0], tuple):
            # Multi-dimensional bounds, assuming the last tuple represents the z-dimension
            z_vals = self.bounds[-1]
        else:
            # Single-dimensional bounds
            z_vals = self.bounds

        return z_vals

    def get_data_grid(self):
        """
        Return the model data in meshgrid form.

        This method reshapes the flattened data array into a 3D meshgrid that matches the
        model's x, y, and z grid.

        Returns
        -------
        np.ndarray
            The model data reshaped into the form of the 3D meshgrid.
        """
        return self.data.reshape(self.X.shape)

    def fill_nans(self, value=EMPTY_VALUE):
        """
        Replace NaN values in the model data array with a specified value.

        This method identifies NaN values within the model's data array and replaces them
        with the specified value.

        Parameters
        ----------
        value : int or float, optional
            The value to replace NaNs with. Default is EMPTY_VALUE.

        Returns
        -------
        np.ndarray
            The data array with NaNs replaced by the specified value.
        """
        assert self.data is not None, "Data array is empty."
        indnan = np.isnan(self.data)
        self.data[indnan] = value
        return self.data

    def add_topography(self, mesh):
        """
        Add a topography mesh to the model.

        This method integrates a topography mesh into the 3D model by interpolating the
        2D topography mesh to match the model's resolution, and then applying it to
        adjust the z-values in the model's data.

        Parameters
        ----------
        mesh : np.ndarray
            A 2D numpy array representing the topography mesh.

        Raises
        ------
        ValueError
            If the mesh dimensions do not match the model's resolution.
        """
        # Interpolate the topography mesh to match the model resolution
        resampled_mesh = resample_mesh(mesh, self.resolution[:2])

        # Expand the 2D topography mesh to match the 3D volume
        expanded_mesh = np.repeat(resampled_mesh[:, :, np.newaxis], self.resolution[2], axis=2)

        # Set all z-values higher than the corresponding topo point at the xy column to np.nan
        above_topo_mask = self.Z > expanded_mesh

        # Reshape the data mesh points into a volume
        data = self.get_data_grid()
        # Add the topography mesh to the model by setting

        data[above_topo_mask] = np.nan
        self.data = data.flatten()

    @classmethod
    def from_tensor(cls, data_tensor, bounds=None):
        """
        Create a GeoModel instance from a 1xXxYxZ or XxYxZ shaped data tensor.

        This special initializer allows the creation of a GeoModel instance directly from
        a PyTorch tensor, setting up the model's meshgrid and data. It also initializes
        the model's history with a `NullProcess()` to indicate no known geological history.

        Parameters
        ----------
        data_tensor : torch.Tensor
            The data tensor to initialize the model with; can be 1xXxYxZ or XxYxZ tensor.
        bounds : tuple, optional
            The bounds of the model in measurement units. If not provided, defaults to the
            resolution of the tensor.

        Returns
        -------
        GeoModel
            An instance of the GeoModel class.

        Raises
        ------
        AssertionError
            If the data tensor does not have the correct dimensions.
        """
        # Check and adjust tensor dimensions if needed
        if data_tensor.dim() == 4 and data_tensor.size(0) == 1:
            data_tensor = data_tensor.squeeze(0)  # Remove the singleton dimension

        # Ensure now the tensor is 3D
        assert data_tensor.dim() == 3, "Data tensor must be either 1xXxYxZ or XxYxZ after squeezing."

        # Extract resolution from tensor shape
        resolution = data_tensor.shape  # Already a tuple of three dimensions
        if bounds is None:
            bounds = (0, resolution[0]), (0, resolution[1]), (0, resolution[2])

        instance = cls(bounds, resolution)
        # Setup mesh for X, Y, Z coordinates and flattened xyz array
        instance._setup_mesh()
        # Insert torch tensor data into model
        instance.data = (
            data_tensor.detach().numpy().flatten()
        )  # Convert tensor to numpy array and flatten it
        # Set history to [NullProcess()] signifying no known geological history
        instance.history = [NullProcess()]

        return instance
