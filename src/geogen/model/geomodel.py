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
    """A 3D geological model that can be built up from geological processes.

    Parameters:
        bounds (Tuple): (allmin, allmax) or ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        resolution (int): Number of divisions in each dimension,
                            or a tuple of 3 values for x, y, and z dimensions
        dtype (dtype): Data type for the model data array
        name (str): Optional name of the model
        height_tracking (bool): Whether to track height above and below the model for renormalization
    """

    EMPTY_VALUE = -1
    EXT_FACTOR = (
        3  # Factor of z-range to extend above and below model for depth measurement
    )
    RES = 128  # Resolution of the extension bars (number of points computed above and below model)

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
        self.height_tracking = height_tracking
        self.extra_points = (
            0  # Number of extra points currently added for height tracking
        )
        self.history = []
        # modified history with deferred parameters resolved (post-computation)
        self.processed_history = []
        # Placeholders for mesh data
        self.data = np.empty(0)  # Vector of data values on mesh points
        self.xyz = np.empty((0, 0))  # nx3 matrix of mesh points (x, y, z)
        self.X = np.empty((0, 0, 0))  # 3D meshgrid for X coordinates
        self.Y = np.empty((0, 0, 0))  # 3D meshgrid for Y coordinates
        self.Z = np.empty((0, 0, 0))  # 3D meshgrid for Z coordinates
        self.mesh_snapshots = np.empty(
            (0, 0, 0, 0)
        )  # 4D array to store intermediate mesh states
        self.data_snapshots = np.empty(
            (0, 0)
        )  # 2D array to store intermediate data states

        self._validate_model_params()

    def _validate_model_params(self):
        """Validate the model parameters."""
        # Check and accept resolution as a single value or a tuple of 3 values
        if isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution, self.resolution)
        elif isinstance(self.resolution, tuple):
            assert (
                len(self.resolution) == 3
            ), "Resolution must be a single value or a tuple of 3 values."
        else:
            raise ValueError(
                "Resolution must be a single value or a tuple of 3 values."
            )

        # Check and accept bounds as a single tuple of 2 values or a tuple of 3 tuples
        if isinstance(self.bounds[0], tuple):
            assert (
                len(self.bounds) == 3
            ), "Bounds must be a tuple of 3 tuples for x, y, and z dimensions."
        elif isinstance(self.bounds, tuple):
            assert (
                len(self.bounds) == 2
            ), "Bounds must be a tuple of 2 values for a single dimension."
            self.bounds = (self.bounds, self.bounds, self.bounds)
        else:
            raise ValueError(
                "Bounds must be a tuple of 2 values or a tuple of 3 tuples."
            )

    def __repr__(self):
        return f"GeoModel(name={self.name}, bounds={self.bounds}, resolution={self.resolution})"

    def __str__(self):
        return f"GeoModel: {self.name}\nBounds: {self.bounds}\nResolution: {self.resolution}\nHistory: {self.get_history_string()}"

    def _repr_html_(self):
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

    def setup_mesh(self):
        """Sets up the 3D meshgrid based on given bounds and resolution."""
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
        self.xyz = np.column_stack(
            (self.X.flatten(), self.Y.flatten(), self.Z.flatten())
        )

        # Initialize data array with NaNs
        self.data = np.full(self.xyz.shape[0], np.nan, dtype=self.dtype)

    def add_history(self, history):
        """Add one or more geological processes to model history.

        Parameters:
            history (GeoProcess or list of GeoProcess): A GeoProcess instance or a list of GeoProcess instances to be added to the model's history.
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

    def get_history_string(self):
        """Returns a string describing the complete geological history of the model."""
        if not self.history:
            return "No geological history to display."

        history_str = "Geological History:\n"

        # Print from the processed history if available
        ref_history = (
            self.history if not self.processed_history else self.processed_history
        )

        for index, process in enumerate(ref_history):
            history_str += f"{index + 1}: {str(process)}\n"

        return history_str.strip()  # Remove the trailing newline

    def clear_history(self):
        """Clear all geological process from history."""
        self.history = []
        self.processed_history = []

    def clear_data(self):
        """Clear the model but retain build parameters."""
        self.mesh_snapshots = np.empty((0, 0, 0, 0))
        self.data_snapshots = np.empty((0, 0))
        self.data = np.empty(0)
        self.xyz = np.empty((0, 0))
        self.X = np.empty((0, 0, 0))
        self.Y = np.empty((0, 0, 0))
        self.Z = np.empty((0, 0, 0))

    def compute_model(
        self, normalize=False, low_res=(8, 8, 64), max_iter=10, keep_snapshots=True
    ):
        """
        Compute the present day model based on the geological history with an option to normalize the height.

        Parameters:
            - normalize (boolean) : Whether to auto-normalize the model's height to fit in view field.
            - low_res (tuple) : The resolution for the preliminary model used for renormalization.
            - max_iter (int) : Maximum iterations for the renormalization loop.
            - keep_snapshots (boolean) : Whether to keep snapshots of the mesh during computation.

        Returns:
            - None
        """
        if normalize:
            normed_history = self._get_lowres_normalized_history(
                low_res=low_res, max_iter=max_iter
            )
            self.clear_history()
            self.add_history(normed_history)

        # Run the actual model computation (whether normalized or not)
        self._apply_history_computation(keep_snapshots)

    def _apply_history_computation(self, keep_snapshots=True):
        """Compute the present day model based on the geological history

        Snapshots:
        The xyz mesh is saved to a preallocated array for use in the forward pass.
        The starting state [0] is always required, additional snapshots are needed
        at the start of any deposition process.

        Backward pass:
        The xyz mesh is backtracked through history using the transformations
        stored in the history list. Intermediate states are stored at required
        intervals for the forward pass.

        Forward pass:
        The deposition events are applied to the xyz mesh in its intermediate
        transformed state.

        Conditional height tracking:
        If height tracking is enabled, additional points are added to the model so that
        additional low resolution context is known to help with renormalization of model height.

        Deferred Parameters:
        The GeoProcess are allowed to use DeferredParameters, they are process parameters that
        are resolved at the time of computation since they depend on the context of the history.
        For example it could be tracking an origin point backwards through history to get its
        equivalent position in the past. History and index are passed for this purpose.
        """
        if len(self.history) == 0:
            raise ValueError("No geological history to compute.")

        # Clear the model data before recomputing
        self.clear_data()
        # Allocate memory for the mesh and data
        self.setup_mesh()

        if getattr(
            self, "height_tracking", False
        ):  # Handle earlier versions of GeoModel without height tracking
            n_tracking_bar_points = self._add_height_tracking_bars()

        # Make a copy of the history to be processed to avoid modifying orignal history
        # with any deferred parameters.
        self.processed_history = copy.deepcopy(self.history)

        # Unpack all compound events into atomic components
        history_unpacked = []
        for event in self.processed_history:
            if isinstance(event, CompoundProcess):
                history_unpacked.extend(event.unpack())
            else:
                history_unpacked.append(event)

        # Determine how many snapshots are needed for memory pre-allocation
        self._prepare_snapshots(history_unpacked)
        # Backward pass to reverse mesh grid of points
        self._backward_pass(history_unpacked)
        # Forward pass to apply deposition events
        self._forward_pass(history_unpacked)

        # Clean up snapshots taken during the backward pass
        if not keep_snapshots:
            self.snapshots = np.empty((0, 0, 0, 0))

        # Remove the height tracking bars from the model
        if getattr(
            self, "height_tracking", False
        ):  # Handle earlier versions of GeoModel without height tracking
            self.xyz = self.xyz[:-n_tracking_bar_points]
            self.data = self.data[:-n_tracking_bar_points]
            self.data_snapshots = self.data_snapshots[:, :-n_tracking_bar_points]
            self.mesh_snapshots = self.mesh_snapshots[:, :-n_tracking_bar_points]

    def _prepare_snapshots(self, history):
        """Determine when to take snapshots of the mesh during the backward pass.

        Snapshots of the xyz mesh should be taken at end of a transformation sequence
        """
        # Always include the oldest time state of mesh
        snapshot_indices = [0]
        for i in range(1, len(history)):
            if isinstance(history[i], Deposition) and isinstance(
                history[i - 1], Transformation
            ):
                snapshot_indices.append(i)

        self.snapshot_indices = snapshot_indices

        self.mesh_snapshots = np.empty((len(self.snapshot_indices), *self.xyz.shape))
        self.data_snapshots = np.empty((len(self.snapshot_indices), *self.data.shape))
        log.debug(f"Intermediate mesh states will be saved at {self.snapshot_indices}")
        log.debug(
            f"Total gigabytes of memory required: {self.mesh_snapshots.nbytes * 1e-9:.2f}"
        )

        return snapshot_indices

    def _backward_pass(self, history):
        """Backtrack the xyz mesh through the geological history using transformations."""
        # Make a copy of the model xyz mesh to apply transformations
        current_xyz = self.xyz.copy()

        i = len(history) - 1
        for event in reversed(history):
            # Store snapshots of the mesh at required intervals
            if i in self.snapshot_indices:
                # The final state (index 0) uses the actual xyz since no further modifications are made,
                # avoiding unnecessary copying for efficiency.
                if i != 0:
                    self.mesh_snapshots[self.snapshot_indices.index(i)] = np.copy(
                        current_xyz
                    )
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
        """Apply deposition events to the mesh based on the geological history."""
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

    def _add_height_tracking_bars(self):
        """Add height tracking bars that extend from the center and corners of the bounds above and below the model.
        The class attributes EXT_FACTOR and RES control the number of points and the extension factor.
        """

        z_bounds = self.bounds[-1]
        # Calculate x, y coords for center and corners
        x_center = (self.bounds[0][0] + self.bounds[0][1]) / 2
        y_center = (self.bounds[1][0] + self.bounds[1][1]) / 2

        corners = [
            (self.bounds[0][0], self.bounds[1][0]),  # Bottom-left
            (self.bounds[0][0], self.bounds[1][1]),  # Top-left
            (self.bounds[0][1], self.bounds[1][0]),  # Bottom-right
            (self.bounds[0][1], self.bounds[1][1]),  # Top-right
        ]

        # create upper and lower bars
        z_range = z_bounds[1] - z_bounds[0]
        z_lower = np.linspace(
            z_bounds[0] - self.EXT_FACTOR * z_range,
            z_bounds[0],
            num=self.RES,
            dtype=self.dtype,
        )
        z_upper = np.linspace(
            z_bounds[1],
            z_bounds[1] + self.EXT_FACTOR * z_range,
            num=self.RES,
            dtype=self.dtype,
        )

        # Generate bars from center, adds 2 bars
        bars = [
            np.column_stack(
                (np.full(self.RES, x_center), np.full(self.RES, y_center), z_lower)
            ),
            np.column_stack(
                (np.full(self.RES, x_center), np.full(self.RES, y_center), z_upper)
            ),
        ]

        # Generate bars from corners, adds 8 bars
        for x, y in corners:
            lower_bar = np.column_stack(
                (np.full(self.RES, x), np.full(self.RES, y), z_lower)
            )
            upper_bar = np.column_stack(
                (np.full(self.RES, x), np.full(self.RES, y), z_upper)
            )
            bars.extend([lower_bar, upper_bar])

        # Stack all bars together
        all_bars = np.vstack(bars)
        self.xyz = np.vstack((self.xyz, all_bars))
        self.data = np.concatenate((self.data, np.full(all_bars.shape[0], np.nan)))
        self.extra_points = all_bars.shape[0]
        return self.extra_points

    def _get_lowres_normalized_history(self, low_res=(8, 8, 64), max_iter=10):
        """Normalize the model to a new maximum height through iterative correction."""
        # Step 1: Generate a low-resolution model to estimate renormalization
        temp_model = GeoModel(self.bounds, resolution=low_res)
        temp_model.add_history(self.history)
        temp_model._apply_history_computation(
            keep_snapshots=False
        )  # Run without snapshots for efficiency

        # Step 2: Normalize the temporary model's height to 50% filled height through iterative correction
        # The goal is to get the model rock/air boundary to be within the model bounds where it can be renormalized
        new_max = temp_model.get_target_normalization(target_max=0.50, std_dev=0.)
        model_max_zbound = temp_model.bounds[2][1]  # Get the maximum z height of the model

        while True and max_iter > 0:
            # Shift the model so that all the points are shifted up or down aiming 
            # for the model to fill only 10% of the sample window height
            observed_max = temp_model.renormalize_height(new_max=new_max)
            if observed_max < model_max_zbound:
                break # If the observed data is lowered enough to be within the model's bounds, break
            max_iter -= 1

        # Assumes that the model rock/air boundary is in frame now
        # Step 3: Rerun normalization to reach the ~85% filled target specified by auto
        for _ in range(3):
            temp_model.renormalize_height(auto=True)

        # Step 4: Return the normalized history to be re-run at full resolution
        normed_history = temp_model.history

        del temp_model  # Clean up the temporary model

        return normed_history

    def fill_nans(self, value=EMPTY_VALUE):
        assert self.data is not None, "Data array is empty."
        indnan = np.isnan(self.data)
        self.data[indnan] = value
        return self.data

    def renormalize_height(self, new_max=0, auto=False, recompute=True):
        """Shift the model vertically so that the highest point in view field is at a new maximum height.
        Note this operation can be expensive since it requires recomputing the model.

        Parameters:
            new_max (float): The new maximum height for the model.
            auto (boolean): Automatically select a new maximum height based on the model's current height.
            recompute: Recompute the model after renormalization.

        Returns:
            The current maximum height of the model.
        """
        assert self.data is not None, "Data array is empty."
        # Find the highest point
        valid_indices = ~np.isnan(self.data)
        valid_z_values = self.xyz[valid_indices, 2]
        try:
            current_max_z = np.max(valid_z_values)
        except ValueError:
            # traceback.print_exc()
            zmin, zmax = self.get_z_bounds()
            current_max_z = zmin  # Defaulting to zmin if no valid max found

        if auto:
            new_max = self.get_target_normalization()

        # Calculate the model shift required to shift to a desired maximum height
        shift_z = new_max - current_max_z

        zmin, zmax = self.get_z_bounds()
        # print(f"Renormalizing model to new maximum height percent: {(new_max-zmin)/ (zmax - zmin):.2f}")

        # Add a shift transformation to the history and recompute
        self.add_history(Shift([0, 0, shift_z]))
        if recompute:
            self.clear_data()
            self._apply_history_computation()

        return current_max_z

    def get_target_normalization(self, target_max=0.85, std_dev=0.05):
        """Get the normalization factor to scale the model to a target maximum height.

        Parameters:
        - target_max: The target maximum height for the model as a fraction of total height.
        - std_dev: The standard deviation of the normal distribution used to add variation.
        """
        bounds = self.get_z_bounds()
        zmin, zmax = bounds
        z_range = zmax - zmin
        target_height = zmin + z_range * (
            target_max + np.abs(np.random.normal(0, std_dev))
        )
        log.debug(f"Normalization Target Height: {target_height}")
        return target_height

    def get_z_bounds(self):
        """Return the minimum and maximum z-coordinates of the model."""
        # Check if bounds is a tuple of tuples (multi-dimensional)
        if isinstance(self.bounds[0], tuple):
            # Multi-dimensional bounds, assuming the last tuple represents the z-dimension
            z_vals = self.bounds[-1]
        else:
            # Single-dimensional bounds
            z_vals = self.bounds

        return z_vals

    def get_data_grid(self):
        """Return the model data in meshgrid form."""
        return self.data.reshape(self.X.shape)

    def add_topography(self, mesh):
        """Add a topography mesh to the model.

        Parameters:
        - mesh: A 2D numpy array representing the topography mesh.
        """
        # Interpolate the topography mesh to match the model resolution
        resampled_mesh = resample_mesh(mesh, self.resolution[:2])

        # Expand the 2D topography mesh to match the 3D volume
        expanded_mesh = np.repeat(
            resampled_mesh[:, :, np.newaxis], self.resolution[2], axis=2
        )

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
        Special initializer to create a GeoModel instance from a 1xXxYxZ or XxYxZ shaped data tensor.
        Initializes history to [NullProcess()] and sets up the mesh and data to allow GeoModel
        plotting tools and processing methods to be used.

        Args:
            data_tensor (torch.Tensor): The data tensor to initialize the model with; can be 1xXxYxZ or XxYxZ tensor.
            bounds (tuple, optional): The bounds of the model in measurement units, if not provided defaults to the resolution of the tensor.

        Returns:
            GeoModel: An instance of the GeoModel class.
        """
        # Check and adjust tensor dimensions if needed
        if data_tensor.dim() == 4 and data_tensor.size(0) == 1:
            data_tensor = data_tensor.squeeze(0)  # Remove the singleton dimension

        # Ensure now the tensor is 3D
        assert (
            data_tensor.dim() == 3
        ), "Data tensor must be either 1xXxYxZ or XxYxZ after squeezing."

        # Extract resolution from tensor shape
        resolution = data_tensor.shape  # Already a tuple of three dimensions
        if bounds is None:
            bounds = (0, resolution[0]), (0, resolution[1]), (0, resolution[2])

        instance = cls(bounds, resolution)
        # Setup mesh for X, Y, Z coordinates and flattened xyz array
        instance.setup_mesh()
        # Insert torch tensor data into model
        instance.data = (
            data_tensor.detach().numpy().flatten()
        )  # Convert tensor to numpy array and flatten it
        # Set history to [NullProcess()] signifying no known geological history
        instance.history = [NullProcess()]

        return instance