# Geomodel.py
### Parameters
#### Input parameters: 
- **Bounds**: (0,16) - absolute values of bounds of model (can be tuple of 3)
- **Resolution**: (128) - number of divisions between bounds (can be tuple of 3)
- **dtype**: np.float32
- **name:** "model"
- **height_tracking**: true

#### Computed parameters: 
History:
- **history**
- **history_unpacked**

Mesh:
- **data**: nx1 vector of data values
- **xyz**: nx3 matrix of mesh points
- **X**: 3d meshgrid representing X
- **Y**: 3d meshgrid representing Y
- **Z**: 3d meshgrid representing Z
- **mesh_snapshots**: 4d array with mesh states
- **data_snapshots**: 2d array with data states 

#### Other parameters: 
- Empty_value (-1), HEIGHT_BAR_EXT_FACTOR (3), HEIGHT_BAR_RESOLUTION (128), HEIGHT_NORMALIZATION_FILL_TARGET (.85), HEIGHT_NORMALIZATION_STD_DEV (.05)


#### Functions
#### Print functions: 
- **__repr__**: "GeoModel(name={self.name}, bounds={self.bounds}, resolution={self.resolution})"
- **__str__**: "GeoModel: {self.name}\nBounds: {self.bounds}\nResolution: {self.resolution}\nHistory: {self.get_history_string()}"
- **__repr_html_**: display name, data, bounds, resolution, history in html

#### Getter functions:
- **get_history_string**: prints entire history unpacked
- **get_data_grid**: return model reshaped on 3d meshgrid
- **get_z_bounds**: return min and max z-coords of model
- **get_max_filled_height**: Get highest z-value among non-NaN data points 

#### Setup functions:
- **validate_model_params**: validate resolutions and bounds 
- **_setup_mesh**: sets up all meshes: X, Y, Z, xyz, data
- **fill_nans**: replace NaNs with EMPTY_VALUE
- **_unpack_history**: create a copy of history that has all compound events unpacked to simple ones
- **_prepare_snapshots**: TODO

#### Height tracking functions: 
- **_remove_tracking_points**: remove height tracking bars from xyz, data, data_snapshots, mesh_snapshots, 
                           reset num_tracking_points, height_tracking_indices
                           occurs after the computations 
- **renormalize_height**: TODO
- **_get_lowres_z_shift_normalization**: TODO 
- **_add_height_tracking_bars**: TODO
- **get_target_normalization**: TODO

#### Clear functions: 
- **clear_history**: set history and history_unpacked to empty arrays
- **clear_data**: resets mesh_snapshots, data_snapshots, data, xyz, X, Y, Z

#### Model Creation functions:
- **compute_model**: Applies z norm and computes full model
            1. applies normalization if bool is set to true
                    - finds the z shift using _get_lowres_z_shift_normalization
                    - adds a z shift event in the history (Shift class)
            2.  runs _apply_history_computation 
- **_apply_history_computation**: Does full computation of model (after z shift)
            1. initial setup: clear_data, _setup_mesh, _add_height_tracking_bars,_unpack-history, _prepare_snapshots
            2. runs _backward_pass
            3. runs _forward_pass
            4. clean model up - _remove_tracking_points (if specified) and removes data for mesh_snapshots, data_snapshots (if specified)
- **_backward_pass**: Run a full backwards pass, while saving snapshots
TODO
- **_forward_pass**: TODO

#### Other functions:
- **from_tensor**: Create geomodel from pytorch tensor 
- **add_topography**: TODO

Geoprocess.py

