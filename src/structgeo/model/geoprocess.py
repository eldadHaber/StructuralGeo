import numpy as np
import itertools

from structgeo.model.util import *

class GeoProcess:
    """Base class for all geological processes.
    
    Conventions:
    Strike, dip, and rake are in degrees.    
    """    
    pass

class Deposition(GeoProcess):
    """
    Base class for all deposition processes, such as layers and dikes.
    
    Depositions modify the geological data (e.g., rock types) associated with a mesh point
    without altering or transforming the mesh.
    """
    def run(self, xyz, data):
        raise NotImplementedError()
    
class Transformation(GeoProcess):
    """
    Base class for all transformation processes, such as tilting and folding.
    
    Transformations modify the mesh coordinates without altering the data contained at each point.
    Frame convention is that north is strike on positive x-axis
    """
    def run(self, xyz, data):
        raise NotImplementedError()
        
class Layer(Deposition):
    """ Fill the model with a layer of rock. """
    def __init__(self, base, width, value):
        self.base = base
        self.width = width
        self.value = value

    def run(self, xyz, data):
        # Extract z-coordinates
        z_coords = xyz[:, 2]

        # Create a mask where z is within the specified range and data is None
        mask = (self.base <= z_coords) & (z_coords <= self.base + self.width) & (np.isnan(data))

        # Apply the mask and update data where condition is met
        data[mask] = self.value

        # Return the unchanged xyz and the potentially modified data
        return xyz, data
    
class Shift(Transformation):
    """ Shift the model by a given vector. """
    def __init__(self, vector):
        self.vector = np.array(vector)
        
    def __str__(self):
        return f"Shift: vector {self.vector}"
    
    def run(self, xyz, data):
        # Apply the shift to the xyz points (inverse operation is negative vector)
        xyz_transformed = xyz - self.vector
        return xyz_transformed, data
    
class Rotate(Transformation):
    """ Rotate the model by a given angle about an axis. """
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
    """ Fill the model with a base level of bedrock.
    
    Parameters:
    - base: The z-coordinate of the base level (top of the rock layer)
    - value: The rock type value to assign to the bedrock layer
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
    
    def last_value(self, data):
        return self.value
    
class Sedimentation(Deposition):
    """ Fill with layers of sediment, with rock values and thickness of layers given as lists.
    The sedimentation starts at the lowest nan point in the mesh and builds up sequentially.
    If mesh is all NaN, an optional floor value can be set to start sedimentation from.
    
        Parameters:
              value_list (iterable float): a list of values to sample from for rock types
              value_selector (iterable float): an optional object to sample values from the list
              base (float): Optional floor value for sedimentation, itherwise starts at lowest NaN mesh point
    """
    def __init__(self, value_list, thickness_list, base=np.nan):
        self.value_list = value_list
        self.thickness_list = thickness_list
        # Initialize the base and top of layer, to be computed at generation time
        self.base = base
        
    def __str__(self):
        values_summary = ", ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in self.value_list[:3])
        values_summary += "..." if len(self.value_list) > 3 else ""
        thicknesses = ", ".join(f"{t:.3f}" for t in self.thickness_list[:3])
        thicknesses += "..." if len(self.thickness_list) > 3 else ""
        return (f"Sedimentation: rock type values [{values_summary}], "
                f"and thicknesses {thicknesses}.")  
        
    def run(self, xyz, data):
        if np.isnan(self.base):            
            # Get the lowest mesh point to build layers from the bottom up
            current_base = self.get_lowest_nan(xyz, data)
        else:
            current_base = self.base
            
        # Form a cycling iterator for thicknesses
        thickness= itertools.cycle(self.thickness_list)
        
        for value in self.value_list:
            current_top = current_base + next(thickness)                       
            # Mask for the current layer of deposit
            mask = (xyz[:, 2] < current_top) & (xyz[:, 2] >= current_base) & (np.isnan(data))            
            # Assign mapped value to the layer
            if np.any(mask):
                data[mask] = value            
            # Update the base for the next layer (sequential deposition)
            current_base = current_top
            
        return xyz, data
    
    def get_lowest_nan(self, xyz, data):
        # Get the lowest mesh point with NaN in data to build layers from the bottom up
        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            lowest = np.min(xyz[nan_mask, 2])
            return lowest
        else:
            return float('Inf')  
        
    def last_value(self):
        return self.value_list[-1]
     
class Dike(Deposition):
    """ Insert a dike to overwrite existing data values.
    
    Parameters:
    strike (float): Strike angle in CW degrees (center-line of the dike) from north
    dip (float): Dip angle in degrees
    width (float): Net Width of the dike
    origin (float): Origin point of the local coordinate frame
    value (float): Value of rock-type to assign to the dike 
    """
    def __init__(self, strike=0., dip=90., width=1., origin=(0,0,0), value=0.):
        self.strike = np.radians(strike)
        self.dip = np.radians(dip)
        self.width = width
        self.origin = np.array(origin)
        self.value = value
        
    def __str__(self):
        # Convert radians back to degrees for more intuitive understanding
        strike_deg = np.degrees(self.strike)
        dip_deg = np.degrees(self.dip)
        origin_str = ", ".join(f"{coord:.1f}" for coord in self.origin)  # Format each coordinate component

        return (f"Dike: strike {strike_deg:.1f}°, dip {dip_deg:.1f}°, width {self.width:.1f}, "
                f"origin ({origin_str}), value {self.value:.1f}.")

    def run(self, xyz, data):
        # Calculate rotation matrices
        M1 = rotate([0, 0, 1.], -self.strike)  # Rotation around z-axis for strike
        M2 = rotate([0., 1., 0], self.dip)     # Rotation around y-axis in strike frame for dip

        N = M1 @ M2 @ [0.0, 0.0, 1.0]
        N /= np.linalg.norm(N)  # Normalize the normal vector

        # Calculate distances from the dike plane
        dists = np.dot(xyz - self.origin, N)

        # Update data based on whether points are within the width of the dike and only where there
        # is existing data to avoid sky dikes
        mask = (np.abs(dists) <= self.width / 2.0) & (~np.isnan(data))
        
        data[mask] = self.value

        return xyz, data
    
    def last_value(self):
        return self.value
    
class UnconformityBase(Deposition):
    """ Erode the model from a given base level upwards
    
    Parameters:
    base (float): base level of erosion
    """
    
    def __init__(self, base, value=np.nan):
        self.base = base
        self.value = value
                
    def run(self, xyz, data):                
        # Mask for points above the base level
        mask = xyz[:, 2] > self.base        
        # Apply the mask and update data where condition is met
        data[mask] = self.value

        # Return the unchanged xyz and the potentially modified data
        return xyz, data
    
class UnconformityDepth(Deposition):
    """ Erode the model from the highest point downwards by a thickness.
    
    Parameters:
    depth (float): Thickness of the erosion layer, measured from the peak of the surface mesh
    """
    
    def __init__(self, depth, value=np.nan):
        self.depth = depth
        self.value = value
        self.peak = None
    
    def run(self, xyz, data):
        # Find the peak of non-NaN data
        self.peak = np.max(xyz[:, 2][~np.isnan(data)])
        
        # Erode down
        mask = xyz[:, 2] > self.peak - self.depth
        data[mask] = self.value

        # Return the unchanged xyz and the potentially modified data
        return xyz, data
   
class Tilt(Transformation):
    """ Tilt the model by a given strike and dip and an origin point.
    
    Parameters:
    strike (float): Strike angle in CW degrees (center-line of the dike) from north
    dip    (float): Dip of the tilt in degrees (CW from the strike axis)
    origin (tuple): Origin point for the tilt (x,z,y)
    """
    def __init__(self, strike, dip, origin=(0,0,0)):
        self.strike = np.radians(strike)  # Convert degrees to radians
        self.dip = np.radians(dip)  # Convert degrees to radians
        self.origin = np.array(origin)
        
    def __str__(self):
        # Convert radians back to degrees for more intuitive understanding
        strike_deg = np.degrees(self.strike)
        dip_deg = np.degrees(self.dip)
        origin_str = ", ".join(f"{coord:.1f}" for coord in self.origin)  # Format each coordinate component

        return (f"Tilt: strike {strike_deg:.1f}°, dip {dip_deg:.1f}°,"
                f"origin ({origin_str})")

    def run(self, xyz, data):
        # Calculate rotation axis from strike (rotation around z-axis)
        axis = rotate([0, 0, 1], -self.strike) @ [0, 1., 0]
        # Calculate rotation matrix from dip (tilt)
        R = rotate(axis, -self.dip)
        
        # Apply rotation about origin -> translate to origin, rotate, translate back
        # Y = R * (X - O) + O
        xyz = xyz @ R.T + (-self.origin @ R.T + self.origin)

        # Apply rotation to xyz points
        return xyz, data  # Assuming xyz is an Nx3 numpy array 
      
class Fold(Transformation):
    """ Generalized fold transformation. 
    Convention is that 0 rake with 90 dip fold creates vertical folds.
    
    Parameters:
    strike: Strike in degrees
    dip: Dip in degrees
    rake: Rake in degrees
    period: Period of the fold in units of the mesh
    amplitude: Amplitude of the fold (motion along slip_vector)
    phase: Phase shift of the fold (in units of the period) [0,1)
    shape: Shape parameter for the fold (enhances 3rd harmonic component in the fold)
    origin: Origin point for the fold
    periodic_func: Custom periodic function for the fold (replaces default cosine function)
                    User provided function should be 1D and accept an array of n_cycles
                    Does not require being periodic, but should be normalized to 1 amplitude
    """
    
    def __init__(self, strike = 0., 
                 dip = 90., rake = 0., 
                 period = 50., 
                 amplitude = 10., 
                 phase = 0.0,
                 shape = 0.0, 
                 origin=(0, 0, 0),
                 periodic_func=None):
        self.strike = np.radians(strike)
        self.dip = np.radians(dip)
        self.rake = np.radians(rake)
        self.period = period
        self.amplitude = amplitude
        self.phase = phase
        self.shape = shape
        self.origin = np.array(origin)
        # Accept a custom periodic function or use the default otherwise
        self.periodic_func = periodic_func if periodic_func else self.periodic_func_default
        
    def __str__(self):
        # Convert radians back to degrees for more intuitive understanding
        strike_deg = np.degrees(self.strike)
        dip_deg = np.degrees(self.dip)
        rake_deg = np.degrees(self.rake)
        origin_str = ", ".join(f"{coord:.1f}" for coord in self.origin)
        
        return (f"Fold: strike {strike_deg:.1f}°, dip {dip_deg:.1f}°, rake {rake_deg:.1f}°, period {self.period:.1f},"
                f"amplitude {self.amplitude:.1f}, origin ({origin_str}).")

    def run(self, xyz, data):
        # Adjust the rake to have slip vector (fold amplitude) perpendicular to the strike
        slip_vector, normal_vector = slip_normal_vectors(self.rake + np.pi/2, self.dip, self.strike)

        # Translate points to origin coordinate frame
        v0 = xyz - self.origin
        # Orthogonal distance from origin along U
        fU = np.dot(v0, normal_vector)
        # Calculate the number of cycles orthogonal distance
        n_cycles =  fU / self.period
        # Get the displacement as a function for n_cycles
        displacement_distance = self.amplitude * self.periodic_func(n_cycles)      
        # Calculate total displacement for each point, recast off as a column vector
        displacement_vector = slip_vector * displacement_distance[:, np.newaxis] 
        # Return to global coordinates
        xyz_transformed = xyz + displacement_vector + self.origin

        return xyz_transformed, data  
    
    def periodic_func_default(self, n_cycles):
        """ Default periodic function for the fold transformation. uses shaping from 3rd harmonic."""
        # Normalize to amplitude of 1
        norm = (1 + self.shape**2)**0.5
        func = np.cos(2 * np.pi * n_cycles) + self.shape * np.cos(3 * 2 * np.pi * n_cycles)  
        return func / norm   

class Slip(Transformation):
    """Gereralized slip transformation.
    
    Parameters:
    displacement_func (callable): Custom displacement function for the slip. Function should map 
    a distance from the slip plane to a displacement value.
    strike (float): Strike in degrees
    dip (float): Dip in degrees
    rake (float): Rake in degrees
    amplitude (float): Amplitude of the slip (motion along slip_vector)
    origin (tuple): Origin point for the slip (local coordinate frame), (x,y,z)
    
    """
    def __init__(self,                 
                displacement_func,
                strike = 0., 
                dip = 90., 
                rake = 0., 
                amplitude = 2., 
                origin=(0, 0, 0),
                ):
        self.strike = np.radians(strike)
        self.dip = np.radians(dip)
        self.rake = np.radians(rake)
        self.amplitude = amplitude
        self.origin = np.array(origin)
        self.displacement_func = displacement_func
        
    def __str__(self):
        strike_deg = np.degrees(self.strike)
        dip_deg = np.degrees(self.dip)
        rake_deg = np.degrees(self.rake)
        origin_str = ", ".join(map(str, self.origin))
        return (f"{self.__class__.__name__} with strike {strike_deg:.1f}°, dip {dip_deg:.1f}°, rake {rake_deg:.1f}°, "
                f"amplitude {self.amplitude:.1f}, origin ({origin_str}).")
    
    def default_displacement_func(self, distances):
        # A simple linear displacement function as an example
        return np.zeros(np.shape(distances))  # Displaces positively where the distance is positive

    def run(self, xyz, array):
        # Slip is measured from dip vector, while the slip_normal convention is from strike vector, add 90 degrees
        slip_vector, normal_vector = slip_normal_vectors(self.rake , self.dip, self.strike)
        
        # Translate points to origin coordinate frame
        v0 = xyz - self.origin        
        # Orthogonal distance from origin along U
        distance_to_slip = np.dot(v0, normal_vector)
        # Apply the displacement function to the distances along normal
        displacements = self.amplitude * self.displacement_func(distance_to_slip)
        # Calculate the movement vector along slip direction
        displacement_vectors = displacements[:, np.newaxis] * slip_vector
        # Return to global coordinates and apply the displacement
        xyz_transformed = xyz + displacement_vectors + self.origin
        return xyz_transformed, array
    
class Fault(Slip):
    """
    A subclass of Slip specifically for modeling brittle fault transformations where
    displacement occurs as a sharp step function across the fault plane. This class
    implements a binary displacement function that represents the sudden shift
    characteristic of brittle faults.

    Parameters:
    - strike (float): Strike angle in degrees
    - dip (float): Dip angle in degrees
    - rake (float): Rake angle in degrees, convention is 0 for side-to-side motion
    - amplitude (float): The maximum displacement magnitude along the slip_vector.
    - origin (tuple of float): The x, y, z coordinates from which the fault originates within the local coordinate frame.
    
    This implementation causes a displacement strictly on one side of the fault, making it suitable for
    simulating scenarios where a clear delineation between displaced and stationary geological strata is necessary.
    
    Example:
        # Creating a Fault instance with specific geological parameters
        fault = Fault(strike=30, dip=60, rake=90, amplitude=5, origin=(0, 0, 0))
    """
    def __init__(self, 
                strike = 0., 
                dip = 90., 
                rake = 0., 
                amplitude = 2., 
                origin=(0, 0, 0),
                ):
        super().__init__(self.fault_displacement, strike, dip, rake, amplitude, origin)
        self.rotation = 0
        
    def fault_displacement(self, distances):
        return self.amplitude * np.sign(distances)
    
class Shear(Slip):
    """ A subclass of Slip for modeling shear transformations, a plastic deformation process.
    Displacement is modeled as a sigmoid function that increases with distance from the slip plane.
    
    Parameters:
    - strike (float): Strike angle in degrees
    - dip (float): Dip angle in degrees
    - rake (float): Rake angle in degrees, convention is 0 for side-to-side motion
    - amplitude (float): The maximum displacement magnitude along the slip_vector.
    - origin (tuple of float): The x, y, z coordinates from which the fault originates within the local coordinate frame.
    - steepness (float): The steepness of the sigmoid function, controlling the rate of change of displacement.
    """
    def __init__(self, 
                strike = 0., 
                dip = 90., 
                rake = 0., 
                amplitude = 2., 
                steepness = 1.,
                origin=(0, 0, 0),
                ):
        self.steepness = steepness
        super().__init__(self.shear_displacement, strike, dip, rake, amplitude, origin)
        
    def shear_displacement(self, distances):
        # The sigmoid function will be centered around zero and will scale with amplitude
        return self.amplitude * (1 / (1 + np.exp(-self.steepness * distances)))
    
    def run(self, xyz, array):
        # Apply the shear transformation
        xyz_transformed, array = super().run(xyz, array)
        return xyz_transformed, array