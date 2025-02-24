""" A collection of curated geological words that combine random variables and geoprocesses into a single generator class. """

import copy
from abc import ABC as _ABC
from abc import abstractmethod
from typing import List, Union

import numpy as np

import geogen.model as geo
import geogen.probability as rv
from geogen.probability import FourierWaveGenerator, MarkovSedimentHelper

# Expected model bounds
BOUNDS_X = (-3840, 3840)
BOUNDS_Y = (-3840, 3840)
BOUNDS_Z = (-1920, 1920)
MAX_BOUNDS = (BOUNDS_X, BOUNDS_Y, BOUNDS_Z)  # Intended maximum bounds for the model
X_RANGE = BOUNDS_X[1] - BOUNDS_X[0]
Z_RANGE = BOUNDS_Z[1] - BOUNDS_Z[0]

# Rock category mapping
BED_ROCK_VAL = 0
SEDIMENT_VALS = [1, 2, 3, 4, 5]
DIKE_VALS = [6, 7, 8]
INTRUSION_VALS = [9, 10, 11]
BLOB_VALS = [12, 13]

# A target mean for random sedimentation depth
MEAN_SEDIMENTATION_DEPTH = Z_RANGE / 8


class GeoWord(_ABC):
    """
    Base class for generating geological events within a hierarchical structure.

    The `GeoWord` class forms the foundation for constructing tree-like histories of geological processes.
    Each instance represents a node in this structure, which can either branch into further `GeoWord` events
    or terminate with one or more defined `GeoProcess` instances.


    Parameters
    ----------
    seed : Optional[int]
        An optional seed for the random number generator, ensuring reproducibility.

    Attributes
    ----------
    hist : List[Union[geo.GeoProcess, GeoWord]]
        A list of geological processes forming the history of the GeoWord, to be randomly sampled.
    seed : Optional[int]
        The seed for the random number generator, ensuring reproducibility.
    rng : np.random.Generator
        The random number generator used to sample random variables.
    """

    def __init__(self, seed: int = None):
        self.hist = []
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def build_history(self):
        """
        Constructs the geological history for the GeoWord.

        This method should be overridden by subclasses to define the geological history for the event.
        History is built by adding GeoProcess, GeoWords, or lists of these using the `add_process` method.
        """
        pass

    def generate(self):
        """
        Generates the geological history by building and compiling it into a CompoundProcess.

        Returns
        -------
        geo.CompoundProcess
            A sampled geological history snippet with a CompoundProcess wrapper
        """
        self.hist.clear()
        self.build_history()
        geoprocess = geo.CompoundProcess(self.hist.copy(), name=self.__class__.__name__)
        return geoprocess

    def add_process(
        self,
        item: Union[geo.GeoProcess, "GeoWord", List[Union[geo.GeoProcess, "GeoWord"]]],
    ):
        """
        Adds a process or event to the GeoWord history.

        This method supports adding individual `GeoProcess` or `GeoWord` instances, as well as lists of them.
        Items are added in chronological order from earliest to latest event. Recursive calls to other GeoWords
        are supported by calling their own generate methods to build their sub-histories.

        Parameters
        ----------
        item : Union[geo.GeoProcess, GeoWord, List[Union[geo.GeoProcess, GeoWord]]]
            The process or event to be added to the history.

        Raises
        ------
        ValueError
            If the item is not a `GeoProcess`, `GeoWord`, or a list of these.
        """
        if isinstance(item, GeoWord):
            self.hist.extend(item.generate().history)
        elif isinstance(item, geo.GeoProcess):
            self.hist.append(item)
        elif isinstance(item, list):
            for sub_item in item:
                self.add_process(sub_item)
        else:
            raise ValueError(f"Expected GeoWord, GeoProcess, or list of these, got {type(item)}")


""" Identity word for generating a null event. """


class NullWord(GeoWord):  # Validated
    """A null geological event, generating a process that does nothing."""

    def build_history(self):
        self.add_process(geo.NullProcess())


""" Infinite foundation layer(s) for initial model"""


class InfiniteBasement(GeoWord):  # Validated
    """A foundational bedrock layer to simulate an infinite basement."""

    def build_history(self):
        # Generate a simple basement layer
        self.add_process(geo.Bedrock(base=0, value=BED_ROCK_VAL))


class InfiniteSedimentUniform(GeoWord):  # Validated
    """
    A large sediment accumulation to simulate deep sedimentary layers.

    Effects:
    --------
    This word fills the model with deep sedimentary layers of uniform thickness. Each layer has a
    random sediment type, and the sedimentation continues until the total depth is filled.
    Bedrock is placed below to ensure full coverage underneath the sediment.
    """

    def build_history(self):
        # Choose a large depth that runs beyond the model's height extension bars
        depth = (Z_RANGE) * (3 * geo.GeoModel.HEIGHT_BAR_EXT_FACTOR)  # Pseudo-infinite using a large depth
        sediment_base = -depth  # The sediment base is located so that it builds back up to z=0

        # calculate layer thicknesses to fill the depth of sediment
        vals = []
        thicks = []
        while depth > 0:
            vals.append(self.rng.choice(SEDIMENT_VALS))
            thicks.append(self.rng.uniform(50, Z_RANGE / 4))
            depth -= thicks[-1]

        # Bedrock ensures full coverage underneath sediment in all cases
        self.add_process(geo.Bedrock(base=sediment_base, value=BED_ROCK_VAL))
        self.add_process(geo.Sedimentation(vals, thicks, base=sediment_base))


class InfiniteSedimentMarkov(GeoWord):  # Validated
    """
    A large sediment accumulation to simulate deep sedimentary layers with dependency on previous layers.

    Random Variables (RVs)
    ----------------------
    - `thickness_variance` : Controls the variability in thickness
      of sediment layers. Higher values introduce more thickness variability.

    - `dirichlet_alpha` : Controls the likelihood of repeating sediment types.
      Lower values increase repeatability, while higher values increase variability.

    - `minimum_layer_thickness` : Represents the smallest possible thickness for a sediment layer.

    - `maximum_layer_thickness` : Represents the largest possible thickness for a sediment layer.

    - `anticorrelation_factor` : Fixed at 0.05. Low factor ensures change in sediment type between layers.

    Effects:
    --------
    The Markov process introduces a correlation between the layers, meaning the characteristics of one layer
    influence the next. This results in a more realistic sedimentary sequence where the variability in layer
    thickness and sediment type is dependent on the preceding layers. Bedrock is placed below the sediment to
    ensure full coverage underneath the layers.
    """

    def build_history(self):
        # Caution, the depth needs to extend beyond the bottom of the model mesh,
        # Including height bar extensions for height tracking, or it will leave a gap underneath
        depth = (Z_RANGE) * (3 * geo.GeoModel.HEIGHT_BAR_EXT_FACTOR)
        sediment_base = -depth

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        # Explanation can be found in the helper class
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(200, Z_RANGE / 4),
            thickness_variance=self.rng.uniform(0.1, 0.6),
            dirichlet_alpha=self.rng.uniform(0.6, 1.2),
            anticorrelation_factor=0.05,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)

        # Bedrock ensures full coverage underneath sediment in all cases
        self.add_process(geo.Bedrock(base=sediment_base, value=BED_ROCK_VAL))
        self.add_process(geo.Sedimentation(vals, thicks, base=sediment_base))


class InfiniteSedimentTilted(GeoWord):  # Validated
    """
    A large sediment accumulation to simulate deep sedimentary layers with a tilt.

    Random Variables (RVs)
    ----------------------

    - `thickness_variance` : Controls the variability in thickness of the sediment layers.

    - `dirichlet_alpha` : Controls the likelihood of repeating sediment types, or formation of mutual loops.
      Lower values increase the pattern repeatability, while higher values increase variability between layers.

    - `minimum_layer_thickness` : Represents the smallest possible thickness for a sediment layer.

    - `maximum_layer_thickness` : Represents the largest possible thickness for a sediment layer.

    - `anticorrelation_factor` : Fixed at 0.05. Low factor ensures change in sediment type between layers.

    Effects:
    --------
    These RVs work together to simulate deep sedimentary layers that are tilted and then truncated at the bottom.
    The strike and dip control the orientation and angle of the tilt, while the sedimentation process creates layers
    of varying thickness and composition.
    """

    def build_history(self):
        # Sediment parameters
        depth = (Z_RANGE) * (
            3 * geo.GeoModel.HEIGHT_BAR_EXT_FACTOR
        )  # Large depth to simulate deep sediment accumulation
        sediment_base = -0.5 * depth  # Overbuild to allow room for tilting and subsequent truncation

        minimum_layer_thickness = 200  # Minimum thickness for sediment layers (meters)
        maximum_layer_thickness = Z_RANGE / 4  # Maximum thickness for sediment layers (based on Z_RANGE)
        thickness_variance = self.rng.uniform(0.1, 0.6)  # Variance in layer thickness
        dirichlet_alpha = self.rng.uniform(0.6, 1.2)  # Parameter controlling repeatability of layers
        anticorrelation_factor = 0.05  # Bias to prevent successive layers from being too similar

        # Tilt parameters
        strike = self.rng.uniform(0, 360)  # Random strike direction
        dip = self.rng.normal(0, 10)  # Tilt angle sampled from a normal distribution

        # Generate sediment layers using a Markov process
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(minimum_layer_thickness, maximum_layer_thickness),
            thickness_variance=thickness_variance,
            dirichlet_alpha=dirichlet_alpha,
            anticorrelation_factor=anticorrelation_factor,
        )
        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)

        # Sediment process
        sed = geo.Sedimentation(vals, thicks, base=sediment_base)
        tilt = geo.Tilt(strike=strike, dip=dip, origin=geo.BacktrackedPoint((0, 0, 0)))

        # Erosion process: truncating sediment at the base
        unc = geo.UnconformityBase(1000)  # Unconformity cuts off sediment above a base level

        # Add processes in the geological history
        self.add_process(geo.Bedrock(base=sediment_base, value=BED_ROCK_VAL))  # Bedrock underneath sediment
        self.add_process([sed, tilt, unc])  # Add sedimentation, tilt, and unconformity to history


""" Sediment Acumulation Events"""


class FineRepeatSediment(GeoWord):  # Validated
    """
    A series of thin sediment layers with repeating values.

    Random Variables (RVs)
    ----------------------
    - `depth` : Log-normal distributed total sedimentation depth with target mean and std.

    - `thickness_variance` : Controls the variability in thickness
      of sediment layers. Higher values introduce more thickness variability.

    - `dirichlet_alpha` : Controls the likelihood of repeating sediment types.
      Lower values increase repeatability, while higher values increase variability.

    - `minimum_layer_thickness` : Represents the smallest possible thickness for a sediment layer.

    - `maximum_layer_thickness` : Represents the largest possible thickness for a sediment layer.

    - `anticorrelation_factor` : Fixed at 0.05. Low factor ensures change in sediment type between layers.

    Effects:
    --------
    These RVs work together to create a sedimentary sequence that can vary in total depth, layer thickness,
    and repeatability of sediment types. Fine-tuning the RVs will affect the appearance of the simulated
    geological history.
    """

    def build_history(self):
        # Random variables with detailed annotations
        # Log-normal depth, with a mean and standard deviation designed for sedimentation layers
        depth = self.rng.lognormal(
            *rv.log_normal_params(mean=MEAN_SEDIMENTATION_DEPTH, std_dev=MEAN_SEDIMENTATION_DEPTH / 3)
        )
        # Markov helper random parameters
        minimum_layer_thickness = 100  # Minimum thickness for sediment layers (meters)
        maximum_layer_thickness = 400  # Maximum thickness for sediment layers
        thickness_variance = self.rng.uniform(
            0.1, 0.3
        )  # Range for sediment layer thickness variance in a set
        dirichlet_alpha = self.rng.uniform(
            0.6, 2.0
        )  # Dirichlet distribution parameter for layer transitions
        anticorrelation_factor = 0.05  # Fixed anticorrelation factor

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(minimum_layer_thickness, maximum_layer_thickness),
            thickness_variance=thickness_variance,
            dirichlet_alpha=dirichlet_alpha,
            anticorrelation_factor=anticorrelation_factor,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)

        self.add_process(geo.Sedimentation(vals, thicks))


class CoarseRepeatSediment(GeoWord):  # Validated
    """
    A series of thick sediment layers with repeating values.

    Random Variables (RVs)
    ----------------------
    - `thickness_variance` : Controls the variability in thickness
      of sediment layers. Higher values introduce more thickness variability.

    - `dirichlet_alpha` : Controls the likelihood of repeating sediment types.
      Lower values increase repeatability, while higher values increase variability.

    - `minimum_layer_thickness` : Represents the smallest possible thickness for a sediment layer.

    - `maximum_layer_thickness` : Represents the largest possible thickness for a sediment layer.

    - `anticorrelation_factor` : Fixed at 0.05. Low factor ensures change in sediment type between layers.

    Effects:
    --------
    This word simulates a sequence of thick, repeating sediment layers. The Markov process introduces consistency
    in sediment type, while variability in thickness is controlled by the random variables. The process continues
    until the total depth is filled with sediment.
    """

    def build_history(self):
        # Get a log-normal depth for the sediment block
        depth = self.rng.lognormal(
            # This helper calculates the required parameters to hit target mean and std for distro
            *rv.log_normal_params(mean=MEAN_SEDIMENTATION_DEPTH, std_dev=MEAN_SEDIMENTATION_DEPTH / 3)
        )

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(Z_RANGE / 12, Z_RANGE / 6),
            thickness_variance=self.rng.uniform(0.1, 0.2),
            dirichlet_alpha=self.rng.uniform(0.8, 1.2),  # Low alpha for high repeatability
            anticorrelation_factor=0.05,  # Low factor gives low repeatability
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)

        self.add_process(geo.Sedimentation(vals, thicks))


class SingleRandSediment(GeoWord):
    """
    A single sediment layer with a random value and thickness.

    Effects:
    --------
    A single sedimentation event occurs with a randomly selected sediment type and thickness.
    This is useful for generating isolated, standalone sediment layers in the geological model.
    """

    def build_history(self):
        val = self.rng.choice(SEDIMENT_VALS)
        sediment = geo.Sedimentation(
            [val],
            [self.rng.normal(MEAN_SEDIMENTATION_DEPTH, MEAN_SEDIMENTATION_DEPTH / 3)],
        )
        self.add_process(sediment)


""" Erosion events"""


class _BaseErosionWord(GeoWord):  # Validated
    """
    Reusable generic class for calculating total depth of erosion events.

    Random Variables (RVs)
    ----------------------
    - `erosion_factor` : Log-normal distributed scaling factor for erosion depth. Scales the MEAN_DEPTH

    Effects:
    --------
    The class calculates an erosion depth based on a log-normal distribution, allowing for a range of possible
    erosion depths. The depth is then applied in child classes to simulate different types of erosion events.
    """

    MEAN_DEPTH = Z_RANGE / 8

    def __init__(self, seed=None):
        super().__init__(seed)

    def calculate_depth(self):
        erosion_factor = self.rng.lognormal(
            *rv.log_normal_params(mean=1, std_dev=0.5)
        )  # Generally between .25 and 2.5
        erosion_factor = np.clip(erosion_factor, 0.25, 3)
        return erosion_factor * self.MEAN_DEPTH


class FlatUnconformity(_BaseErosionWord):  # Validated
    """
    Flat unconformity down to a random depth.

    Random Variables (RVs)
    ----------------------
    - `total_depth` : Log-normal distributed erosion depth, calculated using the base class method.
      The depth represents how much material is eroded from the geological model starting from the top.

    Effects:
    --------
    A flat erosion event removes material down to a certain depth across the entire model. This creates a
    uniform, flat unconformity surface in the model.
    """

    def build_history(self):
        total_depth = self.calculate_depth()
        unconformity = geo.UnconformityDepth(total_depth)
        self.add_process(unconformity)


class TiltedUnconformity(_BaseErosionWord):  # Validated
    """
    Slightly tilted unconformity down to a random depth.

    Random Variables (RVs)
    ----------------------
    - `num_tilts` : Determines the number of tilt-erosion iterations applied to the unconformity surface.

    - `total_depth` : Log-normal distributed erosion depth, calculated using the base class method.
      Represents the total depth to which the unconformity will erode.

    - `depths` : Dirichlet distributed depths for each tilt. The depths are distributed to sum to the total depth

    Effects:
    --------
    This word generates an unconformity where the surface is tilted in one or more directions,
    creating a varied erosion depth across the model. The tilting introduces asymmetry in the erosion pattern.
    """

    def build_history(self):
        num_tilts = self.rng.integers(1, 4)
        total_depth = self.calculate_depth()
        depths = np.random.dirichlet(alpha=[1] * num_tilts) * total_depth

        for depth in depths:
            strike = self.rng.uniform(0, 360)
            dip = self.rng.normal(0, 3)
            x, y, z = rv.random_point_in_ellipsoid(MAX_BOUNDS)
            origin = geo.BacktrackedPoint((x, y, 0))
            tilt_in = geo.Tilt(strike=strike, dip=dip, origin=origin)
            tilt_out = geo.Tilt(strike=strike, dip=-dip, origin=origin)

            unconformity = geo.UnconformityDepth(depth)

            self.add_process([tilt_in, unconformity, tilt_out])


class WaveUnconformity(_BaseErosionWord):
    """
    Wavy unconformity created by two orthogonal folds.

    Random Variables (RVs)
    ----------------------
    - `total_depth` : Log-normal distributed depth, scaled down to 80% of the standard depth.

    - `orientation` : Determines the principal orientation of the wave folds.

    - `num_harmonics` : Controls how many sine wave harmonics are combined to create folded surface

    - `smoothness` : Controls the balance between high and low frequency harmonics in the folded surface

    - `fold_period` : Determines the wavelength of the fold.

    - `amplitude` : Controls the height of the waves in the unconformity surface.

    - `springback_factor` : Determines how much the fold springs back. Full spring back
    leaves the strata flat, while partial spring back retains some of the fold's shape
    correlated with the unconformity.

    Effects:
    --------
    The unconformity creates a wavy surface by folding the geological structure in two orthogonal directions.
    The folding amplitude and period determine the size and frequency of the waves, while the springback factor
    controls how much the fold amplitude is reduced. This leads to a dynamic and non-uniform unconformity surface.
    """

    def build_history(self):
        total_depth = self.calculate_depth() * 0.8
        orientation = self.rng.uniform(0, 360)  # Principal orientation of the waves
        fold_in1, fold_out1 = self.get_fold_pair(strike=orientation, dip=90)
        fold_in2, fold_out2 = self.get_fold_pair(strike=orientation + 90, dip=90)
        unconformity = geo.UnconformityDepth(total_depth)
        self.add_process([fold_in1, fold_in2, unconformity, fold_out2, fold_out1])

    def get_fold_pair(self, strike, dip):
        wave_generator = FourierWaveGenerator(num_harmonics=np.random.randint(3, 5), smoothness=1)
        period = self.rng.uniform(0.5, 2) * X_RANGE
        min_amp = period * 0.001
        max_amp = period * 0.04
        amp = rv.beta_min_max(a=1.5, b=1.5, min_val=min_amp, max_val=max_amp)
        fold_params = {
            "strike": strike,
            "dip": dip,  # average of dike dip and 90
            "rake": self.rng.normal(0, 10),  # Bias to vertical folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold_in = geo.Fold(**fold_params)
        fold_out = copy.deepcopy(fold_in)
        springback_factor = np.clip(self.rng.normal(0.75, 0.1), a_min=0, a_max=1)
        fold_out.amplitude *= -1 * springback_factor
        return fold_in, fold_out


""" Dike Events"""


class DikePlaneWord(GeoWord):  # Validated
    """
    Base GeoWord for forming dikes with organic thickness variations.

    Random Variables (RVs)
    ----------------------
    - `width` : overall thickness of the dike

    - `length` : the length of the dike along long axis

    - `wobble_factor` : Introduces randomness in the thickness variation,
      with higher values leading to more pronounced variations.

    - `amp` : Controls the unevenness of the dike thickness, larger values introducing more variability in thickness.

    - `expo` : Controls the sharpness of tapering at the ends of the dike.  Higher values lead
    to more abrupt tapering at the edges, while lower values create smoother transitions.

    Effects:
    --------
    This word generates a dike with non-uniform thickness along its length and width, shaped by a combination
    of Fourier-generated variability and organic tapering at the ends. The dike parameters—such as width, length,
    and tapering—are controlled by the random variables, making each dike unique in terms of its shape and orientation.
    The dike is embedded into the model at a random origin point and extends along its defined strike and dip.
    """

    class OrganicDikeThicknessFunc:
        def __init__(self, length, expo, amp, x_var, y_var):
            self.length = length
            self.expo = expo
            self.amp = amp
            self.x_var = x_var
            self.y_var = y_var

        def __call__(self, x, y):
            # Elliptical tapering thickness 0 at ends
            taper_factor = np.sqrt(np.maximum(1 - np.abs((2 * y / self.length)) ** self.expo, 0))
            # The thickness modifier combines 2d fourier with tapering at ends
            return (
                (1 + self.amp * self.x_var(x / Z_RANGE))
                * (1 + self.amp * self.y_var(y / X_RANGE))
                * taper_factor
            )

    def get_organic_thickness_func(self, length, wobble_factor=1.0):
        """
        Thickness provides a multiplier for the thickness of the dike using the
        local x-y coordinates of the dike plane itself. A zero multiplier closes the dike.

        The x and y directions have thickness variations introduced using low pass fourier waves.
        The exponent controls the sharpness of the tapering at the ends of the dike.

        The taper factor is and elliptical cross section shape 1 = t^2 - (y/L)^exponent

        The wobble and elliptical tapering are multiplicatively combined to shape the dike.
        """
        # Make a fourier based modifier for both x and y
        fourier = rv.FourierWaveGenerator(num_harmonics=4, smoothness=1)
        x_var = fourier.generate()
        y_var = fourier.generate()
        amp = self.rng.uniform(0.1, 0.2) * wobble_factor  # unevenness of the dike thickness
        expo = self.rng.uniform(4, 10)  # Hyper ellipse exponent controls tapering sharpness

        return self.OrganicDikeThicknessFunc(length, expo, amp, x_var, y_var)

    def build_history(self):
        width = rv.beta_min_max(2, 4, 50, 500)
        length = rv.beta_min_max(2, 2, 300, 16000)
        origin = rv.random_point_in_ellipsoid(MAX_BOUNDS)
        back_origin = geo.BacktrackedPoint(origin)  # Use a backtracked point to ensure origin is in view
        dike_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 10),  # Bias towards vertical dikes
            "origin": back_origin,
            "width": width,
            "value": self.rng.choice(DIKE_VALS),
            "thickness_func": self.get_organic_thickness_func(
                length, wobble_factor=self.rng.uniform(0.5, 1.5)
            ),
        }
        dike = geo.DikePlane(**dike_params)

        self.add_process(dike)


class SingleDikeWarped(DikePlaneWord):  # Validated
    """
    A single dike with organic thickness and a serpentine length.

    Random Variables (RVs)
    ----------------------
    - `width` : overall thickness of the dike

    - `length` : the length of the dike along long axis

    - `wobble_factor` : Increases the degree of randomness in the thickness variation, creating
      a more pronounced organic, uneven appearance.

    - `fold_period` :Controls the wavelength of the fold warping on the dike.

    - `amplitude` : Determines the amplitude of the fold warping on the dike.

    Effects:
    --------
    The generated dike has an organic, serpentine structure with varying thickness and a warped path. The
    `wobble_factor` introduces randomness to the thickness, while the folding process applies a wave-like
    distortion along the length of the dike, making it appear more natural and irregular. The random strike,
    dip, width, and length add further variability to the dike's final shape.
    """

    def build_history(self):
        strike = self.rng.uniform(0, 360)
        dip = self.rng.normal(90, 10)
        width = rv.beta_min_max(2, 4, 50, 500)
        length = rv.beta_min_max(2, 2, 300, 16000)
        dike_params = {
            "strike": strike,
            "dip": dip,  # Bias towards vertical dikes
            "origin": geo.BacktrackedPoint(rv.random_point_in_ellipsoid(MAX_BOUNDS)),
            "width": width,
            "value": self.rng.choice(DIKE_VALS),
            "thickness_func": self.get_organic_thickness_func(length, wobble_factor=1.5),
        }
        dike = geo.DikePlane(**dike_params)

        # Wrap the dike in a change of coordinates via fold
        fold_in = self.get_fold(strike, dip)
        fold_out = copy.deepcopy(fold_in)
        fold_out.amplitude *= -1

        self.add_process([fold_in, dike, fold_out])

    def get_fold(self, dike_strike, dike_dip):
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(4, 8), smoothness=np.random.normal(1.2, 0.2)
        )
        period = self.rng.uniform(0.5, 2) * X_RANGE
        amp = self.rng.uniform(10, 250)
        fold_params = {
            "strike": dike_strike + 90,
            "dip": (2 * 90 + dike_dip) / 3,  # weighted average of dike dip and 90
            "rake": self.rng.normal(90, 5),  # Bias to lateral folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        return fold


class DikeGroup(DikePlaneWord):  # Validated
    """
    A correlated grouping of vertical dikes with varying thicknesses and lengths.

    Random Variables (RVs)
    ----------------------
    - `num_dikes` : At least 2 dikes are generated, with a 30% chance of adding more dikes to the group.

    - `origin` : Random point in an ellipsoid within model bounds, starting location of first dike.
    Subsequent dikes are placed orthogonally to the strike direction of the previous dike with variations.

    - `strike` : Random strike direction of the first dike, with subsequent dikes adjusted slightly.

    - `width` : Width of the first dike, with subsequent dikes adjusted by a random scaling proportional to prev

    - `dip` : Normal distribution with a bias of 90 degrees and std of 8 degrees.

    - `spacing_avg` : Log-normal distributed average spacing between dikes, with variations in spacing introduced.

    Effects:
    --------
    This word generates a group of vertical dikes with varying thicknesses and lengths. The dikes are placed
    sequentially parallel with small variations in strike, width, and dip.
    """

    def build_history(self):
        # Geometric distribution with probability of failure 0.7 (success rate 0.3), min of 2 dikes
        num_dikes = self.rng.geometric(p=0.7) + 1

        # Starting parameters, to be sequentially modified
        origin = rv.random_point_in_ellipsoid(MAX_BOUNDS)
        strike = self.rng.uniform(0, 360)
        width = rv.beta_min_max(1.5, 4, 40, 350)
        dip = self.rng.normal(90, 8)
        value = self.rng.choice(INTRUSION_VALS)
        spacing_avg = self.rng.lognormal(*rv.log_normal_params(mean=1200, std_dev=400))

        # Setup slight wave transform
        fold_in = self.get_fold(strike, dip)
        fold_out = copy.deepcopy(fold_in)
        fold_out.amplitude *= -1

        self.add_process(fold_in)

        for _ in range(num_dikes):
            length = rv.beta_min_max(2, 2, 600, 16000)
            dike_params = {
                "strike": strike,
                "dip": dip,
                "origin": geo.BacktrackedPoint(origin),  # Backtracked point ensures dike is in view
                "width": width,
                "value": value,
                "thickness_func": self.get_organic_thickness_func(length, wobble_factor=0.5),
            }
            dike = geo.DikePlane(**dike_params)
            self.add_process(dike)

            # Modify parameters for next dike
            origin = self.get_next_origin(origin, strike, spacing_avg)
            strike += self.rng.normal(0, 4)
            width *= np.maximum(self.rng.normal(1, 0.1), 0.8)
            dip += self.rng.normal(0, 1)

        # Add final fold out
        self.add_process(fold_out)

    def get_next_origin(self, origin, strike, spacing_avg):
        # Move orthogonally to strike direction (strike measured from y-axis)
        orth_vec = np.array([np.cos(np.radians(strike)), -np.sin(np.radians(strike)), 0])
        orth_distance = spacing_avg * self.rng.uniform(0.9, 1.3)
        # Shift a bit parallel to strike as well
        par_vec = np.array([-orth_vec[1], orth_vec[0], 0])
        par_distance = spacing_avg * self.rng.uniform(-0.2, 0.2)

        new_origin = origin + orth_distance * orth_vec + par_distance * par_vec
        return new_origin

    def get_fold(self, dike_strike, dike_dip):
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(4, 8), smoothness=np.random.normal(1.2, 0.2)
        )
        period = self.rng.uniform(0.5, 2) * X_RANGE
        amp = self.rng.uniform(30, 60)
        fold_params = {
            "strike": dike_strike + 90,
            "dip": (2 * 90 + dike_dip) / 3,  # weighted average of dike dip and 90
            "rake": self.rng.normal(90, 5),  # Bias to lateral folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        return fold


""" Intrusion Events: Sills"""


class SillWord(GeoWord):
    """A sill construction mechanism using horizontal dike planes

    Random Variables (RVs)
    ----------------------
    - `width` : the vertical thickness of the sill

    - `x_length, y_length` : the two principal axes of the hyper ellipsoid sill

    - `origin` : the center point of the hyper ellipsoid sill

    Effects:
    --------
    Generates a puddle-like warped sill based off of a warped hyper ellipsoid. The z-axis is normalized to 1,
    while the x and y lengths are specified relative to it. The width parameter of the Dike controls the scaling
    of the sill thickness from the baseline of 1. Radial warping is controlled using a fourier series
    of harmonic waves to give a rippled edge, while the thickness is also warped with the same technique.
    The overall horizontal intrusion will be place about the specified origin point.
    """

    class EllipsoidShapingFunction:
        def __init__(
            self,
            x_length,
            y_length,
            wobble_factor,
            x_var,
            y_var,
            radial_var,
            amp,
            exp_x,
            exp_y,
            exp_z,
        ):
            self.x_length = x_length
            self.y_length = y_length
            self.wobble_factor = wobble_factor
            self.x_var = x_var
            self.y_var = y_var
            self.radial_var = radial_var
            self.amp = amp
            self.exp_x = exp_x
            self.exp_y = exp_y
            self.exp_z = exp_z

        def __call__(self, x, y):
            # 3d ellipse with thickness axis of 1 and hyper ellipse tapering in x and y
            theta = np.arctan2(y, x)
            ellipse_factor = (
                (1 + 0.6 * self.radial_var(theta / (2 * np.pi)))
                - np.abs(x / self.x_length) ** self.exp_x
                - np.abs(y / self.y_length) ** self.exp_y
            )
            ellipse_factor = (np.maximum(ellipse_factor, 0)) ** (1 / self.exp_z)

            # The thickness modifier combines 2d fourier with tapering at ends
            return (
                (1 + self.amp * self.x_var(x / X_RANGE))
                * (1 + self.amp * self.y_var(y / X_RANGE))
                * ellipse_factor
            )

    def get_ellipsoid_shaping_function(self, x_length, y_length, wobble_factor=1.0):
        """Organic Sheet Maker

        variance is introduced through 3 different fourier waves. x_var and y_var add ripple to the sheet thickness,
        while the radial_var adds a ripple around the edges of the sheet in the distance that it extends.

        The exponents (p) control the sharpness of the hyper ellipse:
        $$ 1 = (\frac{|z|}{d_z})^{p_z} + (\frac{|y|}{d_y})^{p_y} + (\frac{|x|}{d_x})^{p_x}$$

        This function has the z dimension normalized to 1, and the x and y dimensions are
        normalized to the x_length and y_length. A sharp taper off at the edges is controlled with a
        higher exp_z value.
        """
        # Make a fourier based modifier for both x and y
        fourier = rv.FourierWaveGenerator(num_harmonics=4, smoothness=1)
        x_var = fourier.generate()
        y_var = fourier.generate()
        radial_var = fourier.generate()
        amp = np.random.uniform(0.1, 0.2) * wobble_factor  # unevenness of the dike thickness
        exp_x = np.random.uniform(1.5, 4)  # Hyper ellipse exponent controls tapering sharpness
        exp_y = np.random.uniform(1.5, 4)  # Hyper ellipse exponent controls tapering sharpness
        exp_z = np.random.uniform(3, 6)  # Hyper ellipse exponent controls tapering sharpness

        # Return an instance of the EllipsoidShapingFunction class
        return self.EllipsoidShapingFunction(
            x_length,
            y_length,
            wobble_factor,
            x_var,
            y_var,
            radial_var,
            amp,
            exp_x,
            exp_y,
            exp_z,
        )

    def build_history(self):
        width = rv.beta_min_max(2, 4, 50, 250)
        x_length = rv.beta_min_max(2, 2, 600, 5000)
        y_length = self.rng.normal(1, 0.2) * x_length
        origin = geo.BacktrackedPoint(rv.random_point_in_ellipsoid(MAX_BOUNDS))

        dike_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(0, 0.1),  # Bias towards horizontal sills
            "origin": origin,
            "width": width,
            "value": self.rng.choice(INTRUSION_VALS),
            "thickness_func": self.get_ellipsoid_shaping_function(x_length, y_length, wobble_factor=0.0),
        }
        dike = geo.DikePlane(**dike_params)

        self.add_process(dike)


class SillSystem(SillWord):
    """A sill construction mechanism using horizontal dike planes

    Random Variables (RVs)
    ----------------------
    - `origins` : List of origin points for each sill in the system

    - `sediment` : Sedimentation object to be used as a substrate for the sills

    Effects:
    --------
    This is a complex word that generates a sedimentary substrate followed by sills intruding in between layers
    and finally a systemp of pipes conneting the sills from their centers. It requires a deferred parameter that
    can fetch the placement of the sill origins after the sedimentation has been computed. See SillWord for more
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.rock_val = None
        self.origins = []
        self.sediment = None

    def build_history(self):
        # Build a sediment substrate to sill into
        self.build_sedimentation()
        self.add_process(self.sediment)
        # Choose a random rock value and place sills into boundary layers, then link them
        self.rock_val = self.rng.choice(INTRUSION_VALS)
        indices = self.get_layer_indices()
        origins = self.build_sills(indices)
        self.link_sills(origins)

    def build_sedimentation(self) -> geo.Sedimentation:
        """Sediment building portion of word"""
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(Z_RANGE / 18, Z_RANGE / 6),
            thickness_variance=self.rng.uniform(0.1, 0.4),
            dirichlet_alpha=self.rng.uniform(0.8, 1.2),  # Low alpha for high repeatability
            anticorrelation_factor=0.05,  # Low factor gives low repeatability
        )

        depth = Z_RANGE / 2
        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)
        sed = geo.Sedimentation(vals, thicks)
        self.sediment = sed

    def build_sills(self, indices):
        origins = []
        for i, boundary in enumerate(indices):
            x_loc = self.rng.uniform(BOUNDS_X[0], BOUNDS_X[1]) * 0.75
            y_loc = self.rng.uniform(BOUNDS_Y[0], BOUNDS_Y[1]) * 0.75
            sill_origin = geo.SedimentConditionedOrigin(x=x_loc, y=y_loc, boundary_index=boundary)
            origins.append(sill_origin)

            width = rv.beta_min_max(2, 4, 40, 250)
            x_length = rv.beta_min_max(2, 2, 600, 4000)
            y_length = self.rng.lognormal(*rv.log_normal_params(mean=1, std_dev=0.2)) * x_length

            dike_params = {
                "strike": self.rng.uniform(0, 360),
                "dip": self.rng.normal(0, 1),  # Bias towards horizontal sills
                "origin": sill_origin,  # WARNING: This requires sediment to compute first
                "width": width,
                "value": self.rock_val,
                "thickness_func": self.get_ellipsoid_shaping_function(
                    x_length, y_length, wobble_factor=0.0
                ),
            }
            sill = geo.DikePlane(**dike_params)

            self.add_process([sill])

        return origins

    def link_sills(self, origins):
        """Link the sills with some connector between their origins"""
        # Pair up the sill origins with dike cols, add a final endpoint to the mantle
        end_points = origins[1:]
        x_loc = self.rng.uniform(BOUNDS_X[0], BOUNDS_X[1]) * 0.75
        y_loc = self.rng.uniform(BOUNDS_Y[0], BOUNDS_Y[1]) * 0.75
        final_origin = (x_loc, y_loc, -10000)
        end_points.append(final_origin)
        channels = zip(origins, end_points)

        for i, (start, end) in enumerate(channels):
            col_params = {
                "origin": start,
                "end_point": end,
                "diam": self.rng.uniform(300, 800),
                "minor_axis_scale": self.rng.uniform(0.15, 0.4),
                "rotation": self.rng.uniform(0, 360),
                "value": self.rock_val,
                "clip": True,
            }
            col = geo.DikeColumn(**col_params)
            self.add_process(col)

    def get_layer_indices(self):
        """A helper function to select layer boundaries for sill placement"""
        # Create a range from 1 to len(layers) - 2 (inclusive)
        valid_indices = np.arange(1, len(self.sediment.thickness_list) - 1)

        # Randomly select n unique layers to place sills in
        n = self.rng.integers(1, 5)
        n = np.clip(n, 1, len(valid_indices))

        selected_indices = self.rng.choice(valid_indices, size=n, replace=False)

        # Sort in ascending order, then reverse it to get descending order
        selected_indices = np.sort(selected_indices)[::-1]

        return selected_indices


""" Intrusion Events: Plutons"""


class _HemiPushedWord(GeoWord):
    """A generic pushed hemisphere word providing an organic warping function for the hemisphere"""

    class HemiFunction:
        def __init__(self, wobble_factor, x_var, y_var, exp_x, exp_y, exp_z, radial_var):
            self.wobble_factor = wobble_factor
            self.x_var = x_var
            self.y_var = y_var
            self.exp_x = exp_x
            self.exp_y = exp_y
            self.exp_z = exp_z
            self.radial_var = radial_var

        def __call__(self, x, y):
            x = (1 + self.wobble_factor * self.x_var(x)) * x
            y = (1 + self.wobble_factor * self.y_var(y)) * y
            r = 1 + 0.1 * self.radial_var(np.arctan2(y, x) / (2 * np.pi))
            inner = r**2 - np.abs(x) ** self.exp_x - np.abs(y) ** self.exp_y
            z_surf = np.maximum(0, inner) ** (1 / self.exp_z)
            return z_surf

    def __init__(self, seed=None):
        super().__init__(seed)
        self.rock_val = None
        self.origin = None

    def get_hemi_function(self, wobble_factor=0.1):
        """Organic looking warping of hemispheres

        The hemisphere coordinates xyz have been normalized to a simple hemisphere case where
        1=z^2+x^2+y^2 will give a default hemisphere, the purpose is to distort the default z surface
        """

        fourier = rv.FourierWaveGenerator(num_harmonics=4, smoothness=1)
        x_var = fourier.generate()
        y_var = fourier.generate()
        exp_x = np.random.uniform(1.5, 4)
        exp_y = np.random.uniform(1.5, 4)
        exp_z = np.random.uniform(1.5, 3)
        radial_var = fourier.generate()

        # Return an instance of the HemiFunction class
        return self.HemiFunction(wobble_factor, x_var, y_var, exp_x, exp_y, exp_z, radial_var)

    def build_history(self):
        raise NotImplementedError()


class Laccolith(_HemiPushedWord):  # Validated
    """
    A large laccolith intrusion with a pushed hemisphere shape above

    Random Variables (RVs)
    ----------
    - `diam` : Diameter of the laccolith, primary axis of hemisphere

    - `height` : Height of the laccolith dome

    - `rotation` : Random rotation of the laccolith's primary axis

    - `min_axis_scale` : Random scaling of the minor vs major axis of the hemisphere

    Effects:
    --------
    Builds a flattened lens shaped intrusion with an upward pushed direction and a
    feeder column dike below
    """

    def build_history(self):
        self.rock_val = self.rng.choice(INTRUSION_VALS)

        diam = self.rng.uniform(1000, 15000)
        height = self.rng.uniform(250, 1000)
        self.origin = self.get_origin(height)  # places the self.origin parameter
        rotation = self.rng.uniform(0, 360)
        min_axis_scale = rv.beta_min_max(2, 2, 0.5, 1.5)

        hemi_params = {
            "origin": self.origin,
            "diam": diam,
            "height": height,
            "minor_axis_scale": min_axis_scale,
            "rotation": rotation,
            "value": self.rock_val,
            "upper": True,
            "clip": True,
            "z_function": self.get_hemi_function(wobble_factor=0.1),
        }
        hemi = geo.DikeHemispherePushed(**hemi_params)

        # Add a column underneath as a feeder dike
        col_params = {
            "origin": self.origin,
            "diam": diam / 5 * self.rng.lognormal(*rv.log_normal_params(mean=1, std_dev=0.2)),
            "minor_axis_scale": min_axis_scale / 2 * self.rng.normal(1, 0.1),
            "rotation": rotation + self.rng.normal(0, 10),
            "value": self.rock_val,
            "clip": True,
        }
        col = geo.DikeColumn(**col_params)

        self.fold = self.get_fold()
        fold_out = copy.deepcopy(self.fold)
        fold_out.amplitude *= -1

        self.add_process([self.fold, hemi, col, fold_out])

    def get_origin(self, height):
        # Use a deferred parameter to measure a height off the floor of the mesh in the past frame
        x_loc = self.rng.uniform(BOUNDS_X[0], BOUNDS_X[1])
        y_loc = self.rng.uniform(BOUNDS_Y[0], BOUNDS_Y[1])
        z_loc = BOUNDS_Z[0] + self.rng.uniform(
            -height, Z_RANGE / 2
        )  # Sample from just out of view to mid-model
        origin = geo.BacktrackedPoint((x_loc, y_loc, z_loc))
        return origin

    def get_fold(self):
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(4, 8), smoothness=np.random.normal(1.2, 0.2)
        )
        period = self.rng.uniform(0.5, 2) * X_RANGE
        amp = self.rng.uniform(100, 300)
        fold_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 5),  # weighted average of dike dip and 90
            "rake": self.rng.normal(90, 5),  # Bias to lateral folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        return fold


class Lopolith(_HemiPushedWord):
    """
    Lopoliths are larger than laccoliths and have a pushed hemisphere downward

    Random Variables (RVs)
    ----------
    - `diam` : Diameter of the lopolith, primary axis of hemisphere

    - `height` : Height of the lopolith dome

    - `rotation` : Random rotation of the lopolith's primary axis

    - `min_axis_scale` : Random scaling of the minor vs major axis of the hemisphere

    Effects:
    --------
    Builds a flattened lens shaped intrusion with an downward pushed direction and a
    feeder column dike below
    """

    def build_history(self):
        self.rock_val = self.rng.choice(INTRUSION_VALS)

        diam = self.rng.uniform(5000, 30000)
        height = 0.3 * self.rng.uniform(1e-2, 1e-1) + 0.7 * self.rng.uniform(200, 800)
        self.origin = self.get_origin(height)  # places the self.origin parameter
        rotation = self.rng.uniform(0, 360)
        min_axis_scale = rv.beta_min_max(2, 2, 0.5, 1.5)

        hemi_params = {
            "origin": self.origin,
            "diam": diam,
            "height": height,
            "minor_axis_scale": min_axis_scale,
            "rotation": rotation,
            "value": self.rock_val,
            "upper": False,
            "clip": True,
            "z_function": self.get_hemi_function(wobble_factor=0.1),
        }
        hemi = geo.DikeHemispherePushed(**hemi_params)

        # Add a plug underneath as a feeder dike
        col_params = {
            "origin": self.origin,
            "diam": diam / 10 * self.rng.lognormal(*rv.log_normal_params(mean=1, std_dev=0.2)),
            "minor_axis_scale": min_axis_scale / 2 * self.rng.normal(1, 0.1),
            "rotation": rotation + self.rng.normal(0, 10),
            "value": self.rock_val,
            "clip": True,
        }
        col = geo.DikeColumn(**col_params)

        self.fold = self.get_fold()
        fold_out = copy.deepcopy(self.fold)
        fold_out.amplitude *= -1

        self.add_process([self.fold, hemi, col, fold_out])

    def get_origin(self, height):
        # Use a deferred parameter to measure a height off the floor of the mesh in the past frame
        x_loc = self.rng.uniform(BOUNDS_X[0], BOUNDS_X[1])
        y_loc = self.rng.uniform(BOUNDS_Y[0], BOUNDS_Y[1])
        z_loc = BOUNDS_Z[0] + self.rng.uniform(
            -height, Z_RANGE / 2
        )  # Sample from just out of view to mid-model
        origin = geo.BacktrackedPoint((x_loc, y_loc, z_loc))
        return origin

    def get_fold(self):
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(4, 8), smoothness=np.random.normal(1.2, 0.2)
        )
        period = self.rng.uniform(0.5, 2) * X_RANGE
        amp = self.rng.uniform(100, 300)
        fold_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 10),  # weighted average of dike dip and 90
            "rake": self.rng.normal(90, 3),  # Bias to lateral folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        return fold


# TODO: The push factor from PlugPush GeoProcess is not working at this scale, for now using regular plug
# Some sort of deformation factor would be useful in the future
class VolcanicPlug(GeoWord):
    """
    A volcanic plug that is resistant to erosion

    Random Variables (RVs)
    ----------
    - `diam` : Diameter of the plug, not a well defined quantity from how the plug
    is constructed and requires tuning

    - `rotation` : Random rotation of the plug about z-axis

    - `min_axis_scale` : Random scaling of the minor vs major axis of the plug

    - `shape` : Shape of the plug determines steepness of the sides

    Effects:
    --------
    Builds a resistant (no-clip) volcanic plug that is a parabaloid style intrusion from below

    """

    def build_history(self):
        rock_val = self.rng.choice(INTRUSION_VALS)

        diam = self.rng.lognormal(*rv.log_normal_params(mean=1, std_dev=0.2)) * 200
        origin = geo.BacktrackedPoint(
            rv.random_point_in_ellipsoid((BOUNDS_X, BOUNDS_Y, (BOUNDS_Z[0], BOUNDS_Z[1] * 0.8)))
        )
        rotation = self.rng.uniform(0, 360)
        min_axis_scale = rv.beta_min_max(2, 2, 0.2, 1.8)

        plug_params = {
            "origin": origin,
            "diam": diam,
            "minor_axis_scale": min_axis_scale,
            "rotation": rotation,
            "shape": 5,
            # "push" : 1000,
            "value": rock_val,
            "clip": False,
        }
        hemi = geo.DikePlug(**plug_params)

        self.add_process(hemi)


""" Intrusion Events: Blobs/Ore Bodies"""


class BlobWord(GeoWord):
    """
    A single blob intrusion event

    Random Variables (RVs)
    ----------
    - `n_balls` : Number of balls in the blob, determines the complexity of the blob

    - `scale_factor` : Scaling factor for the blob, determines the size of the blob,
    adjusting the radius relative to the number of balls

    - `blg` : Ball list generator object to generate a local coordinate system markov chain of points

    Effects:
    --------
    Generates a blob intrusion with a correlated set of balls, mimicking ore body deposits
    The blobs can form as multiple trees originating from a single point. More information
    can be found in the MetaBall class and associated objects.

    This operation is expensive if not using pruning, for this reason a fast_filter is used
    to prune out potential deposit points that are too far away from the origin.

    """

    def __init__(self, seed=None, origin=None, value=None):
        super().__init__(seed)
        self.rock_val = value
        self.origin = origin
        self.blg = None

    def build_history(self):
        # Pick a rock value from the blob types
        if self.rock_val is None:
            self.rock_val = self.rng.choice(BLOB_VALS)
        if self.origin is None:
            self.origin = geo.BacktrackedPoint(tuple(rv.random_point_in_box(MAX_BOUNDS)))

        # Ball list generator is a markov chain maker for point distribution
        n_balls = int(rv.beta_min_max(2, 2, 8, 60))
        scale_factor = 0.5 ** ((n_balls - 30) / 40)  # Heuristically tuned to adjust radius
        blg = geo.BallListGenerator(
            step_range=(10, 25),
            rad_range=(
                10 * scale_factor,
                20 * scale_factor,
            ),  # Correlate the radius with the number of balls
            goo_range=(0.5, 0.7),
        )

        # Blobs look better with multi-branched approach
        itr = self.rng.integers(2, 4)
        for _ in range(itr):

            ball_list = blg.generate(n_balls=n_balls, origin=(0, 0, 0), variance=0.8)
            blob = geo.MetaBall(
                balls=ball_list,
                threshold=1,
                value=self.rock_val,
                reference_origin=self.origin,
                clip=True,
                fast_filter=True,
            )
            self.add_process(blob)


class BlobCluster(GeoWord):
    """
    This word generates a correlated set of blob clusters mimicking ore body deposits.

    Random Variables (RVs)
    ----------
    - `n_blobs` : number of clusters to generate total

    Effects:
    --------
    Adds several blob intrusions that are in close proximity using a markov chain to step
    between blob origins
    """

    def build_history(self):
        n_blobs = self.rng.integers(2, 7)
        blob_val = self.rng.choice(BLOB_VALS)
        starting_origin = rv.random_point_in_ellipsoid(MAX_BOUNDS)

        # generate a set of origins for the blobs using a markov stepping algorithm
        origin_list = [starting_origin]
        for _ in range(n_blobs - 1):
            origin_list.append(self.get_next_origin(starting_origin))

        # Process each sampled point into a blob
        for origin in origin_list:
            origin = geo.BacktrackedPoint(origin)
            blob_word = BlobWord(seed=self.seed, origin=origin, value=blob_val)
            sub_hist = blob_word.generate()
            self.add_process(sub_hist)

    def get_next_origin(self, origin):
        """
        Determine the next origin point for the next blob using a correlated random walk.

        Random Variables (RVs)
        ----------
        origin : geo.BacktrackedPoint
            The current origin point from which the next origin is determined.

        Returns
        -------
        geo.BacktrackedPoint
            The next origin point, adjusted to stay within the model bounds.
        """
        MAX_ATTEMPTS = 5
        # Define the step size distribution (mean step size can be adjusted)
        step_min = 100
        step_max = 1000

        for _ in range(MAX_ATTEMPTS):
            step_size = rv.beta_min_max(1.3, 2, step_min, step_max)
            # Random direction on the unit sphere
            direction = self.rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            # Calculate the new origin
            new_origin = np.array(origin) + step_size * direction

            # Check if the new origin is within the model bounds
            x, y, z = new_origin
            if (
                BOUNDS_X[0] < x < BOUNDS_X[1]
                and BOUNDS_Y[0] < y < BOUNDS_Y[1]
                and BOUNDS_Z[0] < z < BOUNDS_Z[1]
            ):
                break
            else:
                continue

        # Either max iterations reached or an in bounds was found, return the new origin
        return tuple(new_origin)


""" Tilting Events"""


# TODO This is a very useful weathering word that should be added to deposits, erosions or end of models
class TiltCutFill(GeoWord):
    """
    A combined cluster of tilt, erosion, fill, with weathering

    Random Variables (RVs)
    ----------
    - `dip` : The total permanent dip of the old strata

    -  `edge_displacement` : The amount of displacement at the edge of the model caused by the dip

    - `erosion_depth` : The amount of erosion to cut off the top of the model

    - `fill_depth` : The amount of fill to add to the lower areas of the model

    Effects:
    --------
    An initial tilt of strata is constructed, followed by an estimate of the depth needed
    to create an erosion-fill scheme to prevent unnatural looking models. Erosion-fill is
    done through a 2d fourier surface transform to introduce variety using unconformity with
    a sedimentation fill.
    """

    def build_history(self):
        strike = self.rng.uniform(0, 360)
        dip = self.rng.normal(0, 10)
        # The origin does not change the transform for an auto-height normed model,
        # most efficient is center of mesh
        origin = geo.BacktrackedPoint((0, 0, 0))

        tilt = geo.Tilt(strike=strike, dip=dip, origin=origin)
        self.add_process(tilt)
        self.fill_tilt(dip)

    def fill_tilt(self, dip):
        """Large scale tilt creates unnatural effect without a following erosion/sediment"""
        # Change of basis to get 2d rippled erosion surface
        fold_strike = self.rng.uniform(0, 360)
        fold_in, fold_out = self.get_fold_pair(fold_strike)
        orth_fold_in, orth_fold_out = self.get_fold_pair(fold_strike + 90)

        # estimate the depth based on the height differnce caused by the dip.
        edge_displacement = X_RANGE / 2 * np.abs(np.sin(np.radians(dip)))

        # Erosion step to cut top of tilted model
        erosion_depth = edge_displacement * self.rng.normal(1, 0.05)
        erosion = geo.UnconformityDepth(depth=erosion_depth)

        # Fill in lower areas with sediment
        fill_depth = erosion_depth * self.rng.normal(1, 0.05)
        fill = self.get_fill(fill_depth)
        self.add_process([fold_in, orth_fold_in, erosion, fill, orth_fold_out, fold_out])

    def get_fill(self, depth):
        """Generalized sediment fill"""

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(100, Z_RANGE / 4),
            thickness_variance=self.rng.uniform(0.1, 0.5),
            dirichlet_alpha=self.rng.uniform(0.6, 1.6),  # Low alpha for high repeatability
            anticorrelation_factor=0.05,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)
        return geo.Sedimentation(vals, thicks)

    def get_fold_pair(self, strike):
        dip = self.rng.normal(90, 10)

        wave_generator = FourierWaveGenerator(num_harmonics=np.random.randint(3, 5), smoothness=1)
        period = self.rng.uniform(0.5, 2) * X_RANGE
        min_amp = period * 0.003
        max_amp = period * 0.02
        amp = rv.beta_min_max(a=1.5, b=1.5, min_val=min_amp, max_val=max_amp)
        fold_params = {
            "strike": strike,
            "dip": dip,  # average of dike dip and 90
            "rake": self.rng.normal(0, 10),  # Bias to vertical folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold_in = geo.Fold(**fold_params)
        fold_out = copy.deepcopy(fold_in)
        springback_factor = rv.beta_min_max(a=1.5, b=1.5, min_val=0.98, max_val=1)
        fold_out.amplitude *= -1 * springback_factor
        return fold_in, fold_out


""" Folding Events"""


class MicroNoise(GeoWord):  # Validated
    """A thin layer of noise to simulate small-scale sedimentary features."""

    def build_history(self):
        # Fourier wave generator creates randomized waves with a bias towards lower frequencies
        wave_generator = FourierWaveGenerator(num_harmonics=self.rng.integers(4, 6), smoothness=0.8)

        for _ in range(self.rng.integers(3, 7)):
            period = self.rng.uniform(100, 1000)
            amplitude = period * self.rng.uniform(0.002, 0.005) + 5
            fold_params = {
                "origin": geo.BacktrackedPoint(rv.random_point_in_ellipsoid(MAX_BOUNDS)),
                "strike": self.rng.uniform(0, 360),
                "dip": self.rng.uniform(0, 360),
                "rake": self.rng.uniform(0, 360),
                "period": period,
                "amplitude": amplitude,
                "periodic_func": wave_generator.generate(),
            }
            fold = geo.Fold(**fold_params)
            self.add_process(fold)


class SimpleFold(GeoWord):  # validated
    """A simple fold structure with random orientation and amplitude."""

    def build_history(self):
        period = rv.beta_min_max(a=1.4, b=2.1, min_val=100, max_val=14000)
        min_amp = period * 0.04
        max_amp = period * (0.18 - 0.07 * period / 10000)  # Linear interp, 1000 -> .17 , 11000 -> .10
        amp = self.rng.beta(a=2.1, b=1.4) * (max_amp - min_amp) + min_amp

        fold_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 45),  # Preference towards vertical fold planes
            "rake": self.rng.uniform(0, 360),
            "period": period,
            "amplitude": amp,
            "periodic_func": None,
            "phase": self.rng.uniform(0, 2 * np.pi),
            "origin": geo.BacktrackedPoint(rv.random_point_in_ellipsoid(MAX_BOUNDS)),
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)


class ShapedFold(GeoWord):  # Validated
    """A fold structure with a random shape factor."""

    def build_history(self):
        true_period = rv.beta_min_max(a=2.1, b=2.1, min_val=1000, max_val=11000)
        shape = self.rng.normal(0.3, 0.1)
        harmonic_weight = shape / np.sqrt(1 + shape**2)
        period = (1 - (2 / 3) * harmonic_weight) * true_period  # Effective period due to shape
        min_amp = period * 0.04
        max_amp = period * (0.18 - 0.07 * period / 10000)  # Linear interp, 1000 -> .17 , 11000 -> .10
        amp = self.rng.beta(a=1.2, b=2.1) * (max_amp - min_amp) + min_amp

        fold_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 45),  # Preference towards vertical fold planes
            "rake": self.rng.uniform(0, 360),
            "period": true_period,
            "amplitude": amp,
            "shape": shape,
            "periodic_func": None,
            "phase": self.rng.uniform(0, 2 * np.pi),
            "origin": geo.BacktrackedPoint(rv.random_point_in_ellipsoid(MAX_BOUNDS)),
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)


class FourierFold(GeoWord):  # Validated
    """A fold structure with a random number of harmonics."""

    def build_history(self):

        period = self.rng.uniform(3000, 15000)
        mu_smoothness = 1.4 - 0.1 * period / 10000
        # Fourier wave generator creates randomized waves with a bias towards lower frequencies
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(3, 6),
            smoothness=np.random.normal(mu_smoothness, 0.2),
        )
        min_amp = period * 0.04
        max_amp = period * (0.18 - 0.09 * period / 10000)  # Linear interp
        amp = self.rng.beta(a=1.4, b=2.1) * (max_amp - min_amp) + min_amp
        fold_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 25),
            "rake": self.rng.uniform(0, 360),
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)


""" Fault Events"""


def _typical_fault_amplitude():
    """Get a typical fault amplitude based on a beta distribution."""
    min_amp = 60
    max_amp = 1000
    return rv.beta_min_max(1.8, 5.5, min_amp, max_amp)


class FaultRandom(GeoWord):
    """
    A somewhat unconstrained fault event.

    Random Variables (RVs)
    ----------
    - `rake`: Rake of the fault, angle of slip direction relative to strike.

    - `amplitude`: The vertical displacement along the fault.

    Effects:
    --------
    Generates a random fault structure with random values for strike, dip, rake, and amplitude,
    with little to no constraints. Typically results in a faulting event without a preferred
    orientation or structure.
    """

    def build_history(self):
        strike = self.rng.uniform(0, 360)
        dip = self.rng.uniform(0, 90)
        rake = self.rng.uniform(0, 360)
        amplitude = _typical_fault_amplitude()
        origin = rv.random_point_in_box(MAX_BOUNDS)

        fault_params = {
            "strike": strike,
            "dip": dip,
            "rake": rake,
            "amplitude": amplitude,
            "origin": geo.BacktrackedPoint(tuple(origin)),
        }

        fault = geo.Fault(**fault_params)
        self.add_process(fault)


class FaultNormal(GeoWord):
    """
    Normal faulting with a bias towards vertical dip.

    Random Variables (RVs)
    ----------
    - `dip`: Dip of the fault is controlled to create a fault that is vertical without overhang

    - `rake`: Rake of the fault, the angle of slip direction along the fault plane.

    - `amplitude`: The vertical displacement along the fault.

    Effects:
    --------
    Generates a normal faulting event where the hanging wall moves down relative to the footwall.
    A folding change of coordinates is included to add non linear deformation to the event
    """

    def build_history(self):
        strike = self.rng.uniform(0, 360)
        dip = 90 - np.abs(self.rng.normal(0, 20))
        rake = self.rng.normal(90, 5)

        fault_params = {
            "strike": strike,
            "dip": dip,
            "rake": rake,
            "amplitude": _typical_fault_amplitude(),
            "origin": geo.BacktrackedPoint(tuple(rv.random_point_in_box(MAX_BOUNDS))),
        }

        fold_amp = self.rng.uniform(0, 200)
        fold_in = self.get_fold(strike, fold_amp)
        fold_out = copy.deepcopy(fold_in)
        fold_out.amplitude *= -1

        fault = geo.Fault(**fault_params)
        self.add_process([fold_in, fault, fold_out])

    def get_fold(self, strike, amp):
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(3, 5), smoothness=np.random.normal(1.2, 0.2)
        )
        period = self.rng.uniform(0.5, 2) * X_RANGE
        fold_params = {
            "strike": strike + 90,
            "dip": self.rng.normal(90, 0),
            "rake": self.rng.normal(90, 3),  # Bias to lateral folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        return fold


class FaultReverse(GeoWord):
    """Reverse Faulting with a bias towards vertical dip

    Random Variables (RVs)
    ----------
    - `dip`: Dip of the fault, typically steep for reverse faults.

    - `rake`: Rake of the fault, the angle of slip direction along the fault plane.

    - `amplitude`: The vertical displacement along the fault.

    Effects:
    --------
    Generates a reverse faulting event where the hanging wall moves up relative to the footwall.
    This event is associated with compressional tectonic settings. Includes a folding process
    to represent the deformation caused by faulting.
    """

    def build_history(self):
        strike = self.rng.uniform(0, 360)
        dip = 90 + np.abs(self.rng.normal(0, 15))
        rake = self.rng.normal(90, 5)

        fault_params = {
            "strike": strike,
            "dip": dip,
            "rake": rake,
            "amplitude": _typical_fault_amplitude(),
            "origin": geo.BacktrackedPoint(tuple(rv.random_point_in_box(MAX_BOUNDS))),
        }

        fold_amp = self.rng.uniform(0, 200)
        fold_in = self.get_fold(strike, fold_amp)
        fold_out = copy.deepcopy(fold_in)
        fold_out.amplitude *= -1

        fault = geo.Fault(**fault_params)
        self.add_process([fold_in, fault, fold_out])

    def get_fold(self, strike, amp):
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(3, 5), smoothness=np.random.normal(1.2, 0.2)
        )
        period = self.rng.uniform(0.5, 2) * X_RANGE
        fold_params = {
            "strike": strike + 90,
            "dip": self.rng.normal(90, 0),
            "rake": self.rng.normal(90, 3),  # Bias to lateral folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        return fold


class FaultHorstGraben(GeoWord):
    """
    Horst and Graben faulting

    Random Variables (RVs)
    ----------
    - `dip`: Dip of the fault on either side of the Horst and Graben structure, veritcal bias

    - `rake`: Rake of the fault, the angle of slip direction along the fault plane.

    - `amplitude`: The vertical displacement along each fault.

    - `distance`: Distance between faults based on the amplitude and dip offset.

    Effects:
    --------
    Generates a Horst and Graben structure, characterized by two parallel normal faults
    with opposing dips. One fault moves downward (graben) while the other moves upward
    (horst), creating a distinct topographical depression and uplift.
    """

    def build_history(self):
        strike = self.rng.uniform(0, 360)
        dip_offset = np.abs(self.rng.normal(0, 10))
        rake = self.rng.normal(90, 3)
        amplitude = _typical_fault_amplitude()
        origin = rv.random_point_in_box(MAX_BOUNDS)

        # Throw distance between faults, correlated with amplitude
        distance = rv.beta_min_max(2, 2, 2, 8) * amplitude * (1 + dip_offset / 5)

        fault1_params = {
            "strike": strike,
            "dip": 90 + dip_offset,
            "rake": rake,
            "amplitude": -amplitude,
            "origin": geo.BacktrackedPoint(tuple(origin)),
        }

        origin = self.get_next_origin(origin, strike, distance)

        fault2_params = {
            "strike": strike + self.rng.normal(0, 3),
            "dip": 90 - dip_offset + self.rng.normal(0, 2),
            "rake": rake + self.rng.normal(0, 3),
            "amplitude": amplitude * self.rng.uniform(0.9, 1.1),
            "origin": geo.BacktrackedPoint(tuple(origin)),
        }

        fault1 = geo.Fault(**fault1_params)
        fault2 = geo.Fault(**fault2_params)

        # Handle folded warping
        fold_amp = self.rng.uniform(0, 200)
        fold_in = self.get_fold(strike, fold_amp)
        fold_out1 = copy.deepcopy(fold_in)
        fold_out2 = copy.deepcopy(fold_in)
        rev_amplitude = -np.abs(self.rng.normal(1, 0.1))
        residual_amplitude = 1 - rev_amplitude
        fold_out1.amplitude *= rev_amplitude
        fold_out2.amplitude *= residual_amplitude

        self.add_process([fold_in, fault1, fold_out1, fault2, fold_out2])

    def get_next_origin(self, origin, strike, orth_distance):
        origin = np.array(origin)  # Cast to array
        # Move orthogonally to strike direction (strike measured from y-axis)
        orth_vec = np.array([np.cos(np.radians(strike)), -np.sin(np.radians(strike)), 0])

        # Shift a bit parallel to strike as well
        par_vec = np.array([-orth_vec[1], orth_vec[0], 0])
        par_distance = orth_distance * self.rng.uniform(-0.2, 0.2)

        new_origin = origin + orth_distance * orth_vec + par_distance * par_vec
        return new_origin

    def get_fold(self, strike, amp):
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(3, 5), smoothness=np.random.normal(1.2, 0.2)
        )
        period = self.rng.uniform(0.5, 2) * X_RANGE
        fold_params = {
            "strike": strike + 90,
            "dip": self.rng.normal(90, 0),
            "rake": self.rng.normal(90, 3),  # Bias to lateral folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        return fold


class FaultStrikeSlip(GeoWord):
    """
    A classic strike-slip faulting event

    Random Variables (RVs)
    ----------
    - `dip`: Dip of the fault, biased to vertical

    - `rake`: Rake of the fault, biased to slip along the fault

    - `amplitude`: The horizontal displacement along the fault.

    Effects:
    --------
    Generates a strike-slip fault where lateral movement occurs along the fault plane.
    The displacement is typically horizontal and occurs along a near-vertical fault plane.
    """

    def build_history(self):
        strike = self.rng.uniform(0, 360)
        dip_offset = np.abs(self.rng.normal(0, 20))
        rake = self.rng.normal(0, 5)
        direction = self.rng.choice([-1, 1])
        # Similar to lognormal distribution in shape,
        # most values within 30-250m, but outliers up to 2km
        amplitude = rv.beta_min_max(1.4, 10, 45, 2000) * direction
        origin = rv.random_point_in_box(MAX_BOUNDS)

        fault_params = {
            "strike": strike,
            "dip": 90 + dip_offset,
            "rake": rake,
            "amplitude": amplitude,
            "origin": geo.BacktrackedPoint(tuple(origin)),
        }

        fault = geo.Fault(**fault_params)

        self.add_process(fault)


class FaultSequence(GeoWord):  # Inherits from FaultRandom for base faulting behavior
    """
    A correlated sequence of faults with varying characteristics.

    Random Variables (RVs)
    ----------------------
    - `num_faults`: At least 2 faults are generated, with a geometric probability distribution of adding more faults.

    - `origin`: Random point within model bounds, starting location of the first fault.
    Subsequent faults are placed orthogonally or parallel to the previous fault with variations.

    - `strike`: Strike direction of the first fault, with subsequent faults adjusted slightly.

    - `dip`: Normal distribution of dips for each fault, modified by a small random factor.

    - `rake`: The angle of slip direction for each fault.

    - `amplitude`: Displacement amplitude along the fault. Amplitude varies slightly between faults.

    - `spacing_avg`: Average spacing between faults, controlled by a log-normal distribution.

    Effects:
    --------
    This class generates a sequence of faults where each fault is related but slightly varies in its strike, dip, and
    amplitude. The faults are placed sequentially with a small random offset in origin and spacing.
    """

    def build_history(self):
        # Geometric distribution with a 0.7 probability of stopping, ensuring at least 2 faults
        num_faults = self.rng.geometric(p=0.7) + 1

        # Starting parameters for the first fault
        origin = rv.random_point_in_box(MAX_BOUNDS)
        strike = self.rng.uniform(0, 360)
        dip = self.rng.normal(90, 20)
        rake = self.rng.normal(90, 30)
        amplitude = _typical_fault_amplitude() / (num_faults - 1)
        spacing_avg = self.rng.lognormal(*rv.log_normal_params(mean=600, std_dev=900))

        # Setup slight wave transform for deformation along the fault sequence
        fold_in = self.get_fold(strike, dip)
        fold_out = copy.deepcopy(fold_in)
        fold_out.amplitude *= -1

        self.add_process(fold_in)

        for _ in range(num_faults):
            fault_params = {
                "strike": strike,
                "dip": dip,
                "rake": rake,
                "amplitude": amplitude,
                "origin": geo.BacktrackedPoint(origin),  # Ensures faults remain visible in the model
            }

            fault = geo.Fault(**fault_params)
            self.add_process(fault)

            # Update parameters for the next fault in the sequence
            origin = self.get_next_origin(origin, strike, spacing_avg)
            strike += self.rng.normal(0, 5)
            dip += self.rng.normal(0, 2)
            amplitude *= self.rng.uniform(0.8, 1.2)

        # Add final deformation fold
        self.add_process(fold_out)

    def get_next_origin(self, origin, strike, spacing_avg):
        # Move orthogonally to the strike direction (strike measured from y-axis)
        orth_vec = np.array([np.cos(np.radians(strike)), -np.sin(np.radians(strike)), 0])
        orth_distance = spacing_avg * self.rng.uniform(0.9, 1.2)
        # Shift a bit parallel to strike as well
        par_vec = np.array([-orth_vec[1], orth_vec[0], 0])
        par_distance = spacing_avg * self.rng.uniform(-0.3, 0.3)

        new_origin = origin + orth_distance * orth_vec + par_distance * par_vec
        return new_origin

    def get_fold(self, fault_strike, fault_dip):
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(3, 6), smoothness=np.random.normal(1.2, 0.2)
        )
        period = self.rng.uniform(0.5, 2) * X_RANGE
        amp = self.rng.uniform(10, 250)
        fold_params = {
            "strike": fault_strike + 90,
            "dip": (2 * 90 + fault_dip) / 3,  # weighted average of dip and 90
            "rake": self.rng.normal(90, 5),  # Bias to lateral folds
            "period": period,
            "amplitude": amp,
            "periodic_func": wave_generator.generate(),
        }
        fold = geo.Fold(**fold_params)
        return fold
