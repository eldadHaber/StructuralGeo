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
        Items are added in chronological order from earliest to latest event.

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
            raise ValueError(
                f"Expected GeoWord, GeoProcess, or list of these, got {type(item)}"
            )


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
    """A large sediment accumulation to simulate deep sedimentary layers."""

    def build_history(self):
        # Choose a large depth that runs beyond the model's height extension bars
        depth = (Z_RANGE) * (
            3 * geo.GeoModel.HEIGHT_BAR_EXT_FACTOR
        )  # Pseudo-infinite using a large depth
        sediment_base = (
            -depth
        )  # The sediment base is located so that it builds back up to z=0

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
    """A large sediment accumulation to simulate deep sedimentary layers with dependency on previous layers."""

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

    Fills entire model with sediment, then tilts the model, then truncates sediment to the bottom of the model.
    """

    def build_history(self):
        # Choose a large depth that runs beyond the model's height extension bars below and above
        depth = (Z_RANGE) * (6 * geo.GeoModel.HEIGHT_BAR_EXT_FACTOR)
        sediment_base = (
            -0.5 * depth
        )  # In this case we overbuild up and down to allow for tilting and eroding after

        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(200, Z_RANGE / 4),
            thickness_variance=self.rng.uniform(0.1, 0.6),
            dirichlet_alpha=self.rng.uniform(0.6, 1.2),
            anticorrelation_factor=0.05,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)
        # Sediment needs to go above and below model bounds for tilt, build from bottom up through extended model
        sed = geo.Sedimentation(vals, thicks, base=sediment_base)

        tilt = geo.Tilt(
            strike=self.rng.uniform(0, 360),
            dip=self.rng.normal(0, 10),
            origin=geo.BacktrackedPoint((0, 0, 0)),
        )
        # Shave off all excess sediment above the tilt operation
        unc = geo.UnconformityBase(1000)

        # Bedrock ensures full coverage underneath sediment in all cases
        self.add_process(geo.Bedrock(base=sediment_base, value=BED_ROCK_VAL))
        self.add_process([sed, tilt, unc])


""" Sediment Acumulation Events"""


class FineRepeatSediment(GeoWord):  # Validated
    """A series of thin sediment layers with repeating values."""

    def build_history(self):
        # Get a log-normal depth for the sediment block
        depth = self.rng.lognormal(
            # This helper calculates the required parameters to hit target mean and std for distro
            *rv.log_normal_params(
                mean=MEAN_SEDIMENTATION_DEPTH, std_dev=MEAN_SEDIMENTATION_DEPTH / 3
            )
        )

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(100, Z_RANGE / 10),
            thickness_variance=self.rng.uniform(0.1, 0.3),
            dirichlet_alpha=self.rng.uniform(
                0.6, 2.0
            ),  # Low alpha for high repeatability
            anticorrelation_factor=0.05,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)

        self.add_process(geo.Sedimentation(vals, thicks))


class CoarseRepeatSediment(GeoWord):  # Validated
    """A series of thick sediment layers with repeating values."""

    def build_history(self):
        # Get a log-normal depth for the sediment block
        depth = self.rng.lognormal(
            # This helper calculates the required parameters to hit target mean and std for distro
            *rv.log_normal_params(
                mean=MEAN_SEDIMENTATION_DEPTH, std_dev=MEAN_SEDIMENTATION_DEPTH / 3
            )
        )

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(Z_RANGE / 12, Z_RANGE / 6),
            thickness_variance=self.rng.uniform(0.1, 0.2),
            dirichlet_alpha=self.rng.uniform(
                0.8, 1.2
            ),  # Low alpha for high repeatability
            anticorrelation_factor=0.05,  # Low factor gives low repeatability
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)

        self.add_process(geo.Sedimentation(vals, thicks))


class SingleRandSediment(GeoWord):
    """A single sediment layer with a random value and thickness."""

    def build_history(self):
        val = self.rng.integers(1, 5)
        sediment = geo.Sedimentation(
            [val],
            [self.rng.normal(MEAN_SEDIMENTATION_DEPTH, MEAN_SEDIMENTATION_DEPTH / 3)],
        )
        self.add_process(sediment)


""" Erosion events"""


class _BaseErosionWord(GeoWord):  # Validated
    """Reusable generic class for calculating total depth of erosion events."""

    MEAN_DEPTH = Z_RANGE / 8

    def __init__(self, seed=None):
        super().__init__(seed)

    def calculate_depth(self):
        factor = self.rng.lognormal(
            *rv.log_normal_params(mean=1, std_dev=0.5)
        )  # Generally between .25 and 2.5
        factor = np.clip(factor, 0.25, 3)
        return factor * self.MEAN_DEPTH


class FlatUnconformity(_BaseErosionWord):  # Validated
    """Flat unconformity down to a random depth"""

    def build_history(self):
        total_depth = self.calculate_depth()
        unconformity = geo.UnconformityDepth(total_depth)
        self.add_process(unconformity)


class TiltedUnconformity(_BaseErosionWord):  # Validated
    """Slightly tilted unconformity down to a random depth"""

    def build_history(self):
        num_tilts = self.rng.integers(1, 4)
        total_depth = self.calculate_depth()
        depths = np.random.dirichlet(alpha=[1] * num_tilts) * total_depth

        for depth in depths:
            strike = self.rng.uniform(0, 360)
            tilt_angle = self.rng.normal(0, 3)
            x, y, z = rv.random_point_in_ellipsoid(MAX_BOUNDS)
            origin = geo.BacktrackedPoint((x, y, 0))
            tilt_in = geo.Tilt(strike=strike, dip=tilt_angle, origin=origin)
            tilt_out = geo.Tilt(strike=strike, dip=-tilt_angle, origin=origin)

            unconformity = geo.UnconformityDepth(depth)

            self.add_process([tilt_in, unconformity, tilt_out])


class WaveUnconformity(_BaseErosionWord):
    """Change of coordinates/basis with two orthogonal folds to create wavy unconformity"""

    def build_history(self):
        total_depth = self.calculate_depth() * 0.8
        orientation = self.rng.uniform(0, 360)  # Principal orientation of the waves
        fold_in1, fold_out1 = self.get_fold_pair(strike=orientation, dip=90)
        fold_in2, fold_out2 = self.get_fold_pair(strike=orientation + 90, dip=90)
        unconformity = geo.UnconformityDepth(total_depth)
        self.add_process([fold_in1, fold_in2, unconformity, fold_out2, fold_out1])

    def get_fold_pair(self, strike, dip):
        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(3, 5), smoothness=1
        )
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
    """Base GeoWord for forming dikes with organic thickness variations."""
    
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
        amp = (
            self.rng.uniform(0.1, 0.2) * wobble_factor
        )  # unevenness of the dike thickness
        expo = self.rng.uniform(
            4, 10
        )  # Hyper ellipse exponent controls tapering sharpness

        return self.OrganicDikeThicknessFunc(length, expo, amp, x_var, y_var)

    def build_history(self):
        width = rv.beta_min_max(2, 4, 50, 500)
        length = rv.beta_min_max(2, 2, 300, 16000)
        origin = rv.random_point_in_ellipsoid(MAX_BOUNDS)
        back_origin = geo.BacktrackedPoint(
            origin
        )  # Use a backtracked point to ensure origin is in view
        dike_params = {
            "strike": self.rng.uniform(0, 360),
            "dip": self.rng.normal(90, 10),  # Bias towards vertical dikes
            "origin": back_origin,
            "width": width,
            "value": self.rng.choice(INTRUSION_VALS),
            "thickness_func": self.get_organic_thickness_func(
                length, wobble_factor=self.rng.uniform(0.5, 1.5)
            ),
        }
        dike = geo.DikePlane(**dike_params)

        self.add_process(dike)


class SingleDikeWarped(DikePlaneWord):  # Validated
    """A single dike with organic thickness and a more serpentine length"""

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
            "value": self.rng.choice(INTRUSION_VALS),
            "thickness_func": self.get_organic_thickness_func(
                length, wobble_factor=1.5
            ),
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
    """A correlated grouping of vertical dikes with varying thicknesses and lengths."""

    def build_history(self):
        num_dikes = 2
        while self.rng.uniform() < 0.3:
            num_dikes += 1

        # Starting parameters, to be sequrntially modified
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
                "strike": strike + self.rng.normal(0, 2),
                "dip": dip,
                "origin": geo.BacktrackedPoint(
                    origin
                ),  # Backtracked point ensures dike is in view
                "width": width,
                "value": value,
                "thickness_func": self.get_organic_thickness_func(
                    length, wobble_factor=0.5
                ),
            }
            dike = geo.DikePlane(**dike_params)
            self.add_process(dike)

            # Modify parameters for next dike
            origin = self.get_next_origin(origin, strike, spacing_avg)
            strike += self.rng.normal(0, 2)
            width *= np.maximum(self.rng.normal(1, 0.1), 0.8)
            dip += self.rng.normal(0, 1)

        # Add final fold out
        self.add_process(fold_out)

    def get_next_origin(self, origin, strike, spacing_avg):
        # Move orthogonally to strike direction (strike measured from y-axis)
        orth_vec = np.array(
            [np.cos(np.radians(strike)), -np.sin(np.radians(strike)), 0]
        )
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
    """A sill construction mechanism using horizontal dike planes"""
    
    class EllipsoidShapingFunction:
        def __init__(self, x_length, y_length, wobble_factor, x_var, y_var, radial_var, amp, exp_x, exp_y, exp_z):
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
        return self.EllipsoidShapingFunction(x_length, y_length, wobble_factor, x_var, y_var, radial_var, amp, exp_x, exp_y, exp_z)

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
            "value": self.rng.choice(DIKE_VALS),
            "thickness_func": self.get_ellipsoid_shaping_function(
                x_length, y_length, wobble_factor=0.0
            ),
        }
        dike = geo.DikePlane(**dike_params)

        self.add_process(dike)



class SillSystem(SillWord):
    """A sill construction mechanism using horizontal dike planes"""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.rock_val = None
        self.origins = []
        self.sediment = None

    def build_history(self):
        # Build a sediment substrate to sill into
        self.build_sedimentation()
        self.add_process(self.sediment)
        self.rock_val = self.rng.choice(DIKE_VALS)
        indices = self.get_layer_indices()
        origins = self.build_sills(indices)
        self.link_sills(origins)

    def build_sills(self, indices):
        origins = []
        for i, boundary in enumerate(indices):
            x_loc = self.rng.uniform(BOUNDS_X[0], BOUNDS_X[1]) * 0.75
            y_loc = self.rng.uniform(BOUNDS_Y[0], BOUNDS_Y[1]) * 0.75
            sill_origin = geo.SedimentConditionedOrigin(
                x=x_loc, y=y_loc, boundary_index=boundary
            )
            origins.append(sill_origin)

            width = rv.beta_min_max(2, 4, 40, 250)
            x_length = rv.beta_min_max(2, 2, 600, 4000)
            y_length = (
                self.rng.lognormal(*rv.log_normal_params(mean=1, std_dev=0.2))
                * x_length
            )

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

    def build_sedimentation(self) -> geo.Sedimentation:
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(Z_RANGE / 18, Z_RANGE / 6),
            thickness_variance=self.rng.uniform(0.1, 0.4),
            dirichlet_alpha=self.rng.uniform(
                0.8, 1.2
            ),  # Low alpha for high repeatability
            anticorrelation_factor=0.05,  # Low factor gives low repeatability
        )

        depth = Z_RANGE / 2
        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)
        sed = geo.Sedimentation(vals, thicks)
        self.sediment = sed

    def get_layer_indices(self):
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
    """A generic pushed hemisphere word providing an organic warping function"""
    
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
    """A large laccolith intrusion with a pushed hemisphere shape above"""

    def build_history(self):
        self.rock_val = self.rng.choice(INTRUSION_VALS)

        diam = self.rng.uniform(1000, 15000)
        height = 0.5 * self.rng.uniform(5e-2, 2e-1) + 0.5 * self.rng.uniform(500, 2000)
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

        # Add a plug underneath as a feeder dike
        col_params = {
            "origin": self.origin,
            "diam": diam
            / 5
            * self.rng.lognormal(*rv.log_normal_params(mean=1, std_dev=0.2)),
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
    """Lopoliths are larger than laccoliths and have a pushed hemisphere downward"""

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
            "diam": diam
            / 10
            * self.rng.lognormal(*rv.log_normal_params(mean=1, std_dev=0.2)),
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


# TODO: The push factor is not working at this scale, for now using regular plug
class VolcanicPlug(GeoWord):
    """A volcanic plug that is resistant to erosion"""

    def build_history(self):
        rock_val = self.rng.choice(INTRUSION_VALS)

        diam = self.rng.lognormal(*rv.log_normal_params(mean=1, std_dev=0.2)) * 200
        origin = geo.BacktrackedPoint(
            rv.random_point_in_ellipsoid(
                (BOUNDS_X, BOUNDS_Y, (BOUNDS_Z[0], BOUNDS_Z[1] * 0.8))
            )
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
    """A single blob intrusion event

    Parameters
    ----------
    seed : int
        The seed value for the random number generator.
    origin : tuple (optional)
        The origin point of the blob intrusion. Randomly generated if not provided.
    value : float (optional)
        The rock value of the blob intrusion. Randomly selected if not provided.
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
            self.origin = geo.BacktrackedPoint(
                tuple(rv.random_point_in_box(MAX_BOUNDS))
            )

        # Ball list generator is a markov chain maker for point distribution
        n_balls = int(rv.beta_min_max(2, 2, 8, 60))
        scale_factor = 0.5 ** (
            (n_balls - 30) / 40
        )  # Heuristically tuned to adjust radius
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
    """A clustering of blob intrusions with correlated centers and rock values

    This word generates a correlated set of blob clusters mimicking ore body deposits.
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

        Parameters
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


class TiltCutFill(GeoWord):
    """
    A combined cluster of tilt, erosion, fill, with weathering

    An initial tilt of strata is constructed, followed by an estimate of the depth needed
    to create an erosion-fill scheme to prevent unnatural looking models. Erosion-fill is
    done through a 2d fourier surface transform to introduce variety
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
        self.add_process(
            [fold_in, orth_fold_in, erosion, fill, orth_fold_out, fold_out]
        )

    def get_fill(self, depth):
        """Generalized sediment fill"""

        # Get a markov process for selecting next layer type, gaussian differencing for thickness
        markov_helper = MarkovSedimentHelper(
            categories=SEDIMENT_VALS,
            rng=self.rng,
            thickness_bounds=(100, Z_RANGE / 4),
            thickness_variance=self.rng.uniform(0.1, 0.5),
            dirichlet_alpha=self.rng.uniform(
                0.6, 1.6
            ),  # Low alpha for high repeatability
            anticorrelation_factor=0.05,
        )

        vals, thicks = markov_helper.generate_sediment_layers(total_depth=depth)
        return geo.Sedimentation(vals, thicks)

    def get_fold_pair(self, strike):
        dip = self.rng.normal(90, 10)

        wave_generator = FourierWaveGenerator(
            num_harmonics=np.random.randint(3, 5), smoothness=1
        )
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
        wave_generator = FourierWaveGenerator(
            num_harmonics=self.rng.integers(4, 6), smoothness=0.8
        )

        for _ in range(self.rng.integers(3, 7)):
            period = self.rng.uniform(100, 1000)
            amplitude = period * self.rng.uniform(0.002, 0.005) + 5
            fold_params = {
                "origin": geo.BacktrackedPoint(
                    rv.random_point_in_ellipsoid(MAX_BOUNDS)
                ),
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
        max_amp = period * (
            0.18 - 0.07 * period / 10000
        )  # Linear interp, 1000 -> .17 , 11000 -> .10
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
        period = (
            1 - (2 / 3) * harmonic_weight
        ) * true_period  # Effective period due to shape
        min_amp = period * 0.04
        max_amp = period * (
            0.18 - 0.07 * period / 10000
        )  # Linear interp, 1000 -> .17 , 11000 -> .10
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
    max_amp = 800
    return rv.beta_min_max(1.8, 5.5, min_amp, max_amp)


class FaultRandom(GeoWord):
    """A somewhat unconstrained fault event"""

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
    """Normal faulting"""

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
    """Normal faulting"""

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
    """Horst and Graben faulting"""

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
        orth_vec = np.array(
            [np.cos(np.radians(strike)), -np.sin(np.radians(strike)), 0]
        )

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
    """A classic strike-slip faulting event"""

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
