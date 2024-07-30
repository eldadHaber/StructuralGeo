""" A collection of curated geological words that combine random variables and geoprocesses into a single generator class. """

import numpy as np
from typing import List, Union
from structgeo.model import GeoProcess
import structgeo.model as geo
import structgeo.probability as rv

BOUNDS_X = (-3840, 3840)
BOUNDS_Y = (-3840, 3840)
BOUNDS_Z = (-1920, 1920)
BED_ROCK_VAL  = 0
SEDIMENT_VALS = [1,2,3,4,5]
INTRUSION_VALS = [6,7,8,9,10]
class GeoWord:
    """ 
    Base class providing structure for generating events 
    """
    
    def __init__(self):
        self.hist = []
    
    def build_history(self):
        """
        Builds a geological history for the word. Must be implemented by subclasses.
        """
        raise NotImplementedError()
            
    def generate(self):
        """ 
        Generate geological word assigning all rvs to geoprocesses
        """
        self.hist.clear()
        self.build_history()
        geoprocess = geo.CompoundProcess(self.hist.copy(), name=self.__class__.__name__)
        return geoprocess
    
    def add_process(self, item: Union[geo.GeoProcess, 'GeoWord', List[Union[geo.GeoProcess, 'GeoWord']]]):
        """
        Adds a GeoProcess, GeoWord or a list of processes/words to the GeoWord history. Used during the build_history method.

        Parameters
        ----------
        item : Union[GeoProcess, 'GeoWord', List[Union[GeoProcess, 'GeoWord']]]
            The item to add, which can be a GeoProcess, GeoWord, or a list containing either.

        Raises
        ------
        ValueError
            If the provided item is not a GeoWord, GeoProcess, or a list of these.
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
class NullWord(GeoWord):
    """A null geological event, generating a process that does nothing."""
    def build_history(self):
        self.add_process(geo.NullProcess())

""" Infinite foundation layer(s) for initial model"""        
class InfiniteBasement(GeoWord):
    """A foundational bedrock layer to simulate an infinite basement."""
    def build_history(self):
        # Generate a simple basement layer
        self.add_process(geo.Bedrock(base=BOUNDS_Z[0], value=0))
        
class InfiniteSediment(GeoWord):
    """A large sediment accumulation to simulate deep sedimentary layers."""
    def build_history(self):
        depth = BOUNDS_Z[1]*6 # Pseudo-infinite using a large depth
        vals =[]
        thicks = []
        while depth > 0:
            vals.append(np.random.choice(SEDIMENT_VALS))
            thicks.append(np.random.uniform(50,1000))
            depth -= thicks[-1]
            
        self.add_process(geo.Sedimentation(vals, thicks, base=depth))
            
""" Sediment blocks"""    
class FineRepeatSediment(GeoWord):
    """A series of thin sediment layers with repeating values."""
    def build_history(self):
        sb = geo.SedimentBuilder(start_value=1, total_thickness=np.random.normal(1000,200), min_layers=2, max_layers=5, std=0.5) 
        for _ in range(np.random.randint(1, 3)):
            sediment = geo.Sedimentation(*sb.build_layers())
            self.add_process(sediment)
    
class CoarseRepeatSediment(GeoWord):
    """A series of thick sediment layers with repeating values."""          
    def build_history(self):   
        sb = geo.SedimentBuilder(start_value=1, total_thickness=np.random.normal(1000,300), min_layers=2, max_layers=5, std=0.5)   
        for _ in range(np.random.randint(1, 2)):
            sediment = geo.Sedimentation(*sb.build_layers())
            self.add_process(sediment)
            
class SingleRandSediment(GeoWord):
    """A single sediment layer with a random value and thickness."""
    def build_history(self):
        val = np.random.randint(1, 5)
        sediment = geo.Sedimentation([val], [np.random.normal(1000, 100)])
        self.add_process(sediment)
             
class MicroNoise(GeoWord):
    """A thin layer of noise to simulate small-scale sedimentary features.""" 
    def build_history(self):
        wave_generator = rv.FourierWaveGenerator(num_harmonics=np.random.randint(4, 6), smoothness=0.8)
        for _ in range(np.random.randint(3, 7)):
            fold_params = {
                'strike': np.random.uniform(0, 360),
                'dip': np.random.uniform(0, 360),
                'rake': np.random.uniform(0, 360),
                'period': np.random.uniform(200, 4000),
                'amplitude': np.random.uniform(5, 10),
                'periodic_func': wave_generator.generate()
            }
            fold = geo.Fold(**fold_params)
            self.add_process(fold)
          
class SimpleFold(GeoWord):
    """A simple fold structure with random orientation and amplitude."""
    def build_history(self):
        period = np.random.uniform(100, 11000)
        proportion = np.random.normal(.05,.03)
        fold_params = {
            'strike': np.random.uniform(0, 360),
            'dip': np.random.uniform(0, 360),
            'rake': np.random.uniform(0, 360),
            'period': period,
            'amplitude': period*proportion,
            'periodic_func': None
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)

class ShapedFold(GeoWord):
    """ A fold structure with a random shape factor."""
    def build_history(self):
        period = np.random.uniform(100, 11000)
        proportion = np.random.normal(.05,.03)
        fold_params = {
            'strike': np.random.uniform(0, 360),
            'dip': np.random.uniform(0, 360),
            'rake': np.random.uniform(0, 360),
            'period': period,
            'amplitude': period*proportion,
            'shape':  np.random.normal(1,1),
            'periodic_func': None,
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)
        
class FourierFold(GeoWord):
    """ A fold structure with a random number of harmonics."""
    def build_history(self):
        wave_generator = rv.FourierWaveGenerator(num_harmonics=np.random.randint(4, 6), smoothness=np.random.normal(1,.2))
        period = np.random.uniform(500, 22000)
        proportion = np.random.normal(.05,.03)
        fold_params = {
            'strike': np.random.uniform(0, 360),
            'dip': np.random.uniform(0, 360),
            'rake': np.random.uniform(0, 360),
            'period': period,
            'amplitude': period*proportion,
            'periodic_func': wave_generator.generate()
        }
        fold = geo.Fold(**fold_params)
        self.add_process(fold)

""" Erosion events"""        
class FlatUnconformity(GeoWord):
    def build_history(self):
        unconformity = geo.UnconformityDepth(np.random.uniform(0, 2000))
        self.add_process(unconformity)
        
class TippedUnconformity(GeoWord):
    #TODO: Use tile process instead of rotate
    def build_history(self):
        tilt_angle = np.random.normal(0,20)
        theta = np.random.uniform(0, 360)
        x,y = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        rot_axis = np.array([x,y,0])
        
        rot_in = geo.Rotate(rot_axis, tilt_angle)        
        unconformity = geo.UnconformityDepth(np.random.uniform(0, 2000))
        rot_out = geo.Rotate(rot_axis, -tilt_angle)        
        self.add_process([rot_in, unconformity, rot_out])
        
class WaveUnconformity(GeoWord):
    def build_history(self):
        fold_params = {
            'strike': np.random.uniform(0, 360),
            'dip': np.random.uniform(0, 360),
            'rake': np.random.uniform(0, 360),
            'period': np.random.uniform(100, 11000),
            'amplitude': np.random.uniform(200, 5000),
            'periodic_func': None
        }
        fold_in = geo.Fold(**fold_params)
        fold_out = fold_in.copy()
        fold_out.amplitude *= -1*np.random.normal(.8,.1)
        unconformity = geo.UnconformityDepth(np.random.uniform(200, 2000))        
        self.add_process([fold_in, unconformity, fold_out])
            
        
        

        
    
    


        
        