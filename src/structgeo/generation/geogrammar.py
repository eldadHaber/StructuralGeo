""" Sentence structures and generation processes for geo histories. """

import yaml
import numpy as np
import random
from typing import List, NamedTuple
import structgeo.model as geo
import structgeo.probability as rv
from .geowords import *
from . import geowords as geowords_module

       
class SentenceSelector:
    def __init__(self, grammar_data):
        self.grammar_data = grammar_data
        self.categories = list(grammar_data.keys())
        self.weights = np.array([grammar_data[category]['weight'] for category in self.categories])
        self.weights /= self.weights.sum()  # Normalize weights

    def select_category(self, n_samples: int = 1):
        """Select multiple grammar categories based on defined weights in a batch."""
        return np.random.choice(self.categories, size=n_samples, p=self.weights)

    def select_grammar(self, n_samples: int = 1):
        """Select multiple grammar from chosen categories in a batch."""
        categories = self.select_category(n_samples)
        structures = []
        for category in categories:
            selected_structure = random.choice(self.grammar_data[category]['structures'])
            structures.append(selected_structure)
        return structures

class WordSelector:
    def __init__(self, vocab_data):
        """Initialize with vocabulary data where keys map to class names and their weights."""
        self.processed_vocab = {}
        for grammar_key, entries in vocab_data.items():
            # Retrieve and store the class constructors along with their respective weights
            words = [getattr(geowords_module, name) for name, _ in entries]
            weights = np.array([weight for _, weight in entries])
            probabilities = weights / weights.sum()  # Normalize weights
            self.processed_vocab[grammar_key] = (words, probabilities)

    def select_word(self, grammar_key)->GeoWord:
        """Select and instantiate GeoWord based on defined weights and grammar key."""
        words, probabilities = self.processed_vocab[grammar_key] # Get word options and probabilities
        selected_word = np.random.choice(words, p=probabilities) # Choose a word based on probabilities
        return selected_word() # Instantiate the selected GeoWord

    def fill_grammar_with_words(self, structure: List[str])->List[GeoWord]:
        """Select and instantiate words for each category in the given structure."""
        return [self.select_word(grammar_key) for grammar_key in structure]

class GeologicalModelLoader:
    def __init__(self, config_path, model_bounds=((-3840,3840),(-3840,3840),(-1920,1920)), model_resolution=(256,256,128)):
        self.config_path = config_path
        self.data = self.load_yaml(config_path)        
        self._validate_yaml()
        self.bounds = model_bounds
        self.resolution = model_resolution
        self.sentence_selector = SentenceSelector(self.data['grammar'])
        self.word_selector = WordSelector(self.data['vocab'])

    @staticmethod
    def load_yaml(cfg_path):
        """Load and return the YAML configuration file."""
        try:
            with open(cfg_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Failed to load YAML file: {e}")

    def _validate_yaml(self):
        """Validate the structure and data of the YAML file, check classes exist in geowords."""
        if 'vocab' not in self.data or 'grammar' not in self.data:
            raise ValueError("YAML file must include 'vocab' and 'grammar' sections")
        
        # Validate vocab structure and class existence
        for category, entries in self.data['vocab'].items():
            for entry in entries:
                class_name, weight = entry
                if not isinstance(weight, (int, float)):
                    raise ValueError(f"Weight for {class_name} in '{category}' must be a number")
                try:
                    # Check if the class exists in the geo module
                    if not getattr(geowords_module, class_name, None):
                        print(f"Warning: The class {class_name} does not exist in the geo module")
                except AttributeError:
                    print(f"Warning: The class {class_name} does not exist in the geo module")

    def sample_sentences(self, n_samples: int = 1) -> List[List[GeoWord]]:
        """Sample multiple grammar structures and fill them with corresponding vocab.
        
        Parameters:
        ----------
        n_samples : int
            Number of samples to generate.
        
        Returns:
        -------
        List length n_samples containing lists of GeoWords ready to generate a geological history.
        """
        # Choose general sentence structures i.e. ['Sediment', 'Noise', 'Sediment']
        sentence_structures = self.sentence_selector.select_grammar(n_samples)
        # Fill each structure with words from the vocab, i.e. [CoarseSediment(), MicroNoise(), FineSediment()]
        filled_sentences = [self.word_selector.fill_grammar_with_words(structure) for structure in sentence_structures]
        return filled_sentences
    
    def sentence_to_history(self, sentence: List[GeoWord]) -> List[geo.GeoProcess]:
        """Generate a geological history from a sentence."""
        return [word.generate() for word in sentence]
    
    def history_to_model(self, hist: List[geo.GeoProcess]) -> geo.GeoModel:
        """Generate a model from a history and normalize the height."""
        
        # Generate a low resolution model to estimate the renormalization
        model = geo.GeoModel(bounds=self.bounds, resolution=(8,8,32))
        model.add_history(hist)
        model.compute_model()
        
        # First squash the model downwards until it is below the top of the bounds
        new_max = model.get_target_normalization(target_max = .1)
        model_max = model.bounds[2][1]
        max_iter = 10
        while True and max_iter > 0:
            observed_max = model.renormalize_height(new_max=new_max)
            max_iter -= 1
            if observed_max < model_max:
                break
        
        # Now renormalize the model to the correct height
        model.renormalize_height(auto=True)
        # Copy the vertical shift required to normalize the model    
        normed_hist = model.history.copy()
        
        # Generate the final model with the correct resolution and normalized height
        model = geo.GeoModel(bounds=self.bounds, resolution=self.resolution)
        model.add_history(normed_hist)  
        model.clear_data()
        model.compute_model()
        return model

    def generate_model_batch(self, n_samples: int = 1,) -> List[geo.GeoModel]:
        """Generate multiple geological models from sampled sentences."""
        filled_sentences = self.sample_sentences(n_samples)
        model_histories = [self.sentence_to_history(sentence) for sentence in filled_sentences]
        models = [self.history_to_model(hist) for hist in model_histories]
        return models      


def generate_sentence(vocabulary, grammar_structure):
    """ Generate a sentence from a grammar structure. """   
    sentence = [np.random.choice(vocabulary[word])() for word in grammar_structure]    
    return sentence

def generate_history(sentence: List[GeoWord]) -> List[geo.GeoProcess]:
    """ Generate a list of geological histories from a sentence. """
    h = [word.generate() for word in sentence]
    return h

def generate_normalized_model(hist: List[geo.GeoProcess], bounds=(-1920,1920), resolution=128) -> geo.GeoModel:
    """ Generate a model from a history and normalize the height. Use a low resolution to estimate renormalization."""
    
    # Generate a low resolution model to estimate the renormalization
    model = geo.GeoModel(bounds=bounds, resolution=(8,8,64))
    model.add_history(hist)
    model.compute_model()
    
    # First squash the model downwards until it is below the top of the bounds
    new_max = model.get_target_normalization(target_max = .1)
    model_max = model.bounds[2][1]
    max_iter = 10
    while True and max_iter > 0:
        observed_max = model.renormalize_height(new_max=new_max)
        max_iter -= 1
        if observed_max < model_max:
            break
        
    # Now renormalize the model to the correct height
    model.renormalize_height(auto=True)
    # Copy the vertical shift required to normalize the model    
    normed_hist = model.history.copy()
    
    # Generate the final model with the correct resolution and normalized height
    model = geo.GeoModel(bounds=bounds, resolution=resolution)
    model.add_history(normed_hist)  
    model.clear_data()
    model.compute_model()
    return model

    

