"""
PyTorch DataLoader for streaming GeoWord geological histories.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Two types of geological model generators provided
from geogen.generation import MarkovGeostoryGenerator

_DEFAULT_GENERATOR_CLASS = MarkovGeostoryGenerator


class GeoData3DStreamingDataset(Dataset):
    """
    A PyTorch Dataset wrapper for streaming geological data from a GeostoryGenerator object.

    Parameters
    ----------
    model_bounds : tuple
        Bounds of the model as ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
    model_resolution : tuple
        Resolution of the model as (x_res, y_res, z_res).
    generator_config : str
        A path to a configuration file for the particular generator class.
    dataset_size : int
        The total number of samples in one epoch.
    device : str
        Torch device where data is loaded.
    """

    def __init__(
        self,
        model_bounds=((-3840, 3840), (-3840, 3840), (-1920, 1920)),
        model_resolution=(256, 256, 128),
        generator_config=None,
        dataset_size=int(1e6),
        device="cpu",
        transform=None,  # Add the transform parameter
    ):
        self.model_generator = _DEFAULT_GENERATOR_CLASS(
            model_bounds=model_bounds,
            model_resolution=model_resolution,
            config=generator_config,
        )
        self.device = device
        self.size = dataset_size
        self.transform = transform  # Store the transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        model = self.model_generator.generate_model()
        model.fill_nans()
        data = model.get_data_grid()
        data_tensor = torch.from_numpy(data).float().unsqueeze(0).to(self.device)

        if self.transform:
            data_tensor = self.transform(data_tensor)  # Apply the transform

        return data_tensor


class OneHotTransform:
    def __init__(self, num_classes=15, min_val=-1):
        """
        Args:
            num_classes (int): Number of classes for one-hot encoding.
            min_val (int): Minimum value in the tensor.
        """
        self.num_classes = num_classes
        self.min_val = min_val

    def __call__(self, sample):
        """
        Args:
            sample (torch.Tensor): Input tensor of shape (C, X, Y, Z) with C=1.
        Returns:
            torch.Tensor: One-hot encoded tensor of shape (num_classes, X, Y, Z).
        """
        # Ensure the tensor has a single channel
        if sample.shape[0] != 1:
            raise ValueError(f"Expected channel dimension to be 1, but got {sample.shape[0]}")

        tensor = sample.squeeze(0)  # Shape: (X, Y, Z)
        tensor_shifted = tensor - self.min_val  # Shift to start from 0
        tensor_shifted = torch.clamp(tensor_shifted, 0, self.num_classes - 1)  # Clamp to range
        tensor_shifted = tensor_shifted.long()  # Convert to long for indexing

        # One-hot encode: (X, Y, Z) -> (X, Y, Z, num_classes)
        one_hot = F.one_hot(tensor_shifted, num_classes=self.num_classes)  # Shape: (X, Y, Z, num_classes)
        one_hot = one_hot.permute(3, 0, 1, 2).contiguous().float()

        return one_hot
