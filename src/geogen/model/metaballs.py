"""Collection of classes for implementing metaball blobs in a geological model."""

from typing import List

import numpy as np

from geogen.model.geoprocess import Deposition


class Ball:
    """A single metaball object with a given origin, radius, and goo factor. Base building class for Blob"""

    def __init__(self, origin, radius, goo_factor=1.0):
        self.origin = np.array(origin).astype(np.float32)  # Keep precision low for performance
        self.radius = radius
        self.goo_factor = goo_factor

    def potential(self, points):  # Pass a reference offset to the ball
        # Calculate the distance from the points to the ball's origin
        distances = np.sum((points - self.origin) ** 2, axis=1)
        return (self.radius / distances) ** self.goo_factor


class BallListGenerator:
    """A generator class for metaballs. Generates a list of Ball objects with random parameters.

    Parameters:
    ----------------
    step_range (tuple): The range of uniform sampled step sizes for the metaballs.
    rad_range (tuple): The range of uniform sampled radii for the metaballs. Radius is distance to potential of 1.
    goo_range (tuple): The range of uniform sampled goo factors for the metaballs.
    """

    def __init__(self, step_range, rad_range, goo_range):
        self.step_range = step_range
        self.rad_range = rad_range
        self.goo_range = goo_range

    def generate(self, n_balls, origin, variance=1):
        """Generate a list of n Ball objects with random parameters starting at seeded origin.

        Parameters:
        ----------------
        n_balls (int): Number of Ball objects to generate.
        origin (tuple): Starting point for the first Ball.
        variance (float): Variance for the Gaussian distribution to add randomness to the direction.
        """
        balls = []
        # Set start point and unit direction
        current_point = np.array(origin, dtype=float)
        previous_direction = np.random.normal(size=3)
        previous_direction /= np.linalg.norm(previous_direction)

        for _ in range(n_balls):
            radius = np.random.uniform(*self.rad_range)
            goo_factor = np.random.uniform(*self.goo_range)
            balls.append(Ball(current_point, radius, goo_factor))

            # Generate the next point with a Gaussian bias towards the previous direction
            random_variation = np.random.normal(loc=0, scale=1, size=3)
            direction = previous_direction + variance * random_variation
            direction /= np.linalg.norm(direction)
            step = direction * np.random.uniform(*self.step_range)
            current_point += step

            # Update the previous direction
            previous_direction = direction

        return balls


class MetaBall(Deposition):
    """
    A Blob geological process that modifies points within a specified potential range.

    A fast filter option is provided for cases where all the balls are close to eachother.
    The filter will prune the mesh based on distance vs a factor of the average radius of the balls
    to speed up computation.

    Parameters
    ----------
    balls : List[Ball]
        A list of Ball objects defining the metaball.
    threshold : float
        The threshold potential below which points will be relabeled.
    value : int
        The value to assign to points below the threshold potential.
    reference_origin : tuple, optional
        The reference origin to normalize the points, by default (0, 0, 0).
    clip : bool, optional
        If True, ensures that data points that are NaN will not be overwritten, by default True.
    fast_filter : bool, optional
        If True, apply a mesh filter to prune the points and speed up computation, by default False.
    """

    def __init__(
        self,
        balls: List[Ball],
        threshold,
        value,
        reference_origin=(0, 0, 0),
        clip=True,
        fast_filter=False,
    ):
        self.balls = balls
        self.threshold = threshold
        self.value = value
        self.reference_origin = reference_origin
        self.clip = clip
        self.fast_filter = fast_filter  # A flag to use pruning on the mesh to speed up computation

    def __str__(self):
        return (
            f"Metaball: threshold {self.threshold:.1f}, value {self.value:.1f}, "
            f"with {len(self.balls)} balls."
        )

    def run(self, xyz, data):
        # Change of coordinates to the reference origin
        xyz_p = (xyz - self.reference_origin).astype(np.float32)  # Normalize points to the reference origin

        # Conditional filtering of the mesh, if enabled the mesh is crudely pruned to eliminate far away points
        if self.fast_filter:
            mask = self.mesh_filter(xyz_p)
        else:
            # No filtering-- mask is all true
            mask = np.ones(xyz_p.shape[0], dtype=bool)

        if self.clip:
            mask = mask & (~np.isnan(data))

        # apply mask to reduce the computation size
        data_filtered = data[mask]
        xyz_filtered = xyz_p[mask]

        # Compute the net potential for each point in mesh
        # Vectorizing this computation did not yield a significant speedup, left as for-loop
        potentials = np.zeros(xyz_filtered.shape[0])
        for ball in self.balls:
            potentials += ball.potential(xyz_filtered)

        # Filter which points will be included in the blob and clip if necessary
        pot_mask = potentials > self.threshold

        # Apply the transformation to the filtered data
        data_filtered[pot_mask] = self.value

        # Apply the changes to the original data
        data[mask] = data_filtered

        return xyz, data

    def mesh_filter(self, xyz_p):
        """Perform a crude filter on the mesh to reduce number of points to compute potential"""

        ball_origins = np.array([b.origin for b in self.balls])
        avg_origin = np.mean(ball_origins, axis=0).astype(np.float32)
        ball_radii = np.array([b.radius for b in self.balls])
        avg_radius = np.mean(ball_radii).astype(np.float32)
        n_balls = len(self.balls)

        # Calculate the distance from the points to the ball's origins
        dist = np.linalg.norm(xyz_p - avg_origin, axis=1)
        mask = dist < np.sum(ball_radii) * 0.5
        return mask
