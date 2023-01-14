from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class CarDimensions(ABC):
    @property
    @abstractmethod
    def bounding_box_size(self) -> Tuple[float, float]:
        """
        If the car can be approximated using a rectangle in 2D space (i.e. a bounding box),
        these are the bounds of such approximation (i.e. width and length)
        """
        pass

    @property
    @abstractmethod
    def radius(self) -> float:
        """
        If the car can be approximated using two circles in 2D space,
        this is the radius of each of the circles.
        """
        pass

    @property
    @abstractmethod
    def circle_centers(self) -> np.ndarray:
        """
        If the car can be approximated using circles in 2D space,
        these are the offsets of the circle centers from the center of the vehicle.
        """
        pass


class PriusDimensions(CarDimensions):

    def __init__(self, skip_back_circle_collision_checking=False):
        self.skip_back_circle_collision_checking = skip_back_circle_collision_checking

    @property
    def bounding_box_size(self) -> Tuple[float, float]:
        # this was determined empirically by simulating and placing obstacles around the prius
        return 2.05, 4.85

    @property
    def radius(self) -> float:
        width, _ = self.bounding_box_size
        return width / (2 ** .5)

    @property
    def circle_centers(self) -> np.ndarray:
        width, length = self.bounding_box_size
        offset = length / 2 - width / 2

        if self.skip_back_circle_collision_checking:
            return np.array([[offset, 0]])
        else:
            return np.array([
                [offset, 0],
                [-offset, 0],
            ])
