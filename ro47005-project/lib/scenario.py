from dataclasses import dataclass
from typing import Tuple, List

from lib.obstacles import Obstacle


@dataclass
class Scenario:
    start: Tuple[float, float, float]
    obstacles: List[Obstacle]
    goal_area: Obstacle
    goal_point: Tuple[float, float, float]
