import math


def normalize_angle(theta: float) -> float:
    theta = theta % math.tau  # tau == 2*pi

    if theta >= math.pi:
        theta -= math.tau

    return theta
