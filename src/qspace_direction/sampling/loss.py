import numpy as np


def covering_radius_upper_bound(num: int):
    """Upper bound of covering radius for a scheme with num points

    Args:
        num (int): Number of points

    Returns:
        float: Upper bound of covering radius
    """
    if isinstance(num, int) and num < 3:
        return np.pi / 2
    upperBoundEuc = np.sqrt(4 - (1 / np.sin(np.pi * num / (6 * (num - 2)))) ** 2)
    ub = np.arccos((2 - upperBoundEuc**2) / 2)
    return ub


def covering_radius(vects: np.ndarray, antipodal=True):
    """Covering radius of a point set

    Args:
        vects (np.ndarray): Given point set.
        antipodal (bool, optional): Whether or not to consider antipodal constraint. Defaults to True.

    Returns:
        float: Covering radius
    """
    innerProductAll = np.abs(vects @ vects.T) if antipodal else vects @ vects.T
    return np.arccos((np.clip(np.max(np.triu(innerProductAll, 1)), -1, 1)))


def packing_density_loss(vects: np.ndarray, start: np.ndarray):
    """Packing density increment of points after appending vects to start

    Args:
        vects (np.ndarray): Points to calculate packing density
        start (np.ndarray): Existing points

    Returns:
        float: Loss value
    """
    cons = np.concatenate([start, vects]) if len(start) > 0 else vects
    return np.sum(
        [
            (1 - np.cos(covering_radius(cons[:k]))) / 2 * (k + len(start))
            for k in range(1, len(vects) + 1)
        ]
    )