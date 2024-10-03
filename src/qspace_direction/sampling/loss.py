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


def f_multi_shell(vects, f, w, *args, **kargs):
    return w / len(vects) * sum(f(v, *args, **kargs) for v in vects) + (1 - w) * f(
        np.concatenate(vects), *args, **kargs
    )


def electrostatic_energy(
    vects: np.ndarray,
    order=2,
    antipodal=True,
):
    """Electrostatic energy of a given point set

    Parameters
    ----------
    vects : np.ndarray
        Given point set.
    order : int, optional
        order for calculating electrostatic energy, by default 2
    antipodal : bool, optional
        whether ot consider antipodal energy, by default True

    Returns
    -------
    _type_
        _description_
    """
    epsilon = 1e-9
    N = len(vects)
    energy = 0.0
    for i in range(N - 1):
        diffs = ((vects[i + 1 :] - vects[i]) ** order).sum(1)
        energy += (1.0 / (diffs + epsilon)).sum()
        if antipodal:
            sums = ((vects[i + 1 :] + vects[i]) ** order).sum(1)
            energy += (1.0 / (sums + epsilon)).sum()

    return energy


def norm_of_mean(vects: np.ndarray):
    """Calucalte norm of mean vector of a given point set

    Parameters
    ----------
    vects : np.ndarray

    Returns
    -------
    float
        norm of mean vector
    """
    mean = vects.mean(axis=0)
    return np.linalg.norm(mean)


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
