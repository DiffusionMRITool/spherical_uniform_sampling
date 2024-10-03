import numpy as np
from scipy import optimize as scopt

from .geem import compute_weights
from .geem import optimize as geem_optimize


def initialize(points_per_shell, antipodal) -> np.ndarray:
    """use geem algorithm to generate initialization

    Args:
        points_per_shell (List[int]): number of points on each shell
        antipodal (bool): whether or not considering antipodal constraint
        iprint (int, optional): whether to output. Defaults to 0.

    Returns:
        Array: array shaped (K, 3), where K is the total number of points
    """
    nb_shells = len(points_per_shell)
    K = np.sum(points_per_shell)

    shell_groups = [[i] for i in range(nb_shells)]
    shell_groups.append(range(nb_shells))
    alphas = np.ones(len(shell_groups))

    weights = compute_weights(nb_shells, points_per_shell, shell_groups, alphas)

    points = geem_optimize(
        nb_shells,
        points_per_shell,
        weights,
        antipodal=antipodal,
        max_iter=1000,
    )

    return points


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


def inequality_constraints_9b(vects, *args):
    """The covering radius of s-th shell is smaller then the angle between every point pairs in s-th shell. This corresponds to eq. (9b) in [1]

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        differce between $\theta_s$ and each angle in s-th shell

    Reference
    ---------
    1. Jian Cheng, Dinggang Shen, Pew-Thian Yap and Peter J. Basser, "Single- and Multiple-Shell Uniform Sampling Schemes for Diffusion MRI Using Spherical Codes," in IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 185-199
    """
    N = args[1]
    spherical_index = args[4]
    transform = args[8]
    points = vects[: 3 * N].reshape((N, 3))
    return np.array(
        [
            np.arccos(np.clip(transform(np.dot(points[i], points[j])), -1, 1))
            - vects[3 * N + s]
            for s, i, j in spherical_index
        ]
    )


def grad_inequality_constraints_9b(vects, *args):
    """This is the gradient of :func:`inequality_constraints_9b`.

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        Grdients
    """
    S = args[0]
    N = args[1]
    spherical_index = args[4]
    eps = args[7]
    grad_transform = args[9]
    points = vects[: 3 * N].reshape((N, 3))
    grad = np.zeros((len(spherical_index), N * 3 + S + 1))
    for index, (s, i, j) in enumerate(spherical_index):
        dot = np.dot(points[i], points[j])
        dot = dot / np.linalg.norm(points[i]) * np.linalg.norm(points[j])
        dot = np.clip(dot, -1, 1)
        d = np.sqrt(1 - dot * dot + eps)
        grad[index, 3 * i : 3 * (i + 1)] = points[j] * (-grad_transform(dot) / d)
        grad[index, 3 * j : 3 * (j + 1)] = points[i] * (-grad_transform(dot) / d)
        grad[index, 3 * N + s] = -1
    return grad


def inequality_constraints_9c(vects, *args):
    """The covering radius of combined shell is smaller then the angle between every point pairs. This corresponds to eq. (9c) in [1]

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        differce between $\theta_0$ and every angle between point pairs

    Reference
    ---------
    1. Jian Cheng, Dinggang Shen, Pew-Thian Yap and Peter J. Basser, "Single- and Multiple-Shell Uniform Sampling Schemes for Diffusion MRI Using Spherical Codes," in IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 185-199
    """
    N = args[1]
    cross_spherical_index = args[5]
    transform = args[8]
    points = vects[: 3 * N].reshape((N, 3))
    return np.array(
        [
            np.arccos(np.clip(transform(np.dot(points[i], points[j])), -1, 1))
            - vects[-1]
            for i, j in cross_spherical_index
        ]
    )


def grad_inequality_constraints_9c(vects, *args):
    """This is the gradient of :func:`inequality_constraints_9c`.

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        Grdients
    """
    S = args[0]
    N = args[1]
    cross_spherical_index = args[5]
    eps = args[7]
    grad_transform = args[9]
    points = vects[: 3 * N].reshape((N, 3))
    grad = np.zeros((len(cross_spherical_index), N * 3 + S + 1))
    for index, (i, j) in enumerate(cross_spherical_index):
        dot = np.dot(points[i], points[j])
        dot = dot / np.linalg.norm(points[i]) * np.linalg.norm(points[j])
        dot = np.clip(dot, -1, 1)
        d = np.sqrt(1 - dot * dot + eps)
        grad[index, 3 * i : 3 * (i + 1)] = points[j] * (-grad_transform(dot) / d)
        grad[index, 3 * j : 3 * (j + 1)] = points[i] * (-grad_transform(dot) / d)
        grad[index, -1] = -1
    return grad


def inequality_constraints_9d(vects, *args):
    """The diffence between target and initialization is no more then $\delta_0$. This corresponds to eq. (9d) in [1]

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        $\delta_0$ minus differce between target and initialization 

    Reference
    ---------
    1. Jian Cheng, Dinggang Shen, Pew-Thian Yap and Peter J. Basser, "Single- and Multiple-Shell Uniform Sampling Schemes for Diffusion MRI Using Spherical Codes," in IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 185-199
    """
    N = args[1]
    init = args[2]
    delta = args[3]
    transform = args[8]
    points = vects[: 3 * N].reshape((N, 3))
    diff = (points * init).sum(1)
    return delta - np.arccos(np.clip(transform(diff), -1, 1))


def grad_inequality_constraints_9d(vects, *args):
    """This is the gradient of :func:`inequality_constraints_9d`.

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        Grdients
    """
    S = args[0]
    N = args[1]
    init = args[2]
    eps = args[7]
    grad_transform = args[9]
    points = vects[: 3 * N].reshape((N, 3))
    grad = np.zeros((N, N * 3 + S + 1))
    for i in range(N):
        dot = np.dot(points[i], init[i])
        dot = dot / np.linalg.norm(points[i]) * np.linalg.norm(init[i])
        dot = np.clip(dot, -1, 1)
        grad[i, 3 * i : 3 * (i + 1)] = -init[i] * (
            -grad_transform(dot) / np.sqrt(1 - dot * dot + eps)
        )
    return grad


def inequality_constraints_9e(vects, *args):
    """Covering radius in each individual shell is greater then that of combined shell. This corresponds to eq. (9c) in [1]

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        differce between $\theta_s$ and $\theta_0$

    Reference
    ---------
    1. Jian Cheng, Dinggang Shen, Pew-Thian Yap and Peter J. Basser, "Single- and Multiple-Shell Uniform Sampling Schemes for Diffusion MRI Using Spherical Codes," in IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 185-199
    """
    N = args[1]
    s = args[0]
    return vects[3 * N : 3 * N + s] - vects[-1]


def grad_inequality_constraints_9e(vects, *args):
    """This is the gradient of :func:`inequality_constraints_9e`.

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        Grdients
    """
    S = args[0]
    N = args[1]
    grad = np.zeros((S, N * 3 + S + 1))
    for i in range(S):
        grad[i, 3 * N + i] = 1
        grad[i, -1] = -1
    return grad


def equality_constraints(vects, *args):
    """Each points is on the unit shpere. This corresponds to eq. (9f) in [1]

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        Difference between squared vector norms and 1.

    Reference
    ---------
    1. Jian Cheng, Dinggang Shen, Pew-Thian Yap and Peter J. Basser, "Single- and Multiple-Shell Uniform Sampling Schemes for Diffusion MRI Using Spherical Codes," in IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 185-199
    """
    N = args[1]
    points = vects[: 3 * N].reshape((N, 3))
    return (points**2).sum(1) - 1.0


def grad_equality_constraints(vects, *args):
    """This is the gradient of :func:`equality_constraints`.

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        Grdients
    """
    s = args[0]
    N = args[1]
    points = vects[: 3 * N].reshape((N, 3))
    grad = np.zeros((N, N * 3 + s + 1))
    for i in range(N):
        grad[i, 3 * i : 3 * (i + 1)] = 2 * points[i]
    return grad


def inequality_constraints(vects, *args):
    """This zip all inequality constraints together.

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        every inequality constraints
    """
    return np.concatenate(
        (
            inequality_constraints_9b(vects, *args),
            inequality_constraints_9c(vects, *args),
            inequality_constraints_9d(vects, *args),
            inequality_constraints_9e(vects, *args),
        )
    )


def grad_inequality_constraints(vects, *args):
    """This zip all grdient of inequality constraints together.

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        every grdient of inequality constraints
    """
    return np.concatenate(
        (
            grad_inequality_constraints_9b(vects, *args),
            grad_inequality_constraints_9c(vects, *args),
            grad_inequality_constraints_9d(vects, *args),
            grad_inequality_constraints_9e(vects, *args),
        )
    )


def cost(vects, *args):
    """Weighted average of covering radius in each individual shell and combined shell. This corresponds to eq. (9a) in [1]

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    float
        Value of loss function

    Reference
    ---------
    1. Jian Cheng, Dinggang Shen, Pew-Thian Yap and Peter J. Basser, "Single- and Multiple-Shell Uniform Sampling Schemes for Diffusion MRI Using Spherical Codes," in IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 185-199
    """
    s = args[0]
    N = args[1]
    w = args[6]
    return -(np.sum(vects[3 * N : 3 * N + s]) * w / s + (1 - w) * vects[-1])


def grad_cost(vects, *args):
    """This is the gradient of :func:`cost`.

    Parameters
    ----------
    vects : Array shaped (3 * N + S,)

    Returns
    -------
    Array
        Grdients
    """
    s = args[0]
    N = args[1]
    w = args[6]
    return np.concatenate(
        (np.zeros(3 * N), -np.ones(s) * w / s, -np.ones((1)) * (1 - w))
    )


def cnlo_optimize(
    points_per_shell,
    initialization=None,
    antipodal=True,
    delta=0.1,
    w=0.5,
    max_iter=1000,
    iprint=1,
):
    """Generate a set of uniform sampling schemes given the number on each shell. This uses CNLO algorithm in [1]

    Parameters
    ----------
    points_per_shell : List[int]
        Number of points on each shell
    initialization : Array, optional
        Optional initialization for CNLO algorithm, by default None, which will then use GEEM as initialization
    antipodal : bool, optional
        Whether or not to consider antipodal constraint, by default True
    delta : float, optional
        Maximum distance beween initialization and optimized result, by default 0.1
    w : float, optional
        Balance for indivial shell and combined shell, by default 0.5
    max_iter : int, optional
        Max round of iteration for SLSQP, by default 1000
    iprint : int, optional
        Whether or not to print message by SLSQP, by default 1

    Returns
    -------
    Array
        Set of points generated for each shell 

    Reference
    ---------
    1. Jian Cheng, Dinggang Shen, Pew-Thian Yap and Peter J. Basser, "Single- and Multiple-Shell Uniform Sampling Schemes for Diffusion MRI Using Spherical Codes," in IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 185-199
    """
    nb_points = np.sum(points_per_shell)
    nb_shells = len(points_per_shell)
    if initialization is None:
        initialization = initialize(points_per_shell, antipodal)
    upper_bound_spherical = covering_radius_upper_bound(
        2 * np.array(points_per_shell) if antipodal else np.array(points_per_shell)
    )
    upper_bound_spherical_all = covering_radius_upper_bound(
        2 * np.sum(points_per_shell) if antipodal else np.sum(points_per_shell)
    )
    indices = np.cumsum(points_per_shell).tolist()
    indices.insert(0, 0)
    transform = np.abs if antipodal else lambda x: x
    grad_transform = np.sign if antipodal else lambda _: 1
    eps = 1e-8

    init_theta = [
        covering_radius(initialization[indices[s] : indices[s + 1]], antipodal)
        for s in range(nb_shells)
    ]
    init_theta.append(covering_radius(initialization, antipodal))

    spherical_index = []
    for s, Ks in enumerate(points_per_shell):
        for i in range(Ks - 1):
            for j in range(i + 1, Ks):
                if transform(
                    np.dot(
                        initialization[indices[s] + i], initialization[indices[s] + j]
                    )
                ) >= np.cos(2 * delta + upper_bound_spherical[s]):
                    spherical_index.append((s, indices[s] + i, indices[s] + j))

    cross_spherical_index = []
    for s, Ks in enumerate(points_per_shell):
        for t, Kt in enumerate(points_per_shell[s + 1 :]):
            for i in range(Ks):
                for j in range(Kt):
                    if transform(
                        np.dot(
                            initialization[indices[s] + i],
                            initialization[indices[t + s + 1] + j],
                        )
                    ) >= np.cos(2 * delta + upper_bound_spherical_all):
                        cross_spherical_index.append(
                            (indices[s] + i, indices[t + s + 1] + j)
                        )

    args = (
        nb_shells,  # nb_shells
        nb_points,  # nb_points
        initialization,  # initialization
        delta,  # delta_0
        spherical_index,
        cross_spherical_index,
        w,
        eps,
        transform,
        grad_transform,
    )
    vects = np.concatenate([initialization.flatten(), np.array(init_theta)])

    vects = scopt.fmin_slsqp(
        cost,
        vects,
        fprime=grad_cost,
        f_eqcons=equality_constraints,
        fprime_eqcons=grad_equality_constraints,
        f_ieqcons=inequality_constraints,
        fprime_ieqcons=grad_inequality_constraints,
        iter=max_iter,
        acc=1.0e-5,
        args=args,
        iprint=iprint,
    )

    print(vects[-len(points_per_shell) - 1 :] * 180 / np.pi)

    return vects[: 3 * nb_points].reshape((nb_points, 3))
