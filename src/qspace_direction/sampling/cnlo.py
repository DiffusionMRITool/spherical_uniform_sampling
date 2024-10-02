import numpy as np
from scipy import optimize as scopt

from .geem import compute_weights
from .geem import optimize as geem_optimize


def initialize(points_per_shell, antipodal, iprint=0) -> np.ndarray:
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
        iprint=iprint,
    )

    return points


def covering_radius_upper_bound(num):
    if isinstance(num, int) and num < 3:
        return np.pi / 2
    upperBoundEuc = np.sqrt(4 - (1 / np.sin(np.pi * num / (6 * (num - 2)))) ** 2)
    ub = np.arccos((2 - upperBoundEuc**2) / 2)
    return ub


def covering_radius(vects: np.ndarray, antipodal=True):
    innerProductAll = np.abs(vects @ vects.T) if antipodal else vects @ vects.T
    return np.arccos((np.clip(np.max(np.triu(innerProductAll, 1)), -1, 1)))


def inequality_constraints_9b(vects, *args):
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
    N = args[1]
    init = args[2]
    delta = args[3]
    transform = args[8]
    points = vects[: 3 * N].reshape((N, 3))
    diff = (points * init).sum(1)
    return delta - np.arccos(np.clip(transform(diff), -1, 1))


def grad_inequality_constraints_9d(vects, *args):
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
    N = args[1]
    s = args[0]
    return vects[3 * N : 3 * N + s] - vects[-1]


def grad_inequality_constraints_9e(vects, *args):
    S = args[0]
    N = args[1]
    grad = np.zeros((S, N * 3 + S + 1))
    for i in range(S):
        grad[i, 3 * N + i] = 1
        grad[i, -1] = -1
    return grad


def equality_constraints(vects, *args):
    N = args[1]
    points = vects[: 3 * N].reshape((N, 3))
    return (points**2).sum(1) - 1.0


def grad_equality_constraints(vects, *args):
    s = args[0]
    N = args[1]
    points = vects[: 3 * N].reshape((N, 3))
    grad = np.zeros((N, N * 3 + s + 1))
    for i in range(N):
        grad[i, 3 * i : 3 * (i + 1)] = 2 * points[i]
    return grad


def inequality_constraints(vects, *args):
    return np.concatenate(
        (
            inequality_constraints_9b(vects, *args),
            inequality_constraints_9c(vects, *args),
            inequality_constraints_9d(vects, *args),
            inequality_constraints_9e(vects, *args),
        )
    )


def grad_inequality_constraints(vects, *args):
    return np.concatenate(
        (
            grad_inequality_constraints_9b(vects, *args),
            grad_inequality_constraints_9c(vects, *args),
            grad_inequality_constraints_9d(vects, *args),
            grad_inequality_constraints_9e(vects, *args),
        )
    )


def cost(vects, *args):
    s = args[0]
    N = args[1]
    w = args[6]
    return -(np.sum(vects[3 * N : 3 * N + s]) * w / s + (1 - w) * vects[-1])


def grad_cost(vects, *args):
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
    nb_points = np.sum(points_per_shell)
    nb_shells = len(points_per_shell)
    if initialization is None:
        initialization = initialize(points_per_shell, antipodal, iprint)
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
