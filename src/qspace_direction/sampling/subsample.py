from itertools import combinations, product
from typing import List

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .cnlo import covering_radius_upper_bound


def identity(x):
    """identity funtion

    Args:
        x (Any): Any

    Returns:
        x: Any
    """
    return x


def multiple_subset_from_single_set(
    points: np.ndarray,
    points_per_shell: np.ndarray,
    lb=None,
    w: float = 0.5,
    antipodal=True,
    time_limit=600,
    output_flag=1,
):
    """Given a single set of points, uniformly select multiple sets of samples from it

    Args:
        points (np.ndarray): A single set of points to choose from
        points_per_shell (np.ndarray): Number of sample to select for each shell
        lb (np.ndarray, optional): Lower bound for covering radius of each shell. This is used to help GUROBI to generate better result. Defaults to None.
        w (float, optional): Balance for indivial shell and combined shell. Defaults to 0.5.
        antipodal (bool, optional): Whether or not to consider antipodal constraint. Defaults to True.
        time_limit (int, optional): Time limit for GUROBI to run. Defaults to 600.
        output_flag (int, optional): GUROBI output flag. Defaults to 1.

    Returns:
        List: Set of points chosen for each shell
    """
    N = len(points)
    S = len(points_per_shell)
    M = 2

    transform = np.abs if antipodal else identity
    dis = np.arccos(np.clip(transform(points @ points.T), -1, 1))
    lb = lb if lb else np.zeros(S + 1)

    m = gp.Model("PDMS")
    m.Params.timeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = output_flag

    h = m.addVars(N, S, vtype=GRB.BINARY, name="h")
    theta_s = m.addVars(
        S,
        lb=lb[:S],
        ub=covering_radius_upper_bound(2 * points_per_shell),
        vtype=GRB.CONTINUOUS,
        name="theta_s",
    )
    theta_0 = m.addVar(
        lb=lb[-1],
        ub=covering_radius_upper_bound(2 * np.sum(points_per_shell)),
        vtype=GRB.CONTINUOUS,
        name="theta_0",
    )

    m.addConstrs(
        (
            theta_s[s] - M * (2 - h[i, s] - h[j, s]) <= dis[i][j]
            for s, (i, j) in product(range(S), combinations(range(N), 2))
        ),
        name="11b",
    )
    m.addConstrs(
        (
            theta_0 - M * (2 - h[i, s] - h[j, t]) <= dis[i][j]
            for s, t, (i, j) in product(range(S), range(S), combinations(range(N), 2))
        ),
        name="11c",
    )
    m.addConstrs((h.sum("*", s) == points_per_shell[s] for s in range(S)), name="11e1")
    # m.addConstrs((h.sum(i, "*") <= 1 for i in range(N)), name="11e2")
    for i in range(N):
        m.addSOS(GRB.SOS_TYPE1, [h[i, s] for s in range(S)])

    m.setObjective(w / S * theta_s.sum() + (1 - w) * theta_0, GRB.MAXIMIZE)

    m.optimize()

    l = [[] for _ in range(S)]
    for i in range(N):
        for s in range(S):
            if m.getVarByName(f"h[{i},{s}]").X == 1:
                l[s].append(points[i])

    return l


def single_subset_from_single_set(
    points: np.ndarray,
    K: int,
    lb=None,
    antipodal=True,
    time_limit=600,
    output_flag=1,
):
    """Given a single set of points, uniformly select K samples from the given N samples


    Args:
        points (np.ndarray): A single set of points to choose from
        K (int): Number of points to choose from
        lb (float, optional): Lower bound for covering radius of each shell. This is used to help GUROBI to generate better result. Defaults to None.
        antipodal (bool, optional): Whether or not to consider antipodal constraint. Defaults to True.
        time_limit (int, optional): Time limit for GUROBI to run. Defaults to 600.
        output_flag (int, optional): GUROBI output flag. Defaults to 1.

    Returns:
        Array: Array shaped (K, 3), the chosen K points
    """
    N = len(points)
    M = 2
    transform = np.abs if antipodal else identity
    dis = np.arccos(np.clip(transform(points @ points.T), -1, 1))
    lb = lb if lb else 0

    m = gp.Model("PDSS")
    m.Params.timeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = output_flag

    h = m.addVars(N, vtype=GRB.BINARY, name="h")
    theta = m.addVar(
        lb=lb, ub=covering_radius_upper_bound(2 * K), vtype=GRB.CONTINUOUS, name="theta"
    )

    m.addConstrs(
        (
            theta - M * (2 - h[i] - h[j]) <= dis[i][j]
            for i, j in combinations(range(N), 2)
        ),
        name="10b",
    )
    m.addConstr(h.sum() == K, name="10d")

    m.setObjective(theta, GRB.MAXIMIZE)

    m.optimize()

    l = []
    for i in range(N):
        if m.getVarByName(f"h[{i}]").X == 1:
            l.append(points[i])

    return l


def multiple_subset_from_multiple_set(
    points: List[np.ndarray],
    points_per_shell: np.ndarray,
    lb=None,
    w: float = 0.5,
    antipodal: bool = True,
    time_limit: float = 600,
    output_flag: int = 1,
):
    """Given multiple sets of points, uniformly select K_s points from the N_s samples for the s-th shell

    Args:
        points (List[np.ndarray]): Multiple sets of points to choose from
        points_per_shell (np.ndarray): Number of sample to select for each shell
        lb (np.ndarray, optional): Lower bound for covering radius of each shell. This is used to help GUROBI to generate better result. Defaults to None.
        w (float, optional): Balance for indivial shell and combined shell. Defaults to 0.5.
        antipodal (bool, optional): Whether or not to consider antipodal constraint. Defaults to True.
        time_limit (int, optional): Time limit for GUROBI to run. Defaults to 600.
        output_flag (int, optional): GUROBI output flag. Defaults to 1.

    Returns:
        List: Set of points chosen for each shell
    """
    N_s = [len(l) for l in points]
    S = len(points_per_shell)
    M = 2

    points = np.concatenate(points)
    transform = np.abs if antipodal else identity
    dis = np.arccos(np.clip(transform(points @ points.T), -1, 1))
    lb = lb if lb else np.zeros(S + 1)
    indices = np.cumsum(N_s).tolist()
    indices.insert(0, 0)

    m = gp.Model("PDMM")
    m.Params.timeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = output_flag

    h = m.addVars(
        [(i, s) for s in range(S) for i in range(N_s[s])],
        vtype=GRB.BINARY,
        name="h",
    )
    theta_s = m.addVars(
        S,
        lb=lb[:S],
        ub=covering_radius_upper_bound(2 * points_per_shell),
        vtype=GRB.CONTINUOUS,
        name="theta_s",
    )
    theta_0 = m.addVar(
        lb=lb[-1],
        ub=covering_radius_upper_bound(2 * np.sum(points_per_shell)),
        vtype=GRB.CONTINUOUS,
        name="theta_0",
    )

    for s in range(S):
        m.addConstrs(
            (
                theta_s[s] - M * (2 - h[i, s] - h[j, s])
                <= dis[indices[s] + i][indices[s] + j]
                for i, j in combinations(range(N_s[s]), 2)
            ),
            name="12a",
        )
    for s, t in combinations(range(S), 2):
        m.addConstrs(
            (
                theta_0 - M * (2 - h[i, s] - h[j, t])
                <= dis[indices[s] + i][indices[t] + j]
                for i, j in product(range(N_s[s]), range(N_s[t]))
            ),
            name="12b",
        )
    m.addConstrs((h.sum("*", s) == points_per_shell[s] for s in range(S)), name="11e")

    m.setObjective(w / S * theta_s.sum() + (1 - w) * theta_0, GRB.MAXIMIZE)

    m.optimize()

    l = [[] for _ in range(S)]
    for s in range(S):
        for i in range(N_s[s]):
            if m.getVarByName(f"h[{i},{s}]").X == 1:
                l[s].append(points[indices[s] + i])

    return l
