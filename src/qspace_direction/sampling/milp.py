from itertools import combinations, product
from typing import List

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .cnlo import covering_radius_upper_bound


def identity(x):
    return x


def pdms(
    points: np.ndarray,
    points_per_shell: np.ndarray,
    lb=None,
    w: float = 0.5,
    antipodal=True,
    time_limit=600,
    output_flag=1,
):
    """_summary_

    Args:
        points (np.ndarray): _description_
        points_per_shell (np.ndarray): _description_
        lb (_type_, optional): _description_. Defaults to None.
        w (float, optional): _description_. Defaults to 0.5.
        antipodal (bool, optional): _description_. Defaults to True.
        time_limit (int, optional): _description_. Defaults to 600.
        output_flag (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
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


def pdmsEEM(
    points: np.ndarray,
    points_per_shell: np.ndarray,
    order: int = 1,
    w: float = 0.5,
    antipodal=True,
    time_limit=600,
    output_flag=1,
):
    N = len(points)
    S = len(points_per_shell)
    M = 2

    transform = np.abs if antipodal else identity
    dis = np.clip(transform(points @ points.T), -1, 1)
    f = lambda x: 1 / ((1 - x) ** order) + 1 / ((1 + x) ** order)
    M = max(f(dis[i, j]) for i, j in combinations(range(N), 2))

    m = gp.Model("PDMS")
    m.Params.timeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = output_flag

    h = m.addVars(N, S, vtype=GRB.BINARY, name="h")
    e_s = m.addVars(
        S,
        N,
        N,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="e_s",
    )
    e_0 = m.addVars(
        N,
        N,
        lb=0,
        vtype=GRB.CONTINUOUS,
        name="e_0",
    )

    m.addConstrs(
        (
            e_s[s, i, j] >= f(dis[i, j]) - M * (2 - h[i, s] - h[j, s])
            for s, (i, j) in product(range(S), combinations(range(N), 2))
        ),
        name="11b",
    )
    m.addConstrs(
        (
            e_0[s, i, j] >= f(dis[i, j]) - M * (2 - h[i, s] - h[j, t])
            for s, t, (i, j) in product(range(S), range(S), combinations(range(N), 2))
        ),
        name="11c",
    )
    m.addConstrs((h.sum("*", s) == points_per_shell[s] for s in range(S)), name="11e1")
    # m.addConstrs((h.sum(i, "*") <= 1 for i in range(N)), name="11e2")
    for i in range(N):
        m.addSOS(GRB.SOS_TYPE1, [h[i, s] for s in range(S)])

    m.setObjective(
        w
        * gp.quicksum(
            [
                1 / (points_per_shell[s] ** 2) * e_s[s, i, j]
                for s, (i, j) in product(range(S), combinations(range(N), 2))
            ]
        )
        + (1 - w)
        / (N**2)
        * gp.quicksum([e_0[i, j] for i, j in combinations(range(N), 2)]),
        GRB.MINIMIZE,
    )

    m.optimize()

    l = [[] for _ in range(S)]
    for i in range(N):
        for s in range(S):
            if m.getVarByName(f"h[{i},{s}]").X == 1:
                l[s].append(points[i])

    return l


def pdss(
    points: np.ndarray,
    K: int,
    lb=None,
    antipodal=True,
    time_limit=600,
    output_flag=1,
):
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


def pdssEEM(
    points: np.ndarray,
    K: int,
    order=1,
    antipodal=True,
    time_limit=600,
    output_flag=1,
):
    N = len(points)
    M = 2
    transform = np.abs if antipodal else identity
    dis = np.clip(transform(points @ points.T), -1, 1)
    f = lambda x: 1 / ((1 - x) ** order) + 1 / ((1 + x) ** order)
    M = max(f(dis[i, j]) for i, j in combinations(range(N), 2))

    m = gp.Model("PDSS")
    m.Params.timeLimit = time_limit
    m.Params.MIPFocus = 1
    m.Params.OutputFlag = output_flag

    h = m.addVars(N, vtype=GRB.BINARY, name="h")
    e = m.addVars(N, N, vtype=GRB.CONTINUOUS, lb=0, name="e")

    m.addConstrs(
        (
            e[i, j] >= f(dis[i, j]) - M * (2 - h[i] - h[j])
            for i, j in combinations(range(N), 2)
        ),
    )
    m.addConstr(h.sum() == K, name="10d")

    m.setObjective(
        gp.quicksum([e[i, j] for i, j in combinations(range(N), 2)]), GRB.MINIMIZE
    )

    m.optimize()

    l = []
    for i in range(N):
        if m.getVarByName(f"h[{i}]").X == 1:
            l.append(points[i])

    return l


def pdmm(
    points: List[np.ndarray],
    points_per_shell: np.ndarray,
    lb=None,
    w: float = 0.5,
    antipodal: bool = True,
    time_limit: float = 600,
    output_flag: int = 1,
):
    """_summary_

    Args:
        points (List[np.ndarray]): _description_
        points_per_shell (np.ndarray): _description_
        lb (_type_, optional): _description_. Defaults to None.
        w (float, optional): _description_. Defaults to 0.5.
        antipodal (bool, optional): _description_. Defaults to True.
        time_limit (float, optional): _description_. Defaults to 600.
        output_flag (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
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
